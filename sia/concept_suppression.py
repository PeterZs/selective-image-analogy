"""Concept Suppression Vector (CSV): a difference-of-means direction in the text-embedding space.

Per edit (paper Sec. 4.2, eq. 7; after Diffusion Sliders, Ekin & Gandelsman, arXiv:2603.17998):

1. the backbone's own text encoder, used as a causal LM, writes N contrastive sentence pairs
   ``{pos_style, neg_style, pos, neg}`` (``generate_contrast_pairs``); if too few parse, generic
   scaffold pairs pad the set;
2. each sentence is encoded exactly as at inference and mean-pooled over the rows of its style
   identifier (``steering_pool``);
3. ``v = normalize(mean(pos) - mean(neg))`` is cached on disk with the pairs.

At inference ``apply_csv`` removes (``ablate``) or pushes against (``add``) that direction at the
suppressed edits' token rows of ``prompt_embeds``.

This module holds everything that does not depend on the model. The backbone module
(``sia.backbones.flux2_klein`` or ``sia.backbones.qwen_image21``) is passed in by the caller and
supplies ``generate_contrast_pairs``, ``steering_pool`` and ``cache_key``; nothing here imports a
backbone, so each environment only ever loads its own.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch

# Bump whenever the direction-building logic changes materially (fallback pairs, pooling, DoM
# formula, ...). A cached direction with an older version is rebuilt on the next load.
_CACHE_VERSION = 2


def default_cache_dir(backbone) -> Path:
    """``~/.cache/sia/csv/<backbone>``: where directions are cached unless a directory is given."""
    return Path.home() / ".cache" / "sia" / "csv" / backbone.__name__.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# Contrastive pairs
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"pos_style", "neg_style", "pos", "neg"}


def _build_generation_prompt(edit: str, n_pairs: int) -> str:
    """The paper's unbiased-dataset system prompt, lightly adapted for local LLMs.

    Source: diffusion-sliders/dataset/generate.py::build_prompt and
    diffusion-sliders/system_prompts/unbiased_dataset_generation.txt.
    """
    return f"""You are an advanced data generation assistant.

Your task is to create a contrastive dataset of {n_pairs} examples for \
computing a steering vector.

The steering concept to focus on is: {edit}

Output exactly {n_pairs} JSON objects, one per line (JSON Lines), with no list \
brackets, no extra commentary, and no markdown. Each line must be:
{{"pos_style": "<positive identifier>", "neg_style": "<negative identifier>", \
"pos": "<positive full sentence>", "neg": "<negative full sentence>"}}

Rules (STRICT):
- PARALLELISM: "pos" and "neg" MUST share the same syntactic skeleton, subject, \
setting and composition. Only the identifier tokens differ.
- MINIMAL DELTA: The only difference between "pos" and "neg" is the tokens that \
express the concept contrast.
- STYLE NEUTRALITY: Do NOT change rendering domain, lighting, camera, or layout.
- IDENTIFIERS: Use the SAME "pos_style" and "neg_style" identifiers for all \
{n_pairs} lines, and the identifiers MUST appear verbatim in the corresponding \
sentences.
- DIVERSITY: Vary the subject, setting, viewpoint and wording across the \
{n_pairs} lines so the steering direction generalises.
- Use neutral terms for people ("person", "figure"); avoid age, gender, race \
unless the concept itself requires them.

Example (concept: "bright vs dark"):
{{"pos_style": "bright", "neg_style": "dark", "pos": "A bright living room \
with large windows.", "neg": "A dark living room with large windows."}}

Now generate EXACTLY {n_pairs} JSONL lines for the concept \"{edit}\" following \
these instructions. Do not number the lines. Do not wrap them in code fences.
"""


def _parse_jsonl_records(text: str) -> List[dict]:
    """Parse the LLM reply into pair records, robust to code fences, commentary and trailing
    commas. A record is kept only if each style identifier occurs in its sentence
    (case-insensitive)."""
    records: List[dict] = []
    # Drop markdown code-fence wrappers if present
    text = re.sub(r"^```[a-zA-Z0-9]*\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Tolerate trailing commas
        line = line.rstrip(",").strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not REQUIRED_KEYS.issubset(obj.keys()):
            continue
        # Validate identifiers appear in sentences (case-insensitive). Skip otherwise.
        if obj["pos_style"].lower() not in obj["pos"].lower():
            continue
        if obj["neg_style"].lower() not in obj["neg"].lower():
            continue
        records.append({k: obj[k] for k in ("pos_style", "neg_style", "pos", "neg")})
    return records


_FALLBACK_SCAFFOLDS = [
    "A photo of a {X}.",
    "A detailed close-up of a {X}.",
    "A cinematic portrait showing a {X}.",
    "An outdoor scene featuring a {X}.",
    "A studio photograph of a {X}.",
    "A high-resolution image of a {X}.",
    "An editorial shot of a {X}.",
    "A candid photograph of a {X}.",
]


_NEG_STYLE_VARIANTS = [
    "unchanged", "plain", "original", "unmodified",
    "standard", "normal", "as-is", "unaltered",
]


def _fallback_pairs(edit: str, n_pairs: int) -> List[dict]:
    """Last-resort: build trivial parallel pairs by scaffolding the raw edit
    against a generic "no edit" control. Used only if the LLM's JSONL output is
    unparseable; produces a much weaker steering signal than the full LLM
    pipeline but keeps the run going.

    neg_style cycles through several synonyms for "no change" so that the
    negative embeddings have non-zero variance across pairs — a fixed
    neg_style="unchanged" in every sentence collapses the negative mean to a
    single point and produces a degenerate DoM direction.
    """
    pos_style = edit
    out = []
    i = 0
    while len(out) < n_pairs:
        scaffold = _FALLBACK_SCAFFOLDS[i % len(_FALLBACK_SCAFFOLDS)]
        neg_style = _NEG_STYLE_VARIANTS[i % len(_NEG_STYLE_VARIANTS)]
        pos = scaffold.format(X=f"subject showing {pos_style}")
        neg = scaffold.format(X=f"subject, {neg_style}")
        out.append({"pos_style": pos_style, "neg_style": neg_style,
                    "pos": pos, "neg": neg})
        i += 1
    return out


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------------------------------------------------------------------------
# Difference of means
# ---------------------------------------------------------------------------

def _unit_dom(pos_vecs, neg_vecs):
    diff = torch.stack(pos_vecs).float().mean(0) - torch.stack(neg_vecs).float().mean(0)
    pre_norm = float(diff.norm())
    if pre_norm < 1e-8:
        raise RuntimeError(f"DoM direction has near-zero norm ({pre_norm:.2e}); pairs may be degenerate.")
    return diff / pre_norm, pre_norm


@torch.no_grad()
def compute_dom_vector(backbone, pairs: Sequence[dict], pipe, device) -> Tuple[torch.Tensor, dict]:
    """Unit-norm difference of means over ``{pos, neg, pos_style, neg_style}`` records.

    Each sentence is pooled with ``backbone.steering_pool``; a pair whose identifier cannot be
    located is skipped. Returns ``(v, stats)`` with ``v`` on ``device`` and
    ``stats = {"n_usable", "pre_norm"}`` (``pre_norm`` = |mean(pos) - mean(neg)|, the gap
    ``csv_k`` is measured in).
    """
    pos_vecs, neg_vecs = [], []
    for rec in pairs:
        p = backbone.steering_pool(rec["pos"], rec["pos_style"], pipe, device)
        n = backbone.steering_pool(rec["neg"], rec["neg_style"], pipe, device)
        if p is not None and n is not None:
            pos_vecs.append(p)
            neg_vecs.append(n)
    if not pos_vecs:
        raise RuntimeError("compute_dom_vector: no pooled embeddings could be computed.")
    v_unit, pre_norm = _unit_dom(pos_vecs, neg_vecs)
    return v_unit.to(device), {"n_usable": len(pos_vecs), "pre_norm": pre_norm}


@torch.no_grad()
def compute_incontext_dom_vector(
    backbone, full_prompt: str, edit_text: str, all_edits: List[str], edit_index: int,
    pipe, device, build_prompt_fn: Callable[[List[str], List[str]], str], vl_images: Optional[list] = None,
) -> Tuple[torch.Tensor, dict]:
    """Difference of means inside the actual inference input.

    In a causal text encoder every hidden state depends on the preceding tokens, so a direction
    taken from short isolated sentences need not transfer to the long analogy prompt. Here the
    positive is the full prompt (with ``vl_images`` as condition images, for an encoder that sees
    them) pooled at ``edit_text``; the negatives are the same prompt with the edit replaced by
    each neutral word in ``_NEG_STYLE_VARIANTS``, pooled at that word.
    """
    pos = backbone.steering_pool(full_prompt, edit_text, pipe, device, vl_images)
    if pos is None:
        raise RuntimeError(f"Could not locate edit text {edit_text!r} in the full prompt.")
    neg_vecs = []
    for neg_word in _NEG_STYLE_VARIANTS:
        edits = list(all_edits)
        edits[edit_index] = neg_word
        neg = backbone.steering_pool(build_prompt_fn(edits, edits), neg_word, pipe, device, vl_images)
        if neg is not None:
            neg_vecs.append(neg)
    if not neg_vecs:
        raise RuntimeError("No neutral replacement could be located in the modified prompts.")
    v_unit, pre_norm = _unit_dom([pos], neg_vecs)
    return v_unit.to(device), {"n_usable": 1 + len(neg_vecs), "pre_norm": pre_norm, "neg_count": len(neg_vecs)}


# ---------------------------------------------------------------------------
# Cached directions
# ---------------------------------------------------------------------------

def get_or_build_steering_vector(
    backbone, edit: str, n_pairs: int, pipe, device, cache_dir=None, regen: bool = False, *,
    incontext_prompt: Optional[str] = None, all_edits: Optional[List[str]] = None,
    edit_index: Optional[int] = None, build_prompt_fn: Optional[Callable] = None,
    vl_images: Optional[list] = None,
) -> Tuple[torch.Tensor, dict]:
    """Return the CSV direction of ``edit`` as ``(v_unit, meta)``, from the cache or built now.

    Args:
        backbone: the backbone module (supplies ``cache_key``, ``generate_contrast_pairs`` and
            ``steering_pool``).
        edit: the edit text.
        n_pairs: contrastive pairs to generate when the direction is not cached (paper: 64).
        pipe: the loaded pipeline; only its text encoder and tokenizer/processor are used.
        device: device of the returned vector.
        cache_dir: cache root; ``None`` means ``default_cache_dir(backbone)``. Each direction is a
            folder ``<cache_dir>/<backbone.cache_key(...)>/`` holding ``vector.pt``, ``meta.json``
            and (isolated mode) ``pairs.jsonl``.
        regen: rebuild even if cached.
        incontext_prompt, all_edits, edit_index, build_prompt_fn, vl_images: when the first four
            are given, the direction is computed in context (``compute_incontext_dom_vector``) and
            the cache key also covers the prompt (and, for Qwen-Image-2.1, the images).

    New contrast pairs are sampled (without a fixed seed) only when the direction is not cached.
    ``meta["pre_normalization_norm"]`` is the mean-pole gap used by the additive form.
    """
    use_incontext = None not in (incontext_prompt, all_edits, edit_index, build_prompt_fn)
    cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir(backbone)
    edit_dir = cache_dir / backbone.cache_key(edit, incontext_prompt if use_incontext else None,
                                              vl_images if use_incontext else None)
    vec_path, meta_path = edit_dir / "vector.pt", edit_dir / "meta.json"

    if not regen and vec_path.exists():
        meta = {"source": "cache", "vec_path": str(vec_path)}
        if meta_path.exists():
            meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
        if meta.get("cache_version", 0) >= _CACHE_VERSION:
            # The cache key does not include the pair count, so a direction built with another
            # count is reused as is (use regen to rebuild it).
            cached_pairs = meta.get("n_requested")
            if cached_pairs is not None and cached_pairs != n_pairs:
                print(f"[Steering DoM] Warning: cached direction for edit {edit!r} was built from "
                      f"{cached_pairs} pairs, not the requested {n_pairs}; using the cached one.")
            return torch.load(vec_path, map_location=device).to(device=device, dtype=torch.float32), meta
        print(f"[Steering DoM] Stale cache for edit {edit!r} — regenerating …")

    edit_dir.mkdir(parents=True, exist_ok=True)
    if use_incontext:
        print(f"[Steering DoM] In-context DoM for edit {edit_index + 1} ({edit!r}) …")
        v_unit, stats = compute_incontext_dom_vector(
            backbone, incontext_prompt, edit, all_edits, edit_index, pipe, device, build_prompt_fn, vl_images)
        meta = {"edit": edit, "method": "incontext", "source": "incontext", "n_neg_variants": stats["neg_count"]}
    else:
        print(f"[Steering DoM] Generating {n_pairs} contrastive pairs for edit: {edit!r}")
        records, n_requested = backbone.generate_contrast_pairs(edit, pipe, n_pairs, device)
        n_parsed = len(records)
        used_fallback = n_parsed < max(4, n_pairs // 2)
        if used_fallback:
            print(f"[Steering DoM] Parse yield {n_parsed}/{n_requested} too low — padding with scaffold fallback.")
            seen = {(r["pos"], r["neg"]) for r in records}
            for fb in _fallback_pairs(edit, n_pairs - n_parsed):
                if (fb["pos"], fb["neg"]) not in seen:
                    records.append(fb)
                    seen.add((fb["pos"], fb["neg"]))
        v_unit, stats = compute_dom_vector(backbone, records, pipe, device)
        _write_jsonl(edit_dir / "pairs.jsonl", records)
        meta = {"edit": edit, "method": "qwen_isolated", "source": "generated",
                "n_requested": n_requested, "n_parsed_from_llm": n_parsed, "n_records_saved": len(records),
                "used_fallback_padding": used_fallback}

    torch.save(v_unit.detach().cpu().float(), vec_path)
    meta.update({"cache_version": _CACHE_VERSION,
                 "n_usable_after_pooling": stats["n_usable"],
                 "pre_normalization_norm": stats["pre_norm"]})
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Steering DoM] Cached → {edit_dir}")
    return v_unit.to(device=device, dtype=torch.float32), meta


def build_steering_vector_from_pairs(backbone, edit: str, pairs_path, pipe, device,
                                     cache_dir) -> Tuple[torch.Tensor, dict]:
    """Build and cache the CSV direction of ``edit`` from existing contrast pairs.

    ``pairs_path`` is a ``pairs.jsonl`` of another cache (typically the Qwen-Image-2.1 one); its
    sentences are pooled with ``backbone``'s own encoder. Two backbones then share the same
    (unseeded) pairs, so their directions differ only by the encoder. The result is written to
    ``<cache_dir>/<backbone.cache_key(edit)>/`` like any cached direction, so later runs read it
    through ``get_or_build_steering_vector``. Returns ``(v_unit, meta)``.
    """
    records = _read_jsonl(Path(pairs_path))
    v_unit, stats = compute_dom_vector(backbone, records, pipe, device)
    edit_dir = Path(cache_dir).expanduser() / backbone.cache_key(edit)
    edit_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(edit_dir / "pairs.jsonl", records)
    torch.save(v_unit.detach().cpu().float(), edit_dir / "vector.pt")
    meta = {"edit": edit, "method": "reused_pairs", "cache_version": _CACHE_VERSION,
            "pairs_from": str(pairs_path), "n_records_saved": len(records), "n_usable_after_pooling": stats["n_usable"],
            "pre_normalization_norm": stats["pre_norm"], "source": "built"}
    (edit_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return v_unit, meta


def reusable_pairs(cache_dir) -> dict:
    """``{edit text: pairs.jsonl path}`` for every direction in ``cache_dir`` that kept its pairs."""
    pairs = {}
    for meta_path in sorted(Path(cache_dir).expanduser().glob("*/meta.json")):
        if (meta_path.parent / "pairs.jsonl").exists():
            pairs[json.loads(meta_path.read_text())["edit"]] = meta_path.parent / "pairs.jsonl"
    return pairs


# ---------------------------------------------------------------------------
# Applying the direction to prompt embeddings
# ---------------------------------------------------------------------------

STEERING_SCOPES = ("tokens", "prompt")
STEERING_MODES  = ("add", "ablate")


def apply_csv(
    prompt_embeds: torch.Tensor,
    edit_positions: list,
    steering_vecs: list,
    alpha_per_edit: list,
    scope: str = "tokens",
    all_prompt_positions: list = None,
    mode: str = "add",
    max_push_norm_ratio: float = 0.0,
    ablate_scale: float = 1.0,
) -> torch.Tensor:
    """Return a steered copy of ``prompt_embeds`` using the chosen steering
    mode.

    scope
    -----
      - ``"tokens"`` : apply only at the suppressed edit's own token positions
        (``edit_positions[i]``). Same footprint as the ExLoRA mask.
      - ``"prompt"`` : apply at every non-padding token position in the prompt
        (``all_prompt_positions``).

    mode
    ----
      - ``"add"`` (default):
            x' = x + alpha * v̂
        Pure additive push. Requires a large ``alpha`` to overcome whatever
        component of the concept is already encoded in x.
      - ``"ablate"`` (directional ablation + optional push):
            x' = x - (x · v̂) v̂ + alpha * v̂
        First projects out the component of x along v̂ — i.e. *deletes* the
        concept from that 1-D subspace regardless of its magnitude — then
        adds a controlled anti-push of magnitude ``alpha``. With alpha = 0
        this is pure concept removal (Arditi et al.-style ablation). Usually
        dramatically stronger than ``"add"`` for the same ``|alpha|``.

    alpha_per_edit semantics
    ------------------------
    ``alpha_per_edit[i]`` controls edit ``i``:
      - ``None``  : skip this edit entirely (no push, no ablation). Use this
        for edits that are NOT being suppressed in the current sweep row.
      - ``0.0``   : no additive push. In ``mode="ablate"`` this still
        performs pure concept removal; in ``mode="add"`` it is a no-op.
      - any other float: scaled push / push-after-ablation magnitude.

    Important: passing ``alpha=0`` for a non-suppressed edit in ``"ablate"``
    mode would *still* project its concept out of the prompt. Use ``None``
    for edits you do not want to touch.

    max_push_norm_ratio
    -------------------
    When > 0, clamps the push magnitude so it never exceeds
    ``max_push_norm_ratio * ‖x_t‖`` (the norm of the embedding at each
    token position *before* any modification). This prevents extreme
    out-of-distribution embeddings that arise when ``|alpha|`` far exceeds
    the embedding norm — a common cause of paradoxical concept
    amplification.
    Set to 0 (default) to disable clamping.

    ablate_scale
    ------------
    ``lambda`` of the paper's eq. (CSV): ``x' = x - lambda (x · v̂) v̂``. The
    default 1.0 is the orthogonal projection (pure removal of the component);
    larger values over-subtract it, which is how the paper's strengths
    {75, ..., 300} act. Only used in ``mode="ablate"``.

    v̂ must be unit-norm (which is what ``compute_dom_vector`` already
    produces). ``alpha`` here is the pre-multiplied value
    ``alpha_i * embedding_scale_factor`` coming from the caller.

    When multiple edits are non-None (e.g. scope=="prompt" and multiple
    suppressed edits), each (ablate + push) is applied in sequence. If you
    want the edit vectors to be independent, Gram-Schmidt them upstream —
    this function treats each ``v_i`` as-is.
    """
    if scope not in STEERING_SCOPES:
        raise ValueError(f"Unknown steering scope {scope!r}; "
                         f"expected one of {STEERING_SCOPES}.")
    if mode not in STEERING_MODES:
        raise ValueError(f"Unknown steering mode {mode!r}; "
                         f"expected one of {STEERING_MODES}.")

    effective = [a for a in alpha_per_edit if a is not None]
    if not effective:
        return prompt_embeds
    # Additive mode with all zero alphas is a no-op. Ablation mode is
    # meaningful even at alpha==0 (pure concept removal), so don't skip.
    if mode == "add" and not any(abs(a) > 0 for a in effective):
        return prompt_embeds
    if not any(v is not None for v in steering_vecs):
        return prompt_embeds

    steered = prompt_embeds.clone()
    for positions, v, alpha in zip(edit_positions, steering_vecs, alpha_per_edit):
        if v is None or alpha is None:
            continue
        if mode == "add" and alpha == 0:
            continue  # nothing to add

        if scope == "prompt":
            target_positions = all_prompt_positions if all_prompt_positions else positions
        else:
            target_positions = positions
        if not target_positions:
            continue

        v_hat = v.to(dtype=steered.dtype)
        raw_push = (float(alpha) * v_hat) if alpha != 0 else None
        for t in target_positions:
            if t >= steered.shape[1]:
                continue
            x_t = steered[0, t, :]
            if max_push_norm_ratio > 0 and raw_push is not None:
                x_norm = x_t.norm().item()
                max_mag = max_push_norm_ratio * max(x_norm, 1e-6)
                push_mag = abs(float(alpha))
                if push_mag > max_mag:
                    push = (float(alpha) * max_mag / push_mag) * v_hat
                else:
                    push = raw_push
            else:
                push = raw_push
            if mode == "ablate":
                proj_scalar = torch.dot(x_t, v_hat)
                x_t = x_t - ablate_scale * proj_scalar * v_hat
            if push is not None:
                x_t = x_t + push
            steered[0, t, :] = x_t
    return steered


def parse_embedding_schedule(raw):
    """Normalize an ``embedding_schedule`` spec into a sorted list of
    ``(step_boundary, alpha)`` tuples.

    Accepts any of:
      - ``None`` / empty -> returns ``None``
      - list of ``"boundary:alpha"`` strings  (CLI form)
      - list of ``[boundary, alpha]`` pairs    (YAML form)
      - list of ``{"until_step": b, "alpha": a}`` dicts (YAML form)

    Semantics: for denoising step index ``t`` (0-indexed), the first entry with
    ``t < boundary`` wins — i.e. piecewise-constant, left-closed / right-open.
    """
    if raw is None:
        return None
    try:
        items = list(raw)
    except TypeError:
        return None
    if not items:
        return None
    parsed = []
    for it in items:
        if isinstance(it, str):
            s = it.strip()
            if not s:
                continue
            if ":" not in s:
                raise ValueError(
                    f"embedding_schedule entry {it!r} must be 'step_boundary:alpha'")
            a, b = s.split(":", 1)
            parsed.append((int(a), float(b)))
        elif isinstance(it, dict) or hasattr(it, "keys"):
            boundary = it.get("until_step", it.get("step", it.get("boundary")))
            alpha = it.get("alpha", it.get("value", it.get("weight")))
            if boundary is None or alpha is None:
                raise ValueError(
                    f"embedding_schedule entry {dict(it)!r} must have "
                    "keys 'until_step' and 'alpha'")
            parsed.append((int(boundary), float(alpha)))
        else:
            seq = list(it)
            if len(seq) != 2:
                raise ValueError(
                    f"embedding_schedule entry {it!r} must be a pair "
                    "[step_boundary, alpha]")
            parsed.append((int(seq[0]), float(seq[1])))
    if not parsed:
        return None
    parsed.sort(key=lambda x: x[0])
    return parsed


def alpha_at_step(schedule, step_idx: int) -> float:
    """Piecewise-constant lookup for the schedule."""
    for boundary, alpha in schedule:
        if step_idx < boundary:
            return alpha
    return schedule[-1][1]


def format_schedule_tag(schedule) -> str:
    """Short filename-safe descriptor for a schedule."""
    parts = []
    for boundary, alpha in schedule:
        tag_a = f"{alpha:+.1f}".replace(".", "p").replace("+", "")
        parts.append(f"{boundary}a{tag_a}")
    return "sched_" + "_".join(parts)
