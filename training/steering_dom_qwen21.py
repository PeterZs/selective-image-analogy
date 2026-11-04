#!/usr/bin/env python3
"""
steering_dom_qwen21.py

Concept Suppression Vector (difference-of-means direction) for Qwen-Image-2.1.

Same procedure as steering_dom.py (paper §4.2, eq. 7), with the encoder path
of QwenImage21Pipeline:

  1. The pipeline's text encoder (Qwen3VLForConditionalGeneration, which
     keeps its lm_head) generates N contrastive pos/neg sentence pairs, using
     the generation prompt / parser / fallback from steering_dom.py.
  2. Each sentence is encoded with ``pipe._get_qwen_prompt_embeds`` — the
     exact embedding path of the pipeline (last decoder layer before the final
     norm, system prompt dropped) — and pooled at the ``pos_style`` /
     ``neg_style`` rows located by ``qwen21_find_substring_token_indices``.
  3. v = normalize(mean(pos) - mean(neg)), cached on disk.

The vector has the encoder's hidden size (4096) and is applied to rows of the
``prompt_embeds`` that the transformer receives.

In-context mode encodes the full analogy prompt *with* the condition images
(as at inference) and contrasts the edit against neutral replacements. Since
Qwen3-VL text states depend on the images, that cache key includes them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from training.steering_dom import (
    _CACHE_VERSION,
    _NEG_STYLE_VARIANTS,
    _build_generation_prompt,
    _fallback_pairs,
    _parse_jsonl_records,
    _write_jsonl,
)
from sia.backbones.qwen_image21 import qwen21_find_substring_token_indices


@torch.no_grad()
def generate_pairs_with_qwen21(edit: str, pipe, n_pairs: int, device,
                               max_new_tokens_per_pair: int = 90) -> Tuple[List[dict], int]:
    """Ask the pipeline's Qwen3-VL text encoder for ``n_pairs`` contrastive
    pos/neg sentences about ``edit`` (text only, its own chat template)."""
    tokenizer = pipe.processor.tokenizer
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": _build_generation_prompt(edit, n_pairs)}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_out = pipe.text_encoder.generate(
        **inputs,
        max_new_tokens=max(256, n_pairs * max_new_tokens_per_pair),
        do_sample=True, temperature=1.0, top_p=0.95,
        pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id, use_cache=True,
    )
    reply = tokenizer.decode(gen_out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_jsonl_records(reply), n_pairs


@torch.no_grad()
def pool_edit_embedding_qwen21(sentence: str, style: str, pipe, device,
                               vl_images: Optional[list] = None) -> Optional[torch.Tensor]:
    """Encode ``sentence`` (with ``vl_images`` as condition images, or text
    only) and mean-pool ``prompt_embeds`` over the rows covering ``style``.
    Case-insensitive fallback, since the pair parser checks containment with
    ``.lower()``. Returns ``None`` if ``style`` cannot be located."""
    sizes = [img.size for img in vl_images] if vl_images else []
    positions = None
    for case_sensitive in (True, False):
        try:
            positions = qwen21_find_substring_token_indices(
                pipe, sentence, style, sizes, case_sensitive=case_sensitive)
            break
        except AssertionError:
            continue
    if not positions:
        return None
    prompt_embeds, _, _ = pipe._get_qwen_prompt_embeds([sentence], vl_images or None, device)
    return prompt_embeds[0, positions, :].float().mean(dim=0).cpu()


def _unit_dom(pos_vecs, neg_vecs):
    diff = torch.stack(pos_vecs).float().mean(0) - torch.stack(neg_vecs).float().mean(0)
    pre_norm = float(diff.norm())
    if pre_norm < 1e-8:
        raise RuntimeError(f"DoM direction has near-zero norm ({pre_norm:.2e}); pairs may be degenerate.")
    return diff / pre_norm, pre_norm


@torch.no_grad()
def compute_dom_vector_qwen21(pairs: Sequence[dict], pipe, device) -> Tuple[torch.Tensor, dict]:
    """Unit-norm difference-of-means over ``{pos, neg, pos_style, neg_style}`` records."""
    pos_vecs, neg_vecs = [], []
    for rec in pairs:
        p = pool_edit_embedding_qwen21(rec["pos"], rec["pos_style"], pipe, device)
        n = pool_edit_embedding_qwen21(rec["neg"], rec["neg_style"], pipe, device)
        if p is not None and n is not None:
            pos_vecs.append(p)
            neg_vecs.append(n)
    if not pos_vecs:
        raise RuntimeError("compute_dom_vector_qwen21: no pooled embeddings could be computed.")
    v_unit, pre_norm = _unit_dom(pos_vecs, neg_vecs)
    return v_unit.to(device), {"n_usable": len(pos_vecs), "pre_norm": pre_norm}


@torch.no_grad()
def compute_incontext_dom_vector_qwen21(
    full_prompt: str, edit_text: str, all_edits: List[str], edit_index: int,
    pipe, device, build_prompt_fn: Callable[[List[str], List[str]], str], vl_images: list,
) -> Tuple[torch.Tensor, dict]:
    """DoM inside the actual inference input: the full prompt with the
    condition images, pooled at ``edit_text``, versus the same input with the
    edit replaced by each neutral word in ``_NEG_STYLE_VARIANTS``."""
    pos = pool_edit_embedding_qwen21(full_prompt, edit_text, pipe, device, vl_images)
    if pos is None:
        raise RuntimeError(f"Could not locate edit text {edit_text!r} in the full prompt.")
    neg_vecs = []
    for neg_word in _NEG_STYLE_VARIANTS:
        edits = list(all_edits)
        edits[edit_index] = neg_word
        neg = pool_edit_embedding_qwen21(build_prompt_fn(edits, edits), neg_word, pipe, device, vl_images)
        if neg is not None:
            neg_vecs.append(neg)
    if not neg_vecs:
        raise RuntimeError("No neutral replacement could be located in the modified prompts.")
    v_unit, pre_norm = _unit_dom([pos], neg_vecs)
    return v_unit.to(device), {"n_usable": 1 + len(neg_vecs), "pre_norm": pre_norm}


def _cache_dir(cache_root: Path, edit: str, prompt: Optional[str] = None,
               vl_images: Optional[list] = None) -> Path:
    h = hashlib.sha1(b"qwen-image-2.1\x00")
    h.update(f"{prompt}\x00{edit}".encode("utf-8") if prompt else edit.encode("utf-8"))
    for img in vl_images or []:
        h.update(f"{img.size}{img.mode}".encode())
        h.update(img.tobytes())
    return Path(cache_root) / h.hexdigest()[:12]


def get_or_build_steering_vector_qwen21(
    edit: str, n_pairs: int, pipe, device, cache_root: Path, regen: bool = False, *,
    incontext_prompt: Optional[str] = None, all_edits: Optional[List[str]] = None,
    edit_index: Optional[int] = None, build_prompt_fn: Optional[Callable] = None,
    vl_images: Optional[list] = None,
) -> Tuple[torch.Tensor, dict]:
    """Qwen-Image-2.1 counterpart of ``steering_dom.get_or_build_steering_vector``.
    Returns ``(v_unit, meta)``."""
    use_incontext = None not in (incontext_prompt, all_edits, edit_index, build_prompt_fn, vl_images)
    edit_dir = _cache_dir(cache_root, edit, incontext_prompt if use_incontext else None,
                          vl_images if use_incontext else None)
    vec_path, meta_path = edit_dir / "vector.pt", edit_dir / "meta.json"

    if not regen and vec_path.exists():
        meta = {"source": "cache", "vec_path": str(vec_path)}
        if meta_path.exists():
            meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
        if meta.get("cache_version", 0) >= _CACHE_VERSION:
            return torch.load(vec_path, map_location=device).to(device=device, dtype=torch.float32), meta
        print(f"[Steering DoM] Stale cache for edit {edit!r} — regenerating …")

    edit_dir.mkdir(parents=True, exist_ok=True)
    if use_incontext:
        print(f"[Steering DoM] In-context DoM for edit {edit_index + 1} ({edit!r}) …")
        v_unit, stats = compute_incontext_dom_vector_qwen21(
            incontext_prompt, edit, all_edits, edit_index, pipe, device, build_prompt_fn, vl_images)
        meta = {"edit": edit, "method": "incontext", "source": "incontext"}
    else:
        print(f"[Steering DoM] Generating {n_pairs} contrastive pairs for edit: {edit!r}")
        records, n_requested = generate_pairs_with_qwen21(edit, pipe, n_pairs, device)
        n_parsed = len(records)
        used_fallback = n_parsed < max(4, n_pairs // 2)
        if used_fallback:
            print(f"[Steering DoM] Parse yield {n_parsed}/{n_requested} too low — padding with scaffold fallback.")
            seen = {(r["pos"], r["neg"]) for r in records}
            for fb in _fallback_pairs(edit, n_pairs - n_parsed):
                if (fb["pos"], fb["neg"]) not in seen:
                    records.append(fb)
                    seen.add((fb["pos"], fb["neg"]))
        v_unit, stats = compute_dom_vector_qwen21(records, pipe, device)
        _write_jsonl(edit_dir / "pairs.jsonl", records)
        meta = {"edit": edit, "method": "qwen_isolated", "source": "generated",
                "n_requested": n_requested, "n_parsed_from_llm": n_parsed,
                "used_fallback_padding": used_fallback}

    torch.save(v_unit.detach().cpu().float(), vec_path)
    meta.update({"model": "qwen-image-2.1", "cache_version": _CACHE_VERSION,
                 "n_usable_after_pooling": stats["n_usable"],
                 "pre_normalization_norm": stats["pre_norm"]})
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Steering DoM] Cached → {edit_dir}")
    return v_unit.to(device=device, dtype=torch.float32), meta
