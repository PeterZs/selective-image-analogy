"""SIA Multi-Edit Analogy data, read from the Hugging Face Hub with ``datasets``.

Configs of the dataset: ``subjects`` (train; one row per subject with its source
image, the full edit, the partial edits and the edit texts), ``analogy_benchmark``
(test; (A, A', B) cases without ground truth) and ``full_transfer`` (test;
(A, A', B, B') cases with ground truth).

The two training datasets draw their samples with Python's global ``random``;
for a fixed seed they reproduce the samples of the original folder-based loaders
exactly (same pairs, same images, same order).
"""
import copy
import functools
import io
import itertools
import random

from PIL import Image
from torch.utils.data import Dataset

from sia.prompt import build_analogy_prompt

DATASET = "selective-image-analogy/SIA-Multi-Edit-Analogy"
SPLITS = ("analogy_benchmark", "dev_train")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resize_to_match(img, target_w, target_h):
    """Resize image to (target_w, target_h), both divisible by 16."""
    target_w = (target_w // 16) * 16
    target_h = (target_h // 16) * 16
    return img.resize((target_w, target_h), Image.LANCZOS)


def resize_keep_ratio(img, max_side=512):
    """Resize so the longer side equals max_side, rounding both sides to multiples of 16."""
    w, h = img.size
    scale = max_side / max(w, h)
    new_w = max(16, round(w * scale / 16) * 16)
    new_h = max(16, round(h * scale / 16) * 16)
    if (w, h) != (new_w, new_h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def harmonize_images(a, a_prime, b, b_prime, max_side=512):
    """Normalise all four images for ExLoRA training (FLUX.2-Klein).

    * A is resized to max_side (long-side) while preserving its aspect ratio,
      rounded to multiples of 16.
    * A' is snapped to A's exact pixel dimensions (same subject, same crop).
    * B is resized independently with the same max_side / ratio rule so its
      own aspect ratio is respected.
    * B' is snapped to B's exact pixel dimensions so that the denoising target
      always has the same spatial size as the B condition image.
    """
    a       = resize_keep_ratio(a, max_side)
    a_w, a_h = a.size
    a_prime = resize_to_match(a_prime, a_w, a_h)

    b       = resize_keep_ratio(b, max_side)
    b_w, b_h = b.size
    b_prime = resize_to_match(b_prime, b_w, b_h)

    return a, a_prime, b, b_prime


def harmonize_images_to_a(a, a_prime, b, b_prime):
    """Resize A', B, B' to A's size (AnaLoRA training, FLUX.2-Klein)."""
    target_w, target_h = a.size
    a_prime = resize_to_match(a_prime, target_w, target_h)
    b = resize_to_match(b,       target_w, target_h)
    b_prime = resize_to_match(b_prime, target_w, target_h)
    return a, a_prime, b, b_prime


# ---------------------------------------------------------------------------
# Hub access
# ---------------------------------------------------------------------------

def _img(cell):
    return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")


def _no_decode(feature):
    """Copy of ``feature`` with every Image (at any nesting depth) set to decode=False."""
    from datasets import Image as ImageFeature
    if isinstance(feature, ImageFeature):
        return ImageFeature(decode=False)
    if isinstance(feature, dict):
        return type(feature)({k: _no_decode(v) for k, v in feature.items()})
    if isinstance(feature, list):
        return [_no_decode(f) for f in feature]
    if hasattr(feature, "feature"):  # List / Sequence / LargeList
        clone = copy.copy(feature)
        clone.feature = _no_decode(feature.feature)
        return clone
    return feature


def _load(dataset, config, split, revision=None):
    """One split of the dataset with its images kept as PNG bytes; they are
    decoded (with ``_img``) only when a sample uses them."""
    # Imported here so that importing this module does not fix the HF cache /
    # offline configuration before a script has set it.
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split, revision=revision)
    return ds.cast(_no_decode(ds.features))


def load_subjects(dataset=DATASET, revision=None):
    """The ``subjects`` training split (images undecoded, see ``subject_image``)."""
    return _load(dataset, "subjects", "train", revision)


def subject_image(subjects, ref):
    """Decoded RGB image of a ``subjects`` row. ``ref = (row, key)`` with key
    ``"source"``, ``"full_edit"`` or the index of a partial edit."""
    row, key = ref
    record = subjects[row]
    cell = record["partial_edits"][key]["image"] if isinstance(key, int) else record[key]
    return _img(cell)


def _subjects_by_setting(subjects):
    """Group the ``subjects`` rows as ``{category: {setting: [subject, ...]}}``.

    ``subject_id`` is ``<category>/<setting>/<item>``. Categories, settings and
    items are ordered by plain string comparison of these names (``setting10``
    before ``setting2``). The seeded samplers index into these lists, so the
    order is part of their behaviour.

    Each subject is ``{"name", "row", "edits", "combined"}`` where ``combined``
    lists ``(kept edit numbers, image key)``: the full edit first, then the
    partial edits in stored order.
    """
    groups = {}
    meta = subjects.select_columns(["subject_id", "num_edits", "edits", "partial_edits"])
    for row, r in enumerate(meta):
        category, setting, name = r["subject_id"].split("/")
        combined = [(list(range(1, r["num_edits"] + 1)), "full_edit")]
        combined += [(p["kept_edits"], j) for j, p in enumerate(r["partial_edits"])]
        groups.setdefault(category, {}).setdefault(setting, []).append(
            {"name": name, "row": row, "edits": r["edits"], "combined": combined})
    return {c: {s: sorted(groups[c][s], key=lambda x: x["name"]) for s in sorted(groups[c])}
            for c in sorted(groups)}


# ---------------------------------------------------------------------------
# ExLoRA training data
# ---------------------------------------------------------------------------

class SelectiveAnalogyDataset(Dataset):
    """Selective image analogy dataset for ExLoRA training.

    Each subject of the ``subjects`` split (``<category>/<setting>/<item>``)
    has a source image ``I^∅``, the full edit and partial edits (images with
    a subset of the edits applied). Each item is stored as a dict with:

    * ``name``     — item name (e.g. ``data3``)
    * ``input``    — image ref of the source image (see ``subject_image``)
    * ``edits``    — ``{1: "text1", 2: "text2", …}`` (1-based int keys)
    * ``combined`` — ``{frozenset(kept edits): image ref, …}``

    Sampling produces an analogy pair ``(item_a, item_b)`` from the same
    ``<category>/<setting>`` and selects a ``(demo_set, kept_set)`` pair
    where both combined images exist in both items and ``kept_set`` is a
    strict proper non-empty subset of ``demo_set``.  The suppression set is
    ``suppress_set = demo_set − kept_set``.

    * ``A``, ``B``          — input images
    * ``A_prime``           — ``item_a``'s ``combined[demo_set]`` (the demo A')
    * ``A_prime_total``     — same as ``A_prime``
    * ``A_prime_partial``   — ``item_a``'s ``combined[kept_set]`` if available,
                              else ``None`` (trainer falls back to ``A_prime``)
    * ``B_prime``           — ``item_b``'s ``combined[kept_set]`` (partial target
                              B'_R); equals ``B`` when ``kept_set`` is empty
                              (all-suppress fallback)
    * ``B_total``           — ``item_b``'s ``combined[demo_set]`` (full demo B')
    * ``pair_a_edits``      — list of edit strings for the *demo* edits, in
                              ``demo_set`` index order; length = ``|demo_set|``
    * ``suppress_indices``  — 0-based indices into ``pair_a_edits`` of edits
                              that are suppressed (``suppress_set`` positions)
    * ``suppress_edit_texts`` — the actual edit-text strings for every
                              suppressed edit (``[pair_a_edits[i] for i in
                              suppress_indices]``); always a list, never a
                              single string

    Demo sets are weighted by ``|demo_set|³`` so the model sees larger (more
    complete) demonstrations most of the time, matching §4.3 of the SIA paper.
    When no valid ``(demo, kept)`` pair exists across both items, the largest
    common demo is used with ``kept_set = ∅`` (all-suppress fallback).

    Args:
        dataset: Hub id (or local path) of the dataset.
        revision: optional dataset revision (branch, tag or commit).
        harmonize_fn: resizes (A, A', B, B') after loading; defaults to the
            FLUX.2-Klein rule ``harmonize_images``.
    """

    def __init__(self, dataset=DATASET, revision=None, harmonize_fn=None):
        self.subjects = load_subjects(dataset, revision)
        # Other backbones pass their own rule (e.g. Qwen-Image-2.1 resizes at
        # encode time, see train_exlora_qwen21).
        self.harmonize_fn = harmonize_fn or functools.partial(harmonize_images, max_side=512)
        self.categories: dict = {}
        n_total = 0
        n_skipped = 0

        for cat, settings in _subjects_by_setting(self.subjects).items():
            self.categories[cat] = {}
            for setting, subjects in settings.items():
                items = []
                for subject in subjects:
                    info = self._load_item(subject)
                    if info is None:
                        n_skipped += 1
                        continue
                    items.append(info)
                    n_total += 1

                if len(items) >= 2:
                    self.categories[cat][setting] = items

        self.cat_names = sorted(self.categories.keys())
        print(f"SelectiveAnalogyDataset: {len(self.cat_names)} categories, "
              f"{n_total} items kept ({n_skipped} skipped).")
        for cat in self.cat_names:
            n = sum(len(items) for items in self.categories[cat].values())
            print(f"  {cat}: {len(self.categories[cat])} settings, {n} items")

    # ------------------------------------------------------------------
    @staticmethod
    def _load_item(subject):
        """Parse one subject. Returns None when the item is unusable."""
        edits = {i: e.strip() for i, e in enumerate(subject["edits"], 1) if e and e.strip()}
        if len(edits) < 2:
            return None

        # Only combined images whose indices are all valid edit indices.
        # Keys are inserted sorted by the digits of the kept edits ("1" < "12"
        # < "2"): insertion order can decide the iteration order of the sets
        # built from these keys, which the seeded sampler draws from.
        valid_indices = set(edits.keys())
        combined_map: dict = {}
        for kept, key in sorted(subject["combined"], key=lambda c: "".join(map(str, c[0]))):
            fs = frozenset(kept)
            if fs and fs.issubset(valid_indices):
                combined_map[fs] = (subject["row"], key)

        if not combined_map:
            return None

        return {
            'name':     subject["name"],
            'input':    (subject["row"], "source"),
            'edits':    edits,        # {1: "text1", 2: "text2", …}
            'combined': combined_map, # {frozenset(indices): image ref}
        }

    # ------------------------------------------------------------------
    def __len__(self):
        return sum(len(items) for c in self.categories.values()
                   for items in c.values())

    # ------------------------------------------------------------------
    def _sample_two_items(self):
        """Pick a category/setting and two distinct items from that setting.

        Categories are picked uniformly at random; settings are picked
        uniformly within a category; items are picked without replacement
        within a setting. Settings with < 2 items are excluded.
        """
        cat = random.choice(self.cat_names)
        settings_with_pairs = [
            s for s, items in self.categories[cat].items() if len(items) >= 2
        ]
        if not settings_with_pairs:
            # Fall back to any category that has at least one usable setting.
            valid_cats = [
                c for c in self.cat_names
                if any(len(items) >= 2 for items in self.categories[c].values())
            ]
            cat = random.choice(valid_cats)
            settings_with_pairs = [
                s for s, items in self.categories[cat].items() if len(items) >= 2
            ]
        setting = random.choice(settings_with_pairs)
        items = self.categories[cat][setting]
        item_a, item_b = random.sample(items, 2)
        return cat, setting, item_a, item_b

    # ------------------------------------------------------------------
    @staticmethod
    def _pick_demo_kept(common_sets):
        """Sample a ``(demo_set, kept_set)`` pair from combined sets present
        in both items.

        All valid pairs ``(demo, kept)`` where ``kept`` is a non-empty proper
        subset of ``demo`` are enumerated.  Pairs are weighted by
        ``|demo|³`` to strongly prefer larger (more complete) demos.
        Falls back to ``(largest_demo, frozenset())`` when no strict-subset
        pair exists (all-suppress branch).

        Returns ``(demo_set, kept_set)`` — both frozensets.
        """
        valid_pairs = [
            (demo, kept)
            for demo in common_sets
            for kept in common_sets
            if len(kept) > 0 and kept < demo   # strict proper subset
        ]
        if valid_pairs:
            weights = [len(demo) ** 3 for demo, _ in valid_pairs]
            return random.choices(valid_pairs, weights=weights, k=1)[0]
        # No valid partial pair — use the largest common combined as the
        # demo, leave kept empty (all-suppress fallback).
        demo_set = max(common_sets, key=lambda s: (len(s), tuple(sorted(s))))
        return demo_set, frozenset()

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        for _ in range(8):
            sample = self._try_sample()
            if sample is not None:
                return sample
        raise RuntimeError(
            "Failed to sample a valid analogy pair after 8 attempts.")

    # ------------------------------------------------------------------
    def _try_sample(self):
        cat, setting, item_a, item_b = self._sample_two_items()

        a_edits    = item_a['edits']
        a_combined = item_a['combined']
        b_combined = item_b['combined']

        common_sets = set(a_combined) & set(b_combined)
        if not common_sets:
            return None

        demo_set, kept_set = self._pick_demo_kept(common_sets)
        suppress_set = demo_set - kept_set

        # ---- Load images ----
        a_img      = subject_image(self.subjects, item_a['input'])
        b_img      = subject_image(self.subjects, item_b['input'])
        a_demo_img = subject_image(self.subjects, a_combined[demo_set])
        b_demo_img = subject_image(self.subjects, b_combined[demo_set])

        if kept_set:
            b_kept_img = subject_image(self.subjects, b_combined[kept_set])
            # A's matching partial for the AnaLoRA-partial step (may be absent).
            a_kept_img = (subject_image(self.subjects, a_combined[kept_set])
                          if kept_set in a_combined else None)
        else:
            b_kept_img = None
            a_kept_img = None

        # ---- Harmonize sizes ----
        a_img, a_demo_img, b_img, b_demo_img = self.harmonize_fn(
            a_img, a_demo_img, b_img, b_demo_img)
        if b_kept_img is not None:
            b_kept_img = resize_to_match(b_kept_img, *b_img.size)
        if a_kept_img is not None:
            a_kept_img = resize_to_match(a_kept_img, *a_img.size)

        # ---- Build edit text lists ----
        demo_indices     = sorted(demo_set)
        suppress_indices = sorted(suppress_set)

        # demo_edit_texts: one entry per demo edit, in index order.
        # suppress_edit_texts: sub-list for edits being suppressed — every
        # edit in suppress_set gets its own entry (never collapsed to one).
        demo_edit_texts     = [a_edits[i] for i in demo_indices if i in a_edits]
        suppress_edit_texts = [a_edits[i] for i in suppress_indices if i in a_edits]

        if not demo_edit_texts:
            return None

        # 0-based positions within demo_edit_texts for the suppressed edits.
        demo_idx_to_pos = {idx: pos for pos, idx in enumerate(demo_indices)}
        suppress_indices_0b = [demo_idx_to_pos[i] for i in suppress_indices
                               if i in demo_idx_to_pos]

        # B_prime: partial target (kept edits only); B itself when all-suppress.
        b_prime = b_kept_img if b_kept_img is not None else b_img

        combined_prompt = build_analogy_prompt(demo_edit_texts, demo_edit_texts)

        return {
            'A':                   a_img,
            'A_prime':             a_demo_img,  # demo A' (same as A_prime_total)
            'A_prime_total':       a_demo_img,
            'A_prime_partial':     a_kept_img,  # matched partial A', or None
            'B':                   b_img,
            'B_prime':             b_prime,     # partial target B'_R
            'B_total':             b_demo_img,  # full demo B' (AnaLoRA total step)
            'edit_name':           f"{cat}/{setting}",
            'prompt':              combined_prompt,
            'pair_a_edits':        demo_edit_texts,
            'pair_b_edits':        demo_edit_texts,
            # 0-based positions in pair_a_edits for the edits being suppressed.
            'suppress_indices':    suppress_indices_0b,
            # Actual text strings for ALL suppressed edits (one per edit, not
            # collapsed) — used directly by build_token_mask without re-indexing.
            'suppress_edit_texts': suppress_edit_texts,
        }


def collate_selective(batch):
    """Collate ``SelectiveAnalogyDataset`` samples into lists per field."""
    return {
        'A':                   [item['A']                   for item in batch],
        'A_prime':             [item['A_prime']             for item in batch],
        'A_prime_total':       [item['A_prime_total']       for item in batch],
        'A_prime_partial':     [item['A_prime_partial']     for item in batch],
        'B':                   [item['B']                   for item in batch],
        'B_prime':             [item['B_prime']             for item in batch],
        'B_total':             [item['B_total']             for item in batch],
        'prompts':             [item['prompt']              for item in batch],
        'pair_a_edits':        [item['pair_a_edits']        for item in batch],
        'pair_b_edits':        [item['pair_b_edits']        for item in batch],
        # 0-based positions in pair_a_edits; each entry is a list (multi-edit).
        'suppress_indices':    [item['suppress_indices']    for item in batch],
        # Actual suppressed edit strings per sample — list[list[str]].
        'suppress_edit_texts': [item['suppress_edit_texts'] for item in batch],
    }


# ---------------------------------------------------------------------------
# AnaLoRA training data
# ---------------------------------------------------------------------------

def _build_prompt_from_indices(edit_indices: list, prompt_dict: dict) -> str:
    """Build the analogy prompt from selected 1-based edit indices.

    e.g. edit_indices=[1,2], prompt_dict={"edit1":"add hat","edit2":"add glasses"}
    →  "Image 1 is the original … Edit 1: add hat. Edit 2: add glasses. Apply …"
    """
    edits_part = " ".join(
        f"Edit {i}: {prompt_dict[f'edit{i}']}."
        for i in sorted(edit_indices)
        if prompt_dict.get(f'edit{i}', '')
    )
    return (
        "Image 1 is the original and image 2 is the edited version. "
        f"{edits_part} "
        "Apply the same edits to image 3 to produce the output."
    )


def _tier_probs(n_tiers: int) -> list:
    """Per-tier sampling probabilities, longest tier first.

    AnaLoRA is the "full-transfer" model (paper §4.1, §4.3 eq. 11), so we
    strongly bias toward the largest available subset (= full edit set
    ``I^E``). The tail is kept non-zero so AnaLoRA also sees partial
    subsets occasionally — useful when partial inputs exist and as light
    augmentation against overfitting to a single edit configuration.

    1 → [1.0]
    2 → [0.92, 0.08]
    3 → [0.85, 0.10, 0.05]
    4+ → [0.85, 0.10, 0.05/(n-2), ...]
    """
    if n_tiers <= 1:
        return [1.0] * max(n_tiers, 1)
    if n_tiers == 2:
        return [0.92, 0.08]
    if n_tiers == 3:
        return [0.85, 0.10, 0.05]
    tail = 0.05 / (n_tiers - 2)
    return [0.85, 0.10] + [tail] * (n_tiers - 2)


def _sample_combined(combined_files: list):
    """Sample one ``(image ref, edit_indices)`` using tier probabilities.

    Tier = number of edits applied (len of edit_indices).
    Longest tier → highest probability.  Uniform within each tier.
    """
    if not combined_files:
        raise ValueError("combined_files is empty")
    tier_dict: dict = {}
    for fname, indices in combined_files:
        t = len(indices)
        tier_dict.setdefault(t, []).append((fname, indices))
    sorted_tiers = sorted(tier_dict.keys(), reverse=True)
    probs = _tier_probs(len(sorted_tiers))
    chosen_tier = random.choices(sorted_tiers, weights=probs)[0]
    return random.choice(tier_dict[chosen_tier])


class AnalogyPairDataset(Dataset):
    """Image analogy pairs (A, A', B, B') for AnaLoRA training.

    Sampling procedure for one training pair:

    1. Pick an edit category uniformly at random.
    2. Pick a setting within that category uniformly at random.
    3. Pick two distinct subjects from that setting.
    4. Intersect the available edit-index sets (full and partial edits)
       across both subjects.
    5. Sample one edit-index set using tier probabilities (``_tier_probs``).
    6. Use A/A' from subject 1 and B/B' from subject 2 with that set.
    7. Build the prompt from the selected edit indices; 15 % of the prompts
       are dropped (empty string).

    Args:
        dataset: Hub id (or local path) of the dataset.
        revision: optional dataset revision (branch, tag or commit).
        harmonize_fn: resizes (A, A', B, B') after loading; defaults to the
            FLUX.2-Klein rule ``harmonize_images_to_a``.
    """

    def __init__(self, dataset=DATASET, revision=None, harmonize_fn=None):
        self.subjects = load_subjects(dataset, revision)
        self.harmonize_fn = harmonize_fn or harmonize_images_to_a

        # categories[cat][setting] = list of (input ref, combined, prompt_dict),
        # combined = [(image ref, sorted edit indices), ...]: the full edit, then
        # the partial edits in stored order (the seeded sampler depends on it).
        self.categories: dict = {}
        for cat, settings in _subjects_by_setting(self.subjects).items():
            self.categories[cat] = {}
            for setting, subjects in settings.items():
                samples = []
                for subject in subjects:
                    row = subject["row"]
                    combined = [((row, key), sorted(kept)) for kept, key in subject["combined"]]
                    prompt_dict = {f"edit{i}": e for i, e in enumerate(subject["edits"], 1)}
                    samples.append(((row, "source"), combined, prompt_dict))
                if len(samples) >= 2:
                    self.categories[cat][setting] = samples

        self.cat_names = sorted(self.categories.keys())
        total = sum(len(s) for c in self.categories.values() for s in c.values())
        print(f"AnalogyPairDataset: {len(self.cat_names)} categories, {total} samples")
        for cat in self.cat_names:
            n = sum(len(s) for s in self.categories[cat].values())
            print(f"  {cat}: {len(self.categories[cat])} settings, {n} samples")

    def __len__(self):
        return sum(len(s) for c in self.categories.values() for s in c.values())

    def sample_val_pair(self):
        """Return decoded A, A', B, B' (unresized) and the prompt for validation.

        Uses the same matched-pair sampling as ``__getitem__`` so the
        validation A' and B' always refer to the same combined-edit subset.
        """
        (cat, setting, input_a, input_b,
         ref_a, ref_b, indices, prompt_a) = self._sample_matched_pair()
        prompt = _build_prompt_from_indices(indices, prompt_a)
        return {
            'a':         subject_image(self.subjects, input_a),
            'a_prime':   subject_image(self.subjects, ref_a),
            'b':         subject_image(self.subjects, input_b),
            'b_prime':   subject_image(self.subjects, ref_b),
            'edit_name': f"{cat}/{setting}",
            'prompt':    prompt,
        }

    def _sample_matched_pair(self, max_tries: int = 32):
        """Sample (cat, setting, input_a, input_b, ref_a, ref_b, indices, prompt_a)
        such that A' and B' refer to the *same* combined-edit subset.

        Subjects have heterogeneous partial edits (e.g. one 4-edit subject may
        have edit set {3} while another only has {1, 2}). To guarantee A→A'
        and B→B' demonstrate the same transformation, we only sample pairs
        whose available subsets intersect; we retry a few times before giving
        up and falling back to the longest combined image on both sides.
        """
        for _ in range(max_tries):
            cat = random.choice(self.cat_names)
            settings = self.categories[cat]
            setting = random.choice(list(settings.keys()))
            samples = settings[setting]

            sample_a, sample_b = random.sample(samples, 2)
            input_a, combined_a, prompt_a = sample_a
            input_b, combined_b, _ = sample_b

            map_a = {frozenset(idx): ref for ref, idx in combined_a}
            map_b = {frozenset(idx): ref for ref, idx in combined_b}
            common = set(map_a) & set(map_b)
            if not common:
                continue

            pool = [(map_a[k], sorted(k)) for k in common]
            ref_a, indices = _sample_combined(pool)
            ref_b = map_b[frozenset(indices)]
            return (cat, setting, input_a, input_b,
                    ref_a, ref_b, indices, prompt_a)

        # Final fallback: pick *any* setting; default to the longest combined
        # available on each side independently (visually they may differ but
        # this keeps the dataset usable when intersection is repeatedly
        # empty). In a healthy dataset this branch should be unreachable.
        cat = random.choice(self.cat_names)
        settings = self.categories[cat]
        setting = random.choice(list(settings.keys()))
        sample_a, sample_b = random.sample(settings[setting], 2)
        input_a, combined_a, prompt_a = sample_a
        input_b, combined_b, _ = sample_b
        ref_a, indices_a = max(combined_a, key=lambda x: len(x[1]))
        ref_b, _ = max(combined_b, key=lambda x: len(x[1]))
        return (cat, setting, input_a, input_b,
                ref_a, ref_b, indices_a, prompt_a)

    def __getitem__(self, idx):
        (cat, setting, input_a, input_b,
         ref_a, ref_b, indices, prompt_a) = self._sample_matched_pair()

        a       = subject_image(self.subjects, input_a)
        a_prime = subject_image(self.subjects, ref_a)
        b       = subject_image(self.subjects, input_b)
        b_prime = subject_image(self.subjects, ref_b)
        a, a_prime, b, b_prime = self.harmonize_fn(a, a_prime, b, b_prime)

        # Build the listed-edits prompt; 15 % of the time drop it entirely so
        # AnaLoRA also learns to infer the A→A' transformation purely from
        # the visual demonstration (no edit-text supervision).
        combined_prompt = _build_prompt_from_indices(indices, prompt_a)
        if random.random() < 0.15:
            combined_prompt = ""

        return {
            'A':         a,
            'A_prime':   a_prime,
            'B':         b,
            'B_prime':   b_prime,
            'edit_name': f"{cat}/{setting}",
            'prompt':    combined_prompt,
        }


def collate_pairs(batch):
    """Collate ``AnalogyPairDataset`` samples into lists per field."""
    return {
        'A': [item['A'] for item in batch],
        'A_prime': [item['A_prime'] for item in batch],
        'B': [item['B'] for item in batch],
        'B_prime': [item['B_prime'] for item in batch],
        'prompts': [item['prompt'] for item in batch],
    }


# ---------------------------------------------------------------------------
# Evaluation data
# ---------------------------------------------------------------------------

def dev_train_cases(limit=None, n_settings=40, seed=0, dataset=DATASET, revision=None):
    """Development cases for choosing method settings WITHOUT touching the test
    benchmark: training-set subjects, 3- and 4-edit recipes, two distinct items
    per setting as (A, B) with the full edit list as the demo.
    Same dict layout as ``load_split("analogy_benchmark")``. The images were seen
    in training, so absolute scores run high; only differences between methods
    are meaningful."""
    ds = SelectiveAnalogyDataset(dataset, revision)
    rng = random.Random(seed)
    pool = []
    for cat in ds.cat_names:
        if not cat.startswith(("3", "4")):
            continue
        for setting, items in ds.categories[cat].items():
            full = [it for it in items if len(it["edits"]) >= 3 and frozenset(it["edits"]) in it["combined"]]
            if len(full) >= 2:
                pool.append((cat, setting, full))
    rng.shuffle(pool)
    rows = []
    for cat, setting, full in pool[:n_settings]:
        for _ in range(3):            # up to three (A, B) item pairs per setting
            ia, ib = rng.sample(full, 2)
            idx = sorted(ia["edits"])
            if idx != sorted(ib["edits"]) or any(r["id"].endswith(f"{ia['name']}-{ib['name']}")
                                                 and f"/{cat}/{setting}/" in r["id"] for r in rows):
                continue
            rows.append(dict(
                id=f"dev/{cat}/{setting}/{ia['name']}-{ib['name']}", num_edits=len(idx),
                subject_type="non_creature" if "non_creature" in cat else "creature",
                edits=[ia["edits"][i] for i in idx], edit_types=["unknown"] * len(idx), recipe_in_train=True,
                a=subject_image(ds.subjects, ia["input"]),
                a_prime=subject_image(ds.subjects, ia["combined"][frozenset(idx)]),
                b=subject_image(ds.subjects, ib["input"]), src_id=None, dst_id=None))
    rows.sort(key=lambda r: r["id"])
    return rows[:limit] if limit else rows


def load_split(split, limit=None, dataset=DATASET, revision=None):
    """Rows of an evaluation split, sorted by id, with decoded PIL images.

    ``split`` is a test config of the dataset (``analogy_benchmark``,
    ``full_transfer``) or ``dev_train`` (``dev_train_cases``; training data,
    never the test benchmark).
    """
    if split == "dev_train":
        return dev_train_cases(limit, dataset=dataset, revision=revision)
    rows = []
    for r in _load(dataset, split, "test", revision):
        for k in ("a", "a_prime", "b", "b_prime"):
            if k in r:
                r[k] = _img(r[k])
        rows.append(r)
    rows.sort(key=lambda r: r["id"])
    return rows[:limit] if limit else rows


def load_benchmark_cases(ids, dataset=DATASET, revision=None):
    """(A, A', B) cases from the dataset's ``analogy_benchmark`` test split,
    whose subjects never appear in training. Returns dicts in ``ids`` order."""
    rows = {r["id"]: r for r in _load(dataset, "analogy_benchmark", "test", revision) if r["id"] in ids}
    missing = [i for i in ids if i not in rows]
    if missing:
        raise ValueError(f"Benchmark ids not found in {dataset}: {missing}")
    return [{"id": i, "edits": rows[i]["edits"], "recipe_in_train": rows[i]["recipe_in_train"],
             "a": _img(rows[i]["a"]), "a_prime": _img(rows[i]["a_prime"]), "b": _img(rows[i]["b"])}
            for i in ids]


def subsets_for(n):
    """Every non-empty proper subset of ``range(n)``, larger subsets first."""
    return [k for r in range(n - 1, 0, -1) for k in itertools.combinations(range(n), r)]


def covered_by_training(kept, n_edits, subject_type):
    """True when this kept subset has ground truth in the training data: the
    3/4-edit recipes only ship cumulative prefixes {1}, {1,2}, ... and 2-edit
    creature recipes ship both single edits."""
    if kept == tuple(range(len(kept))):
        return True
    return n_edits == 2 and subject_type == "creature"


def distinct_edits(split="analogy_benchmark", dataset=DATASET):
    """Sorted distinct edit texts of an evaluation split."""
    return sorted({e for c in load_split(split, dataset=dataset) for e in c["edits"]})
