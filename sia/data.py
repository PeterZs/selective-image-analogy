"""Training and evaluation data from the SIA Multi-Edit Analogy dataset on the Hugging Face Hub.

Dataset configs: ``subjects`` (train; one row per subject with its source image, the full
edit, the partial edits and the edit texts), ``analogy_benchmark`` (test; (A, A', B) cases
without ground truth) and ``full_transfer`` (test; (A, A', B, B') cases with ground truth).

The two training datasets draw samples from Python's global ``random``. For a fixed seed the
samples depend on the exact order of the ``random`` calls and of the lists they index, so both
are part of the reproducibility contract and must not change.
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
    """Resize ``img`` to ``(target_w, target_h)``, each rounded down to a multiple of 16."""
    target_w = (target_w // 16) * 16
    target_h = (target_h // 16) * 16
    return img.resize((target_w, target_h), Image.LANCZOS)


def resize_keep_ratio(img, max_side=512):
    """Resize so the longer side is ``max_side``, rounding both sides to multiples of 16."""
    w, h = img.size
    scale = max_side / max(w, h)
    new_w = max(16, round(w * scale / 16) * 16)
    new_h = max(16, round(h * scale / 16) * 16)
    if (w, h) != (new_w, new_h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def harmonize_images(a, a_prime, b, b_prime, max_side=512):
    """Resize the four images for ExLoRA training on FLUX.2-Klein.

    A and B are resized independently with ``resize_keep_ratio`` (each keeps its own aspect
    ratio); A' is resized to A's size and B' to B's size, so the denoising target B' always
    matches the spatial size of the B condition image.
    """
    a       = resize_keep_ratio(a, max_side)
    a_w, a_h = a.size
    a_prime = resize_to_match(a_prime, a_w, a_h)

    b       = resize_keep_ratio(b, max_side)
    b_w, b_h = b.size
    b_prime = resize_to_match(b_prime, b_w, b_h)

    return a, a_prime, b, b_prime


def harmonize_images_to_a(a, a_prime, b, b_prime):
    """Resize A', B and B' to A's size (AnaLoRA training on FLUX.2-Klein)."""
    target_w, target_h = a.size
    a_prime = resize_to_match(a_prime, target_w, target_h)
    b = resize_to_match(b,       target_w, target_h)
    b_prime = resize_to_match(b_prime, target_w, target_h)
    return a, a_prime, b, b_prime


# ---------------------------------------------------------------------------
# Hub access
# ---------------------------------------------------------------------------

def _img(cell):
    """Decode an undecoded ``datasets`` image cell to an RGB PIL image."""
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
    """Load one split with its images kept as encoded bytes (decoded lazily with ``_img``)."""
    # Imported lazily so that importing this module does not fix the HF cache or offline
    # configuration before a script has set it.
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split, revision=revision)
    return ds.cast(_no_decode(ds.features))


def load_subjects(dataset=DATASET, revision=None):
    """Load the ``subjects`` training split (images undecoded, see ``subject_image``)."""
    return _load(dataset, "subjects", "train", revision)


def subject_image(subjects, ref):
    """Return the decoded RGB image referenced by ``ref = (row, key)``.

    ``key`` is ``"source"``, ``"full_edit"`` or the integer index of a partial edit of that row.
    """
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
    """Selective analogy samples for ExLoRA training (Stage 2).

    Each subject (``<category>/<setting>/<item>``) has a source image, the full edit and
    partial edits (images with a subset of the edits applied). Each sample draws two distinct
    items ``(item_a, item_b)`` of one setting and a ``(demo_set, kept_set)`` pair of edit sets
    whose images exist for both items, with ``kept_set`` a non-empty proper subset of
    ``demo_set``. The suppressed edits are ``demo_set - kept_set``. Pairs are weighted by
    ``|demo_set|**3`` so most demonstrations are large (paper, Sec. 4.3). If no such pair
    exists, the largest common set is the demo and ``kept_set`` is empty (all edits suppressed).

    A sample is a dict with:

    * ``A``, ``B``: source images of ``item_a`` and ``item_b``.
    * ``A_prime``, ``A_prime_total``: ``item_a`` with ``demo_set`` applied (the demo A').
    * ``A_prime_partial``: ``item_a`` with ``kept_set`` applied, or ``None`` if unavailable.
    * ``B_prime``: ``item_b`` with ``kept_set`` applied (the training target); ``B`` when
      ``kept_set`` is empty.
    * ``B_total``: ``item_b`` with ``demo_set`` applied.
    * ``prompt``: ``build_analogy_prompt`` over the demo edits.
    * ``pair_a_edits``, ``pair_b_edits``: demo edit texts in edit-number order.
    * ``suppress_indices``: 0-based positions in ``pair_a_edits`` of the suppressed edits.
    * ``suppress_edit_texts``: the suppressed edit texts (always a list).
    * ``edit_name``: ``"<category>/<setting>"``.

    ``__getitem__`` ignores ``idx``: every call draws a fresh sample from the global ``random``.

    Args:
        dataset: Hub id (or local path) of the dataset.
        revision: optional dataset revision (branch, tag or commit).
        harmonize_fn: resizes (A, A', B, B') after loading; defaults to the FLUX.2-Klein rule
            ``harmonize_images`` with ``max_side=512``.
    """

    def __init__(self, dataset=DATASET, revision=None, harmonize_fn=None):
        self.subjects = load_subjects(dataset, revision)
        # Qwen-Image-2.1 passes its own rule because it resizes at encode time.
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

    @staticmethod
    def _load_item(subject):
        """Convert one subject to ``{"name", "input", "edits", "combined"}``, or ``None``.

        ``edits`` maps 1-based edit numbers to texts and ``combined`` maps
        ``frozenset(kept edit numbers)`` to an image ref. Subjects with fewer than two non-empty
        edits or no usable combined image are skipped.
        """
        edits = {i: e.strip() for i, e in enumerate(subject["edits"], 1) if e and e.strip()}
        if len(edits) < 2:
            return None

        # Keys are inserted sorted by the digits of the kept edits ("1" < "12" < "2"): the
        # insertion order can decide the iteration order of the sets built from these keys,
        # which the seeded sampler draws from.
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
            'edits':    edits,
            'combined': combined_map,
        }

    def __len__(self):
        return sum(len(items) for c in self.categories.values()
                   for items in c.values())

    def _sample_two_items(self):
        """Return ``(category, setting, item_a, item_b)`` with two distinct items of one setting.

        The category and then the setting are drawn uniformly; the two items are drawn without
        replacement. Only settings with at least two items are eligible.
        """
        cat = random.choice(self.cat_names)
        settings_with_pairs = [
            s for s, items in self.categories[cat].items() if len(items) >= 2
        ]
        if not settings_with_pairs:
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

    @staticmethod
    def _pick_demo_kept(common_sets):
        """Sample ``(demo_set, kept_set)`` (frozensets) from the edit sets shared by both items.

        Every pair with ``kept`` a non-empty proper subset of ``demo`` is a candidate, weighted
        by ``|demo|**3``. Without candidates, returns the largest set and an empty ``kept_set``.
        """
        valid_pairs = [
            (demo, kept)
            for demo in common_sets
            for kept in common_sets
            if len(kept) > 0 and kept < demo
        ]
        if valid_pairs:
            weights = [len(demo) ** 3 for demo, _ in valid_pairs]
            return random.choices(valid_pairs, weights=weights, k=1)[0]
        demo_set = max(common_sets, key=lambda s: (len(s), tuple(sorted(s))))
        return demo_set, frozenset()

    def __getitem__(self, idx):
        for _ in range(8):
            sample = self._try_sample()
            if sample is not None:
                return sample
        raise RuntimeError(
            "Failed to sample a valid analogy pair after 8 attempts.")

    def _try_sample(self):
        """Draw one sample (see the class docstring), or ``None`` if the drawn items do not fit."""
        cat, setting, item_a, item_b = self._sample_two_items()

        a_edits    = item_a['edits']
        a_combined = item_a['combined']
        b_combined = item_b['combined']

        common_sets = set(a_combined) & set(b_combined)
        if not common_sets:
            return None

        demo_set, kept_set = self._pick_demo_kept(common_sets)
        suppress_set = demo_set - kept_set

        a_img      = subject_image(self.subjects, item_a['input'])
        b_img      = subject_image(self.subjects, item_b['input'])
        a_demo_img = subject_image(self.subjects, a_combined[demo_set])
        b_demo_img = subject_image(self.subjects, b_combined[demo_set])

        if kept_set:
            b_kept_img = subject_image(self.subjects, b_combined[kept_set])
            a_kept_img = (subject_image(self.subjects, a_combined[kept_set])
                          if kept_set in a_combined else None)
        else:
            b_kept_img = None
            a_kept_img = None

        a_img, a_demo_img, b_img, b_demo_img = self.harmonize_fn(
            a_img, a_demo_img, b_img, b_demo_img)
        if b_kept_img is not None:
            b_kept_img = resize_to_match(b_kept_img, *b_img.size)
        if a_kept_img is not None:
            a_kept_img = resize_to_match(a_kept_img, *a_img.size)

        demo_indices     = sorted(demo_set)
        suppress_indices = sorted(suppress_set)

        demo_edit_texts     = [a_edits[i] for i in demo_indices if i in a_edits]
        suppress_edit_texts = [a_edits[i] for i in suppress_indices if i in a_edits]

        if not demo_edit_texts:
            return None

        demo_idx_to_pos = {idx: pos for pos, idx in enumerate(demo_indices)}
        suppress_indices_0b = [demo_idx_to_pos[i] for i in suppress_indices
                               if i in demo_idx_to_pos]

        b_prime = b_kept_img if b_kept_img is not None else b_img

        combined_prompt = build_analogy_prompt(demo_edit_texts, demo_edit_texts)

        return {
            'A':                   a_img,
            'A_prime':             a_demo_img,
            'A_prime_total':       a_demo_img,
            'A_prime_partial':     a_kept_img,
            'B':                   b_img,
            'B_prime':             b_prime,
            'B_total':             b_demo_img,
            'edit_name':           f"{cat}/{setting}",
            'prompt':              combined_prompt,
            'pair_a_edits':        demo_edit_texts,
            'pair_b_edits':        demo_edit_texts,
            'suppress_indices':    suppress_indices_0b,
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
        'suppress_indices':    [item['suppress_indices']    for item in batch],
        'suppress_edit_texts': [item['suppress_edit_texts'] for item in batch],
    }


# ---------------------------------------------------------------------------
# AnaLoRA training data
# ---------------------------------------------------------------------------

def _build_prompt_from_indices(edit_indices: list, prompt_dict: dict) -> str:
    """Build the analogy prompt for the given 1-based edit numbers.

    Unlike ``build_analogy_prompt``, edits keep their original numbers (``{2, 3}`` gives
    ``Edit 2: ... Edit 3: ...``). ``prompt_dict`` maps ``"edit<i>"`` to the edit text.
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
    """Sampling probabilities of the edit-count tiers, largest tier first.

    AnaLoRA is the full-transfer model (paper, Sec. 4.1 and Eq. 11), so sampling is strongly
    biased toward the largest available edit set; the small tail keeps partial sets in training
    as light augmentation. Values: 1 tier ``[1.0]``; 2 tiers ``[0.92, 0.08]``; 3 tiers
    ``[0.85, 0.10, 0.05]``; n >= 4 tiers ``[0.85, 0.10]`` followed by ``n - 2`` equal shares
    of 0.05.
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
    """Sample one ``(image ref, edit_indices)`` entry of ``combined_files``.

    A tier is the number of applied edits. The tier is drawn with ``_tier_probs`` (largest tier
    first), then an entry is drawn uniformly within it.
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
    """Analogy pairs (A, A', B, B') for AnaLoRA training (Stage 1).

    One sample is drawn as follows:

    1. Pick a category, then a setting within it, uniformly at random.
    2. Pick two distinct subjects of that setting.
    3. Intersect their available edit sets (full and partial edits).
    4. Sample one shared edit set with the tier probabilities of ``_tier_probs``.
    5. Use A/A' from the first subject and B/B' from the second, both with that edit set.
    6. Build the prompt from the selected edits; 15% of the prompts are dropped (empty string).

    ``__getitem__`` ignores ``idx``: every call draws a fresh sample from the global ``random``.

    Args:
        dataset: Hub id (or local path) of the dataset.
        revision: optional dataset revision (branch, tag or commit).
        harmonize_fn: resizes (A, A', B, B') after loading; defaults to the
            FLUX.2-Klein rule ``harmonize_images_to_a``.
    """

    def __init__(self, dataset=DATASET, revision=None, harmonize_fn=None):
        self.subjects = load_subjects(dataset, revision)
        self.harmonize_fn = harmonize_fn or harmonize_images_to_a

        # categories[cat][setting] = [(input ref, combined, prompt_dict), ...] with
        # combined = [(image ref, sorted edit indices), ...]: the full edit first, then the
        # partial edits in stored order (the seeded sampler depends on this order).
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
        """Draw a validation sample: decoded, unresized A, A', B, B', its edit name and prompt.

        Uses the same matched-pair sampling as ``__getitem__``, so A' and B' show the same
        edit set.
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
        """Sample ``(cat, setting, input_a, input_b, ref_a, ref_b, indices, prompt_a)``.

        Subjects of one setting can have different partial edits, so a pair is accepted only if
        the two subjects share an edit set; A' and B' then show the same edit set. After
        ``max_tries`` failures, each side falls back to its largest edit set independently.
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

        # Reached only if no draw found a shared edit set; keeps sampling from failing.
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

        # Dropping 15% of the prompts teaches AnaLoRA to infer the edit from the images alone.
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
    """Development cases built from training subjects, for tuning without the test benchmark.

    Uses 3- and 4-edit settings: up to three (A, B) item pairs per setting, with the full edit
    list as the demonstration. Rows have the layout of ``load_split("analogy_benchmark")``.
    The images were seen in training, so absolute scores are inflated; only differences
    between methods are meaningful. Sampling uses a private ``random.Random(seed)``.
    """
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
        for _ in range(3):
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
    """Return the rows of an evaluation split, sorted by id, with decoded PIL images.

    ``split`` is a test config of the dataset (``analogy_benchmark``, ``full_transfer``) or
    ``dev_train`` (see ``dev_train_cases``; training data, never the test benchmark).
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
    """Return the ``analogy_benchmark`` cases with the given ids, in ``ids`` order.

    Each case is ``{"id", "edits", "recipe_in_train", "a", "a_prime", "b"}`` with decoded
    images. Raises ``ValueError`` if an id is missing.
    """
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
    """Whether the kept subset (0-based edit indices) has a ground-truth image in training.

    The 3- and 4-edit recipes provide only cumulative prefixes (``(0,)``, ``(0, 1)``, ...); the
    2-edit creature recipes provide both single edits.
    """
    if kept == tuple(range(len(kept))):
        return True
    return n_edits == 2 and subject_type == "creature"


def distinct_edits(split="analogy_benchmark", dataset=DATASET, revision=None):
    """Return the sorted distinct edit texts of an evaluation split."""
    return sorted({e for c in load_split(split, dataset=dataset, revision=revision) for e in c["edits"]})
