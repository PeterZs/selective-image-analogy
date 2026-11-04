"""Sampling invariants of the two training datasets, read from the HF dataset cache.

The samplers draw from the global ``random`` module, so a fixed ``random.seed`` must give the same samples.
"""
import hashlib
import random
import re
from pathlib import Path

import pytest

from sia.data import DATASET, AnalogyPairDataset, SelectiveAnalogyDataset, _build_prompt_from_indices
from sia.prompt import build_analogy_prompt

N_SAMPLES = 8


def dataset_is_cached():
    from datasets import config
    cache_name = DATASET.replace("/", "___").lower()
    return (Path(config.HF_DATASETS_CACHE) / cache_name / "subjects").is_dir()


requires_dataset = pytest.mark.skipif(not dataset_is_cached(), reason=f"{DATASET} is not in the HF cache")


def n_edits_of(edit_name):
    """Number of edits of a subject, from its category name (``3edits_non_creature/setting4`` -> 3)."""
    return int(re.match(r"(\d+)edits", edit_name).group(1))


def image_digest(img):
    return None if img is None else (img.size, hashlib.md5(img.tobytes()).hexdigest())


@pytest.fixture(scope="module")
def selective_dataset():
    return SelectiveAnalogyDataset()


@pytest.fixture(scope="module")
def pair_dataset():
    return AnalogyPairDataset()


@requires_dataset
def test_selective_samples_are_valid(selective_dataset):
    random.seed(0)
    for _ in range(N_SAMPLES):
        s = selective_dataset[0]
        demo, suppress = s["pair_a_edits"], s["suppress_indices"]

        assert 2 <= len(demo) <= n_edits_of(s["edit_name"]) <= 4
        # The kept set is a non-empty proper subset of the demo set.
        assert suppress == sorted(set(suppress))
        assert 0 < len(suppress) < len(demo)
        assert all(0 <= i < len(demo) for i in suppress)
        assert s["suppress_edit_texts"] == [demo[i] for i in suppress]
        # The prompt lists the whole demo; suppression only acts through the token mask.
        assert s["prompt"] == build_analogy_prompt(demo, demo)

        for key in ("A", "A_prime", "B", "B_prime", "B_total"):
            assert s[key].mode == "RGB"
        assert s["A_prime"].size == s["A"].size
        assert s["B_prime"].size == s["B"].size
        if s["A_prime_partial"] is not None:
            assert s["A_prime_partial"].size == s["A"].size


@requires_dataset
def test_pair_samples_are_valid(pair_dataset):
    random.seed(0)
    for _ in range(N_SAMPLES):
        s = pair_dataset[0]
        for key in ("A", "A_prime", "B", "B_prime"):
            assert s[key].mode == "RGB"
        assert s["A_prime"].size == s["A"].size
        if s["prompt"]:
            listed = [int(k) for k in re.findall(r"Edit (\d+):", s["prompt"])]
            assert 1 <= len(listed) <= n_edits_of(s["edit_name"])
            assert set(listed) <= set(range(1, n_edits_of(s["edit_name"]) + 1))


def selective_identities(dataset, seed):
    random.seed(seed)
    out = []
    for _ in range(N_SAMPLES):
        s = dataset[0]
        out.append((s["edit_name"], s["prompt"], tuple(s["suppress_indices"]),
                    image_digest(s["A"]), image_digest(s["B_prime"]), image_digest(s["A_prime_partial"])))
    return out


def pair_identities(dataset, seed):
    random.seed(seed)
    out = []
    for _ in range(N_SAMPLES):
        s = dataset[0]
        out.append((s["edit_name"], s["prompt"], image_digest(s["A"]), image_digest(s["B_prime"])))
    return out


@requires_dataset
def test_selective_sampling_is_deterministic(selective_dataset):
    first = selective_identities(selective_dataset, seed=1)
    assert selective_identities(selective_dataset, seed=1) == first
    assert selective_identities(selective_dataset, seed=2) != first


@requires_dataset
def test_pair_sampling_is_deterministic(pair_dataset):
    first = pair_identities(pair_dataset, seed=1)
    assert pair_identities(pair_dataset, seed=1) == first
    assert pair_identities(pair_dataset, seed=2) != first


def test_pair_prompt_keeps_the_original_edit_numbers():
    # Unlike build_analogy_prompt, the AnaLoRA prompt is NOT renumbered: a partial edit {2} is listed
    # as "Edit 2". Merging the two builders would change the AnaLoRA training prompts.
    edits = {"edit1": "add hat", "edit2": "add glasses", "edit3": ""}
    assert _build_prompt_from_indices([2], edits) == (
        "Image 1 is the original and image 2 is the edited version. "
        "Edit 2: add glasses. "
        "Apply the same edits to image 3 to produce the output."
    )
    assert _build_prompt_from_indices([3, 1], edits) == (
        "Image 1 is the original and image 2 is the edited version. "
        "Edit 1: add hat. "
        "Apply the same edits to image 3 to produce the output."
    )
