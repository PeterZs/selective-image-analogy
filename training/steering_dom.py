"""Old FLUX.2-Klein CSV entry points, kept only for the external regression harness (``klein_generate_dev.py``).

The code now lives in ``sia.concept_suppression`` (backbone-free) and ``sia.backbones.flux2_klein``
(``cache_key``, ``steering_pool``, ``generate_contrast_pairs``). Remove together with the other training/ shims.
"""
from pathlib import Path
from types import SimpleNamespace

from sia import concept_suppression
from sia.backbones import flux2_klein
from sia.concept_suppression import _CACHE_VERSION, _read_jsonl, _write_jsonl  # noqa: F401


def _edit_cache_dir(cache_root, edit, prompt=None):
    return Path(cache_root) / flux2_klein.cache_key(edit, prompt)


def compute_dom_vector(pairs, text_encoder, tokenizer, device, max_sequence_length=512):
    pipe = SimpleNamespace(text_encoder=text_encoder, tokenizer=tokenizer)
    return concept_suppression.compute_dom_vector(flux2_klein, pairs, pipe, device)


def get_or_build_steering_vector(edit, n_pairs, text_encoder, tokenizer, device, cache_root, regen=False):
    pipe = SimpleNamespace(text_encoder=text_encoder, tokenizer=tokenizer)
    return concept_suppression.get_or_build_steering_vector(
        flux2_klein, edit, n_pairs, pipe, device, cache_root, regen)
