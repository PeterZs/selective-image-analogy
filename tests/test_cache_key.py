"""CSV cache keys: the folder name of a cached direction must never change.

The contrast pairs behind a cached direction were sampled without a seed, so a changed key would
silently resample them and change every CSV result. Each backbone has its own scheme (``cache_key``):

- FLUX.2-Klein: ``sha1(edit)`` or ``sha1(prompt + "\\0" + edit)`` (in-context mode), first 12 hex digits;
- Qwen-Image-2.1: the same, prefixed with ``b"qwen-image-2.1\\0"`` and followed by the condition images'
  size, mode and bytes (in-context mode).

The fixed pairs below were computed with the original functions. If ``SIA_REFERENCE_DIR`` points to a
directory holding the original Qwen cache (``steering_cache_qwen21``, 127 folders), every folder name is
recomputed from its ``meta.json`` as well.
"""
import json
import os
from pathlib import Path

import pytest
from PIL import Image

REFERENCE_DIR = os.environ.get("SIA_REFERENCE_DIR")


def backbone(name):
    try:
        return __import__(f"sia.backbones.{name}", fromlist=["cache_key"])
    except ImportError as exc:  # each environment has the diffusers version of one backbone only
        pytest.skip(f"sia.backbones.{name} needs the other environment ({exc})")


def images():
    return [Image.new("RGB", (4, 3), (10, 20, 30)), Image.new("RGBA", (2, 2), (1, 2, 3, 4))]


KLEIN_KEYS = [
    ("add sunglasses", None, "ea7d82db3dfc"),
    ("winter background", None, "63ed020be1ff"),
    ("add a floral wreath crown", None, "939f1b86eb5d"),
    ("yawning expression", "Image 1 is the original. Edit 1: yawning expression.", "820f07b0e032"),
]

# (edit, in-context prompt, number of condition images from ``images()``, key)
QWEN_KEYS = [
    ("add sunglasses", None, 0, "aa3252d218b8"),
    ("winter background", None, 0, "20fcb797d51a"),
    ("add a nose ring", "Edit 1: add a nose ring.", 1, "4302552bd0bf"),
    ("add star sunglasses", "Edit 1: add star sunglasses.", 2, "dd8b49048456"),
]


@pytest.mark.parametrize("edit, prompt, key", KLEIN_KEYS)
def test_klein_cache_key(edit, prompt, key):
    assert backbone("flux2_klein").cache_key(edit, prompt) == key


@pytest.mark.parametrize("edit, prompt, n_images, key", QWEN_KEYS)
def test_qwen_cache_key(edit, prompt, n_images, key):
    vl_images = images()[:n_images] or None
    assert backbone("qwen_image21").cache_key(edit, prompt, vl_images) == key


def test_qwen_cache_folders_hash_to_their_names():
    if not REFERENCE_DIR or not (Path(REFERENCE_DIR) / "steering_cache_qwen21").is_dir():
        pytest.skip("set SIA_REFERENCE_DIR to a directory containing steering_cache_qwen21/")
    QWEN_CACHE = Path(REFERENCE_DIR) / "steering_cache_qwen21"
    cache_key = backbone("qwen_image21").cache_key
    folders = sorted(p for p in QWEN_CACHE.iterdir() if p.is_dir())
    assert len(folders) == 127
    for folder in folders:
        meta = json.loads((folder / "meta.json").read_text())
        assert meta["method"] == "qwen_isolated"   # isolated mode: the key depends on the edit only
        assert (folder / "pairs.jsonl").exists()
        assert cache_key(meta["edit"]) == folder.name, meta["edit"]
