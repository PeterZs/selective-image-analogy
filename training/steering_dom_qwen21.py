"""Old Qwen-Image-2.1 CSV entry point, kept with the other training/ shims until the scripts merge.

The code now lives in ``sia.concept_suppression`` (backbone-free) and ``sia.backbones.qwen_image21``
(``cache_key``, ``steering_pool``, ``generate_contrast_pairs``).
"""
from functools import partial

from sia.backbones import qwen_image21
from sia.concept_suppression import get_or_build_steering_vector

# Same positional arguments as before: (edit, n_pairs, pipe, device, cache_root, regen, ...).
get_or_build_steering_vector_qwen21 = partial(get_or_build_steering_vector, qwen_image21)
