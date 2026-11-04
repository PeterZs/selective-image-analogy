"""Re-export of the FLUX.2-Klein token helpers, now in ``sia.backbones.flux2_klein``.

Kept only for the external regression harness (``klein_generate_dev.py``); remove together with the
training/ shims once the harness imports from ``sia.backbones``.
"""
from sia.backbones.flux2_klein import (  # noqa: F401
    _apply_klein_chat_template,
    klein_find_substring_token_indices,
    klein_templated_prompt_token_positions,
)
