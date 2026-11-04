"""Token-position helpers for Qwen-Image-2.1 (``QwenImage21Pipeline``).

Qwen-Image-2.1 differs from FLUX.2-Klein in two ways that matter for the
token mask shared by ExLoRA and the CSV:

1. The prompt is encoded by Qwen3-VL *together with* the condition images.
   Each ``<|image_pad|>`` placeholder in the template is expanded by the
   processor into one token per vision slot, and the leading system-prompt
   tokens are dropped afterwards (``pipe._drop_idx``). The rows of
   ``prompt_embeds`` are therefore positions in that expanded, truncated
   sequence — not in the raw prompt and not in the chat template.

2. The transformer is single-stream. It builds one joint sequence in which
   every vision slot is replaced by ``_IMG_TOKENS_PER_SLOT`` (=4) latent
   tokens and the target image's tokens are appended. ExLoRA wraps the
   shared projections, so its mask has to be expressed in that joint
   sequence (see ``qwen21_joint_token_mask``).

Every position helper here mirrors ``QwenImage21Pipeline._get_qwen_prompt_embeds``
and is checked against the processor's own ``input_ids`` by
``training/_check_token_alignment_qwen21.py``.
"""

import torch
from diffusers.models.transformers.transformer_qwenimage21 import _IMG_TOKENS_PER_SLOT
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

_VISION_SLOT = "<image{}><|vision_start|><|image_pad|><|vision_end|>"


def qwen21_resize_dims(width: int, height: int, resolution: int):
    """Size the pipeline resizes a condition image to: ``resolution**2`` area
    at the image's own aspect ratio, sides rounded to multiples of 32 (one
    vision slot = one 2x2 group of 16x latent tokens). The output image takes
    the last condition image's size."""
    w, h, _ = calculate_dimensions(resolution * resolution, width / height)
    return w, h


def qwen21_templated_text(pipe, prompt: str, n_images: int) -> str:
    """The exact string ``_get_qwen_prompt_embeds`` hands to the processor for
    one prompt with ``n_images`` condition images (before pad expansion)."""
    # The pipeline swaps an empty prompt for " " (Qwen has no bos token).
    prompt = prompt or " "
    if n_images == 0:
        return pipe.prompt_template_t2i.format(prompt)
    slots = _VISION_SLOT.format(1) + "".join(
        " " + _VISION_SLOT.format(i) for i in range(2, n_images + 1))
    return pipe.prompt_template_ti2i.replace(_VISION_SLOT.format(1), slots).format(prompt)


def qwen21_image_slot_counts(processor, image_sizes) -> list:
    """Vision slots (``<|image_pad|>`` tokens) per image of size ``(w, h)``,
    as computed by the processor's own ``smart_resize``."""
    image_processor = processor.image_processor
    merge = image_processor.merge_size ** 2
    return [image_processor.get_number_of_image_patches(h, w, {}) // merge
            for w, h in image_sizes]


def qwen21_expanded_text(pipe, prompt: str, image_sizes) -> str:
    """Templated text with every ``<|image_pad|>`` repeated once per vision
    slot — the text the processor actually tokenizes."""
    text = qwen21_templated_text(pipe, prompt, len(image_sizes))
    pad = pipe.processor.image_token
    pieces = text.split(pad)
    if len(pieces) != len(image_sizes) + 1:
        raise ValueError(
            f"Expected {len(image_sizes)} {pad} placeholders in the template, "
            f"found {len(pieces) - 1} (does the prompt contain {pad}?).")
    counts = qwen21_image_slot_counts(pipe.processor, image_sizes)
    out = pieces[0]
    for n, piece in zip(counts, pieces[1:]):
        out += pad * n + piece
    return out


def _prompt_start(text: str, has_images: bool) -> int:
    """Character offset where the user prompt begins in the templated text."""
    if has_images:
        return text.rindex("<|vision_end|>") + len("<|vision_end|>")
    marker = "<|im_start|>user\n"
    return text.index(marker) + len(marker)


def qwen21_tokenize_with_offsets(pipe, prompt: str, image_sizes):
    """Tokenize the expanded text exactly as the processor does. Returns
    ``(text, input_ids, offsets)`` over the full (un-dropped) sequence."""
    text = qwen21_expanded_text(pipe, prompt, image_sizes)
    enc = pipe.processor.tokenizer(text, return_offsets_mapping=True)
    return text, enc["input_ids"], enc["offset_mapping"]


def qwen21_find_substring_token_indices(
    pipe,
    prompt: str,
    substr: str,
    image_sizes,
    case_sensitive: bool = True,
):
    """Return the rows of ``prompt_embeds`` covering ``substr`` in ``prompt``.

    ``image_sizes`` are the ``(w, h)`` of the condition images exactly as they
    are passed to the text encoder (after the pipeline's resize); use ``[]``
    for a text-only (t2i template) encode. Indices already account for the
    dropped system-prompt tokens, so they index ``prompt_embeds`` directly.

    Like ``klein_find_substring_token_indices``, the ``"Edit N: <substr>."``
    occurrence is preferred so an edit that also appears elsewhere in the
    prompt is not matched there. Raises ``AssertionError`` if not found.
    """
    text, _, offsets = qwen21_tokenize_with_offsets(pipe, prompt, image_sizes)
    start = _prompt_start(text, bool(image_sizes))
    haystack, needle = (text, substr) if case_sensitive else (text.lower(), substr.lower())

    anchored = haystack.find(f": {needle}.", start)
    char_start = anchored + 2 if anchored != -1 else haystack.find(needle, start)
    if char_start == -1:
        raise AssertionError(f"substr not found in templated prompt text: '{substr[:80]}'")
    char_end = char_start + len(substr)

    token_indices = [i for i, (cs, ce) in enumerate(offsets) if ce > char_start and cs < char_end]
    if not token_indices:
        raise AssertionError(
            f"No tokens overlap with substr chars [{char_start}:{char_end}]: '{substr[:80]}'")
    if token_indices[0] < pipe._drop_idx:
        raise AssertionError(f"'{substr[:80]}' matched inside the dropped system prompt.")
    return [i - pipe._drop_idx for i in token_indices]


def qwen21_prompt_token_positions(pipe, prompt: str, image_sizes):
    """Every text (non-vision-slot) row of ``prompt_embeds`` — the footprint
    of ``csv_scope='prompt'``. Vision-slot rows are skipped: the transformer
    overwrites them with image latents, so steering them has no effect."""
    _, ids, _ = qwen21_tokenize_with_offsets(pipe, prompt, image_sizes)
    ids = ids[pipe._drop_idx:]
    return [i for i, tok in enumerate(ids) if tok != pipe._img_token_id]


def qwen21_text_token_mask(pipe, prompt: str, edit_texts, image_sizes, seq_len: int, device):
    """``(1, seq_len)`` bool mask over ``prompt_embeds`` rows covering every
    edit in ``edit_texts``. Raises if an edit cannot be located — an empty
    mask silently disables ExLoRA."""
    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    for text in edit_texts:
        if not text:
            continue
        try:
            indices = qwen21_find_substring_token_indices(pipe, prompt, text, image_sizes)
        except AssertionError as exc:
            raise RuntimeError(
                f"[TokenMask] Could not locate edit text in prompt:\n"
                f"  edit:   {text!r}\n"
                f"  prompt: {prompt!r}\n"
                f"  underlying: {exc}"
            ) from exc
        if indices[-1] >= seq_len:
            raise RuntimeError(
                f"[TokenMask] Edit {text!r} maps to row {indices[-1]} but prompt_embeds has "
                f"{seq_len} rows — image sizes differ from the ones that were encoded.")
        mask[0, indices] = True
    if edit_texts and not bool(mask.any()):
        raise RuntimeError("[TokenMask] All edit texts produced empty positions.")
    return mask


def qwen21_joint_token_mask(text_mask: torch.Tensor, img_mask: torch.Tensor) -> torch.Tensor:
    """Map a ``(B, S)`` mask over ``prompt_embeds`` rows onto the joint
    sequence built by ``QwenImage21Transformer2DModel.forward``.

    ``img_mask`` is the transformer's ``img_mask`` argument: ``(B, S + T/4)``,
    True at vision slots and at the appended target slots. Each True slot
    becomes ``_IMG_TOKENS_PER_SLOT`` joint tokens, every other position one.
    """
    if bool((text_mask & img_mask[:, : text_mask.shape[1]]).any()):
        raise ValueError("Token mask marks vision-slot positions; it must cover text tokens only.")
    repeats = torch.where(img_mask[0], _IMG_TOKENS_PER_SLOT, 1)
    extended = torch.cat(
        [text_mask, text_mask.new_zeros(text_mask.shape[0], img_mask.shape[1] - text_mask.shape[1])],
        dim=1,
    )
    return torch.repeat_interleave(extended, repeats, dim=1)
