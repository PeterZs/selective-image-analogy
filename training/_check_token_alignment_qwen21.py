#!/usr/bin/env python3
"""
_check_token_alignment_qwen21.py

Qwen-Image-2.1 counterpart of _check_token_alignment.py.

ExLoRA's mask and the CSV both act on the rows of ``prompt_embeds``. For
Qwen-Image-2.1 those rows come from the Qwen3-VL processor's sequence, in
which each condition image is expanded into one ``<|image_pad|>`` per vision
slot and the system prompt is dropped. ExLoRA then needs the mask in the
transformer's joint sequence, where each vision slot becomes 4 latent tokens.

This script needs only the processor (CPU, no model load) and checks:
  - our expanded-text tokenization reproduces the processor's input_ids
    (so offsets computed on it are valid),
  - every edit's positions decode back to the edit text,
  - ExLoRA mask positions == CSV positions, and none sit on a vision slot,
  - the joint-sequence mask lands on the same tokens after slot expansion,
  - the mask builder raises (not silently zeroes) on a missing edit.

Usage:
  python training/_check_token_alignment_qwen21.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from diffusers import QwenImage21Pipeline  # noqa: E402
from diffusers.models.transformers.transformer_qwenimage21 import _IMG_TOKENS_PER_SLOT  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import Qwen3VLProcessor  # noqa: E402

from sia.utils.qwen21 import (  # noqa: E402
    qwen21_find_substring_token_indices,
    qwen21_joint_token_mask,
    qwen21_prompt_token_positions,
    qwen21_resize_dims,
    qwen21_templated_text,
    qwen21_text_token_mask,
    qwen21_tokenize_with_offsets,
)
from sia.prompt import build_analogy_prompt  # noqa: E402
from training._check_token_alignment import EDITS_BY_N, _check, _green, _red  # noqa: E402

# (w, h) of the raw condition images: the dataset's two sizes plus a landscape one.
RAW_SIZES = [(384, 512), (368, 512), (640, 480)]


def _run_checks_for_n(n, edits, pipe, resolution) -> bool:
    print()
    print(f"=== N={n} edits, resolution={resolution} ===")
    prompt = build_analogy_prompt(edits, edits)
    sizes = [qwen21_resize_dims(w, h, resolution) for w, h in RAW_SIZES]
    images = [Image.new("RGB", s, (40 * i, 80, 120)) for i, s in enumerate(sizes)]
    print(f"condition sizes: {sizes}")

    all_ok = True

    # 1. Expanded-text tokenization == processor tokenization (same call the pipeline makes)
    _, ids, _ = qwen21_tokenize_with_offsets(pipe, prompt, sizes)
    proc_ids = pipe.processor(
        text=[qwen21_templated_text(pipe, prompt, len(images))], images=images,
        padding=True, padding_side="left", return_tensors="pt",
    )["input_ids"][0].tolist()
    all_ok &= _check(ids == proc_ids,
                     f"  expanded tokenization matches processor ({len(ids)} tokens)")

    rows = proc_ids[pipe._drop_idx:]          # == rows of prompt_embeds (batch of one)
    image_pad = torch.tensor([t == pipe._img_token_id for t in rows])

    # 2. Each edit aligns, decodes back to itself, and avoids vision slots
    for i, edit in enumerate(edits):
        pos = qwen21_find_substring_token_indices(pipe, prompt, edit, sizes)
        decoded = pipe.processor.tokenizer.decode([rows[p] for p in pos])
        all_ok &= _check(decoded.strip() == edit.strip(),
                         f"  edit {i+1} aligns: rows={pos[0]}..{pos[-1]} decoded={decoded!r}")
        all_ok &= _check(not bool(image_pad[pos].any()),
                         f"  edit {i+1} positions are text tokens")

    # 3. ExLoRA mask == CSV positions, and subset of scope='prompt' positions
    prompt_rows = set(qwen21_prompt_token_positions(pipe, prompt, sizes))
    for i, edit in enumerate(edits):
        mask = qwen21_text_token_mask(pipe, prompt, [edit], sizes, len(rows), "cpu")
        mask_pos = set(torch.nonzero(mask[0]).flatten().tolist())
        csv_pos = set(qwen21_find_substring_token_indices(pipe, prompt, edit, sizes))
        all_ok &= _check(mask_pos == csv_pos and mask_pos <= prompt_rows,
                         f"  edit {i+1} ExLoRA mask positions == CSV positions")

    # 4. Joint-sequence mapping: same tokens, shifted by 3 per vision slot before them
    full_mask = qwen21_text_token_mask(pipe, prompt, edits, sizes, len(rows), "cpu")
    target_tokens = (sizes[-1][0] // 16) * (sizes[-1][1] // 16)
    img_mask = torch.cat([image_pad, torch.ones(target_tokens // _IMG_TOKENS_PER_SLOT, dtype=torch.bool)])[None]
    joint = qwen21_joint_token_mask(full_mask, img_mask)
    n_slots = int(image_pad.sum())
    expected_len = len(rows) - n_slots + n_slots * _IMG_TOKENS_PER_SLOT + target_tokens
    expected = set()
    for p in torch.nonzero(full_mask[0]).flatten().tolist():
        expected.add(p + (_IMG_TOKENS_PER_SLOT - 1) * int(image_pad[:p].sum()))
    all_ok &= _check(joint.shape[1] == expected_len and set(torch.nonzero(joint[0]).flatten().tolist()) == expected,
                     f"  joint mask: len={joint.shape[1]} (expected {expected_len}), "
                     f"{int(joint.sum())} tokens at the shifted positions")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--resolution", type=int, nargs="+", default=[512, 1024])
    args = parser.parse_args()

    processor = Qwen3VLProcessor.from_pretrained(args.pretrained, subfolder="processor")
    # Processor-only pipeline: enough for the templates and _drop_idx.
    pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=None,
                               processor=processor, transformer=None)

    all_ok = True
    for resolution in args.resolution:
        for n in sorted(EDITS_BY_N):
            all_ok &= _run_checks_for_n(n, EDITS_BY_N[n], pipe, resolution)

    print()
    print("=== Regression canaries ===")
    edits = EDITS_BY_N[2]
    prompt = build_analogy_prompt(edits, edits)
    sizes = [qwen21_resize_dims(w, h, 512) for w, h in RAW_SIZES]
    raised = False
    try:
        qwen21_text_token_mask(pipe, prompt, ["this exact phrase is not in the prompt 12345"],
                               sizes, 4096, "cpu")
    except RuntimeError:
        raised = True
    all_ok &= _check(raised, "mask builder RAISES on missing edit (no silent empty mask)")

    # Text-only (t2i template) alignment, used by the isolated-sentence CSV pairs
    sentence = "A bright living room with large windows."
    pos = qwen21_find_substring_token_indices(pipe, sentence, "bright", [])
    _, ids, _ = qwen21_tokenize_with_offsets(pipe, sentence, [])
    decoded = processor.tokenizer.decode([ids[pipe._drop_idx + p] for p in pos])
    all_ok &= _check(decoded.strip() == "bright", f"t2i template alignment: decoded={decoded!r}")

    if all_ok:
        print(_green("\nAll Qwen-Image-2.1 token-alignment checks passed."))
        sys.exit(0)
    print(_red("\nOne or more Qwen-Image-2.1 token-alignment checks FAILED."))
    sys.exit(1)


if __name__ == "__main__":
    main()
