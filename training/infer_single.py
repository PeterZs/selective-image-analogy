#!/usr/bin/env python3
"""
infer_single.py

Single-image analogy inference with all suppression modes.

Given A, A', B and a list of edits [e1, e2, ..., eN], runs:
  - analora          : AnaLoRA only, ExLoRA off
  - exlora (no supp)   : ExLoRA on, no mask
  - exlora (supp e1)   : ExLoRA on, mask e1 tokens
  - exlora (supp e2)   : ExLoRA on, mask e2 tokens
  ...
  - exlora (supp all)  : ExLoRA on, mask all edit tokens

Saves a strip:  A | A' | B | analora | exlora_none | exlora_e1 | ... | exlora_all

Usage:
  python training/infer_single.py \\
      --config   training/configs/train_exlora_flux2_klein.yaml \\
      --image_a       /path/to/A.png \\
      --image_a_prime /path/to/A_prime.png \\
      --image_b       /path/to/B.png \\
      --edits "X" "Y" "Z" \\
      --output /tmp/result.png
"""

import argparse
import itertools
import os
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from sia.backbones.flux2_klein import (
    compute_text_embeddings,
    klein_templated_prompt_token_positions,
    load_pipeline,
    run_inference,
    templated_substring_indices,
)
from sia.prompt import build_analogy_prompt
# Imported for its side effect: the accelerate compatibility patch must be active before models load.
import training.train_exlora_flux2_klein  # noqa: F401
from sia.backbones import flux2_klein
from sia.concept_suppression import (
    STEERING_MODES,
    STEERING_SCOPES,
    alpha_at_step,
    apply_csv,
    format_schedule_tag,
    get_or_build_steering_vector,
    parse_embedding_schedule,
)


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
]

def _load_font(size):
    for fp in _FONT_CANDIDATES:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return None


def _fit_thumbnail(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Scale ``img`` to fit inside (max_w, max_h) while preserving aspect ratio.
    The result is centred on a white background of exactly (max_w, max_h)."""
    src_w, src_h = img.size
    scale = min(max_w / src_w, max_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (max_w, max_h), (255, 255, 255))
    off_x = (max_w - new_w) // 2
    off_y = (max_h - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    return canvas


def _cell_dims(ref_img: Image.Image, cell_size: int) -> tuple:
    """Derive (cell_w, cell_h) for grid cells from a reference image's aspect
    ratio, bounded by ``cell_size``. The larger side is always exactly
    ``cell_size``; the smaller side scales down proportionally."""
    src_w, src_h = ref_img.size
    if src_w >= src_h:
        cell_w = cell_size
        cell_h = max(1, int(cell_size * src_h / src_w))
    else:
        cell_h = cell_size
        cell_w = max(1, int(cell_size * src_w / src_h))
    return cell_w, cell_h


SCALE_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]  # overridden by --scale_values


# ---------------------------------------------------------------------------
# Image annotation helper
# ---------------------------------------------------------------------------

def annotate_image(img, label, suppressed_texts, font_size=16):
    """Return a new image with a white banner below containing label + suppressed edits."""
    font       = _load_font(font_size)
    line_h     = font_size + 4
    col_w      = img.width
    chars_per_line = max(30, int(col_w / (font_size * 0.55)))

    lines = [label.replace("\n", " ")]
    if suppressed_texts:
        supp_str = "Suppressed: " + "; ".join(suppressed_texts)
        lines += textwrap.wrap(supp_str, chars_per_line)
    else:
        lines.append("Suppressed: (none)")

    banner_h = len(lines) * line_h + 8
    out = Image.new("RGB", (col_w, img.height + banner_h), (255, 255, 255))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        color = (160, 0, 0) if i > 0 and suppressed_texts else (0, 0, 0)
        draw.text((4, img.height + 4 + i * line_h), line,
                  fill=color, font=font)
    return out


# ---------------------------------------------------------------------------
# Combined 2-D grid builder
# ---------------------------------------------------------------------------

def build_combined_grid(image_a, image_a_prime, image_b,
                        row_descs,    # [(label, suppress_list), ...]
                        all_preds,    # dict: (row_label, st_s) -> PIL.Image
                        scale_values, # list of exlora scale floats
                        cell_size=256,
                        analora_image=None):
    """
    Build a single 2-D grid:
      Rows    : suppression modes
      Columns : exlora_scale values (left → right: 0 → 1)
    Plus a reference strip (A | A' | B | analora) and column headers above the grid.
    """
    n_rows = len(row_descs)
    n_cols = len(scale_values)
    font_size = 14
    font      = _load_font(font_size)
    line_h    = font_size + 4

    # Cell dims follow B's aspect ratio so outputs and references share shape.
    cell_w, cell_h = _cell_dims(image_b, cell_size)

    header_h    = 32               # title banner
    col_hdr_h   = 24               # column header row
    row_hdr_w   = 140              # left margin for row labels
    ref_strip_h = cell_h + 20

    ref_images = [(image_a, "A"), (image_a_prime, "A'"), (image_b, "B")]
    if analora_image is not None:
        ref_images.append((analora_image, "analora"))
    n_ref_cols = len(ref_images)

    total_w = row_hdr_w + max(n_cols, n_ref_cols) * cell_w
    total_h = header_h + ref_strip_h + col_hdr_h + n_rows * cell_h

    img  = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # --- Title banner ---
    draw.rectangle([0, 0, total_w - 1, header_h - 1], fill=(210, 225, 255))
    draw.text((6, 8), "ExLoRA ablation:  suppress mode (rows) × scale (cols)",
              fill=(0, 0, 100), font=font)

    # --- Reference thumbnails: A | A' | B | analora (same cell size) ---
    ref_y = header_h
    for ci, (ref_img, lbl) in enumerate(ref_images):
        x = row_hdr_w + ci * cell_w
        thumb = _fit_thumbnail(ref_img, cell_w, cell_h)
        img.paste(thumb, (x, ref_y + 20))
        draw.text((x + 4, ref_y + 2), lbl, fill=(0, 0, 0), font=font)

    # --- Column headers: exlora scale ---
    col_hdr_y = header_h + ref_strip_h
    draw.rectangle([0, col_hdr_y, row_hdr_w - 1, col_hdr_y + col_hdr_h - 1],
                   fill=(230, 230, 230))
    draw.text((4, col_hdr_y + 4), "supp \\ scale", fill=(80, 80, 80), font=font)
    for ci, sv in enumerate(scale_values):
        x = row_hdr_w + ci * cell_w
        draw.rectangle([x, col_hdr_y, x + cell_w - 1, col_hdr_y + col_hdr_h - 1],
                       fill=(220, 240, 255))
        draw.text((x + 4, col_hdr_y + 4), f"st={sv:.2g}", fill=(0, 0, 0), font=font)

    # --- Grid cells ---
    grid_y0 = col_hdr_y + col_hdr_h
    for ri, (row_label, suppress) in enumerate(row_descs):
        y  = grid_y0 + ri * cell_h
        bg = (245, 245, 255) if ri % 2 == 0 else (235, 235, 250)
        draw.rectangle([0, y, row_hdr_w - 1, y + cell_h - 1], fill=bg)
        draw.text((4, y + 4), row_label, fill=(0, 0, 100), font=font)
        if suppress:
            supp_txt = "supp: " + "; ".join(s[:30] for s in suppress)
            for li, part in enumerate(textwrap.wrap(supp_txt, 18)):
                draw.text((4, y + 4 + (li + 1) * line_h), part,
                          fill=(160, 0, 0), font=font)

        for ci, sv in enumerate(scale_values):
            x    = row_hdr_w + ci * cell_w
            pred = all_preds.get((row_label, sv))
            if pred is not None:
                img.paste(_fit_thumbnail(pred, cell_w, cell_h), (x, y))
            draw.text((x + 3, y + 3), f"st={sv:.2g}", fill=(255, 255, 200), font=font)

    return img


# ---------------------------------------------------------------------------
# 3-axis grid: suppression row → block of (exlora × embedding) sub-grids
# ---------------------------------------------------------------------------

def build_joint_grid(image_a, image_a_prime, image_b,
                     row_descs,     # [(label, suppress_list, suppress_idx), ...]
                     all_preds,     # dict: (row_label, st_s, emb_s) -> PIL.Image
                     st_scales,     # list of exlora floats (inner rows)
                     emb_scales,    # list of embedding alphas (inner cols)
                     cell_size=256,
                     analora_image=None):
    """
    Outer rows   : suppression mode
    Inner grid   : exlora_scale (rows, top→bottom: first→last) × embedding α (cols)
    Reference strip A | A' | B | analora on top.
    """
    n_sup = len(row_descs)
    n_st  = len(st_scales)
    n_emb = len(emb_scales)
    font_size = 14
    font      = _load_font(font_size)
    line_h    = font_size + 4

    # Cells follow B's aspect ratio so references and outputs share pixel size.
    cell_w, cell_h = _cell_dims(image_b, cell_size)

    header_h    = 32
    col_hdr_h   = 24
    row_hdr_w   = 170               # outer row label
    inner_lbl_w = 54                # inner row label (exlora scale)
    ref_strip_h = cell_h + 20

    ref_images = [(image_a, "A"), (image_a_prime, "A'"), (image_b, "B")]
    if analora_image is not None:
        ref_images.append((analora_image, "analora"))
    n_ref_cols = len(ref_images)

    inner_w = inner_lbl_w + n_emb * cell_w
    inner_h = col_hdr_h + n_st  * cell_h
    total_w = max(row_hdr_w + inner_w, row_hdr_w + n_ref_cols * cell_w)
    total_h = header_h + ref_strip_h + n_sup * inner_h

    img  = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.rectangle([0, 0, total_w - 1, header_h - 1], fill=(210, 225, 255))
    draw.text(
        (6, 8),
        "Joint ablation — outer rows: suppress mode | inner rows: ExLoRA scale "
        "| inner cols: text-embedding alpha",
        fill=(0, 0, 100), font=font,
    )

    # Reference thumbnails A | A' | B | analora (same cell size as outputs)
    ref_y = header_h
    for ci, (ref_img, lbl) in enumerate(ref_images):
        x = row_hdr_w + ci * cell_w
        thumb = _fit_thumbnail(ref_img, cell_w, cell_h)
        img.paste(thumb, (x, ref_y + 20))
        draw.text((x + 4, ref_y + 2), lbl, fill=(0, 0, 0), font=font)

    inner_y0 = header_h + ref_strip_h
    for ri, (row_label, suppress, _idx) in enumerate(row_descs):
        outer_y = inner_y0 + ri * inner_h

        # Outer row label cell
        bg = (245, 245, 255) if ri % 2 == 0 else (235, 235, 250)
        draw.rectangle(
            [0, outer_y, row_hdr_w - 1, outer_y + inner_h - 1], fill=bg)
        draw.text((4, outer_y + 4), row_label, fill=(0, 0, 100), font=font)
        if suppress:
            supp_txt = "supp: " + "; ".join(s[:24] for s in suppress)
            for li, part in enumerate(textwrap.wrap(supp_txt, 22)):
                draw.text(
                    (4, outer_y + 4 + (li + 1) * line_h),
                    part, fill=(160, 0, 0), font=font)

        # Inner column header (embedding alphas)
        col_hdr_y = outer_y
        draw.rectangle(
            [row_hdr_w, col_hdr_y,
             row_hdr_w + inner_lbl_w - 1, col_hdr_y + col_hdr_h - 1],
            fill=(230, 230, 230))
        draw.text((row_hdr_w + 3, col_hdr_y + 4),
                  "st\\emb", fill=(80, 80, 80), font=font)
        for ci, emb_s in enumerate(emb_scales):
            x = row_hdr_w + inner_lbl_w + ci * cell_w
            draw.rectangle(
                [x, col_hdr_y, x + cell_w - 1, col_hdr_y + col_hdr_h - 1],
                fill=(220, 240, 255))
            draw.text((x + 4, col_hdr_y + 4),
                      f"emb={emb_s:.2g}", fill=(0, 0, 0), font=font)

        # Inner grid cells
        cells_y0 = col_hdr_y + col_hdr_h
        for ii, st_s in enumerate(st_scales):
            y = cells_y0 + ii * cell_h

            # Inner row label (exlora scale)
            draw.rectangle(
                [row_hdr_w, y, row_hdr_w + inner_lbl_w - 1, y + cell_h - 1],
                fill=(245, 245, 245))
            draw.text((row_hdr_w + 3, y + 4),
                      f"st={st_s:.2g}", fill=(0, 0, 0), font=font)

            for ci, emb_s in enumerate(emb_scales):
                x = row_hdr_w + inner_lbl_w + ci * cell_w
                pred = all_preds.get((row_label, st_s, emb_s))
                if pred is not None:
                    img.paste(_fit_thumbnail(pred, cell_w, cell_h), (x, y))
                draw.text(
                    (x + 3, y + 3),
                    f"st={st_s:.2g},emb={emb_s:.2g}",
                    fill=(255, 255, 200), font=font)

    return img


# ---------------------------------------------------------------------------
# Per-row grid: one suppression mode — rows=ExLoRA scale, cols=embedding alpha
# ---------------------------------------------------------------------------

def build_row_grid(image_a, image_a_prime, image_b,
                   row_label, suppress,
                   row_preds,    # dict: (st_s, emb_s) -> PIL.Image
                   st_scales, emb_scales,
                   cell_size=256,
                   analora_image=None):
    """Save a self-contained grid for a single suppression row.
    Layout: rows = ExLoRA scale, cols = embedding alpha.
    Same visual style as ``build_joint_grid``'s inner block, plus a reference
    strip (A | A' | B | analora) at the top.
    """
    n_st  = len(st_scales)
    n_emb = len(emb_scales)
    font_size = 14
    font      = _load_font(font_size)

    # Cells follow B's aspect ratio so references and outputs share pixel size.
    cell_w, cell_h = _cell_dims(image_b, cell_size)

    header_h    = 32
    col_hdr_h   = 24
    row_hdr_w   = 70               # inner row label (st scale)
    ref_strip_h = cell_h + 20

    ref_images = [(image_a, "A"), (image_a_prime, "A'"), (image_b, "B")]
    if analora_image is not None:
        ref_images.append((analora_image, "analora"))
    n_ref_cols = len(ref_images)

    total_w = row_hdr_w + max(n_emb, n_ref_cols) * cell_w
    total_h = header_h + ref_strip_h + col_hdr_h + n_st * cell_h

    img  = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.rectangle([0, 0, total_w - 1, header_h - 1], fill=(210, 225, 255))
    title = f"{row_label}  |  rows: ExLoRA scale  |  cols: embedding alpha"
    draw.text((6, 8), title, fill=(0, 0, 100), font=font)

    # Reference thumbnails A | A' | B | analora (same pixel size as output cells)
    ref_y = header_h
    for ci, (ref_img, lbl) in enumerate(ref_images):
        x = row_hdr_w + ci * cell_w
        thumb = _fit_thumbnail(ref_img, cell_w, cell_h)
        img.paste(thumb, (x, ref_y + 20))
        draw.text((x + 4, ref_y + 2), lbl, fill=(0, 0, 0), font=font)

    # Column headers (embedding alpha)
    col_hdr_y = header_h + ref_strip_h
    draw.rectangle([0, col_hdr_y, row_hdr_w - 1, col_hdr_y + col_hdr_h - 1],
                   fill=(230, 230, 230))
    draw.text((4, col_hdr_y + 4), "st\\emb", fill=(80, 80, 80), font=font)
    for ci, emb_s in enumerate(emb_scales):
        x = row_hdr_w + ci * cell_w
        draw.rectangle([x, col_hdr_y, x + cell_w - 1, col_hdr_y + col_hdr_h - 1],
                       fill=(220, 240, 255))
        draw.text((x + 4, col_hdr_y + 4),
                  f"emb={emb_s:.2g}", fill=(0, 0, 0), font=font)

    # Grid cells
    grid_y0 = col_hdr_y + col_hdr_h
    for ii, st_s in enumerate(st_scales):
        y = grid_y0 + ii * cell_h
        draw.rectangle([0, y, row_hdr_w - 1, y + cell_h - 1],
                       fill=(245, 245, 245))
        draw.text((4, y + 4), f"st={st_s:.2g}", fill=(0, 0, 0), font=font)

        for ci, emb_s in enumerate(emb_scales):
            x = row_hdr_w + ci * cell_w
            pred = row_preds.get((st_s, emb_s))
            if pred is not None:
                img.paste(_fit_thumbnail(pred, cell_w, cell_h), (x, y))
            draw.text((x + 3, y + 3),
                      f"st={st_s:.2g},emb={emb_s:.2g}",
                      fill=(255, 255, 200), font=font)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str, required=True)
    parser.add_argument("--image_a",       type=str, required=True)
    parser.add_argument("--image_a_prime", type=str, required=True)
    parser.add_argument("--image_b",       type=str, required=True)
    parser.add_argument("--edits", type=str, nargs="+", required=True,
                        help="Raw edit texts in order (e.g. 'change expression to laugh'). "
                             "The prompt is auto-built from these using the same format as training.")
    parser.add_argument("--scale_values", type=float, nargs="+",
                        default=None,
                        help="LoRA scale sweep values (default: 0 0.25 0.5 0.75 1).")
    parser.add_argument("--embedding_scales", type=float, nargs="+",
                        default=None,
                        help="Text-embedding steering alpha values, applied at suppressed "
                             "edits' token positions (Diffusion-Sliders-style). "
                             "alpha>0 amplifies the edit; alpha<0 suppresses. "
                             "Default [0.0] = pure LoRA interpolation (no embedding steering).")
    parser.add_argument("--embedding_schedule", type=str, nargs="+", default=None,
                        help="Piecewise-constant schedule for the embedding alpha "
                             "across denoising steps. Each entry is "
                             "'step_boundary:alpha'; for step index t the first "
                             "entry with t<boundary wins. Example: "
                             "'10:-40 20:-30 28:-20' uses alpha=-40 for steps 0-9, "
                             "-30 for 10-19, -20 for 20-27. "
                             "When set, overrides --embedding_scales and runs one "
                             "inference per (suppress-mode × exlora-scale). "
                             "The schedule can also be set in the YAML config as "
                             "'embedding_schedule: [[10, -40], [20, -30], [28, -20]]'.")
    parser.add_argument("--embedding_scale_factor", type=float, default=1.0,
                        help="Absolute magnitude multiplier on the unit-norm steering "
                             "direction. Since edit tokens in prompt_embeds can have norm "
                             "~10-50, alpha=1 corresponds to a unit-length push. Raise this "
                             "to amplify steering effect (typical: 5-30).")
    parser.add_argument("--csv_scope", type=str, nargs="+",
                        default=None, choices=list(STEERING_SCOPES),
                        help="Where to add the steering vector in the prompt "
                             "embedding: 'tokens' (only the suppressed edit's "
                             "own token positions — same footprint as ExLoRA), "
                             "or 'prompt' (every non-padding prompt token). "
                             "Pass multiple values to sweep both as an "
                             "ablation — each scope gets its own output "
                             "subdirectory. Default: 'tokens'. "
                             "Can also be set in YAML as "
                             "'csv_scope: \"tokens\"' or "
                             "'csv_scope: [\"tokens\", \"prompt\"]'.")
    parser.add_argument("--csv_mode", type=str, default=None,
                        choices=list(STEERING_MODES),
                        help="How the steering vector modifies the prompt "
                             "embedding at target positions. "
                             "'add' (default): x' = x + alpha*v_hat — pure "
                             "additive push (original behaviour). "
                             "'ablate': x' = x - (x.v_hat)*v_hat + alpha*v_hat "
                             "— directional ablation (project the concept "
                             "component OUT of x) then optional push. Usually "
                             "much stronger for suppression at the same |alpha|. "
                             "With alpha=0, 'ablate' performs pure concept "
                             "removal. Can also be set in YAML as "
                             "'csv_mode: \"ablate\"'.")
    parser.add_argument("--num_steering_pairs", type=int, default=32,
                        help="Number of contrastive pos/neg sentence pairs per edit that "
                             "Qwen3 generates (Diffusion-Sliders paper uses 100 via GPT-4o; "
                             "we default to 32 with a local Qwen to keep runtime modest).")
    parser.add_argument("--steering_cache_dir", type=str, default=None,
                        help="Directory in which to cache generated pairs.jsonl and "
                             "vector.pt per edit (keyed by sha1(edit)); "
                             "default ~/.cache/sia/csv/<backbone>.")
    parser.add_argument("--regen_steering", action="store_true",
                        help="Regenerate the contrastive pairs + DoM vector even if a cached "
                             "vector.pt exists for the edit.")
    parser.add_argument("--max_push_norm_ratio", type=float, default=0.0,
                        help="Clamp the steering push magnitude to this multiple of the "
                             "per-token embedding norm (before modification). Prevents "
                             "extreme out-of-distribution embeddings. 0 = no clamp (default). "
                             "Typical safe value: 1.0-3.0.")
    parser.add_argument("--incontext_steering", action="store_true",
                        help="Compute the DoM steering direction inside the actual inference "
                             "prompt (in-context) instead of from isolated short sentences. "
                             "Fixes the causal-LM context mismatch: the direction is extracted "
                             "in the same representation space where it is applied. Requires "
                             "1+8 text-encoder forward passes per edit (fast) and no Qwen "
                             "generation.")
    parser.add_argument("--output",         type=str, default="result.png")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Stage-2 checkpoint to load ExLoRA/AnaLoRA from: a local dir or a Hub path "
                             "<org>/<repo>/<subfolder>[@revision]. Overrides the auto-discovery "
                             "from output_dir in the config.")
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--cell_size", type=int, default=512)
    parser.add_argument("--seed",      type=int, default=0)
    return parser


def parse_args(cli=None):
    """Build the run config from ``cli`` (parsed ``build_parser()`` args;
    parsed from sys.argv when omitted) merged into the YAML config."""
    if cli is None:
        cli = build_parser().parse_args()

    cfg = OmegaConf.load(cli.config)
    cfg.num_steps  = cli.num_steps
    cfg.cell_size  = cli.cell_size
    cfg.seed       = cli.seed
    cfg.image_a         = cli.image_a
    cfg.image_a_prime   = cli.image_a_prime
    cfg.image_b         = cli.image_b
    cfg.edits           = cli.edits
    # Build the prompt the same way training does — guarantees edits are exact substrings
    cfg.prompt          = build_analogy_prompt(cli.edits, cli.edits)
    cfg.output          = cli.output
    cfg.checkpoint_dir  = cli.checkpoint_dir
    cfg.scale_values    = cli.scale_values if cli.scale_values is not None else SCALE_VALUES
    cfg.embedding_scales = (
        cli.embedding_scales if cli.embedding_scales is not None else [0.0])
    cfg.embedding_scale_factor = cli.embedding_scale_factor
    cfg.num_steering_pairs     = cli.num_steering_pairs
    cfg.steering_cache_dir     = cli.steering_cache_dir
    cfg.regen_steering         = cli.regen_steering
    cfg.incontext_steering     = cli.incontext_steering
    cfg.max_push_norm_ratio    = cli.max_push_norm_ratio

    # Schedule: CLI overrides YAML. Store as a plain list of (boundary, alpha)
    # tuples on cfg.embedding_schedule_parsed; leave the raw form on
    # cfg.embedding_schedule for traceability.
    raw_sched = cli.embedding_schedule
    if raw_sched is None:
        raw_sched = OmegaConf.to_container(cfg.get("embedding_schedule"), resolve=True) \
            if "embedding_schedule" in cfg else None
    cfg.embedding_schedule_parsed = parse_embedding_schedule(raw_sched)

    # Steering scope: CLI overrides YAML. Normalize to a list of valid choices.
    if cli.csv_scope is not None:
        raw_scopes = cli.csv_scope
    elif "csv_scope" in cfg:
        raw_scopes = OmegaConf.to_container(cfg.csv_scope, resolve=True)
    else:
        raw_scopes = ["tokens"]
    if isinstance(raw_scopes, str):
        raw_scopes = [raw_scopes]
    scopes = []
    for s in raw_scopes:
        s = str(s).strip().lower()
        if s not in STEERING_SCOPES:
            raise ValueError(
                f"csv_scope entry {s!r} invalid; "
                f"expected one of {STEERING_SCOPES}.")
        if s not in scopes:   # dedupe, preserve order
            scopes.append(s)
    cfg.csv_scopes = scopes

    # Steering mode (single value): CLI overrides YAML, default "add".
    if cli.csv_mode is not None:
        mode = cli.csv_mode
    elif "csv_mode" in cfg:
        mode = str(cfg.csv_mode)
    else:
        mode = "add"
    mode = str(mode).strip().lower()
    if mode not in STEERING_MODES:
        raise ValueError(f"csv_mode={mode!r} invalid; "
                         f"expected one of {STEERING_MODES}.")
    cfg.csv_mode = mode
    return cfg


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    image_a       = Image.open(args.image_a).convert("RGB")
    image_a_prime = Image.open(args.image_a_prime).convert("RGB")
    image_b       = Image.open(args.image_b).convert("RGB")

    print("Loading models …")
    pipe, tokenizer, text_encoder = load_pipeline(args, device)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Precompute base prompt embeds ----
    with torch.no_grad():
        base_prompt_embeds, _ = compute_text_embeddings(
            [args.prompt], tokenizer, text_encoder)
    max_seq_len = base_prompt_embeds.shape[1]

    # ---- AnaLoRA-only reference image (ExLoRA off) ----
    print("[analora] Running AnaLoRA-only inference for the reference strip …")
    analora_gen = torch.Generator(device=device).manual_seed(int(args.seed))
    analora_image = run_inference(
        pipe, tokenizer, text_encoder,
        image_a, image_a_prime, image_b,
        args.prompt, [],
        args.num_steps, device,
        analora_scale=1.0,
        exlora_scale=0.0,
        prompt_embeds_override=base_prompt_embeds,
        generator=analora_gen,
    )
    analora_ref_path = out_dir / "analora.png"
    analora_image.save(analora_ref_path)
    print(f"  Saved analora reference → {analora_ref_path}")

    # ---- Resolve embedding schedule (piecewise-constant alpha over steps) ----
    schedule = args.get("embedding_schedule_parsed", None)
    if schedule:
        print(f"[Steering schedule] Using time-varying alpha schedule: "
              f"{[(b, a) for b, a in schedule]} "
              f"(overrides --embedding_scales)")

    # ---- Per-edit: templated token positions in the inference prompt ----
    # These positions index into ``base_prompt_embeds`` (which is computed on
    # the Qwen chat-templated sequence), so we align the edit substring against
    # the templated text to get the correct positions for the steering add.
    edit_positions = []
    # We need DoM vectors whenever any steering will happen. For additive
    # mode this only matters when at least one alpha is non-zero. For ablate
    # mode it matters always — pure concept removal (alpha=0) is still
    # meaningful.
    steering_mode_for_gate = str(args.get("csv_mode", "add"))
    any_alpha = (
        steering_mode_for_gate == "ablate"
        or any(abs(float(a)) > 0 for a in args.embedding_scales)
        or (schedule is not None and any(abs(a) > 0 for _, a in schedule))
    )
    for i, edit_text in enumerate(args.edits):
        pos = templated_substring_indices(
            args.prompt, edit_text, tokenizer, max_length=max_seq_len)
        edit_positions.append(pos)
        # Sanity check: decode tokens at those positions and confirm they match
        # the edit text (they should; mismatch means the templated alignment
        # silently failed).
        if pos:
            templated_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            templated_ids = tokenizer(
                templated_prompt, max_length=max_seq_len,
                padding="max_length", truncation=True,
                return_tensors="pt")["input_ids"][0].tolist()
            decoded = tokenizer.decode([templated_ids[p] for p in pos],
                                       skip_special_tokens=False)
            print(f"[Steering] edit_{i+1} positions[{len(pos)}]: "
                  f"{pos[0]}..{pos[-1]}  decoded={decoded!r}")
        else:
            print(f"[Steering] edit_{i+1} ({args.edits[i][:50]}): "
                  "NO tokens aligned in templated prompt — steering disabled for this edit.")

    # ---- Per-edit: DoM steering vector via Qwen3 + caching ----
    steering_vecs = [None] * len(args.edits)
    use_incontext = bool(args.get("incontext_steering", False))
    if use_incontext:
        print("[Steering DoM] In-context mode enabled — DoM will be computed "
              "inside the full inference prompt (causal-LM-aware).")
    if any_alpha:
        hidden_dim = base_prompt_embeds.shape[-1]
        for i, edit_text in enumerate(args.edits):
            if not edit_positions[i]:
                print(f"[Steering DoM] Skipping edit_{i+1} — no positions in prompt.")
                continue
            incontext_kwargs = {}
            if use_incontext:
                incontext_kwargs = dict(
                    incontext_prompt=args.prompt,
                    all_edits=list(args.edits),
                    edit_index=i,
                    build_prompt_fn=build_analogy_prompt,
                )
            v, meta = get_or_build_steering_vector(
                flux2_klein, edit_text, int(args.num_steering_pairs), pipe, device,
                cache_dir=args.steering_cache_dir,
                regen=bool(args.regen_steering),
                **incontext_kwargs,
            )
            if v.shape[-1] != hidden_dim:
                raise RuntimeError(
                    f"Steering vector dim {v.shape[-1]} != prompt_embeds dim "
                    f"{hidden_dim} for edit_{i+1}. Cache from a different "
                    f"encoder config? Pass --regen_steering.")
            steering_vecs[i] = v.to(device=device, dtype=base_prompt_embeds.dtype)
            n_pooled = meta.get("n_usable_after_pooling", "?")
            pre = meta.get("pre_normalization_norm", None)
            src = meta.get("source", "?")
            pre_str = f"{pre:.3f}" if isinstance(pre, (int, float)) else str(pre)
            print(f"[Steering DoM] edit_{i+1} ({edit_text[:50]}): "
                  f"source={src} |v|_pre={pre_str} pooled_pairs={n_pooled} "
                  f"dim={v.shape[-1]}")
    else:
        print("[Steering DoM] All embedding_scales == 0 and mode='add' — "
              "skipping DoM vector computation (pure LoRA mode).")

    # Row descriptors: (row_label, suppress_list, suppress_indices)
    # "no supp" row is excluded — it doesn't demonstrate selective suppression.
    # Ablation is only on ExLoRA suppression; analora is always on at scale=1.
    row_descs = []
    n = len(args.edits)
    for r in range(1, n + 1):          # start at 1 to skip "no supp"
        for indices in itertools.combinations(range(n), r):
            subset = [args.edits[i] for i in indices]
            if r == n and n > 1:
                label = "exlora (supp all)"
            else:
                label = "exlora (supp " + "+".join(f"e{i+1}" for i in indices) + ")"
            row_descs.append((label, subset, list(indices)))

    # When a schedule is active, the per-step alpha comes from the schedule;
    # emb_scales collapses to a single placeholder key so per-row bookkeeping
    # and grid layout (which expects a scalar emb axis) still work.
    if schedule:
        emb_scales = [0.0]
        sched_tag = format_schedule_tag(schedule)
    else:
        emb_scales = list(args.embedding_scales)
        sched_tag = None
    st_scales  = list(args.scale_values)

    # Steering scope ablation axis. "tokens" touches only the suppressed edit's
    # own token positions (same footprint as ExLoRA); "prompt" touches every
    # non-padding prompt token (to push against concept residue that leaked
    # into neighbouring tokens through the encoder's self-attention).
    scopes = list(args.csv_scopes)
    all_prompt_positions = None
    if "prompt" in scopes:
        all_prompt_positions = klein_templated_prompt_token_positions(
            args.prompt, tokenizer, max_length=max_seq_len)
        print(f"[Steering] scope='prompt' will add the vector at "
              f"{len(all_prompt_positions)} non-padding token positions "
              f"(positions {all_prompt_positions[0]}..{all_prompt_positions[-1]}).")
    print(f"[Steering] Scope ablation: {scopes}")
    mode = str(args.csv_mode)
    print(f"[Steering] Mode: {mode}  "
          + ("(additive push: x' = x + alpha * v_hat)"
             if mode == "add"
             else "(directional ablation + push: "
                  "x' = x - (x . v_hat) v_hat + alpha * v_hat)"))

    total_runs = len(scopes) * len(row_descs) * len(st_scales) * len(emb_scales)
    run_idx    = 0
    output_path = Path(args.output)
    combined_paths = []

    for scope in scopes:
        # Per-scope output subdir only when we are actually running multiple
        # scopes; otherwise keep the original flat layout for backward compat.
        if len(scopes) > 1:
            scope_out_dir = out_dir / f"scope_{scope}"
            scope_out_dir.mkdir(parents=True, exist_ok=True)
            scope_output  = scope_out_dir / output_path.name
        else:
            scope_out_dir = out_dir
            scope_output  = output_path
        scope_tag_str = f"scope={scope}"

        all_preds = {}   # (row_label, st_s, emb_s) -> PIL.Image  (per-scope)

        for row_label, suppress, suppress_idx in row_descs:
            row_dir = scope_out_dir / row_label.replace(" ", "_")\
                .replace("/", "-").replace("(", "").replace(")", "")
            row_dir.mkdir(parents=True, exist_ok=True)

            base_s = 1.0       # analora is always at full strength
            row_preds = {}
            for emb_s in emb_scales:
                if schedule:
                    embeds_schedule = []
                    for step_idx in range(int(args.num_steps)):
                        a_t = alpha_at_step(schedule, step_idx)
                        # None for non-suppressed edits → leave their concept
                        # untouched even in ``mode="ablate"`` (where alpha=0
                        # would otherwise mean "pure concept removal").
                        alpha_per_edit = [
                            (a_t * args.embedding_scale_factor) if i in suppress_idx else None
                            for i in range(n)
                        ]
                        embeds_schedule.append(apply_csv(
                            base_prompt_embeds, edit_positions, steering_vecs,
                            alpha_per_edit,
                            scope=scope,
                            all_prompt_positions=all_prompt_positions,
                            mode=mode,
                            max_push_norm_ratio=float(args.max_push_norm_ratio)))
                    steered_embeds = None
                    step_tag = sched_tag
                else:
                    # None for non-suppressed edits → see note above.
                    alpha_per_edit = [
                        (emb_s * args.embedding_scale_factor) if i in suppress_idx else None
                        for i in range(n)
                    ]
                    steered_embeds = apply_csv(
                        base_prompt_embeds, edit_positions, steering_vecs,
                        alpha_per_edit,
                        scope=scope,
                        all_prompt_positions=all_prompt_positions,
                        mode=mode,
                        max_push_norm_ratio=float(args.max_push_norm_ratio))
                    embeds_schedule = None
                    step_tag = f"emb={emb_s:.2g}"

                for st_s in st_scales:
                    run_idx += 1
                    print(f"[{run_idx}/{total_runs}] {scope_tag_str}  "
                          f"{row_label}  suppress={suppress}  "
                          f"base={base_s}  st={st_s}  {step_tag}")
                    # Freshly-seeded generator per cell so the INITIAL NOISE
                    # is identical across the ablation axis. Without this the
                    # global RNG advances with every torch.randn call and each
                    # cell sees different noise — you cannot tell whether an
                    # image changed because of alpha or because of noise.
                    cell_gen = torch.Generator(device=device).manual_seed(
                        int(args.seed))
                    pred = run_inference(
                        pipe, tokenizer, text_encoder,
                        image_a, image_a_prime, image_b,
                        args.prompt, suppress,
                        args.num_steps, device,
                        analora_scale=base_s,
                        exlora_scale=st_s,
                        prompt_embeds_override=steered_embeds,
                        prompt_embeds_schedule=embeds_schedule,
                        generator=cell_gen,
                    )
                    all_preds[(row_label, st_s, emb_s)] = pred
                    row_preds[(st_s, emb_s)] = pred

                    annotated = annotate_image(
                        pred,
                        f"{row_label}  st={st_s:.2g}  {step_tag}  {scope_tag_str}",
                        suppress,
                    )
                    if schedule:
                        fname = (f"st{st_s:.2f}".replace(".", "p")
                                 + f"_{sched_tag}.png")
                    else:
                        fname = (f"st{st_s:.2f}".replace(".", "p")
                                 + f"_emb{emb_s:.2f}".replace(".", "p") + ".png")
                    annotated.save(row_dir / fname)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Per-row grid
            row_grid = build_row_grid(
                image_a, image_a_prime, image_b,
                f"{row_label}  [{scope_tag_str}]", suppress,
                row_preds, st_scales, emb_scales,
                cell_size=args.cell_size,
                analora_image=analora_image,
            )
            row_grid.save(row_dir / "grid.png")
            print(f"  Saved row grid → {row_dir / 'grid.png'}")

        # Combined grid for this scope
        if len(emb_scales) == 1:
            legacy_preds = {(rl, s): all_preds[(rl, s, emb_scales[0])]
                            for rl, _, _ in row_descs for s in st_scales}
            combined = build_combined_grid(
                image_a, image_a_prime, image_b,
                [(rl, sup) for rl, sup, _ in row_descs],
                legacy_preds, st_scales, cell_size=args.cell_size,
                analora_image=analora_image,
            )
        else:
            combined = build_joint_grid(
                image_a, image_a_prime, image_b,
                row_descs, all_preds, st_scales, emb_scales,
                cell_size=args.cell_size,
                analora_image=analora_image,
            )
        combined.save(scope_output)
        combined_paths.append(scope_output)
        print(f"[{scope_tag_str}] Saved combined grid → {scope_output}")

    print("Individual images saved under:")
    for p in combined_paths:
        print(f"  {p.parent}/")


if __name__ == "__main__":
    main()
