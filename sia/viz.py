"""Image grids: the validation and benchmark strips written during training (``render_rows``) and the
suppression-mode x LoRA-scale x CSV-strength grids of ``scripts/infer.py``."""
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
]


def _font(size):
    """The first available TrueType font at ``size``, or ``None`` (PIL's default bitmap font)."""
    for fp in _FONT_CANDIDATES:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return None


def render_rows(rows, cell=384, title=""):
    """``rows``: list of (row_label, [(image, caption), ...]). Images are
    fitted into square cells keeping their aspect ratio."""
    font = _font(16)
    label_w, cap_h = 200, 40
    n_cols = max(len(r[1]) for r in rows)
    title_lines = textwrap.wrap(title, 140) if title else []
    top = 8 + 20 * len(title_lines)
    canvas = Image.new("RGB", (label_w + n_cols * cell, top + len(rows) * (cell + cap_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for li, line in enumerate(title_lines):
        draw.text((8, 4 + 20 * li), line, fill=(0, 0, 160), font=font)
    for ri, (row_label, cells) in enumerate(rows):
        y = top + ri * (cell + cap_h)
        draw.multiline_text((6, y + 6), "\n".join(textwrap.wrap(row_label, 22)), fill=(0, 0, 0), font=font)
        for ci, (img, caption) in enumerate(cells):
            x = label_w + ci * cell
            if img is not None:
                thumb = img.copy()
                thumb.thumbnail((cell, cell), Image.LANCZOS)
                canvas.paste(thumb, (x + (cell - thumb.width) // 2, y + cap_h))
            draw.multiline_text((x + 4, y + 2), caption, fill=(0, 0, 0), font=font)
    return canvas


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


def annotate_image(img, label, suppressed_texts, font_size=16):
    """Return a new image with a white banner below containing label + suppressed edits."""
    font       = _font(font_size)
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
    font      = _font(font_size)
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
    font      = _font(font_size)
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
    font      = _font(font_size)

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
