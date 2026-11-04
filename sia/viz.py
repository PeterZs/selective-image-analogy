"""Image strips for the validation and benchmark visualisations written during training."""
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
