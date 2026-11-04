"""On-disk layout of an evaluation run and the items it scores.

Predictions of a method ``name`` are written to ``<out>/<name>/<safe case id>/keepXY.png`` (``XY`` = the
1-based indices of the kept edits) and ``<out>/<name>/full_transfer/<safe case id>.png``; scores are
appended as JSON lines to ``<out>/judge.jsonl``, ``<out>/clip.jsonl`` and ``<out>/full_transfer.jsonl``.
"""
import json
from pathlib import Path

from PIL import Image

from sia.data import subsets_for


def keep_tag(kept):
    """File tag of a kept-edit subset: 0-based ``(0, 2)`` -> ``"keep13"``."""
    return "keep" + "".join(str(p + 1) for p in kept)


def safe_id(case_id):
    """Case id as a folder name (``/`` -> ``__``)."""
    return case_id.replace("/", "__")


def pred_path(out, name, case_id, kept):
    """Image of method ``name`` for one case and kept-edit subset."""
    return Path(out) / name / safe_id(case_id) / f"{keep_tag(kept)}.png"


def full_path(out, name, case_id):
    """Full-transfer image (nothing suppressed) of method ``name`` for one case."""
    return Path(out) / name / "full_transfer" / f"{safe_id(case_id)}.png"


def jsonl_read(path):
    """Rows of a JSON-lines file; ``[]`` if it does not exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def iter_items(out, names, cases, include_identity):
    """(method, case, kept, image path) for every generated item of ``names`` in ``out``, plus
    (``"identity"``, case, kept, ``None``) for every item when ``include_identity`` is set."""
    for name in names:
        for case in cases:
            for kept in subsets_for(len(case["edits"])):
                p = pred_path(out, name, case["id"], kept)
                if p.exists():
                    yield name, case, kept, p
    if include_identity:
        for case in cases:
            for kept in subsets_for(len(case["edits"])):
                yield "identity", case, kept, None   # output = B, the no-edit control


def load_pred(item, case):
    """The output image of an item: the file ``item``, or B itself for the identity control (``None``)."""
    return case["b"].copy() if item is None else Image.open(item).convert("RGB")
