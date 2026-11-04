"""CPU tests of the evaluation helpers (``sia.eval``) and of the arm parsing of ``scripts/eval.py``.

Expected values were computed with the original functions (``training/eval_benchmark_qwen21.py``)
before they moved: the on-disk layout must not change, or existing evaluation runs stop resuming.
"""
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from sia.data import subsets_for
from sia.eval.judge import build_judge_prompt, parse_judge
from sia.eval.results import _items, _load_pred, full_path, keep_tag, pred_path, safe_id

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_subsets_for():
    assert subsets_for(2) == [(0,), (1,)]
    assert subsets_for(3) == [(0, 1), (0, 2), (1, 2), (0,), (1,), (2,)]
    assert len(subsets_for(4)) == 14 and subsets_for(4)[:5] == [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1)]


def test_output_layout():
    assert keep_tag((0, 2)) == "keep13"
    assert keep_tag(()) == "keep"
    assert safe_id("dev/3edits/setting1/data2-data3") == "dev__3edits__setting1__data2-data3"
    assert pred_path("o", "m", "2edits/data1", (1,)) == Path("o/m/2edits__data1/keep2.png")
    assert full_path("o", "m", "2edits/data1") == Path("o/m/full_transfer/2edits__data1.png")


def test_items_enumerates_existing_predictions_and_identity(tmp_path):
    case = {"id": "2edits/data1", "edits": ["a", "b"], "b": Image.new("RGB", (4, 4), (1, 2, 3))}
    p = pred_path(tmp_path, "m", case["id"], (1,))
    p.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(p)
    items = list(_items(tmp_path, ["m"], [case], include_identity=True))
    assert [(m, k, q) for m, _, k, q in items] == [("m", (1,), p), ("identity", (0,), None), ("identity", (1,), None)]
    assert _load_pred(None, case).getpixel((0, 0)) == (1, 2, 3)


def test_parse_judge():
    reply = ('x {"retained_accuracy": 12, "preservation": 7, "suppression": -1, '
             '"retained_detected": [true, false], "suppressed_visible": [false]} y')
    assert parse_judge(reply, 2, 1) == {"ret_acc": 10.0, "pres": 7.0, "supp": 0.0, "recall": 0.5, "supp_rate": 1.0}
    assert parse_judge(reply, 3, 2) == {"ret_acc": 10.0, "pres": 7.0, "supp": 0.0, "recall": None, "supp_rate": None}
    assert parse_judge("no json", 1, 1) is None


def test_build_judge_prompt_lists_retained_and_suppressed():
    text = build_judge_prompt(["a", "b", "c"], (0, 2))
    assert "  1. a\n  2. b\n  3. c" in text
    assert "  R1: a\n  R2: c" in text and "  S1: b" in text


@pytest.fixture(scope="module")
def eval_script():
    spec = importlib.util.spec_from_file_location("sia_eval_script", SCRIPTS / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_arm(eval_script):
    assert eval_script.parse_arm("full:full") == {"name": "full", "method": "full"}
    assert eval_script.parse_arm("x:add:1.5") == {"name": "x", "method": "add", "csv_k": 1.5, "csv_alpha": 0.0,
                                                  "csv_lambda": 1.0}
    assert eval_script.parse_arm("y:abl:2") == {"name": "y", "method": "abl", "csv_k": None, "csv_alpha": 0.0,
                                                "csv_lambda": 2.0}
    for bad in ("full", "x:foo", "x:add"):
        with pytest.raises(Exception):
            eval_script.parse_arm(bad)


def test_csv_alphas(eval_script):
    gaps = [2.0, 4.0, 8.0]
    add = eval_script.parse_arm("a:add:1")
    assert eval_script.csv_alphas(add, [1], gaps) == [None, -4.0, None]
    abl = eval_script.parse_arm("b:abl:1")
    assert eval_script.csv_alphas(abl, [0, 2], gaps) == [0.0, None, 0.0]
