"""CPU tests of the backbone-free CSV code in ``sia.concept_suppression``.

Expected values were computed with the original functions (``training/infer_single.py`` and
``training/steering_dom*.py``) before they moved.
"""
import json
from types import SimpleNamespace

import pytest
import torch

from sia.concept_suppression import (
    _fallback_pairs,
    _parse_jsonl_records,
    alpha_at_step,
    apply_csv,
    compute_dom_vector,
    compute_incontext_dom_vector,
    format_schedule_tag,
    get_or_build_steering_vector,
    parse_embedding_schedule,
)


def embeds():
    return torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)


VECS = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.6, 0.8, 0.0])]
POSITIONS = [[1, 2], [3]]   # rows of edit 1 and edit 2


@pytest.mark.parametrize("alphas, kwargs, expected", [
    ([-2.0, None], dict(mode="add"),
     [[0, 1, 2], [1, 4, 5], [4, 7, 8], [9, 10, 11], [12, 13, 14]]),
    ([None, 0.0], dict(mode="ablate"),
     [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0.9600000381469727, -0.7200002670288086, 11], [12, 13, 14]]),
    ([0.5, -1.0], dict(mode="ablate", ablate_scale=2.0),
     [[0, 1, 2], [-2.5, 4, 5], [-5.5, 7, 8], [-7.679999828338623, -12.24000072479248, 11], [12, 13, 14]]),
    ([None, 1.0], dict(mode="add", scope="prompt", all_prompt_positions=[0, 4]),
     [[0.6000000238418579, 1.7999999523162842, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
      [12.600000381469727, 13.800000190734863, 14]]),
    ([-100.0, None], dict(mode="add", max_push_norm_ratio=0.5),
     [[0, 1, 2], [-0.5355339050292969, 4, 5], [-0.10327768325805664, 7, 8], [9, 10, 11], [12, 13, 14]]),
])
def test_apply_csv(alphas, kwargs, expected):
    x = embeds()
    out = apply_csv(x, POSITIONS, VECS, alphas, **kwargs)
    assert torch.equal(out, torch.tensor([expected], dtype=torch.float32))
    assert torch.equal(x, embeds())   # the input is never modified


def test_apply_csv_skips_positions_past_the_sequence():
    out = apply_csv(embeds(), [[1, 9]], VECS[:1], [3.0], mode="add")
    expected = embeds()
    expected[0, 1, 0] += 3.0
    assert torch.equal(out, expected)


def test_apply_csv_no_op_returns_the_input():
    x = embeds()
    assert apply_csv(x, POSITIONS, VECS, [None, None]) is x
    assert apply_csv(x, POSITIONS, VECS, [0.0, 0.0], mode="add") is x
    assert apply_csv(x, POSITIONS, [None, None], [1.0, 1.0]) is x


@pytest.mark.parametrize("kwargs", [dict(scope="after"), dict(mode="scale")])
def test_apply_csv_rejects_unknown_scope_or_mode(kwargs):
    with pytest.raises(ValueError):
        apply_csv(embeds(), POSITIONS, VECS, [1.0, None], **kwargs)


@pytest.mark.parametrize("raw, expected", [
    (["10:-2", "5:1.5"], [(5, 1.5), (10, -2.0)]),
    ([[3, 0.5]], [(3, 0.5)]),
    ([{"until_step": 4, "alpha": -1}], [(4, -1.0)]),
    (None, None), ([], None), (["", " "], None), (5, None),
])
def test_parse_embedding_schedule(raw, expected):
    assert parse_embedding_schedule(raw) == expected


@pytest.mark.parametrize("raw", [["3"], [{"alpha": 1}], [[1, 2, 3]]])
def test_parse_embedding_schedule_rejects_malformed_entries(raw):
    with pytest.raises(ValueError):
        parse_embedding_schedule(raw)


def test_alpha_at_step_and_tag():
    schedule = parse_embedding_schedule(["10:-2", "20:1.5"])
    assert [alpha_at_step(schedule, i) for i in (0, 9, 10, 19, 20, 99)] == [-2.0, -2.0, 1.5, 1.5, 1.5, 1.5]
    assert format_schedule_tag(schedule) == "sched_10a-2p0_20a1p5"
    assert format_schedule_tag([(4, 0.25), (8, -0.05)]) == "sched_4a0p2_8a-0p1"


# ---------------------------------------------------------------------------
# Contrast pairs and difference of means (backbone stubbed out)
# ---------------------------------------------------------------------------

def test_parse_jsonl_records_keeps_valid_lines_only():
    good = {"pos_style": "bright", "neg_style": "dark", "pos": "A Bright room.", "neg": "A dark room."}
    text = "\n".join([
        "```json",
        json.dumps(good) + ",",
        json.dumps(dict(good, pos="A room.")),          # pos_style missing from pos
        json.dumps({"pos": "x", "neg": "y"}),           # keys missing
        "not json",
        "```",
    ])
    assert _parse_jsonl_records(text) == [good]


def test_fallback_pairs_cycle_neutral_words():
    pairs = _fallback_pairs("add a hat", 3)
    assert [p["neg_style"] for p in pairs] == ["unchanged", "plain", "original"]
    assert pairs[1] == {"pos_style": "add a hat", "neg_style": "plain",
                        "pos": "A detailed close-up of a subject showing add a hat.",
                        "neg": "A detailed close-up of a subject, plain."}


def stub_backbone(table):
    """A backbone whose ``steering_pool`` looks the sentence up in ``table`` (``None`` = not found)."""
    def steering_pool(sentence, style, pipe, device, vl_images=None):
        return None if table.get(sentence) is None else torch.tensor(table[sentence])

    def cache_key(edit, prompt=None, vl_images=None):
        return "key_" + edit.replace(" ", "_")

    def generate_contrast_pairs(edit, pipe, n_pairs, device):
        raise AssertionError("a cached direction must never sample new pairs")

    return SimpleNamespace(__name__="sia.backbones.stub", steering_pool=steering_pool, cache_key=cache_key,
                           generate_contrast_pairs=generate_contrast_pairs)


def test_compute_dom_vector():
    backbone = stub_backbone({"p1": [3.0, 0.0], "n1": [0.0, 0.0], "p2": [1.0, 4.0], "n2": [0.0, 0.0],
                              "p3": [9.0, 9.0], "n3": None})
    pairs = [dict(pos=f"p{i}", neg=f"n{i}", pos_style="s", neg_style="s") for i in (1, 2, 3)]
    v, stats = compute_dom_vector(backbone, pairs, None, "cpu")
    assert stats == {"n_usable": 2, "pre_norm": pytest.approx(2.8284271247)}
    assert torch.allclose(v, torch.tensor([2.0, 2.0]) / 2.8284271247)


def test_compute_incontext_dom_vector():
    def build(pair_a, pair_b):
        return " | ".join(pair_a)

    neutral = {f"a | {w}": [0.0, 1.0] for w in ("unchanged", "plain", "original", "unmodified",
                                                  "standard", "normal", "as-is", "unaltered")}
    backbone = stub_backbone({"a | b": [0.0, 3.0], **neutral})
    v, stats = compute_incontext_dom_vector(backbone, "a | b", "b", ["a", "b"], 1, None, "cpu", build)
    assert stats == {"n_usable": 9, "pre_norm": 2.0, "neg_count": 8}
    assert torch.equal(v, torch.tensor([0.0, 1.0]))


def test_get_or_build_reads_the_cache_without_sampling(tmp_path):
    backbone = stub_backbone({})
    folder = tmp_path / "key_add_a_hat"
    folder.mkdir()
    torch.save(torch.tensor([0.0, 1.0], dtype=torch.float64), folder / "vector.pt")
    (folder / "meta.json").write_text(json.dumps({"edit": "add a hat", "cache_version": 2,
                                                  "pre_normalization_norm": 7.5}))
    v, meta = get_or_build_steering_vector(backbone, "add a hat", 64, None, "cpu", cache_dir=tmp_path)
    assert v.dtype == torch.float32 and torch.equal(v, torch.tensor([0.0, 1.0]))
    assert meta["source"] == "cache" and meta["pre_normalization_norm"] == 7.5


@pytest.mark.parametrize("requested, warns", [(64, False), (32, True)])
def test_get_or_build_warns_on_pair_count_mismatch(tmp_path, capsys, requested, warns):
    backbone = stub_backbone({})
    folder = tmp_path / "key_add_a_hat"
    folder.mkdir()
    torch.save(torch.tensor([0.0, 1.0]), folder / "vector.pt")
    (folder / "meta.json").write_text(json.dumps({"edit": "add a hat", "cache_version": 2, "n_requested": 64}))
    v, meta = get_or_build_steering_vector(backbone, "add a hat", requested, None, "cpu", cache_dir=tmp_path)
    assert torch.equal(v, torch.tensor([0.0, 1.0]))   # reused, never rebuilt
    assert ("built from 64 pairs, not the requested 32" in capsys.readouterr().out) == warns


def test_default_cache_dir_is_per_backbone(monkeypatch, tmp_path):
    from sia.concept_suppression import default_cache_dir
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_cache_dir(stub_backbone({})) == tmp_path / ".cache" / "sia" / "csv" / "stub"
