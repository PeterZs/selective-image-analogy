"""The analogy prompt text is part of what the adapters were trained on: it must not change."""
import pytest

from sia.prompt import build_analogy_prompt

HEAD = "Image 1 is the original and image 2 is the edited version."
TAIL = "Apply the same edits to image 3 to produce the output."


@pytest.mark.parametrize("edits, expected", [
    (
        ["add hat", "add glasses"],
        f"{HEAD} Edit 1: add hat. Edit 2: add glasses. {TAIL}",
    ),
    (
        ["a", "b", "c"],
        f"{HEAD} Edit 1: a. Edit 2: b. Edit 3: c. {TAIL}",
    ),
    (
        ["a", "b", "c", "d"],
        f"{HEAD} Edit 1: a. Edit 2: b. Edit 3: c. Edit 4: d. {TAIL}",
    ),
])
def test_prompt_lists_every_edit(edits, expected):
    assert build_analogy_prompt(edits, edits) == expected


def test_pair_b_edits_do_not_affect_the_prompt():
    edits = ["add hat", "add glasses"]
    assert build_analogy_prompt(edits, None) == build_analogy_prompt(edits, ["x", "y"])


@pytest.mark.parametrize("suppress, expected", [
    ([1], f"{HEAD} Edit 1: a. Edit 2: c. Edit 3: d. {TAIL}"),
    ([0, 2], f"{HEAD} Edit 1: b. Edit 2: d. {TAIL}"),
    ((3,), f"{HEAD} Edit 1: a. Edit 2: b. Edit 3: c. {TAIL}"),
    (None, f"{HEAD} Edit 1: a. Edit 2: b. Edit 3: c. Edit 4: d. {TAIL}"),
])
def test_suppressed_edits_are_omitted_and_the_rest_renumbered(suppress, expected):
    assert build_analogy_prompt(["a", "b", "c", "d"], None, suppress_indices=suppress) == expected


def test_empty_edit_strings_are_skipped():
    assert build_analogy_prompt(["a", "", "c"], None) == f"{HEAD} Edit 1: a. Edit 2: c. {TAIL}"


def test_suppressing_every_edit_leaves_only_the_frame():
    # Current behaviour: the empty edit list leaves a double space between the two sentences.
    assert build_analogy_prompt(["a", "b"], None, suppress_indices=[0, 1]) == f"{HEAD}  {TAIL}"
