"""The analogy prompt shared by both backbones, training and inference."""


def build_analogy_prompt(pair_a_edits, pair_b_edits, suppress_indices=None):
    """Build the concat-mode analogy prompt for an arbitrary number of edits.

    Supports N∈{2,3,4} (and beyond) by simply iterating over ``pair_a_edits``.
    Only the A pair edits are listed (the model infers B→B' by analogy).

    Args:
        pair_a_edits: list of edit strings for A→A' (length = number of edits).
        pair_b_edits: list of edit strings for B→B' (unused in prompt, kept
            for API compatibility).
        suppress_indices: iterable of 0-based indices into ``pair_a_edits`` to
            omit from the listing (``None`` = keep all).

    Returns:
        Prompt string. The listed edit numbers in the output are renumbered
        contiguously after suppression (Edit 1, Edit 2, …) so the visible
        prompt never references an omitted edit.
    """
    suppress = set(suppress_indices or [])
    kept = [e for i, e in enumerate(pair_a_edits) if i not in suppress and e]
    edits_part = " ".join(f"Edit {n + 1}: {e}." for n, e in enumerate(kept))
    return (
        "Image 1 is the original and image 2 is the edited version. "
        f"{edits_part} "
        "Apply the same edits to image 3 to produce the output."
    )
