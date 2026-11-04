"""The analogy prompt shared by both backbones, training and inference."""


def build_analogy_prompt(pair_a_edits, pair_b_edits, suppress_indices=None):
    """Build the analogy prompt that lists the edits of the A -> A' demonstration.

    Only the A-pair edits are listed; the model infers B -> B' by analogy. Any number of edits
    is supported.

    Args:
        pair_a_edits: edit strings of the A -> A' demonstration, in order.
        pair_b_edits: unused; accepted so that callers can pass both pairs.
        suppress_indices: 0-based indices into ``pair_a_edits`` to leave out (``None`` keeps all).

    Returns:
        The prompt string. Kept edits are renumbered contiguously (``Edit 1``, ``Edit 2``, ...),
        so the prompt never refers to an omitted edit. Empty edit strings are skipped.
    """
    suppress = set(suppress_indices or [])
    kept = [e for i, e in enumerate(pair_a_edits) if i not in suppress and e]
    edits_part = " ".join(f"Edit {n + 1}: {e}." for n, e in enumerate(kept))
    return (
        "Image 1 is the original and image 2 is the edited version. "
        f"{edits_part} "
        "Apply the same edits to image 3 to produce the output."
    )
