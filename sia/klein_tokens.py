"""Token positions in FLUX.2-Klein prompts.

Klein encodes the chat-templated prompt, so every token index (ExLoRA mask, CSV positions) is computed
against that templated text.
"""


def _apply_klein_chat_template(prompt: str, tokenizer) -> str:
    """Wrap ``prompt`` in the same Qwen chat template that
    ``Flux2KleinPipeline._get_qwen3_prompt_embeds`` applies before tokenizing.

    This is the **only** sequence the diffusion transformer ever sees, so
    every token-position computation (ExLoRA mask, steering vector,
    embedding pooling) MUST be done against this string.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def klein_find_substring_token_indices(
    prompt: str,
    substr: str,
    tokenizer,
    max_length: int = 512,
):
    """Return Qwen token indices of ``substr`` inside ``prompt``.

    The substring is aligned against the chat-templated text, which is exactly
    what ``Flux2KleinPipeline._get_qwen3_prompt_embeds`` encodes. The returned
    indices therefore index ``prompt_embeds`` directly (for steering vectors)
    and the token mask consumed by ``ExLoRALinear`` (which sees
    ``encoder_hidden_states`` produced from the same templated sequence).
    """
    text = _apply_klein_chat_template(prompt, tokenizer)

    enc = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"][0].tolist()  # list of (char_start, char_end)

    # Find the character span of the substring in the templated text.
    # Prefer the "Edit N: <substr>." occurrence so an edit that also appears in
    # the preamble ("image") or inside an earlier edit isn't matched there.
    anchored = text.find(f": {substr}.")
    char_start = anchored + 2 if anchored != -1 else text.find(substr)
    if char_start == -1:
        raise AssertionError(
            f"substr not found in templated prompt text: '{substr[:80]}'"
        )
    char_end = char_start + len(substr)

    # Collect token indices whose character span overlaps with the substring.
    # Padding tokens and special tokens have offset (0, 0); skip them after
    # position 0 so we don't accidentally pick up the leading <|im_start|>.
    token_indices = []
    for tok_idx, (cs, ce) in enumerate(offsets):
        if cs == ce == 0 and tok_idx > 0:
            continue
        if ce > char_start and cs < char_end:
            token_indices.append(tok_idx)

    if not token_indices:
        raise AssertionError(
            f"No tokens overlap with substr chars [{char_start}:{char_end}]: "
            f"'{substr[:80]}'"
        )

    return token_indices


def klein_templated_prompt_token_positions(
    prompt: str,
    tokenizer,
    max_length: int = 512,
):
    """Return the list of non-padding token indices in the chat-templated
    prompt — i.e. every "real" text-token position the diffusion encoder
    actually sees (preamble, edits, end-of-turn marker, etc.).

    Used to support the ``scope='prompt'`` steering mode in inference and
    for sanity checks / regression tests.
    """
    text = _apply_klein_chat_template(prompt, tokenizer)
    enc = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    attn = enc["attention_mask"][0].tolist()
    return [i for i, m in enumerate(attn) if m]
