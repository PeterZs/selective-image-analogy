"""The ExLoRA token mask and the CSV token positions must select exactly the tokens of each edit text.

Both are computed on the sequence the text encoder actually sees: the chat-templated prompt for FLUX.2-Klein,
and the Qwen3-VL processor sequence (vision slots expanded, system prompt dropped) for Qwen-Image-2.1.
Only the tokenizer / processor is loaded (CPU, HF cache); the tests skip when it is not cached.
"""
import pytest
import torch

from sia.prompt import build_analogy_prompt

MAX_LEN = 512

# One representative prompt per supported edit count, in the style of the dataset's edit texts.
EDITS_BY_N = {
    2: ["turned-on front lights", "open the front doors"],
    3: ["add open-top crown", "add gold seamless necklace", "add angry expression"],
    4: [
        "add open-top crown without jewelries",
        "add green color hair",
        "add old cobblestone alley background",
        "add Van gogh style",
    ],
}
MISSING_EDIT = "this exact phrase is not in the prompt 12345"


# ---------------------------------------------------------------------------
# FLUX.2-Klein
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def klein():
    return pytest.importorskip("sia.backbones.flux2_klein", exc_type=ImportError)


def skip_unless_real_tokenizer(tokenizer, what):
    """Skip when ``tokenizer`` is a placeholder: offline with an empty cache, transformers can return a tokenizer
    without vocabulary or chat template instead of raising."""
    if tokenizer.chat_template is None or len(tokenizer) < 1000:
        pytest.skip(f"{what} not in the Hugging Face cache")


@pytest.fixture(scope="module")
def klein_tokenizer():
    from transformers import Qwen2TokenizerFast
    try:
        tokenizer = Qwen2TokenizerFast.from_pretrained("black-forest-labs/FLUX.2-klein-base-9B", subfolder="tokenizer")
    except (OSError, ValueError) as exc:
        pytest.skip(f"FLUX.2-Klein tokenizer not available: {exc}")
    skip_unless_real_tokenizer(tokenizer, "FLUX.2-Klein tokenizer")
    return tokenizer


def klein_ids(klein, prompt, tokenizer):
    templated = klein._apply_klein_chat_template(prompt, tokenizer)
    enc = tokenizer(templated, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    return enc["input_ids"][0].tolist()


@pytest.mark.parametrize("n", sorted(EDITS_BY_N))
def test_klein_edit_positions_decode_to_edit_text(klein, klein_tokenizer, n):
    edits = EDITS_BY_N[n]
    prompt = build_analogy_prompt(edits, edits)
    ids = klein_ids(klein, prompt, klein_tokenizer)
    prompt_positions = set(klein.klein_templated_prompt_token_positions(prompt, klein_tokenizer, max_length=MAX_LEN))

    for edit in edits:
        positions = klein.klein_find_substring_token_indices(prompt, edit, klein_tokenizer, max_length=MAX_LEN)
        decoded = klein_tokenizer.decode([ids[p] for p in positions], skip_special_tokens=False)
        assert decoded.strip() == edit
        assert set(positions) <= prompt_positions


@pytest.mark.parametrize("n", sorted(EDITS_BY_N))
def test_klein_exlora_mask_equals_csv_positions(klein, klein_tokenizer, n):
    edits = EDITS_BY_N[n]
    prompt = build_analogy_prompt(edits, edits)
    cpu = torch.device("cpu")

    per_edit = []
    for edit in edits:
        mask = klein.build_token_mask(prompt, [edit], klein_tokenizer, device=cpu, max_seq_len=MAX_LEN)
        mask_positions = set(torch.nonzero(mask[0]).flatten().tolist())
        csv_positions = set(klein.templated_substring_indices(prompt, edit, klein_tokenizer, max_length=MAX_LEN))
        assert mask_positions
        assert mask_positions == csv_positions
        per_edit.append(mask_positions)

    full_mask = klein.build_token_mask(prompt, edits, klein_tokenizer, device=cpu, max_seq_len=MAX_LEN)
    assert set(torch.nonzero(full_mask[0]).flatten().tolist()) == set().union(*per_edit)
    assert int(full_mask.sum()) == sum(len(p) for p in per_edit)


def test_klein_mask_raises_on_missing_edit(klein, klein_tokenizer):
    edits = EDITS_BY_N[2]
    prompt = build_analogy_prompt(edits, edits)
    with pytest.raises(RuntimeError):
        klein.build_token_mask(prompt, [MISSING_EDIT], klein_tokenizer, device=torch.device("cpu"),
                               max_seq_len=MAX_LEN)


# ---------------------------------------------------------------------------
# Qwen-Image-2.1
# ---------------------------------------------------------------------------

# (w, h) of the raw condition images: the dataset's two sizes plus a landscape one.
RAW_SIZES = [(384, 512), (368, 512), (640, 480)]


@pytest.fixture(scope="module")
def qwen():
    return pytest.importorskip("sia.backbones.qwen_image21", exc_type=ImportError)


@pytest.fixture(scope="module")
def qwen_pipe(qwen):
    from diffusers import QwenImage21Pipeline
    from huggingface_hub import snapshot_download
    from transformers import Qwen3VLProcessor
    # Loaded from the snapshot directory: with HF_HUB_OFFLINE=1, ``from_pretrained(repo, subfolder=...)``
    # fails because the processor's tokenizer also looks up a config.json that the repo does not have.
    try:
        snapshot = snapshot_download("Qwen/Qwen-Image-2.1", allow_patterns="processor/*")
        processor = Qwen3VLProcessor.from_pretrained(f"{snapshot}/processor")
    except (OSError, ValueError) as exc:
        pytest.skip(f"Qwen-Image-2.1 processor not available: {exc}")
    skip_unless_real_tokenizer(processor.tokenizer, "Qwen-Image-2.1 processor")
    # A processor-only pipeline is enough for the prompt templates and ``_drop_idx``.
    return QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=None, processor=processor, transformer=None)


def qwen_case(qwen, pipe, n, resolution):
    """Prompt, condition sizes, processor input_ids and the rows of ``prompt_embeds`` for one case."""
    from PIL import Image
    edits = EDITS_BY_N[n]
    prompt = build_analogy_prompt(edits, edits)
    sizes = [qwen.qwen21_resize_dims(w, h, resolution) for w, h in RAW_SIZES]
    images = [Image.new("RGB", s, (40 * i, 80, 120)) for i, s in enumerate(sizes)]
    proc_ids = pipe.processor(
        text=[qwen.qwen21_templated_text(pipe, prompt, len(images))], images=images,
        padding=True, padding_side="left", return_tensors="pt",
    )["input_ids"][0].tolist()
    rows = proc_ids[pipe._drop_idx:]
    return edits, prompt, sizes, proc_ids, rows


@pytest.mark.parametrize("resolution", [512, 1024])
@pytest.mark.parametrize("n", sorted(EDITS_BY_N))
def test_qwen_tokenization_matches_processor(qwen, qwen_pipe, n, resolution):
    _, prompt, sizes, proc_ids, _ = qwen_case(qwen, qwen_pipe, n, resolution)
    _, ids, _ = qwen.qwen21_tokenize_with_offsets(qwen_pipe, prompt, sizes)
    assert ids == proc_ids


@pytest.mark.parametrize("resolution", [512, 1024])
@pytest.mark.parametrize("n", sorted(EDITS_BY_N))
def test_qwen_edit_positions_decode_to_edit_text(qwen, qwen_pipe, n, resolution):
    edits, prompt, sizes, _, rows = qwen_case(qwen, qwen_pipe, n, resolution)
    image_pad = torch.tensor([t == qwen_pipe._img_token_id for t in rows])

    for edit in edits:
        positions = qwen.qwen21_find_substring_token_indices(qwen_pipe, prompt, edit, sizes)
        decoded = qwen_pipe.processor.tokenizer.decode([rows[p] for p in positions])
        assert decoded.strip() == edit
        assert not bool(image_pad[positions].any())


@pytest.mark.parametrize("resolution", [512, 1024])
@pytest.mark.parametrize("n", sorted(EDITS_BY_N))
def test_qwen_exlora_mask_equals_csv_positions(qwen, qwen_pipe, n, resolution):
    edits, prompt, sizes, _, rows = qwen_case(qwen, qwen_pipe, n, resolution)
    prompt_rows = set(qwen.qwen21_prompt_token_positions(qwen_pipe, prompt, sizes))

    for edit in edits:
        mask = qwen.qwen21_text_token_mask(qwen_pipe, prompt, [edit], sizes, len(rows), "cpu")
        mask_positions = set(torch.nonzero(mask[0]).flatten().tolist())
        csv_positions = set(qwen.qwen21_find_substring_token_indices(qwen_pipe, prompt, edit, sizes))
        assert mask_positions
        assert mask_positions == csv_positions
        assert mask_positions <= prompt_rows


@pytest.mark.parametrize("resolution", [512, 1024])
@pytest.mark.parametrize("n", sorted(EDITS_BY_N))
def test_qwen_joint_mask_shifts_by_vision_slots(qwen, qwen_pipe, n, resolution):
    from diffusers.models.transformers.transformer_qwenimage21 import _IMG_TOKENS_PER_SLOT

    edits, prompt, sizes, _, rows = qwen_case(qwen, qwen_pipe, n, resolution)
    image_pad = torch.tensor([t == qwen_pipe._img_token_id for t in rows])
    text_mask = qwen.qwen21_text_token_mask(qwen_pipe, prompt, edits, sizes, len(rows), "cpu")
    target_tokens = (sizes[-1][0] // 16) * (sizes[-1][1] // 16)
    img_mask = torch.cat([image_pad, torch.ones(target_tokens // _IMG_TOKENS_PER_SLOT, dtype=torch.bool)])[None]

    joint = qwen.qwen21_joint_token_mask(text_mask, img_mask)

    # Each vision slot before a text token becomes _IMG_TOKENS_PER_SLOT latent tokens.
    n_slots = int(image_pad.sum())
    assert joint.shape[1] == len(rows) - n_slots + n_slots * _IMG_TOKENS_PER_SLOT + target_tokens
    expected = {p + (_IMG_TOKENS_PER_SLOT - 1) * int(image_pad[:p].sum())
                for p in torch.nonzero(text_mask[0]).flatten().tolist()}
    assert set(torch.nonzero(joint[0]).flatten().tolist()) == expected


def test_qwen_mask_raises_on_missing_edit(qwen, qwen_pipe):
    edits = EDITS_BY_N[2]
    prompt = build_analogy_prompt(edits, edits)
    sizes = [qwen.qwen21_resize_dims(w, h, 512) for w, h in RAW_SIZES]
    with pytest.raises(RuntimeError):
        qwen.qwen21_text_token_mask(qwen_pipe, prompt, [MISSING_EDIT], sizes, 4096, "cpu")


def test_qwen_text_only_template_alignment(qwen, qwen_pipe):
    # Text-to-image template (no condition images), used by the isolated-sentence CSV pairs.
    sentence = "A bright living room with large windows."
    positions = qwen.qwen21_find_substring_token_indices(qwen_pipe, sentence, "bright", [])
    _, ids, _ = qwen.qwen21_tokenize_with_offsets(qwen_pipe, sentence, [])
    decoded = qwen_pipe.processor.tokenizer.decode([ids[qwen_pipe._drop_idx + p] for p in positions])
    assert decoded.strip() == "bright"
