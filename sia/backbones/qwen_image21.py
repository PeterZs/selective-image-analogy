"""Qwen-Image-2.1 backbone (``QwenImage21Pipeline``, diffusers >= 0.42).

Everything that depends on the model family lives here: pipeline and adapter
loading, condition-image preparation, the joint Qwen3-VL prompt encoding and
its token positions, the timestep shift, the transformer forward, the sampler
and the per-sample flow-matching loss.

Qwen-Image-2.1 differs from FLUX.2-Klein in two ways that matter for the
token mask shared by ExLoRA and the CSV:

1. The prompt is encoded by Qwen3-VL *together with* the condition images.
   Each ``<|image_pad|>`` placeholder in the template is expanded by the
   processor into one token per vision slot, and the leading system-prompt
   tokens are dropped afterwards (``pipe._drop_idx``). The rows of
   ``prompt_embeds`` are therefore positions in that expanded, truncated
   sequence, not in the raw prompt and not in the chat template.

2. The transformer is single-stream. It builds one joint sequence in which
   every vision slot is replaced by ``_IMG_TOKENS_PER_SLOT`` (=4) latent
   tokens and the target image's tokens are appended. ExLoRA wraps the
   shared projections, so its mask has to be expressed in that joint
   sequence (see ``qwen21_joint_token_mask``).

Every position helper mirrors ``QwenImage21Pipeline._get_qwen_prompt_embeds``
and is checked against the processor's own ``input_ids`` by
``training/_check_token_alignment_qwen21.py``.

This module needs the Qwen environment; Klein code must never import it at
module level.
"""

import contextlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import QwenImage21Pipeline
from diffusers.models.transformers.transformer_qwenimage21 import (
    _IMG_TOKENS_PER_SLOT,
    QwenImage21KVCache,
)
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from sia.adapters import (
    checkpoint_files,
    checkpoint_lora_rank,
    exlora_token_mask_ctx,
    find_latest_checkpoint,
    inject_analora_modules,
    inject_exlora_modules,
    load_analora_from_peft_checkpoint,
    load_analora_state_dict,
    load_exlora_state_dict,
    load_tensors,
)
from sia.hf import resolve_checkpoint

# ---------------------------------------------------------------------------
# Target modules. Every block is single-stream, so the same projections carry
# text and image tokens. AnaLoRA adapts all of them for every token; ExLoRA
# wraps the same projections but only fires on the masked edit tokens — the
# single-stream equivalent of the Klein ExLoRA on the text-stream layers.
# ---------------------------------------------------------------------------
QWEN21_LORA_TARGET_MODULES = [
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out",
]
ANALORA_TARGET_MODULES = QWEN21_LORA_TARGET_MODULES
EXLORA_TARGET_MODULES = QWEN21_LORA_TARGET_MODULES


# ---------------------------------------------------------------------------
# Prompt token positions
# ---------------------------------------------------------------------------

_VISION_SLOT = "<image{}><|vision_start|><|image_pad|><|vision_end|>"


def qwen21_resize_dims(width: int, height: int, resolution: int):
    """Size the pipeline resizes a condition image to: ``resolution**2`` area
    at the image's own aspect ratio, sides rounded to multiples of 32 (one
    vision slot = one 2x2 group of 16x latent tokens). The output image takes
    the last condition image's size."""
    w, h, _ = calculate_dimensions(resolution * resolution, width / height)
    return w, h


def qwen21_templated_text(pipe, prompt: str, n_images: int) -> str:
    """The exact string ``_get_qwen_prompt_embeds`` hands to the processor for
    one prompt with ``n_images`` condition images (before pad expansion)."""
    # The pipeline swaps an empty prompt for " " (Qwen has no bos token).
    prompt = prompt or " "
    if n_images == 0:
        return pipe.prompt_template_t2i.format(prompt)
    slots = _VISION_SLOT.format(1) + "".join(
        " " + _VISION_SLOT.format(i) for i in range(2, n_images + 1))
    return pipe.prompt_template_ti2i.replace(_VISION_SLOT.format(1), slots).format(prompt)


def qwen21_image_slot_counts(processor, image_sizes) -> list:
    """Vision slots (``<|image_pad|>`` tokens) per image of size ``(w, h)``,
    as computed by the processor's own ``smart_resize``."""
    image_processor = processor.image_processor
    merge = image_processor.merge_size ** 2
    return [image_processor.get_number_of_image_patches(h, w, {}) // merge
            for w, h in image_sizes]


def qwen21_expanded_text(pipe, prompt: str, image_sizes) -> str:
    """Templated text with every ``<|image_pad|>`` repeated once per vision
    slot — the text the processor actually tokenizes."""
    text = qwen21_templated_text(pipe, prompt, len(image_sizes))
    pad = pipe.processor.image_token
    pieces = text.split(pad)
    if len(pieces) != len(image_sizes) + 1:
        raise ValueError(
            f"Expected {len(image_sizes)} {pad} placeholders in the template, "
            f"found {len(pieces) - 1} (does the prompt contain {pad}?).")
    counts = qwen21_image_slot_counts(pipe.processor, image_sizes)
    out = pieces[0]
    for n, piece in zip(counts, pieces[1:]):
        out += pad * n + piece
    return out


def _prompt_start(text: str, has_images: bool) -> int:
    """Character offset where the user prompt begins in the templated text."""
    if has_images:
        return text.rindex("<|vision_end|>") + len("<|vision_end|>")
    marker = "<|im_start|>user\n"
    return text.index(marker) + len(marker)


def qwen21_tokenize_with_offsets(pipe, prompt: str, image_sizes):
    """Tokenize the expanded text exactly as the processor does. Returns
    ``(text, input_ids, offsets)`` over the full (un-dropped) sequence."""
    text = qwen21_expanded_text(pipe, prompt, image_sizes)
    enc = pipe.processor.tokenizer(text, return_offsets_mapping=True)
    return text, enc["input_ids"], enc["offset_mapping"]


def qwen21_find_substring_token_indices(
    pipe,
    prompt: str,
    substr: str,
    image_sizes,
    case_sensitive: bool = True,
):
    """Return the rows of ``prompt_embeds`` covering ``substr`` in ``prompt``.

    ``image_sizes`` are the ``(w, h)`` of the condition images exactly as they
    are passed to the text encoder (after the pipeline's resize); use ``[]``
    for a text-only (t2i template) encode. Indices already account for the
    dropped system-prompt tokens, so they index ``prompt_embeds`` directly.

    Like ``klein_find_substring_token_indices``, the ``"Edit N: <substr>."``
    occurrence is preferred so an edit that also appears elsewhere in the
    prompt is not matched there. Raises ``AssertionError`` if not found.
    """
    text, _, offsets = qwen21_tokenize_with_offsets(pipe, prompt, image_sizes)
    start = _prompt_start(text, bool(image_sizes))
    haystack, needle = (text, substr) if case_sensitive else (text.lower(), substr.lower())

    anchored = haystack.find(f": {needle}.", start)
    char_start = anchored + 2 if anchored != -1 else haystack.find(needle, start)
    if char_start == -1:
        raise AssertionError(f"substr not found in templated prompt text: '{substr[:80]}'")
    char_end = char_start + len(substr)

    token_indices = [i for i, (cs, ce) in enumerate(offsets) if ce > char_start and cs < char_end]
    if not token_indices:
        raise AssertionError(
            f"No tokens overlap with substr chars [{char_start}:{char_end}]: '{substr[:80]}'")
    if token_indices[0] < pipe._drop_idx:
        raise AssertionError(f"'{substr[:80]}' matched inside the dropped system prompt.")
    return [i - pipe._drop_idx for i in token_indices]


def qwen21_prompt_token_positions(pipe, prompt: str, image_sizes):
    """Every text (non-vision-slot) row of ``prompt_embeds`` — the footprint
    of ``csv_scope='prompt'``. Vision-slot rows are skipped: the transformer
    overwrites them with image latents, so steering them has no effect."""
    _, ids, _ = qwen21_tokenize_with_offsets(pipe, prompt, image_sizes)
    ids = ids[pipe._drop_idx:]
    return [i for i, tok in enumerate(ids) if tok != pipe._img_token_id]


def qwen21_text_token_mask(pipe, prompt: str, edit_texts, image_sizes, seq_len: int, device):
    """``(1, seq_len)`` bool mask over ``prompt_embeds`` rows covering every
    edit in ``edit_texts``. Raises if an edit cannot be located — an empty
    mask silently disables ExLoRA."""
    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    for text in edit_texts:
        if not text:
            continue
        try:
            indices = qwen21_find_substring_token_indices(pipe, prompt, text, image_sizes)
        except AssertionError as exc:
            raise RuntimeError(
                f"[TokenMask] Could not locate edit text in prompt:\n"
                f"  edit:   {text!r}\n"
                f"  prompt: {prompt!r}\n"
                f"  underlying: {exc}"
            ) from exc
        if indices[-1] >= seq_len:
            raise RuntimeError(
                f"[TokenMask] Edit {text!r} maps to row {indices[-1]} but prompt_embeds has "
                f"{seq_len} rows — image sizes differ from the ones that were encoded.")
        mask[0, indices] = True
    if edit_texts and not bool(mask.any()):
        raise RuntimeError("[TokenMask] All edit texts produced empty positions.")
    return mask


def qwen21_joint_token_mask(text_mask: torch.Tensor, img_mask: torch.Tensor) -> torch.Tensor:
    """Map a ``(B, S)`` mask over ``prompt_embeds`` rows onto the joint
    sequence built by ``QwenImage21Transformer2DModel.forward``.

    ``img_mask`` is the transformer's ``img_mask`` argument: ``(B, S + T/4)``,
    True at vision slots and at the appended target slots. Each True slot
    becomes ``_IMG_TOKENS_PER_SLOT`` joint tokens, every other position one.
    """
    if bool((text_mask & img_mask[:, : text_mask.shape[1]]).any()):
        raise ValueError("Token mask marks vision-slot positions; it must cover text tokens only.")
    repeats = torch.where(img_mask[0], _IMG_TOKENS_PER_SLOT, 1)
    extended = torch.cat(
        [text_mask, text_mask.new_zeros(text_mask.shape[0], img_mask.shape[1] - text_mask.shape[1])],
        dim=1,
    )
    return torch.repeat_interleave(extended, repeats, dim=1)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def harmonize_images_qwen21(a, a_prime, b, b_prime):
    """Snap A' to A's size and B' to B's size (same subject, same crop).

    Unlike the Klein rule nothing is downscaled here: the final resize happens
    at encode time with the pipeline's own rule (``qwen21_resize_dims``).
    """
    if a_prime.size != a.size:
        a_prime = a_prime.resize(a.size, Image.LANCZOS)
    if b_prime.size != b.size:
        b_prime = b_prime.resize(b.size, Image.LANCZOS)
    return a, a_prime, b, b_prime


def _to_rgba(img):
    # The Qwen-Image-2.1 VAE reads RGBA; the pipeline converts every condition image.
    return img if img.mode == "RGBA" else img.convert("RGBA")


def flatten_rgba(img):
    """Composite an RGBA output over white, as the pipeline does for the encoder."""
    if img.mode != "RGBA":
        return img.convert("RGB")
    white = Image.new("RGB", img.size, (255, 255, 255))
    white.paste(img, mask=img.getchannel("A"))
    return white


# ---------------------------------------------------------------------------
# Qwen-Image-2.1 model helpers (mirror QwenImage21Pipeline.__call__)
# ---------------------------------------------------------------------------

def load_pipeline_qwen21(pretrained_model_name_or_path):
    """Load the full pipeline in bf16. Going through ``from_pretrained`` keeps
    the checkpoint's own config (e.g. Turbo's ``sample_sigmas``)."""
    return QwenImage21Pipeline.from_pretrained(pretrained_model_name_or_path, dtype=torch.bfloat16)


@dataclass
class Qwen21Conditions:
    """The condition images A, A', B as the pipeline prepares them."""
    vl_images: list        # RGBA PIL at encode size — what encode_prompt receives
    image_sizes: list      # (w, h) per condition image
    latents: torch.Tensor  # (1, N_cond, C) packed, normalised VAE latents
    img_shapes: list       # (1, h, w) latent grid per condition image

    @property
    def target_size(self):
        # The output takes the last condition image's (B's) size.
        return self.image_sizes[-1]


@torch.no_grad()
def encode_image_latents(pipe, img, width, height, device):
    """VAE-encode one image exactly as ``QwenImage21Pipeline.prepare_latents``
    does. Returns packed latents ``(1, h*w, C)`` and the grid ``(1, h, w)``."""
    pixels = pipe.image_processor.preprocess(_to_rgba(img), width=width, height=height).unsqueeze(2)
    latents = pipe._encode_vae_image(pixels.to(device=device, dtype=pipe.vae.dtype), generator=None)
    _, channels, _, h, w = latents.shape
    return pipe._pack_latents(latents, 1, channels, h, w), (1, h, w)


@torch.no_grad()
def prepare_conditions(pipe, images, resolution, device) -> Qwen21Conditions:
    """One resize per condition image feeds both the text encoder and the VAE
    (pipeline step 1), at ``resolution**2`` area and the image's own aspect."""
    vl_images, sizes, latents, shapes = [], [], [], []
    for img in images:
        img = _to_rgba(img)
        w, h = qwen21_resize_dims(img.width, img.height, resolution)
        vl_images.append(pipe.image_processor.resize(img, width=w, height=h))
        tokens, shape = encode_image_latents(pipe, img, w, h, device)
        sizes.append((w, h))
        latents.append(tokens)
        shapes.append(shape)
    return Qwen21Conditions(vl_images, sizes, torch.cat(latents, dim=1), shapes)


@torch.no_grad()
def encode_prompt(pipe, prompt, cond: Qwen21Conditions, device):
    """Joint Qwen3-VL encoding of the prompt and the condition images.
    Returns ``prompt_embeds (1, S, D)`` and ``image_pad_mask (1, S)``."""
    prompt_embeds, _, image_pad_mask = pipe._get_qwen_prompt_embeds([prompt], cond.vl_images, device)
    return prompt_embeds, image_pad_mask


def build_img_mask(image_pad_mask, n_target_tokens):
    """The transformer's ``img_mask``: vision slots plus one slot per 2x2
    group of target tokens."""
    target_slots = image_pad_mask.new_ones(image_pad_mask.shape[0], n_target_tokens // _IMG_TOKENS_PER_SLOT)
    return torch.cat([image_pad_mask, target_slots], dim=1)


def qwen21_mu(scheduler, image_seq_len):
    """Resolution-dependent shift, computed as in the pipeline."""
    cfg = scheduler.config
    return calculate_shift(
        image_seq_len,
        cfg.get("base_image_seq_len", 256),
        cfg.get("max_image_seq_len", 4096),
        cfg.get("base_shift", 0.5),
        cfg.get("max_shift", 1.15),
    )


def sample_train_sigma(scheduler, n_target_tokens, weighting_scheme="none", shift=True):
    """Draw one training sigma in [0, 1]. With ``shift`` the uniform (or
    logit-normal) draw goes through the same exponential time shift the
    sampler applies at this target size."""
    u = compute_density_for_timestep_sampling(
        weighting_scheme, batch_size=1, logit_mean=0.0, logit_std=1.0, mode_scale=1.29)
    if shift and scheduler.config.use_dynamic_shifting:
        u = scheduler.time_shift(qwen21_mu(scheduler, n_target_tokens), 1.0, u)
    return u


def predict_velocity(transformer, cond: Qwen21Conditions, target_tokens, timestep,
                     prompt_embeds, img_mask, target_shape, kv_cache=None, kv_cache_mode=None):
    """One transformer call: condition latents first, the target last, as in
    the pipeline. ``timestep`` is sigma in [0, 1]. Returns the target's velocity."""
    out = transformer(
        hidden_states=torch.cat([cond.latents, target_tokens], dim=1),
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=None,  # batch of one: nothing is padded
        img_shapes=[cond.img_shapes + [target_shape]],
        img_mask=img_mask,
        kv_cache=kv_cache,
        kv_cache_mode=kv_cache_mode,
        return_dict=False,
    )[0]
    return out[:, -target_tokens.shape[1]:]


@torch.no_grad()
def decode_latents(pipe, latents, width, height):
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor).to(pipe.vae.dtype)
    z_dim = pipe.vae.config.z_dim
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    std = torch.tensor(pipe.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    image = pipe.vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
    return flatten_rgba(pipe.image_processor.postprocess(image, output_type="pil")[0])


@torch.no_grad()
def generate_qwen21(pipe, cond: Qwen21Conditions, prompt_embeds, image_pad_mask,
                    num_steps, generator=None, exlora_text_mask=None,
                    use_kv_cache=True, prompt_embeds_schedule=None):
    """Sample B' with the pipeline's sampler (sigma grid, dynamic shift, prefix
    KV cache, no CFG). ``exlora_text_mask`` is a ``(1, S)`` mask over
    ``prompt_embeds`` rows; ``None`` leaves ExLoRA off."""
    transformer = pipe.transformer
    device = prompt_embeds.device
    width, height = cond.target_size
    h, w = height // pipe.vae_scale_factor, width // pipe.vae_scale_factor
    channels = transformer.config.in_channels

    latents = randn_tensor((1, 1, channels, h, w), generator=generator, device=device, dtype=prompt_embeds.dtype)
    latents = pipe._pack_latents(latents, 1, channels, h, w)

    sigmas = pipe.config.sample_sigmas
    if sigmas is None:
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, num_steps, device, sigmas=sigmas, mu=qwen21_mu(pipe.scheduler, latents.shape[1]))
    pipe.scheduler.set_begin_index(0)
    if prompt_embeds_schedule is not None and len(prompt_embeds_schedule) != len(timesteps):
        raise ValueError(f"prompt_embeds_schedule has {len(prompt_embeds_schedule)} entries "
                         f"for {len(timesteps)} steps.")

    img_mask = build_img_mask(image_pad_mask, latents.shape[1])
    joint_mask = None
    if exlora_text_mask is not None:
        joint_mask = qwen21_joint_token_mask(exlora_text_mask, img_mask)

    # A per-step embedding schedule changes the prefix every step, so the
    # prefix cannot be cached then.
    cache = None
    if use_kv_cache and transformer.config.causal_condition and prompt_embeds_schedule is None:
        cache = QwenImage21KVCache(len(transformer.transformer_blocks))

    for i, t in enumerate(timesteps):
        kv_mode = ("extract" if i == 0 else "cached") if cache is not None else None
        embeds = prompt_embeds_schedule[i] if prompt_embeds_schedule is not None else prompt_embeds
        # ExLoRA only gates prompt tokens, and a cached step recomputes the
        # target tokens alone — their prefix K/V already carry the update.
        with exlora_token_mask_ctx(transformer, None if kv_mode == "cached" else joint_mask):
            noise_pred = predict_velocity(
                transformer, cond, latents, t.expand(1).to(latents.dtype) / 1000,
                embeds, img_mask, (1, h, w), cache, kv_mode)
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return decode_latents(pipe, latents, width, height)


def sample_flow_loss(pipe, transformer, images, b_prime, prompt, resolution, device,
                     args, suppress_texts=None):
    """Flow-matching loss for one (A, A', B) -> B' sample.

    ``suppress_texts``: edit strings whose tokens gate ExLoRA (paper eq. 14);
    ``None`` runs with ExLoRA off. The mask is left set on return so the
    gradient-checkpointing recompute in backward sees it.
    """
    cond = prepare_conditions(pipe, images, resolution, device)
    target, target_shape = encode_image_latents(pipe, b_prime, *cond.target_size, device)
    prompt_embeds, image_pad_mask = encode_prompt(pipe, prompt, cond, device)
    img_mask = build_img_mask(image_pad_mask, target.shape[1])

    joint_mask = None
    if suppress_texts is not None:
        text_mask = qwen21_text_token_mask(
            pipe, prompt, suppress_texts, cond.image_sizes, prompt_embeds.shape[1], device)
        joint_mask = qwen21_joint_token_mask(text_mask, img_mask)

    sigma = sample_train_sigma(
        pipe.scheduler, target.shape[1],
        weighting_scheme=args.get("weighting_scheme", "none"),
        shift=args.get("shift_train_sigmas", True),
    ).to(device=device, dtype=target.dtype)
    noise = torch.randn_like(target)
    noisy = (1.0 - sigma) * target + sigma * noise

    with exlora_token_mask_ctx(transformer, joint_mask, disable_mask_after=False):
        pred = predict_velocity(transformer, cond, noisy, sigma, prompt_embeds, img_mask, target_shape)
    return F.mse_loss(pred.float(), (noise - target).float())


def backward_per_sample(accelerator, model, losses_fn, bsz, extra_loss_fn=None):
    """Run ``losses_fn(i)`` and backward for each sample in turn, averaging
    over the batch. Gradients sync only on the last backward; ``extra_loss_fn``
    (e.g. the AnaLoRA anchor) joins that one. Returns the mean sample loss."""
    total = 0.0
    for i in range(bsz):
        is_last = i == bsz - 1
        ctx = contextlib.nullcontext() if is_last else accelerator.no_sync(model)
        with ctx:
            loss = losses_fn(i) / bsz
            total += loss.detach().item()
            if is_last and extra_loss_fn is not None:
                loss = loss + extra_loss_fn()
            accelerator.backward(loss)
    return total


# ---------------------------------------------------------------------------
# Inference pipeline with adapters
# ---------------------------------------------------------------------------

def load_pipeline(args, device):
    """Qwen-Image-2.1 pipeline with AnaLoRA + ExLoRA loaded for inference.

    The checkpoint is ``args.checkpoint_dir`` (local dir or Hub path) if set,
    else the latest ``checkpoint-*`` in ``args.output_dir``. A Stage-2
    checkpoint carries its own AnaLoRA (and its rank); otherwise the Stage-1
    PEFT AnaLoRA comes from ``args.analora_path``."""
    pipe = load_pipeline_qwen21(args.pretrained_model_name_or_path)
    transformer = pipe.transformer

    explicit_ckpt = getattr(args, "checkpoint_dir", None)
    ckpt_dir = resolve_checkpoint(explicit_ckpt) if explicit_ckpt else find_latest_checkpoint(args.output_dir)

    # A Stage-2 checkpoint carries its own AnaLoRA; take the rank from it.
    analora_file, exlora_file = checkpoint_files(ckpt_dir) if ckpt_dir else (None, None)
    has_ckpt_base = analora_file is not None and analora_file.exists()
    if has_ckpt_base:
        rank = checkpoint_lora_rank(ckpt_dir, load_tensors(analora_file))
        if rank != args.lora_rank:
            print(f"[Checkpoint] lora_rank {args.lora_rank} in config -> {rank} from checkpoint")
            args.lora_rank = rank

    n_base = inject_analora_modules(transformer, ANALORA_TARGET_MODULES,
                                    r=args.lora_rank, alpha=args.lora_rank, dropout=0.0)
    print(f"[AnaLoRA] Injected {len(n_base)} modules")
    if not has_ckpt_base:
        load_analora_from_peft_checkpoint(transformer, resolve_checkpoint(args.analora_path), device="cpu")

    n_st = inject_exlora_modules(transformer, EXLORA_TARGET_MODULES,
                                 r=args.lora_rank, alpha=args.lora_rank, dropout=0.0)
    print(f"[ExLoRA] Injected {len(n_st)} modules")

    if ckpt_dir:
        print(f"[Checkpoint] Loading from {ckpt_dir}")
        if exlora_file.exists():
            load_exlora_state_dict(transformer, exlora_file)
        elif explicit_ckpt:
            raise FileNotFoundError(
                f"{exlora_file.name} missing from {ckpt_dir} — without it ExLoRA is "
                f"zero-initialised and has no effect.")
        if has_ckpt_base:
            load_analora_state_dict(transformer, analora_file, "cpu")
            print("[Checkpoint] AnaLoRA loaded from checkpoint")
    else:
        print("[Checkpoint] No checkpoint found — using freshly initialised ExLoRA")

    pipe.to(device)
    transformer.eval()
    return pipe
