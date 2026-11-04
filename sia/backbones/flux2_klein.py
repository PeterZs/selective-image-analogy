"""FLUX.2-Klein backbone: model loading, prompt tokens, latents, timestep shift and the denoising loop.

Klein encodes the chat-templated prompt with its Qwen3 text encoder, so every token index (ExLoRA mask,
CSV positions) is computed against that templated text. A, A' and B are VAE-encoded, packed and appended
to the target sequence as reference images (concat mode); the transformer predicts the velocity of B'.

Training (``scripts/train_*.py``): ``train_step`` (flow-matching loss and backward on a batch) and the
validation strips ``log_analora_validation`` / ``log_exlora_validation``.
"""
import hashlib
import itertools
import os
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_timesteps
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from huggingface_hub.constants import HF_HUB_OFFLINE
from PIL import Image, ImageDraw
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from sia.adapters import (
    ANALORA_ADAPTER_NAME,
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
    lora_mode_ctx,
)
from sia.concept_suppression import _build_generation_prompt, _parse_jsonl_records
from sia.data import resize_to_match, subject_image
from sia.hf import resolve_checkpoint
from sia.lora import ExLoRALinear
from sia.prompt import build_analogy_prompt
from sia.viz import _font

# ExLoRA adapts the context (text-stream) layers only; AnaLoRA adapts both streams.
EXLORA_TARGET_MODULES = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]

ANALORA_TARGET_MODULES = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2",
    "ff_context.net.0.proj", "ff_context.net.2",
]

# Training resolution (square side) of the released checkpoints; see configs/exlora_flux2_klein.yaml.
TRAIN_SIZE = 512


# ---------------------------------------------------------------------------
# Prompt tokens
# ---------------------------------------------------------------------------

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


def templated_substring_indices(prompt: str, substr: str, tokenizer,
                                max_length: int = 512):
    """``klein_find_substring_token_indices`` that returns ``[]`` instead of raising when
    ``substr`` is not in the prompt (callers treat "not found" as a soft signal)."""
    try:
        return klein_find_substring_token_indices(
            prompt, substr, tokenizer,
            max_length=max_length,
        )
    except AssertionError:
        return []


def build_token_mask(full_prompt, edit_texts, tokenizer, device, max_seq_len=512):
    """Build a (1, max_seq_len) bool mask covering tokens for each edit text.

    The mask is aligned against the **chat-templated** Qwen prompt — the
    exact sequence ``Flux2KleinPipeline._get_qwen3_prompt_embeds`` feeds
    into the text encoder. Mask positions therefore line up with rows of
    ``prompt_embeds`` and with the ``encoder_hidden_states`` consumed by
    ``ExLoRALinear``.

    Args:
        full_prompt:  full prompt string (raw, pre-template)
        edit_texts:   list of edit strings whose tokens should be masked
        tokenizer:    Qwen2TokenizerFast
        device:       target device
        max_seq_len:  must match max_length used in prompt encoding

    Returns:
        torch.BoolTensor of shape (1, max_seq_len). Raises if an edit cannot be
        located: an empty mask would silently disable ExLoRA.
    """
    mask = torch.zeros((1, max_seq_len), device=device, dtype=torch.bool)
    for text in edit_texts:
        if not text:
            continue
        try:
            indices = klein_find_substring_token_indices(
                full_prompt, text, tokenizer,
                max_length=max_seq_len,
            )
        except AssertionError as exc:
            # Substring not found in the prompt — almost always an edit-text
            # vs. prompt-builder mismatch. Loud failure is mandatory because
            # an empty mask silently disables ExLoRA.
            raise RuntimeError(
                f"[TokenMask] Could not locate edit text in prompt:\n"
                f"  edit:   {text!r}\n"
                f"  prompt: {full_prompt!r}\n"
                f"  underlying: {exc}"
            ) from exc
        for idx in indices:
            if idx < max_seq_len:
                mask[0, idx] = True

    if edit_texts and not bool(mask.any()):
        raise RuntimeError(
            "[TokenMask] All edit texts produced empty positions — ExLoRA "
            "would be a no-op. Check tokenizer / prompt format."
        )
    return mask


# ---------------------------------------------------------------------------
# Models and text encoding
# ---------------------------------------------------------------------------

def load_models(args):
    path = args.pretrained_model_name_or_path
    # The frozen models are cast to bf16 in main() anyway; loading them in bf16
    # avoids a ~70 GB fp32 copy in host RAM (the LoRA layers stay fp32).
    # diffusers 0.38 queries the Hub for sharded weights unless local_files_only is
    # set, so HF_HUB_OFFLINE=1 is passed on explicitly.
    bf16 = dict(torch_dtype=torch.bfloat16, local_files_only=HF_HUB_OFFLINE)
    tokenizer = Qwen2TokenizerFast.from_pretrained(path, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(path, subfolder="text_encoder", **bf16)
    vae = AutoencoderKLFlux2.from_pretrained(path, subfolder="vae", **bf16)
    transformer = Flux2Transformer2DModel.from_pretrained(path, subfolder="transformer", **bf16)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")
    return tokenizer, text_encoder, vae, transformer, noise_scheduler


def load_pipeline(args, device):
    """Klein pipeline with AnaLoRA + ExLoRA loaded.

    Weights come from ``args.checkpoint_dir`` (local dir or Hub path, see ``sia.hf.resolve_checkpoint``)
    or else the latest checkpoint in ``args.output_dir``. A Stage-2 checkpoint carries its own AnaLoRA;
    only without one is the Stage-1 AnaLoRA read from ``args.analora_path``.
    """
    tokenizer, text_encoder, vae, transformer, scheduler = load_models(args)

    explicit_ckpt = getattr(args, "checkpoint_dir", None)
    if explicit_ckpt:
        # Fail loud: a wrong path must not silently leave ExLoRA zero-initialised.
        ckpt_dir = resolve_checkpoint(explicit_ckpt)
    else:
        ckpt_dir = find_latest_checkpoint(args.output_dir)

    # Take the rank from the weights so a config/checkpoint rank mismatch can't break loading.
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
            # Most likely an AnaLoRA-only run picked by mistake: refuse to run with zero-init ExLoRA.
            raise FileNotFoundError(
                f"{exlora_file.name} missing from {ckpt_dir}\n"
                f"  Files present: {sorted(os.listdir(ckpt_dir))}\n"
                f"  Without this file ExLoRA is zero-initialised and has "
                f"no effect (st=0 == st=1).")
        if has_ckpt_base:
            load_analora_state_dict(transformer, analora_file, "cpu")
            print("[Checkpoint] AnaLoRA loaded from checkpoint")
    else:
        print("[Checkpoint] No checkpoint found — using freshly initialised weights")

    pipe = Flux2KleinPipeline(
        transformer=transformer, vae=vae,
        text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler,
    )
    pipe = pipe.to(device=device, dtype=torch.bfloat16)
    pipe.transformer.eval()
    return pipe, tokenizer, text_encoder


@torch.no_grad()
def compute_text_embeddings(prompts, tokenizer, text_encoder):
    prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompts,
    )
    text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds)
    text_ids = text_ids.to(prompt_embeds.device)
    return prompt_embeds, text_ids


# ---------------------------------------------------------------------------
# Timestep shift
# ---------------------------------------------------------------------------

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


# ---------------------------------------------------------------------------
# Latents
# ---------------------------------------------------------------------------

def _native_wh():
    """Fixed image size (w, h) used for every Klein image, or None for the square recipe.

    The released checkpoints were trained at the dataset's native 384x512 resolution, so that is
    the default. Environment variable SIA_NATIVE_RES overrides it: "0" or "" selects the legacy
    square recipe (resolution x resolution), "WxH" selects another fixed size.
    """
    v = os.environ.get("SIA_NATIVE_RES", "1")
    if v in ("", "0"):
        return None
    w, h = (384, 512) if v == "1" else map(int, v.lower().split("x"))
    return w // 16 * 16, h // 16 * 16


@torch.no_grad()
def encode_single_image(pipe, img, cell_size=512):
    """Resize, preprocess, and VAE-encode a single PIL image. Returns (1, C, h, w) latent.

    Unless SIA_NATIVE_RES=0 (legacy square recipe) every image is encoded at the fixed native
    non-square size instead of cell_size x cell_size (our data is 384x512 portrait)."""
    w, h = _native_wh() or (cell_size, cell_size)
    img = resize_to_match(img, w, h)
    tensor = pipe.image_processor.preprocess(img, h, w)   # height first
    tensor = tensor.to(device=pipe._execution_device, dtype=torch.bfloat16)
    return pipe._encode_vae_image(tensor, generator=None)


@torch.no_grad()
def prepare_analogy_latents(pipe, batch, resolution=512):
    """Encode A, A', B, B' through VAE and prepare concat-mode latents + IDs.

    Returns:
        target_latents:    (B, N, C) packed B' latents
        condition_latents: (B, 3*N, C) packed [A, A', B] condition latents
        target_ids:        position IDs for target
        condition_ids:     position IDs for conditions
    """
    target_arr, cond_arr = [], []

    for a, a_prime, b, b_prime in zip(
        batch['A'], batch['A_prime'], batch['B'], batch['B_prime']
    ):
        lat_a       = encode_single_image(pipe, a,       resolution)
        lat_a_prime = encode_single_image(pipe, a_prime, resolution)
        lat_b       = encode_single_image(pipe, b,       resolution)
        lat_b_prime = encode_single_image(pipe, b_prime, resolution)

        target_arr.append(Flux2KleinPipeline._pack_latents(lat_b_prime))
        cond_a  = Flux2KleinPipeline._pack_latents(lat_a).squeeze(0)
        cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime).squeeze(0)
        cond_b  = Flux2KleinPipeline._pack_latents(lat_b).squeeze(0)
        cond_arr.append(
            torch.cat([cond_a, cond_ap, cond_b], dim=0).unsqueeze(0))

    target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_b_prime)
    condition_ids = Flux2KleinPipeline._prepare_image_ids(
        [lat_a, lat_a_prime, lat_b])
        
    return (
        torch.cat(target_arr, dim=0),
        torch.cat(cond_arr,   dim=0),
        target_ids,
        condition_ids,
    )


def _encode_image(pipe, img: Image.Image, resolution: int = TRAIN_SIZE):
    """VAE-encode ``img`` exactly as training does (``encode_single_image``): resized to
    ``resolution`` x ``resolution`` (or the SIA_NATIVE_RES size) without keeping aspect ratio."""
    return encode_single_image(pipe, img, resolution)


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(pipe, tokenizer, text_encoder,
                  image_a, image_a_prime, image_b,
                  prompt, texts_to_suppress, num_steps, device,
                  analora_scale: float = 1.0, exlora_scale: float = 1.0,
                  prompt_embeds_override: torch.Tensor = None,
                  prompt_embeds_schedule: list = None,
                  generator: "torch.Generator | None" = None):
    """
    texts_to_suppress: list of edit strings whose tokens are masked for ExLoRA.
                       Pass [] for no suppression.
    analora_scale / exlora_scale: multiplier in [0, 1] for each LoRA's strength.
    prompt_embeds_override: optional precomputed (possibly steered) prompt embeds
                            to use instead of re-encoding from `prompt`. When
                            provided, `prompt` is still used for token-mask
                            substring alignment.
    prompt_embeds_schedule: optional list of prompt_embed tensors (one per
                            denoising step). When given, the embedding fed to
                            the transformer is swapped per step — used to
                            implement a time-varying steering-alpha schedule.
    generator: optional ``torch.Generator`` used to sample the initial noise.
               Pass a freshly seeded generator per sweep cell to keep the
               initial noise fixed across the ablation axis — otherwise each
               call consumes the global RNG and every cell sees different
               noise, confounding the ablation.

    Images are encoded at TRAIN_SIZE x TRAIN_SIZE, matching training. The
    square output is resized back to B's size before it is returned.
    """
    dtype = torch.bfloat16

    lat_a       = _encode_image(pipe, image_a,       TRAIN_SIZE)
    lat_a_prime = _encode_image(pipe, image_a_prime, TRAIN_SIZE)
    lat_b       = _encode_image(pipe, image_b,       TRAIN_SIZE)

    if prompt_embeds_schedule is not None and len(prompt_embeds_schedule) > 0:
        if len(prompt_embeds_schedule) != num_steps:
            raise ValueError(
                f"prompt_embeds_schedule has {len(prompt_embeds_schedule)} entries "
                f"but num_steps={num_steps} — must be one embed per step.")
        prompt_embeds = prompt_embeds_schedule[0]
        text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(prompt_embeds.device)
    elif prompt_embeds_override is not None:
        prompt_embeds = prompt_embeds_override
        text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(prompt_embeds.device)
    else:
        prompt_embeds, text_ids = compute_text_embeddings([prompt], tokenizer, text_encoder)
    patch_h, patch_w = lat_a.shape[2], lat_a.shape[3]

    cond_a  = Flux2KleinPipeline._pack_latents(lat_a)
    cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime)
    cond_b  = Flux2KleinPipeline._pack_latents(lat_b)
    condition_latents = torch.cat([cond_a, cond_ap, cond_b], dim=1)

    if generator is not None:
        gen_latents = torch.randn(lat_a.shape, device=device, dtype=dtype,
                                  generator=generator)
    else:
        gen_latents = torch.randn(lat_a.shape, device=device, dtype=dtype)
    gen_latents   = Flux2KleinPipeline._pack_latents(gen_latents)
    target_ids    = Flux2KleinPipeline._prepare_latent_ids(lat_a).to(device)
    condition_ids = Flux2KleinPipeline._prepare_image_ids([lat_a, lat_a_prime, lat_b]).to(device)
    combined_ids  = torch.cat([target_ids, condition_ids], dim=1)

    sigmas       = np.linspace(1.0, 1 / num_steps, num_steps)
    mu           = compute_empirical_mu(gen_latents.shape[1], num_steps)
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_steps, device, sigmas=sigmas, mu=mu)

    # The mask is set whenever edits are suppressed, even at exlora_scale == 0
    # (lora_mode_ctx zeroes the ExLoRA delta), so st=0 and st=1 run the same
    # adapter path with different gain.
    if texts_to_suppress:
        token_mask = build_token_mask(
            prompt, texts_to_suppress, tokenizer, device,
            max_seq_len=prompt_embeds.shape[1],
        )
        if exlora_scale > 0.0:
            print(f"[ExLoRA] token_mask covers {int(token_mask.sum().item())} "
                  f"templated-prompt position(s) for {len(texts_to_suppress)} "
                  f"suppressed edit text(s).")
    else:
        token_mask = None

    with lora_mode_ctx(pipe.transformer, analora_scale, exlora_scale), \
         exlora_token_mask_ctx(pipe.transformer, token_mask, disable_mask_after=True):
        for step_idx, t in enumerate(timesteps):
            step_embeds = (prompt_embeds_schedule[step_idx]
                           if prompt_embeds_schedule is not None
                           else prompt_embeds)
            latent_model_input = torch.cat([gen_latents, condition_latents], dim=1)
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t.expand(1).to(dtype) / 1000,
                guidance=None,
                encoder_hidden_states=step_embeds,
                txt_ids=text_ids,
                img_ids=combined_ids,
                return_dict=False,
            )[0][:, :gen_latents.shape[1]]
            gen_latents = pipe.scheduler.step(noise_pred, t, gen_latents, return_dict=False)[0]

    unpacked = gen_latents.permute(0, 2, 1).reshape(1, -1, patch_h, patch_w)
    bn_mean  = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
    bn_std   = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1)
                          + pipe.vae.config.batch_norm_eps).to(unpacked.device, unpacked.dtype)
    unpacked = unpacked * bn_std + bn_mean
    unpacked = Flux2KleinPipeline._unpatchify_latents(unpacked)
    decoded  = pipe.vae.decode(unpacked, return_dict=False)[0]
    out = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    return out.resize(image_b.size, Image.LANCZOS)


# ---------------------------------------------------------------------------
# Training: one optimisation step and the validation strips
# ---------------------------------------------------------------------------

def train_step(accelerator, pipe, transformer, noise_scheduler, batch, prompts, args,
               suppress_texts=None, extra_loss_fn=None):
    """Flow-matching loss on a batch and its backward pass. Returns the loss value.

    ``pipe`` encodes the images and prompts; ``transformer`` is the model returned by
    ``accelerator.prepare``; ``noise_scheduler`` provides the training timesteps and sigmas.
    ``prompts`` holds one prompt per sample. ``suppress_texts`` holds, per sample, the edit texts
    whose tokens gate ExLoRA (paper eq. 14); ``None`` runs with ExLoRA off. ``extra_loss_fn``
    (e.g. the AnaLoRA anchor) is added to the loss before the backward pass.
    """
    device = accelerator.device

    # ---- Encode images (concat mode) ----
    target_latents, condition_latents, target_ids, condition_ids = \
        prepare_analogy_latents(pipe, batch, resolution=args.get("resolution", 512))

    target_ids    = target_ids.to(device)
    condition_ids = condition_ids.to(device)
    bsz = target_latents.shape[0]

    target_ids    = target_ids.expand(bsz, -1, -1)
    condition_ids = condition_ids.expand(bsz, -1, -1)
    combined_ids  = torch.cat([target_ids, condition_ids], dim=1)

    # ---- Sample timestep and add noise ----
    u = compute_density_for_timestep_sampling("none", batch_size=bsz)
    indices = (
        u * (args.max_train_timesteps - args.min_train_timesteps)
        + args.min_train_timesteps
    ).long().clamp(0, len(noise_scheduler.timesteps) - 1)
    timesteps = noise_scheduler.timesteps[indices].to(device)
    sigmas = get_sigmas(
        noise_scheduler, timesteps, device,
        n_dim=target_latents.ndim, dtype=target_latents.dtype,
    )
    noise = torch.randn_like(target_latents)
    noisy_model_input = (1.0 - sigmas) * target_latents + sigmas * noise

    orig_shape = noisy_model_input.shape
    latent_model_input = torch.cat(
        [noisy_model_input, condition_latents], dim=1)

    prompt_embeds, text_ids = compute_text_embeddings(prompts, pipe.tokenizer, pipe.text_encoder)

    if suppress_texts is None:
        mask_ctx = exlora_token_mask_ctx(transformer, None)
    else:
        # ---- Build (bsz, max_seq_len) token mask for suppressed edits ----
        max_seq_len = prompt_embeds.shape[1]
        tokens_mask = torch.zeros((bsz, max_seq_len), device=device, dtype=torch.bool)
        for i in range(bsz):
            single_mask = build_token_mask(
                prompts[i], suppress_texts[i], pipe.tokenizer, device,
                max_seq_len=max_seq_len,
            )
            tokens_mask[i] = single_mask[0]
        # The mask stays set after the forward: the gradient-checkpointing recompute needs it.
        mask_ctx = exlora_token_mask_ctx(transformer, tokens_mask, disable_mask_after=False)

    with mask_ctx:
        model_pred = transformer(
            hidden_states=latent_model_input,
            timestep=timesteps / 1000,
            guidance=None,  # Klein: no guidance embeds
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=combined_ids,
            return_dict=False,
        )[0][:, :orig_shape[1]]

    flow_target = noise - target_latents
    weighting = compute_loss_weighting_for_sd3("none", sigmas=sigmas)
    loss = torch.mean(
        (weighting.float()
         * (model_pred.float() - flow_target.float()) ** 2
         ).reshape(bsz, -1),
        dim=1,
    ).mean()
    if extra_loss_fn is not None:
        loss = loss + extra_loss_fn()

    accelerator.backward(loss)
    return loss.detach().item()


@torch.no_grad()
def log_analora_validation(pipeline, train_dataset, tokenizer, text_encoder,
                           args, accelerator, global_step):
    """Run analogy inference with LoRA scale=1 and log a comparison strip to wandb."""
    print(f"Running analogy validation at step {global_step}...")

    pipe = pipeline
    device = accelerator.device
    dtype = torch.bfloat16

    # Pick a fresh random sample from the multi-edit dataset.
    val_meta = train_dataset.sample_val_pair()
    combined_prompt = val_meta['prompt']
    edit_name       = val_meta['edit_name']

    val_sample = {
        'A':         val_meta['a'],
        'A_prime':   val_meta['a_prime'],
        'B':         val_meta['b'],
        'B_prime':   val_meta['b_prime'],
        'edit_name': edit_name,
        'prompt':    combined_prompt,
    }

    # Encode images through VAE
    lat_a = encode_single_image(pipe, val_sample['A'])
    lat_a_prime = encode_single_image(pipe, val_sample['A_prime'])
    lat_b = encode_single_image(pipe, val_sample['B'])

    # Text embeddings
    prompt_embeds, text_ids = compute_text_embeddings(
        [val_sample['prompt']], tokenizer, text_encoder
    )
    # Klein: no guidance embeds, pass guidance=None
    guidance = None

    def decode_latents(packed, h, w):
        """Decode packed latents (1, N, C) → PIL image, given patchified spatial dims h, w."""
        unpacked = packed.permute(0, 2, 1).reshape(1, -1, h, w)
        bn_mean = pipe.vae.bn.running_mean.view(
            1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(
            1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(unpacked.device, unpacked.dtype)
        unpacked = unpacked * bn_std + bn_mean
        unpacked = Flux2KleinPipeline._unpatchify_latents(unpacked)
        decoded = pipe.vae.decode(unpacked, return_dict=False)[0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    pipe.transformer.set_adapters([ANALORA_ADAPTER_NAME], [1])
    num_inference_steps = args.num_val_inference_steps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

    # Denoise B' from noise
    patch_h, patch_w = lat_a.shape[2], lat_a.shape[3]

    cond_a = Flux2KleinPipeline._pack_latents(lat_a)
    cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime)
    cond_b = Flux2KleinPipeline._pack_latents(lat_b)
    condition_latents = torch.cat([cond_a, cond_ap, cond_b], dim=1)

    gen_latents = torch.randn(lat_a.shape, device=device, dtype=dtype)
    gen_latents = Flux2KleinPipeline._pack_latents(gen_latents)

    target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_a).to(device)
    condition_ids = Flux2KleinPipeline._prepare_image_ids(
        [lat_a, lat_a_prime, lat_b]).to(device)
    combined_ids = torch.cat([target_ids, condition_ids], dim=1)

    image_seq_len = gen_latents.shape[1]
    mu = compute_empirical_mu(image_seq_len, num_inference_steps)
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)

    for t in timesteps:
        latent_model_input = torch.cat(
            [gen_latents, condition_latents], dim=1)
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=t.expand(1).to(dtype) / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=combined_ids,
            return_dict=False,
        )[0][:, :gen_latents.shape[1]]
        gen_latents = pipe.scheduler.step(
            noise_pred, t, gen_latents, return_dict=False)[0]

    pred_b_prime = decode_latents(gen_latents, patch_h, patch_w)

    # Condition images from val_sample (shown once, not repeated)
    cell_w, cell_h = _native_wh() or (512, 512)
    a_img = val_sample['A'].resize((cell_w, cell_h), Image.LANCZOS)
    a_prime_img = val_sample['A_prime'].resize(
        (cell_w, cell_h), Image.LANCZOS)
    b_img = val_sample['B'].resize((cell_w, cell_h), Image.LANCZOS)
    b_prime_gt_img = val_sample['B_prime'].resize(
        (cell_w, cell_h), Image.LANCZOS)

    # Log to wandb: individual images + full strip
    if accelerator.is_main_process:
        vis_dir = Path(args.output_dir) / "validation_images"
        vis_dir.mkdir(parents=True, exist_ok=True)

        all_imgs = [a_img, a_prime_img, b_img, b_prime_gt_img, pred_b_prime]
        labels = ["A", "A'", "B", "B' GT", "B' pred"]

        strip_w = cell_w * len(all_imgs)
        strip_h = cell_h + 30
        strip = Image.new("RGB", (strip_w, strip_h), (255, 255, 255))
        draw = ImageDraw.Draw(strip)
        for i, (img, label) in enumerate(zip(all_imgs, labels)):
            strip.paste(img.resize((cell_w, cell_h)), (i * cell_w, 30))
            draw.text((i * cell_w + 4, 4), label, fill=(0, 0, 0))

        strip_path = vis_dir / f"step_{global_step:06d}.png"
        strip.save(strip_path)
        print(f"Saved validation strip to {strip_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()



@torch.no_grad()
def log_exlora_validation(pipeline, train_dataset, tokenizer, text_encoder,
                           args, accelerator, global_step):
    """Run analogy inference with ExLoRA and log a comparison strip.

    Layout:
      - ONE shared header row at the very top:  A, A', B, B' GT(all_edits)
      - Then one row PER kept_set combination (2^n total, n = num edits
        in the full demo), each row showing:
            supp=[...],
            B' GT(keep=[...], supp=[...]),
            B' (AnaLoRA total prompt),
            B' (AnaLoRA partial prompt),
            B' (ExLoRA)

    The "all suppressed -> expect B" sanity-check column has been removed
    entirely (not computed, not displayed).
    """
    print(f"Running ExLoRA analogy validation at step {global_step}...")

    pipe   = pipeline
    device = accelerator.device
    dtype  = torch.bfloat16

    # ------------------------------------------------------------------
    # 1) Sample one (item_a, item_b) pair using the dataset's own sampler.
    # ------------------------------------------------------------------
    cat, setting, item_a, item_b = train_dataset._sample_two_items()
    edit_name = f"{cat}/{setting}"

    a_edits    = item_a['edits']
    a_combined = item_a['combined']
    b_combined = item_b['combined']

    common_sets = set(a_combined) & set(b_combined)
    if not common_sets:
        print(f"  No common combined sets for pair in {edit_name}; skipping validation.")
        return

    # ------------------------------------------------------------------
    # 2) Fix demo_set = the largest common set (the full n-edit demo).
    # ------------------------------------------------------------------
    demo_set = max(common_sets, key=lambda s: (len(s), tuple(sorted(s))))
    demo_indices = sorted(demo_set)
    n_edits = len(demo_indices)

    pair_a_edits = [a_edits[i] for i in demo_indices if i in a_edits]
    if not pair_a_edits or len(pair_a_edits) != n_edits:
        print(f"  Missing edit text for demo_set {demo_set} in {edit_name}; skipping validation.")
        return
    pair_b_edits = pair_a_edits  # same convention as before (A and B share edit texts)

    # Map demo-set edit index (1-based, original numbering) -> 0-based
    # position within pair_a_edits.
    demo_idx_to_pos = {idx: pos for pos, idx in enumerate(demo_indices)}

    a_demo_ref = a_combined[demo_set]
    b_demo_ref = b_combined[demo_set]

    # ------------------------------------------------------------------
    # 3) Enumerate ALL 2^n subsets of demo_indices as kept_set candidates.
    #    Ordered full set first, empty set last.
    # ------------------------------------------------------------------
    all_kept_sets = []
    for r in range(n_edits, -1, -1):
        for combo in itertools.combinations(demo_indices, r):
            all_kept_sets.append(frozenset(combo))
    # all_kept_sets now has exactly 2^n entries.

    prompt = build_analogy_prompt(pair_a_edits, pair_b_edits)

    val_sample = {
        'A':       subject_image(train_dataset.subjects, item_a['input']),
        'A_prime': subject_image(train_dataset.subjects, a_demo_ref),
        'B':       subject_image(train_dataset.subjects, item_b['input']),
        'B_total': subject_image(train_dataset.subjects, b_demo_ref),  # B' GT (all edits)
        'edit_name': edit_name,
        'prompt':    prompt,
    }

    resolution = args.get("resolution", 512)

    # Encode images through VAE (shared across all kept_set combinations).
    lat_a       = encode_single_image(pipe, val_sample['A'],       resolution)
    lat_a_prime = encode_single_image(pipe, val_sample['A_prime'], resolution)
    lat_b       = encode_single_image(pipe, val_sample['B'],       resolution)

    # Text embeddings for the full prompt (shared across all combinations).
    prompt_embeds, text_ids = compute_text_embeddings(
        [prompt], tokenizer, text_encoder
    )
    guidance = None  # Klein: no guidance embeds

    patch_h, patch_w = lat_a.shape[2], lat_a.shape[3]

    def decode_latents(packed, h, w):
        unpacked = packed.permute(0, 2, 1).reshape(1, -1, h, w)
        bn_mean = pipe.vae.bn.running_mean.view(
            1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(
            1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(unpacked.device, unpacked.dtype)
        unpacked = unpacked * bn_std + bn_mean
        unpacked = Flux2KleinPipeline._unpatchify_latents(unpacked)
        decoded = pipe.vae.decode(unpacked, return_dict=False)[0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    def run_denoising(exlora_scale=1.0, suppress_indices=None,
                      prompt_embeds_override=None, text_ids_override=None,
                      mask_prompt_override=None):
        """Run denoising loop with ExLoRA at the given uniform scale.

        suppress_indices: list of 0-based positions (into pair_a_edits) to
            suppress via the token mask. ``None`` -> no suppression.
        When exlora_scale == 0 the mask is unused (ExLoRA is fully disabled).
        """
        emb     = prompt_embeds_override if prompt_embeds_override is not None else prompt_embeds
        tids    = text_ids_override      if text_ids_override      is not None else text_ids
        raw_pt  = mask_prompt_override   if mask_prompt_override   is not None else prompt

        if exlora_scale != 0.0:
            indices = suppress_indices if suppress_indices is not None else []
            texts_to_mask = [
                pair_a_edits[i]
                for i in indices
                if i < len(pair_a_edits) and pair_a_edits[i]
            ]
            max_seq_len = emb.shape[1]
            exlora_mask = build_token_mask(
                raw_pt, texts_to_mask, tokenizer, device,
                max_seq_len=max_seq_len,
            )  # (1, max_seq_len)
        else:
            exlora_mask = None

        for m in pipe.transformer.modules():
            if isinstance(m, ExLoRALinear):
                m.set_scaling(exlora_scale)

        cond_a  = Flux2KleinPipeline._pack_latents(lat_a)
        cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime)
        cond_b  = Flux2KleinPipeline._pack_latents(lat_b)
        condition_latents = torch.cat([cond_a, cond_ap, cond_b], dim=1)

        gen_latents = torch.randn(lat_a.shape, device=device, dtype=dtype)
        gen_latents = Flux2KleinPipeline._pack_latents(gen_latents)

        target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_a).to(device)
        condition_ids = Flux2KleinPipeline._prepare_image_ids(
            [lat_a, lat_a_prime, lat_b]).to(device)
        combined_ids = torch.cat([target_ids, condition_ids], dim=1)

        image_seq_len = gen_latents.shape[1]
        num_steps = args.num_val_inference_steps
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        mu = compute_empirical_mu(image_seq_len, num_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler, num_steps, device, sigmas=sigmas, mu=mu)

        with exlora_token_mask_ctx(pipe.transformer, exlora_mask, disable_mask_after=False):
            for t in timesteps:
                latent_model_input = torch.cat(
                    [gen_latents, condition_latents], dim=1)
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=t.expand(1).to(dtype) / 1000,
                    guidance=guidance,
                    encoder_hidden_states=emb,
                    txt_ids=tids,
                    img_ids=combined_ids,
                    return_dict=False,
                )[0][:, :gen_latents.shape[1]]
                gen_latents = pipe.scheduler.step(
                    noise_pred, t, gen_latents, return_dict=False)[0]

        for m in pipe.transformer.modules():
            if isinstance(m, ExLoRALinear):
                m.reset_scaling()

        return decode_latents(gen_latents, patch_h, patch_w)

    # ------------------------------------------------------------------
    # Shared header-row images (computed once, shown only at the very top).
    # ------------------------------------------------------------------
    target_w, target_h = _native_wh() or (512, 512)

    def resize_img(img):
        return resize_to_match(img, target_w, target_h)

    a_img_full         = resize_img(val_sample['A'])
    a_prime_img_full   = resize_img(val_sample['A_prime'])
    b_img_full         = resize_img(val_sample['B'])
    b_gt_all_edits_img = resize_img(val_sample['B_total'])

    cell_w, cell_h = target_w, target_h

    def load_combined(combined_map, fset):
        """Return PIL image for combined_map[fset] if present, else None."""
        ref = combined_map.get(fset)
        if ref is None:
            return None
        return subject_image(train_dataset.subjects, ref)

    # ------------------------------------------------------------------
    # Pre-compute per-kept_set metadata (GT image, suppressed prompt, etc.)
    # ------------------------------------------------------------------
    row_groups = []

    for kept_set in all_kept_sets:
        suppress_set = demo_set - kept_set
        kept_indices_sorted    = sorted(kept_set)
        suppress_indices_sorted = sorted(suppress_set)

        suppress_pos_0b = [demo_idx_to_pos[i] for i in suppress_indices_sorted]
        kept_pos_1b     = [demo_idx_to_pos[i] + 1 for i in kept_indices_sorted]
        suppress_pos_1b  = [demo_idx_to_pos[i] + 1 for i in suppress_indices_sorted]

        b_prime_type = f"keep={kept_pos_1b}/supp={suppress_pos_1b}"

        # B' ground truth for this kept_set.
        if kept_set == demo_set:
            b_prime_gt_img = subject_image(train_dataset.subjects, b_demo_ref)
        elif not kept_set:
            b_kept_img = load_combined(b_combined, kept_set)
            b_prime_gt_img = b_kept_img if b_kept_img is not None else val_sample['B']
        else:
            b_kept_img = load_combined(b_combined, kept_set)
            b_prime_gt_img = b_kept_img if b_kept_img is not None else val_sample['B_total']

        suppress_edit_texts = [pair_a_edits[p] for p in suppress_pos_0b if pair_a_edits[p]]
        suppressed_prompt = build_analogy_prompt(
            pair_a_edits, pair_b_edits, suppress_indices=suppress_pos_0b,
        )

        row_groups.append({
            'kept_set':            kept_set,
            'suppress_set':        suppress_set,
            'kept_pos_1b':         kept_pos_1b,
            'suppress_pos_1b':     suppress_pos_1b,
            'b_prime_type':        b_prime_type,
            'b_prime_gt_img':      b_prime_gt_img,
            'suppress_pos_0b':     suppress_pos_0b,
            'suppress_edit_texts': suppress_edit_texts,
            'suppressed_prompt':   suppressed_prompt,
        })

    # ------------------------------------------------------------------
    # Run inference for every row-group's 3 prediction columns:
    #   1. AnaLoRA total prompt    — full prompt, ExLoRA off
    #   2. AnaLoRA partial prompt  — suppressed edit(s) stripped, ExLoRA off
    #   3. ExLoRA                    — full prompt + token mask for this kept_set
    # Column 1 doesn't depend on kept_set, so it's computed once and reused.
    # (The previous "all suppressed -> expect B" column has been removed.)
    # ------------------------------------------------------------------
    pred_base_total_shared = run_denoising(exlora_scale=0.0)

    for rg in row_groups:
        suppressed_prompt = rg['suppressed_prompt']
        suppress_pos_0b   = rg['suppress_pos_0b']

        if not suppress_pos_0b:
            # Nothing suppressed: base-partial == base-total, ExLoRA == no-mask.
            pred_base_partial = pred_base_total_shared
            pred_with_lora    = run_denoising(exlora_scale=1.0, suppress_indices=[])
        else:
            suppressed_prompt_embeds, suppressed_text_ids = compute_text_embeddings(
                [suppressed_prompt], tokenizer, text_encoder
            )
            pred_base_partial = run_denoising(
                exlora_scale=0.0,
                prompt_embeds_override=suppressed_prompt_embeds,
                text_ids_override=suppressed_text_ids,
                mask_prompt_override=suppressed_prompt,
            )
            pred_with_lora = run_denoising(
                exlora_scale=1.0, suppress_indices=suppress_pos_0b)

        rg['pred_base_total']   = pred_base_total_shared
        rg['pred_base_partial'] = pred_base_partial
        rg['pred_with_lora']    = pred_with_lora

    # ------------------------------------------------------------------
    # Render: ONE shared header row, then one row per kept_set combination.
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        vis_dir = Path(args.output_dir) / "validation_images"
        vis_dir.mkdir(parents=True, exist_ok=True)

        font_size        = 20
        label_font_size  = 22
        font       = _font(font_size)
        label_font = _font(label_font_size)

        n_cols = 4
        supp_col_w = 240
        strip_w = supp_col_w + cell_w * n_cols
        label_height = 28

        approx_char_w = (font_size * 0.55) if font else 6
        chars_per_line = max(60, int((strip_w - 8) / approx_char_w))

        def wrap(text):
            return "\n".join(textwrap.wrap(text, chars_per_line))

        row_block_h = label_height + cell_h

        # Top prompt text, shown once.
        top_prompt_lines = wrap(f"Full prompt ({n_edits} edits): {prompt}")
        line_h = font_size + 4
        top_text_h = (top_prompt_lines.count("\n") + 1) * line_h + 16

        # Total height: top text + 1 shared header row + 2^n per-combination rows.
        strip_h = top_text_h + row_block_h + len(row_groups) * row_block_h
        strip   = Image.new("RGB", (strip_w, strip_h), (255, 255, 255))
        draw    = ImageDraw.Draw(strip)

        draw.multiline_text(
            (8, 8), top_prompt_lines, fill=(0, 0, 160),
            font=font if font else None, spacing=4,
        )

        def _paste_row(row_imgs, row_labels, y0, supp_text=""):
            # Left suppression-text column.
            if supp_text:
                supp_lines = textwrap.wrap(str(supp_text), width=22)
                draw.multiline_text(
                    (10, y0 + 4),
                    "\n".join(supp_lines),
                    fill=(0, 0, 0),
                    font=label_font if label_font else None,
                    spacing=2,
                )

            # Optional separator to make the text column visually distinct.
            sep_x = supp_col_w - 1
            draw.line((sep_x, y0, sep_x, y0 + row_block_h), fill=(180, 180, 180), width=2)

            # Four image columns to the right.
            for i, (img, label) in enumerate(zip(row_imgs, row_labels)):
                cell_img = resize_img(img)
                cw, ch = cell_img.size
                x0 = supp_col_w + i * cell_w
                x_off = x0 + (cell_w - cw) // 2
                y_off = y0 + label_height + (cell_h - ch) // 2
                strip.paste(cell_img, (x_off, y_off))
                draw.text(
                    (x0 + 6, y0 + 4),
                    label,
                    fill=(0, 0, 0),
                    font=label_font if label_font else None,
                )

        y_cursor = top_text_h

        # ---- Shared header row (shown ONCE) ----
        header_row_imgs = [
            a_img_full, a_prime_img_full, b_img_full, b_gt_all_edits_img,
        ]
        header_row_labels = ["A", "A'", "B", "B' GT\n(all edits)"]
        _paste_row(header_row_imgs, header_row_labels, y0=y_cursor, supp_text="")
        y_cursor += row_block_h

        # ---- One row per kept_set combination ----
        for rg in row_groups:
            b_prime_type = rg['b_prime_type']

            row_imgs = [
                rg['b_prime_gt_img'],
                rg['pred_base_total'],
                rg['pred_base_partial'],
                rg['pred_with_lora'],
            ]
            row_labels = [
                f"B' GT\n({b_prime_type})",
                "B' (AnaLoRA\ntotal prompt)",
                "B' (AnaLoRA\npartial prompt)",
                "B' (ExLoRA)",
            ]
            supp_text = f"supp={rg['suppress_pos_1b']}"
            _paste_row(row_imgs, row_labels, y0=y_cursor, supp_text=supp_text)
            y_cursor += row_block_h

        strip_path = vis_dir / f"step_{global_step:06d}.png"
        strip.save(strip_path)
        print(f"Saved validation strip ({len(row_groups)} combinations, "
              f"{n_edits} edits) to {strip_path}")

        # Save prompts/metadata to text file alongside the strip.
        prompt_path = vis_dir / f"step_{global_step:06d}_prompt.txt"
        lines = [f"edit: {edit_name}", "", f"full prompt:\n{prompt}", ""]
        for rg in row_groups:
            lines.append(
                f"keep={rg['kept_pos_1b']} supp={rg['suppress_pos_1b']} "
                f"-> partial prompt: {rg['suppressed_prompt']}"
            )
        prompt_path.write_text("\n".join(lines) + "\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()



# ---------------------------------------------------------------------------
# Concept suppression vector (the encoder-specific part; see sia.concept_suppression)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_contrast_pairs(edit: str, pipe, n_pairs: int, device,
                            max_new_tokens_per_pair: int = 90) -> Tuple[List[dict], int]:
    """Ask the Qwen3 text encoder (a ``Qwen3ForCausalLM``) for ``n_pairs`` contrastive pos/neg
    sentences about ``edit``. Sampling is not seeded.

    Returns ``(records, n_requested)``; ``records`` may be shorter than requested (the caller
    pads with ``sia.concept_suppression._fallback_pairs``).
    """
    text_encoder, tokenizer = pipe.text_encoder, pipe.tokenizer
    user_prompt = _build_generation_prompt(edit, n_pairs)
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    was_training = text_encoder.training
    text_encoder.eval()
    try:
        gen_out = text_encoder.generate(
            **inputs,
            max_new_tokens=max(256, n_pairs * max_new_tokens_per_pair),
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    finally:
        if was_training:
            text_encoder.train()

    # Strip the prompt prefix so we only decode the model's reply.
    generated = gen_out[0, inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(generated, skip_special_tokens=True)

    records = _parse_jsonl_records(reply)
    return records, n_pairs


@torch.no_grad()
def _encode_prompt(text: str, text_encoder, tokenizer, device,
                   max_sequence_length: int = 512) -> torch.Tensor:
    """Run a single prompt through the same encoder path as diffusion inference."""
    prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=[text],
        device=device,
        max_sequence_length=max_sequence_length,
    )
    return prompt_embeds  # (1, seq_len, hidden_dim)


def _style_token_indices(
    prompt: str,
    substr: str,
    tokenizer,
    max_length: int = 512,
) -> List[int]:
    """Token indices of a contrast pair's style identifier ``substr`` in the chat-templated
    ``prompt``, or ``[]`` if it is missing (the pair is then skipped).

    Unlike ``templated_substring_indices`` this falls back to a case-insensitive match: the pair
    parser checks identifier containment with ``.lower()``, so "Bright" in a sentence with the
    identifier "bright" is a valid pair.
    """
    try:
        return klein_find_substring_token_indices(
            prompt, substr, tokenizer,
            max_length=max_length,
        )
    except AssertionError:
        pass

    # Case-insensitive fallback: re-locate the char span, then reuse the
    # offset-mapping logic the canonical helper would have applied.
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    char_start = templated.lower().find(substr.lower())
    if char_start < 0:
        return []
    char_end = char_start + len(substr)
    enc = tokenizer(
        templated,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"][0].tolist()
    out = []
    for i, (cs, ce) in enumerate(offsets):
        if cs == ce == 0 and i > 0:
            continue
        if ce > char_start and cs < char_end:
            out.append(i)
    return out


@torch.no_grad()
def steering_pool(sentence: str, style: str, pipe, device, vl_images=None,
                  max_sequence_length: int = 512) -> Optional[torch.Tensor]:
    """Encode ``sentence`` as at inference and mean-pool ``prompt_embeds`` over the tokens of
    ``style``. Returns a float32 CPU vector, or ``None`` if ``style`` cannot be located.
    ``vl_images`` is unused: the Klein text encoder sees no images."""
    text_encoder, tokenizer = pipe.text_encoder, pipe.tokenizer
    positions = _style_token_indices(
        sentence, style, tokenizer, max_length=max_sequence_length)
    if not positions:
        return None
    prompt_embeds = _encode_prompt(
        sentence, text_encoder, tokenizer, device, max_sequence_length)
    # (1, S, D) → (D,) via mean over style positions
    pooled = prompt_embeds[0, positions, :].float().mean(dim=0).cpu()
    return pooled


def cache_key(edit: str, prompt: Optional[str] = None, vl_images=None) -> str:
    """Cache folder name of a CSV direction: ``sha1(edit)``, or ``sha1(prompt + "\\0" + edit)`` in
    in-context mode, first 12 hex digits. Must never change (cached pairs were sampled unseeded).
    ``vl_images`` is unused: the Klein direction does not depend on the images."""
    key = f"{prompt}\x00{edit}" if prompt else edit
    sha = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return sha
