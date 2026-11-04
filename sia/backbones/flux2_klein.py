"""FLUX.2-Klein backbone: model loading, prompt tokens, latents, timestep shift and the denoising loop.

Klein encodes the chat-templated prompt with its Qwen3 text encoder, so every token index (ExLoRA mask,
CSV positions) is computed against that templated text. A, A' and B are VAE-encoded, packed and appended
to the target sequence as reference images (concat mode); the transformer predicts the velocity of B'.
"""
import os

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_timesteps
from huggingface_hub.constants import HF_HUB_OFFLINE
from PIL import Image
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

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
    lora_mode_ctx,
)
from sia.data import resize_to_match
from sia.hf import resolve_checkpoint

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

# Training resolution (square side) of the released checkpoints; see configs/train_exlora_flux2_klein.yaml.
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
    """(w, h) from env SIA_NATIVE_RES ("1" -> 384x512, "WxH"), or None for the square 512 recipe."""
    v = os.environ.get("SIA_NATIVE_RES", "0")
    if v in ("", "0"):
        return None
    w, h = (384, 512) if v == "1" else map(int, v.lower().split("x"))
    return w // 16 * 16, h // 16 * 16


@torch.no_grad()
def encode_single_image(pipe, img, cell_size=512):
    """Resize, preprocess, and VAE-encode a single PIL image. Returns (1, C, h, w) latent.

    With SIA_NATIVE_RES set (1 = 384x512, or "WxH") every image is encoded at that fixed
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


@torch.no_grad()
def prepare_analora_latents(pipe, batch):
    """AnaLoRA-trainer variant of ``prepare_analogy_latents``: always encodes at the default
    512 cell size (or the SIA_NATIVE_RES size).

    Returns:
        target_latents, condition_latents, target_ids, condition_ids
    """
    target_arr, cond_arr = [], []

    for a, a_prime, b, b_prime in zip(batch['A'], batch['A_prime'], batch['B'], batch['B_prime']):
        lat_a = encode_single_image(pipe, a)
        lat_a_prime = encode_single_image(pipe, a_prime)
        lat_b = encode_single_image(pipe, b)
        lat_b_prime = encode_single_image(pipe, b_prime)

        # B' is target, A/A'/B are separate conditions
        target_arr.append(Flux2KleinPipeline._pack_latents(lat_b_prime))
        cond_a = Flux2KleinPipeline._pack_latents(lat_a).squeeze(0)
        cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime).squeeze(0)
        cond_b = Flux2KleinPipeline._pack_latents(lat_b).squeeze(0)
        cond_arr.append(
            torch.cat([cond_a, cond_ap, cond_b], dim=0).unsqueeze(0))

    target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_b_prime)
    condition_ids = Flux2KleinPipeline._prepare_image_ids(
        [lat_a, lat_a_prime, lat_b])

    return (
        torch.cat(target_arr, dim=0),
        torch.cat(cond_arr, dim=0),
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
