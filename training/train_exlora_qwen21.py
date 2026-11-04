"""
train_exlora_qwen21.py

ExLoRA (token-gated LoRA from the SIA paper) trainer for the Selective Image
Analogy project, built on Qwen-Image-2.1. Also holds the Qwen-Image-2.1
helpers (conditioning, prompt encoding, sampler) shared by the Stage 1
trainer and the inference script.

The method is the same as in train_exlora_flux2_klein.py; what changes is how
it maps onto the backbone:
  - A, A' and B are Qwen-Image-2.1 condition images (<image1..3>), so they are
    read by the Qwen3-VL encoder *and* enter the transformer as latent tokens,
    exactly as QwenImage21Pipeline does for multi-image editing.
  - The transformer is single-stream: there are no text-stream projections.
    ExLoRA therefore wraps the shared attention/MLP projections of every block
    and the token mask gates it to the suppressed edit tokens of the joint
    text/image sequence (sia/utils/qwen21.py).
  - Text and condition tokens are modulated from t=0 (causal_condition), so
    their activations — including the ExLoRA update — are step-independent
    and the prefix KV cache stays valid at inference.
  - Each sample goes through the transformer on its own. Samples differ in
    prompt length and image size; batching them would right-pad the prompt in
    the middle of the joint sequence and shift the target's RoPE positions
    away from what single-sample inference sees.
  - Noise levels are drawn with the sampler's resolution-dependent
    exponential shift, so training covers the sigmas the sampler visits.

Usage:
  python training/train_exlora_qwen21.py \
      --config training/configs/train_exlora_qwen21.yaml
"""

import contextlib
import itertools
import math
import os
import random
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

# Ensure the sia package (parent of this training/ dir) is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Importing the Klein trainer sets the wandb switch and the accelerate compatibility shim.
from training.train_exlora_flux2_klein import (  # noqa: E402
    _USE_WANDB,
    parse_args,
    restore_training_state,
)

if _USE_WANDB:
    import wandb

from accelerate import Accelerator  # noqa: E402
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, set_seed  # noqa: E402
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel  # noqa: E402
from diffusers.models.transformers.transformer_qwenimage21 import (  # noqa: E402
    _IMG_TOKENS_PER_SLOT,
    QwenImage21KVCache,
)
from diffusers.optimization import get_scheduler  # noqa: E402
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (  # noqa: E402
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling  # noqa: E402
from diffusers.utils.torch_utils import randn_tensor  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from sia.data import (  # noqa: E402
    DATASET,
    SelectiveAnalogyDataset,
    collate_selective,
    load_benchmark_cases,
    subject_image,
)
from sia.adapters import (  # noqa: E402
    collect_analora_params,
    collect_exlora_params,
    exlora_token_mask_ctx,
    inject_analora_modules,
    inject_exlora_modules,
    load_analora_from_peft_checkpoint,
    load_analora_state_dict,
    load_exlora_state_dict,
    save_analora_state_dict,
    save_exlora_state_dict,
    set_analora_requires_grad,
    set_exlora_requires_grad,
)
from sia.hf import resolve_checkpoint  # noqa: E402
from sia.lora import AnaLoRALinear  # noqa: E402
from sia.prompt import build_analogy_prompt  # noqa: E402
from sia.utils.qwen21 import (  # noqa: E402
    qwen21_joint_token_mask,
    qwen21_resize_dims,
    qwen21_text_token_mask,
)

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
# Strip rendering (validation)
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _font(size):
    for fp in _FONT_CANDIDATES:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return None


def render_rows(rows, cell=384, title=""):
    """``rows``: list of (row_label, [(image, caption), ...]). Images are
    fitted into square cells keeping their aspect ratio."""
    font = _font(16)
    label_w, cap_h = 200, 40
    n_cols = max(len(r[1]) for r in rows)
    title_lines = textwrap.wrap(title, 140) if title else []
    top = 8 + 20 * len(title_lines)
    canvas = Image.new("RGB", (label_w + n_cols * cell, top + len(rows) * (cell + cap_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for li, line in enumerate(title_lines):
        draw.text((8, 4 + 20 * li), line, fill=(0, 0, 160), font=font)
    for ri, (row_label, cells) in enumerate(rows):
        y = top + ri * (cell + cap_h)
        draw.multiline_text((6, y + 6), "\n".join(textwrap.wrap(row_label, 22)), fill=(0, 0, 0), font=font)
        for ci, (img, caption) in enumerate(cells):
            x = label_w + ci * cell
            if img is not None:
                thumb = img.copy()
                thumb.thumbnail((cell, cell), Image.LANCZOS)
                canvas.paste(thumb, (x + (cell - thumb.width) // 2, y + cap_h))
            draw.multiline_text((x + 4, y + 2), caption, fill=(0, 0, 0), font=font)
    return canvas


# ---------------------------------------------------------------------------
# Checkpoint hooks
# ---------------------------------------------------------------------------

def get_save_model_hook(accelerator):
    def hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                unwrapped = accelerator.unwrap_model(model)
                if isinstance(unwrapped, QwenImage21Transformer2DModel):
                    save_exlora_state_dict(unwrapped, os.path.join(output_dir, "selective_lora.pt"))
                    save_analora_state_dict(unwrapped, os.path.join(output_dir, "base_lora.pt"))
                    weights.pop()
    return hook


def get_load_model_hook(accelerator):
    def hook(models, input_dir):
        assert len(models) == 1, "Only transformer should be in the models list."
        transformer = models.pop()
        load_exlora_state_dict(transformer, os.path.join(input_dir, "selective_lora.pt"))
        analora_path = os.path.join(input_dir, "base_lora.pt")
        if os.path.exists(analora_path):
            load_analora_state_dict(transformer, analora_path, accelerator.device)
        cast_training_params([transformer], dtype=torch.float32)
    return hook


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_analogy_validation(pipe, train_dataset, args, accelerator, global_step):
    """Same layout as the Klein ExLoRA validation: a header row (A, A', B,
    B' GT all edits), then one row per kept subset of the full demo with
    B' GT | AnaLoRA total prompt | AnaLoRA partial prompt | ExLoRA."""
    print(f"Running ExLoRA analogy validation at step {global_step}...")
    device = accelerator.device
    resolution = args.get("resolution", 512)

    cat, setting, item_a, item_b = train_dataset._sample_two_items()
    common_sets = set(item_a['combined']) & set(item_b['combined'])
    if not common_sets:
        print(f"  No common combined sets for {cat}/{setting}; skipping validation.")
        return
    demo_set = max(common_sets, key=lambda s: (len(s), tuple(sorted(s))))
    demo_indices = sorted(demo_set)
    edits = [item_a['edits'][i] for i in demo_indices]
    prompt = build_analogy_prompt(edits, edits)

    a = subject_image(train_dataset.subjects, item_a['input'])
    a_prime = subject_image(train_dataset.subjects, item_a['combined'][demo_set])
    b = subject_image(train_dataset.subjects, item_b['input'])
    b_total = subject_image(train_dataset.subjects, item_b['combined'][demo_set])

    cond = prepare_conditions(pipe, [a, a_prime, b], resolution, device)
    prompt_embeds, image_pad_mask = encode_prompt(pipe, prompt, cond, device)

    def gen(embeds=prompt_embeds, pad_mask=image_pad_mask, exlora_mask=None):
        generator = torch.Generator(device=device).manual_seed(int(args.seed))
        return generate_qwen21(pipe, cond, embeds, pad_mask, args.num_val_inference_steps,
                               generator=generator, exlora_text_mask=exlora_mask,
                               use_kv_cache=args.get("use_kv_cache", True))

    pred_total = gen()
    rows = [("demo", [(a, "A"), (a_prime, "A'"), (b, "B"), (b_total, "B' GT\n(all edits)")])]
    for r in range(len(demo_indices), -1, -1):
        for kept in itertools.combinations(demo_indices, r):
            kept_set = frozenset(kept)
            supp_pos = [p for p, i in enumerate(demo_indices) if i not in kept_set]
            supp_texts = [edits[p] for p in supp_pos]
            gt_ref = item_b['combined'].get(kept_set)
            gt = subject_image(train_dataset.subjects, gt_ref) if gt_ref else (b if not kept_set else None)
            if supp_pos:
                partial_prompt = build_analogy_prompt(edits, edits, suppress_indices=supp_pos)
                partial_embeds, partial_pad = encode_prompt(pipe, partial_prompt, cond, device)
                pred_partial = gen(partial_embeds, partial_pad)
                mask = qwen21_text_token_mask(pipe, prompt, supp_texts, cond.image_sizes,
                                              prompt_embeds.shape[1], device)
                pred_exlora = gen(exlora_mask=mask)
            else:
                # Nothing suppressed: an all-zero mask leaves ExLoRA inert.
                pred_partial = pred_exlora = pred_total
            keep_1b = [p + 1 for p, i in enumerate(demo_indices) if i in kept_set]
            supp_1b = [p + 1 for p in supp_pos]
            rows.append((f"supp={supp_1b}", [
                (gt, f"B' GT\n(keep={keep_1b})"),
                (pred_total, "B' (AnaLoRA\ntotal prompt)"),
                (pred_partial, "B' (AnaLoRA\npartial prompt)"),
                (pred_exlora, "B' (ExLoRA)"),
            ]))

    if accelerator.is_main_process:
        vis_dir = Path(args.output_dir) / "validation_images"
        vis_dir.mkdir(parents=True, exist_ok=True)
        strip = render_rows(rows, title=f"{cat}/{setting} — full prompt: {prompt}")
        strip_path = vis_dir / f"step_{global_step:06d}.png"
        strip.save(strip_path)
        (vis_dir / f"step_{global_step:06d}_prompt.txt").write_text(
            f"edit: {cat}/{setting}\n\nfull prompt:\n{prompt}\n")
        print(f"Saved validation strip ({len(rows) - 1} combinations) to {strip_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def log_benchmark_visualization(pipe, cases, args, accelerator, global_step):
    """Unseen-data check (no ground truth): for every benchmark case, the full
    prompt with ExLoRA suppressing each subset of the edits in turn."""
    print(f"Running benchmark visualization at step {global_step}...")
    device = accelerator.device
    resolution = args.get("resolution", 512)
    out_dir = Path(args.output_dir) / "benchmark_images" / f"step_{global_step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        a, a_prime, b = harmonize_images_qwen21(case["a"], case["a_prime"], case["b"], case["b"])[:3]
        edits = case["edits"]
        prompt = build_analogy_prompt(edits, edits)
        cond = prepare_conditions(pipe, [a, a_prime, b], resolution, device)
        prompt_embeds, image_pad_mask = encode_prompt(pipe, prompt, cond, device)

        cells = []
        positions = range(len(edits))
        for r in range(len(edits), -1, -1):
            for kept in itertools.combinations(positions, r):
                supp_texts = [edits[p] for p in positions if p not in kept]
                mask = (qwen21_text_token_mask(pipe, prompt, supp_texts, cond.image_sizes,
                                               prompt_embeds.shape[1], device) if supp_texts else None)
                generator = torch.Generator(device=device).manual_seed(int(args.seed))
                pred = generate_qwen21(pipe, cond, prompt_embeds, image_pad_mask, args.num_val_inference_steps,
                                       generator=generator, exlora_text_mask=mask,
                                       use_kv_cache=args.get("use_kv_cache", True))
                cells.append((pred, f"keep={[p + 1 for p in kept]}"))

        rows = [("demo", [(a, "A"), (a_prime, "A'"), (b, "B")])]
        rows += [("ExLoRA" if i == 0 else "", cells[i:i + 4]) for i in range(0, len(cells), 4)]
        edits_txt = " ".join(f"{n + 1}: {e}." for n, e in enumerate(edits))
        seen = " (recipe seen in training, new subjects)" if case["recipe_in_train"] else ""
        title = f"{case['id']}{seen} @ step {global_step} — {edits_txt}"
        if accelerator.is_main_process:
            render_rows(rows, title=title).save(out_dir / f"{case['id'].replace('/', '_')}.png")

    print(f"Saved benchmark visualization ({len(cases)} cases) to {out_dir}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args(print_args=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
    )

    if accelerator.is_main_process and _USE_WANDB:
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if wandb_key:
            wandb.login(key=wandb_key)
        wandb.init(project=args.tracker_name, name=args.wandb_run_name,
                   config=OmegaConf.to_container(args, resolve=True))
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

    set_seed(args.seed)
    resolution = args.get("resolution", 512)

    # ---- Load models (frozen, bf16) ----
    pipe = load_pipeline_qwen21(args.pretrained_model_name_or_path)
    for m in [pipe.text_encoder, pipe.vae, pipe.transformer]:
        m.requires_grad_(False)
    pipe.to(accelerator.device)
    transformer = pipe.transformer

    # ---- AnaLoRA (Stage 1 weights) on every target projection ----
    base_replaced = inject_analora_modules(
        transformer, ANALORA_TARGET_MODULES, r=args.lora_rank, alpha=args.lora_rank, dropout=0.0)
    print(f"[AnaLoRA] Injected {len(base_replaced)} AnaLoRALinear modules")
    load_analora_from_peft_checkpoint(transformer, resolve_checkpoint(args.analora_path), device="cpu")

    # ---- ExLoRA around the same projections ----
    replaced = inject_exlora_modules(
        transformer, EXLORA_TARGET_MODULES, r=args.lora_rank, alpha=args.lora_rank, dropout=args.lora_dropout)
    print(f"[ExLoRA] Injected {len(replaced)} ExLoRALinear modules")

    if args.get("gradient_checkpointing", True):
        transformer.enable_gradient_checkpointing()
    accelerator.register_save_state_pre_hook(get_save_model_hook(accelerator))
    accelerator.register_load_state_pre_hook(get_load_model_hook(accelerator))
    cast_training_params([transformer], dtype=torch.float32)

    # ---- Snapshot AnaLoRA weights as L2 anchor (prevents drift) ----
    # The freshly injected LoRA layers still live on CPU until accelerator.prepare.
    analora_anchor = {}
    for mod_name, mod in transformer.named_modules():
        if isinstance(mod, AnaLoRALinear):
            analora_anchor[(mod_name, 'A')] = mod.lora_A.weight.detach().clone().to(accelerator.device)
            analora_anchor[(mod_name, 'B')] = mod.lora_B.weight.detach().clone().to(accelerator.device)
    analora_anchor_weight = args.get("analora_anchor_weight", 1e-4)
    print(f"[L2 Anchor] Saved {len(analora_anchor)} AnaLoRA param tensors, λ = {analora_anchor_weight}")

    optimizer = torch.optim.AdamW(
        [
            {"params": collect_exlora_params(transformer), "lr": args.learning_rate},
            {"params": collect_analora_params(transformer), "lr": args.get("analora_learning_rate", 5e-6)},
        ],
        betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8,
    )

    train_dataset = SelectiveAnalogyDataset(args.get("dataset", DATASET), args.get("dataset_revision"),
                                            harmonize_fn=harmonize_images_qwen21)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                  collate_fn=collate_selective, num_workers=2, shuffle=True)

    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler)
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    # Encoding and validation go through the pipeline; training forwards go
    # through the prepared (possibly DDP-wrapped) model.
    pipe.transformer = unwrapped_transformer

    init_ckpt = args.get("init_from_lora_ckpt", None)
    if init_ckpt and args.get("init_optimizer_from_ckpt", False):
        restore_training_state(optimizer, lr_scheduler, init_ckpt, accelerator.device)

    num_update_steps_per_epoch = math.ceil(
        math.ceil(len(train_dataset) / args.train_batch_size) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    print("***** Running ExLoRA training (Qwen-Image-2.1) *****")
    print(f"  Num examples             = {len(train_dataset)}")
    print(f"  Num epochs               = {num_train_epochs}")
    print(f"  Batch size per device    = {args.train_batch_size} (one transformer pass per sample)")
    print(f"  Gradient accumulation    = {args.gradient_accumulation_steps}")
    print(f"  Total optimisation steps = {args.max_train_steps}")
    print(f"  Resolution (area)        = {resolution}^2")

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        existing = os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else []
        dirs = sorted([d for d in existing if d.startswith("checkpoint")], key=lambda x: int(x.split("-")[1]))
        if dirs:
            print(f"Resuming from {dirs[-1]}")
            accelerator.load_state(os.path.join(args.output_dir, dirs[-1]))
            global_step = int(dirs[-1].split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            print(f"No checkpoint found in {args.output_dir}. Starting fresh.")

    benchmark_cases = []
    benchmark_steps = args.get("benchmark_steps", 0)
    if benchmark_steps and args.get("benchmark_ids"):
        benchmark_cases = load_benchmark_cases(list(args.benchmark_ids), args.get("dataset", DATASET),
                                               args.get("dataset_revision"))
        if global_step == 0:
            # ExLoRA is zero-initialised: this is the Stage 1 model, suppression off.
            transformer.eval()
            log_benchmark_visualization(pipe, benchmark_cases, args, accelerator, 0)

    progress_bar = tqdm(range(args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process)
    transformer.train()
    device = accelerator.device

    for epoch in range(first_epoch, num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                pair_a_edits = batch['pair_a_edits']
                bsz = len(pair_a_edits)

                # ---- Decide step type (same distribution as the Klein trainer) ----
                #   [0.00, 0.25) → AnaLoRA partial  (25 %)
                #   [0.25, 0.55) → AnaLoRA total    (30 %)
                #   [0.55, 0.90) → ExLoRA partial   (35 %)
                #   [0.90, 1.00) → ExLoRA all-supp  (10 %)
                _r = random.random()
                is_all_suppressed_step = _r >= 0.90
                is_analora_step = _r < 0.55
                use_total_change = 0.25 <= _r < 0.55
                is_analora_partial = is_analora_step and not use_total_change

                set_exlora_requires_grad(unwrapped_transformer, not is_analora_step)
                set_analora_requires_grad(unwrapped_transformer, is_analora_step)

                # ---- Choose A' and target per step type (see Klein trainer) ----
                a_primes, b_primes = batch['A_prime'], batch['B_prime']
                if use_total_change:
                    b_primes = batch['B_total']
                elif is_all_suppressed_step:
                    b_primes = batch['B']
                elif is_analora_partial:
                    a_primes = [ap if ap is not None else at
                                for ap, at in zip(batch['A_prime_partial'], batch['A_prime_total'])]

                def losses_fn(i):
                    images = [batch['A'][i], a_primes[i], batch['B'][i]]
                    pa = pair_a_edits[i]
                    if is_analora_step:
                        # No suppression mechanism: the suppressed edits are
                        # removed from the prompt text, ExLoRA stays off.
                        supp = None if use_total_change else batch['suppress_indices'][i]
                        prompt = build_analogy_prompt(pa, pa, suppress_indices=supp)
                        suppress_texts = None
                    else:
                        prompt = build_analogy_prompt(pa, pa)
                        idxs = (range(len(pair_a_edits[i])) if is_all_suppressed_step
                                else batch['suppress_indices'][i])
                        suppress_texts = [pair_a_edits[i][j] for j in idxs if pair_a_edits[i][j]]
                    return sample_flow_loss(pipe, transformer, images, b_primes[i], prompt,
                                            resolution, device, args, suppress_texts=suppress_texts)

                anchor_loss = None

                def anchor_fn():
                    nonlocal anchor_loss
                    anchor_loss = sum(
                        torch.sum((p.float() - analora_anchor[(n, ab)].float()) ** 2)
                        for n, mod in unwrapped_transformer.named_modules() if isinstance(mod, AnaLoRALinear)
                        for ab, p in (('A', mod.lora_A.weight), ('B', mod.lora_B.weight))
                    )
                    return analora_anchor_weight * anchor_loss

                use_anchor = is_analora_step and analora_anchor_weight > 0
                loss = backward_per_sample(accelerator, transformer, losses_fn, bsz,
                                           extra_loss_fn=anchor_fn if use_anchor else None)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and _USE_WANDB:
                    wandb.log({"train/loss": loss, "train/lr": lr_scheduler.get_last_lr()[0],
                               "train/epoch": epoch, "train/step": global_step})

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0 or global_step == args.max_train_steps:
                        transformer.eval()
                        log_analogy_validation(pipe, train_dataset, args, accelerator, global_step)
                        transformer.train()

                    if benchmark_cases and (global_step % benchmark_steps == 0
                                            or global_step == args.max_train_steps):
                        transformer.eval()
                        log_benchmark_visualization(pipe, benchmark_cases, args, accelerator, global_step)
                        transformer.train()

            if is_all_suppressed_step:
                step_type = "exlora_all_supp"
            elif use_total_change:
                step_type = "base_total"
            elif is_analora_step:
                step_type = "base_partial"
            else:
                step_type = "exlora"
            logs = {"loss": loss, "lr": lr_scheduler.get_last_lr()[0], "step_type": step_type}
            if anchor_loss is not None:
                logs["anchor_loss"] = (analora_anchor_weight * anchor_loss).item()
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    if accelerator.is_main_process and _USE_WANDB:
        wandb.finish()
    accelerator.end_training()


if __name__ == "__main__":
    main()
