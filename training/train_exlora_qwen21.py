"""
train_exlora_qwen21.py

ExLoRA (token-gated LoRA from the SIA paper) trainer for the Selective Image
Analogy project, built on Qwen-Image-2.1. The backbone code (conditioning,
prompt encoding, sampler, loss) lives in sia/backbones/qwen_image21.py.

The method is the same as in train_exlora_flux2_klein.py; what changes is how
it maps onto the backbone:
  - A, A' and B are Qwen-Image-2.1 condition images (<image1..3>), so they are
    read by the Qwen3-VL encoder *and* enter the transformer as latent tokens,
    exactly as QwenImage21Pipeline does for multi-image editing.
  - The transformer is single-stream: there are no text-stream projections.
    ExLoRA therefore wraps the shared attention/MLP projections of every block
    and the token mask gates it to the suppressed edit tokens of the joint
    text/image sequence (sia/backbones/qwen_image21.py).
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

import itertools
import math
import os
import random
import sys
import textwrap
from pathlib import Path

import torch
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
from diffusers import QwenImage21Transformer2DModel  # noqa: E402
from diffusers.optimization import get_scheduler  # noqa: E402
from diffusers.training_utils import cast_training_params  # noqa: E402
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
from sia.backbones.qwen_image21 import (  # noqa: E402
    ANALORA_TARGET_MODULES,
    EXLORA_TARGET_MODULES,
    backward_per_sample,
    encode_prompt,
    generate_qwen21,
    harmonize_images_qwen21,
    load_pipeline_qwen21,
    prepare_conditions,
    qwen21_text_token_mask,
    sample_flow_loss,
)

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
