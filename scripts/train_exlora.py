"""ExLoRA training (Stage 2 of SIA): the token-gated exclusion LoRA on top of a trained AnaLoRA.

The backbone (``backbone: flux2_klein | qwen_image21``, inferred from ``pretrained_model_name_or_path``
when absent) comes from the config. Each optimizer step draws one of four step types:

  [0.00, 0.25)  AnaLoRA partial  (25 %): suppressed edits removed from the prompt, GT = partial edit
  [0.25, 0.55)  AnaLoRA total    (30 %): every edit in the prompt, GT = all edits applied
  [0.55, 0.90)  ExLoRA partial   (35 %): full prompt, ExLoRA gated to the suppressed edit tokens,
                                         GT = partial edit (paper Sec. 4.3, eq. 14)
  [0.90, 1.00)  ExLoRA all-supp  (10 %): full prompt, every edit suppressed, GT = B

AnaLoRA steps train AnaLoRA only (with an L2 anchor to its Stage 1 weights); ExLoRA steps train
ExLoRA only.

On Qwen-Image-2.1, A, A' and B are condition images read by the Qwen3-VL encoder and enter the
single-stream transformer as latent tokens, so ExLoRA wraps the shared projections of every block
and its mask covers the suppressed edit tokens of the joint sequence. Each sample goes through the
transformer on its own (prompt lengths and image sizes differ). See ``sia/backbones/qwen_image21.py``.

Usage:
  accelerate launch --mixed_precision bf16 scripts/train_exlora.py --config configs/exlora_flux2_klein.yaml
"""
import copy
import math
import random

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params

from sia.adapters import (
    collect_analora_params,
    collect_exlora_params,
    inject_analora_modules,
    inject_exlora_modules,
    load_analora_from_peft_checkpoint,
    set_analora_requires_grad,
    set_exlora_requires_grad,
)
from sia.data import DATASET, SelectiveAnalogyDataset, collate_selective, load_benchmark_cases
from sia.hf import resolve_checkpoint
from sia.inference import backbone_name
from sia.prompt import build_analogy_prompt
from sia.training import (
    TrainingRun,
    analora_anchor_loss,
    due,
    end_run,
    get_exlora_load_hook,
    get_exlora_save_hook,
    load_peft_ckpt_into_exlora,
    log_metrics,
    parse_args,
    resume_from_latest,
    restore_training_state,
    save_checkpoint,
    snapshot_analora,
    start_run,
)


# ---------------------------------------------------------------------------
# Step types (shared by both backbones)
# ---------------------------------------------------------------------------

def draw_step_type():
    """One ``random.random()`` draw -> ``base_partial`` | ``base_total`` | ``exlora`` | ``exlora_all_supp``."""
    r = random.random()
    if r >= 0.90:
        return "exlora_all_supp"
    if r >= 0.55:
        return "exlora"
    if r >= 0.25:
        return "base_total"
    return "base_partial"


def select_images(batch, step_type):
    """The batch with the demonstration A' and the target B' of ``step_type``.

    AnaLoRA-partial swaps A' to the matching partial edit (same kept edits as B'), falling back to
    the full edit where that image is unavailable. The ExLoRA steps keep the full-edit A': the
    selective-suppression step trains the model to suppress edits that REMAIN visible in A'.
    """
    if step_type == "base_total":
        return {**batch, 'B_prime': batch['B_total']}
    if step_type == "exlora_all_supp":
        # All edits suppressed -> the model should reproduce B unchanged.
        return {**batch, 'B_prime': batch['B']}
    if step_type == "base_partial":
        a_prime_matched = [ap if ap is not None else at
                           for ap, at in zip(batch['A_prime_partial'], batch['A_prime_total'])]
        return {**batch, 'A_prime': a_prime_matched}
    return batch


def step_prompts(batch, step_type):
    """Per-sample prompts and ExLoRA suppression texts of ``step_type``.

    AnaLoRA has no suppression mechanism: on its steps the suppressed edits are removed from the
    prompt text and ExLoRA stays off (``suppress_texts`` is ``None``). ExLoRA steps keep the full
    prompt and gate ExLoRA to the tokens of the suppressed edits.
    """
    pair_a_edits, pair_b_edits = batch['pair_a_edits'], batch['pair_b_edits']
    if step_type == "base_total":
        return [build_analogy_prompt(ae, be) for ae, be in zip(pair_a_edits, pair_b_edits)], None
    if step_type == "base_partial":
        return [build_analogy_prompt(ae, be, suppress_indices=idxs)
                for ae, be, idxs in zip(pair_a_edits, pair_b_edits, batch['suppress_indices'])], None

    prompts = [build_analogy_prompt(ae, be) for ae, be in zip(pair_a_edits, pair_b_edits)]
    if step_type == "exlora_all_supp":
        suppress_per_sample = [list(range(len(ae))) for ae in pair_a_edits]
    else:
        suppress_per_sample = batch['suppress_indices']
    suppress_texts = [[ae[j] for j in idxs if j < len(ae) and ae[j]]
                      for ae, idxs in zip(pair_a_edits, suppress_per_sample)]
    return prompts, suppress_texts


# ---------------------------------------------------------------------------
# Backbone-specific setup
# ---------------------------------------------------------------------------

def inject_adapters(transformer, bb, args):
    """AnaLoRA on every target projection with the Stage 1 weights, then ExLoRA around the ExLoRA
    targets (wrapping the AnaLoRA layers where both apply)."""
    base_replaced = inject_analora_modules(
        transformer, bb.ANALORA_TARGET_MODULES, r=args.lora_rank, alpha=args.lora_rank, dropout=0.0)
    print(f"[AnaLoRA] Injected {len(base_replaced)} AnaLoRALinear modules")
    load_analora_from_peft_checkpoint(transformer, resolve_checkpoint(args.analora_path), device="cpu")

    replaced = inject_exlora_modules(
        transformer, bb.EXLORA_TARGET_MODULES, r=args.lora_rank, alpha=args.lora_rank, dropout=args.lora_dropout)
    print(f"[ExLoRA] Injected {len(replaced)} ExLoRALinear modules")


def build_optimizer(transformer, args):
    """Two param groups: ExLoRA at ``learning_rate``, AnaLoRA at ``analora_learning_rate``."""
    exlora_params = collect_exlora_params(transformer)
    analora_params = collect_analora_params(transformer)
    print(f"[Optimizer] ExLoRA params: {len(exlora_params)}, AnaLoRA params: {len(analora_params)}")
    return torch.optim.AdamW(
        [
            {"params": exlora_params, "lr": args.learning_rate},
            {"params": analora_params, "lr": args.get("analora_learning_rate", 5e-6)},
        ],
        betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8,
    )


def build_data(args, accelerator, optimizer, harmonize_fn=None):
    """ExLoRA training dataset, its shuffled DataLoader and the constant learning-rate scheduler."""
    train_dataset = SelectiveAnalogyDataset(args.get("dataset", DATASET), args.get("dataset_revision"),
                                            harmonize_fn=harmonize_fn)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                  collate_fn=collate_selective, num_workers=2, shuffle=True)
    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    return train_dataset, train_dataloader, lr_scheduler


def setup_flux2_klein(args, accelerator):
    """Load FLUX.2-Klein with AnaLoRA + ExLoRA for Stage 2 training; returns a ``TrainingRun``."""
    from diffusers import Flux2KleinPipeline
    from diffusers.models import Flux2Transformer2DModel

    from sia.backbones import flux2_klein as bb

    tokenizer, text_encoder, vae, transformer, noise_scheduler = bb.load_models(args)
    for m in [text_encoder, vae, transformer]:
        m.requires_grad_(False)
    for m in [text_encoder, vae, transformer]:
        m.to(accelerator.device, dtype=torch.bfloat16)

    inject_adapters(transformer, bb, args)
    # Optionally initialise ExLoRA from a PEFT checkpoint.
    init_ckpt = args.get("init_from_lora_ckpt", None)
    if init_ckpt:
        load_peft_ckpt_into_exlora(transformer, init_ckpt, accelerator.device)

    if args.get("gradient_checkpointing", True):
        transformer.enable_gradient_checkpointing()
    accelerator.register_save_state_pre_hook(get_exlora_save_hook(accelerator, Flux2Transformer2DModel))
    accelerator.register_load_state_pre_hook(get_exlora_load_hook(accelerator))
    cast_training_params([transformer], dtype=torch.float32)

    analora_anchor = snapshot_analora(transformer, accelerator.device)
    optimizer = build_optimizer(transformer, args)
    train_dataset, train_dataloader, lr_scheduler = build_data(args, accelerator, optimizer)

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler)
    if init_ckpt and args.get("init_optimizer_from_ckpt", False):
        restore_training_state(optimizer, lr_scheduler, init_ckpt, accelerator.device)

    # VAE / text encoding and validation go through the pipeline. keep_fp32_wrapper=False strips
    # accelerate's autocast wrapper from the model's forward, for the training forwards as well.
    pipeline = Flux2KleinPipeline(
        # Own copy: validation calls retrieve_timesteps() on pipe.scheduler, which would
        # otherwise shrink noise_scheduler.timesteps to the 28 inference steps and
        # collapse training timestep sampling onto the lowest-noise one.
        scheduler=copy.deepcopy(noise_scheduler),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(transformer, keep_fp32_wrapper=False),
    )

    def train_step(batch, prompts, suppress_texts, extra_loss_fn):
        return bb.train_step(accelerator, pipeline, transformer, noise_scheduler, batch, prompts, args,
                             suppress_texts=suppress_texts, extra_loss_fn=extra_loss_fn)

    def visualize(global_step):
        if due(global_step, args.validation_steps, args):
            pipeline.transformer = accelerator.unwrap_model(transformer, keep_fp32_wrapper=False)
            bb.log_exlora_validation(pipeline, train_dataset, tokenizer, text_encoder,
                                     args, accelerator, global_step)
            transformer.train()

    return TrainingRun(transformer, optimizer, lr_scheduler, train_dataloader, train_dataset,
                       train_step, visualize, unwrapped=accelerator.unwrap_model(transformer),
                       analora_anchor=analora_anchor)


def setup_qwen_image21(args, accelerator):
    """Load Qwen-Image-2.1 with AnaLoRA + ExLoRA for Stage 2 training; returns a ``TrainingRun``."""
    from diffusers import QwenImage21Transformer2DModel

    from sia.backbones import qwen_image21 as bb

    pipe = bb.load_pipeline_qwen21(args.pretrained_model_name_or_path)
    for m in [pipe.text_encoder, pipe.vae, pipe.transformer]:
        m.requires_grad_(False)
    pipe.to(accelerator.device)
    transformer = pipe.transformer

    inject_adapters(transformer, bb, args)

    if args.get("gradient_checkpointing", True):
        transformer.enable_gradient_checkpointing()
    accelerator.register_save_state_pre_hook(get_exlora_save_hook(accelerator, QwenImage21Transformer2DModel))
    accelerator.register_load_state_pre_hook(get_exlora_load_hook(accelerator))
    cast_training_params([transformer], dtype=torch.float32)

    # The freshly injected LoRA layers still live on CPU until accelerator.prepare.
    analora_anchor = snapshot_analora(transformer, accelerator.device)
    optimizer = build_optimizer(transformer, args)
    train_dataset, train_dataloader, lr_scheduler = build_data(
        args, accelerator, optimizer, harmonize_fn=bb.harmonize_images_qwen21)

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler)
    unwrapped = accelerator.unwrap_model(transformer)
    # Encoding and validation go through the pipeline; training forwards go
    # through the prepared (possibly DDP-wrapped) model.
    pipe.transformer = unwrapped

    init_ckpt = args.get("init_from_lora_ckpt", None)
    if init_ckpt and args.get("init_optimizer_from_ckpt", False):
        restore_training_state(optimizer, lr_scheduler, init_ckpt, accelerator.device)

    # Unseen-data check (no ground truth) on fixed benchmark cases, every benchmark_steps.
    benchmark_cases = []
    benchmark_steps = args.get("benchmark_steps", 0)
    if benchmark_steps and args.get("benchmark_ids"):
        benchmark_cases = load_benchmark_cases(list(args.benchmark_ids), args.get("dataset", DATASET),
                                               args.get("dataset_revision"))

    def train_step(batch, prompts, suppress_texts, extra_loss_fn):
        return bb.train_step(accelerator, pipe, transformer, batch, prompts, args,
                             suppress_texts=suppress_texts, extra_loss_fn=extra_loss_fn)

    def before_training(global_step):
        if benchmark_cases and global_step == 0:
            # ExLoRA is zero-initialised: this is the Stage 1 model, suppression off.
            transformer.eval()
            bb.log_benchmark_visualization(pipe, benchmark_cases, args, accelerator, 0)

    def visualize(global_step):
        if due(global_step, args.validation_steps, args):
            transformer.eval()
            bb.log_exlora_validation(pipe, train_dataset, args, accelerator, global_step)
            transformer.train()
        if benchmark_cases and due(global_step, benchmark_steps, args):
            transformer.eval()
            bb.log_benchmark_visualization(pipe, benchmark_cases, args, accelerator, global_step)
            transformer.train()

    return TrainingRun(transformer, optimizer, lr_scheduler, train_dataloader, train_dataset,
                       train_step, visualize, before_training=before_training, unwrapped=unwrapped,
                       analora_anchor=analora_anchor)


SETUP = {"flux2_klein": setup_flux2_klein, "qwen_image21": setup_qwen_image21}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    """Train ExLoRA (Stage 2) with the configuration given on the command line."""
    args = parse_args(print_args=True)
    backbone = backbone_name(args)
    accelerator = start_run(args)
    run = SETUP[backbone](args, accelerator)

    analora_anchor_weight = args.get("analora_anchor_weight", 1e-4)
    print(f"[L2 Anchor] Saved {len(run.analora_anchor)} AnaLoRA param tensors, λ = {analora_anchor_weight}")

    num_update_steps_per_epoch = math.ceil(
        math.ceil(len(run.train_dataset) / args.train_batch_size) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    print(f"***** Running ExLoRA training ({backbone}) *****")
    print(f"  Num examples             = {len(run.train_dataset)}")
    print(f"  Num epochs               = {num_train_epochs}")
    print(f"  Batch size per device    = {args.train_batch_size}")
    print(f"  Gradient accumulation    = {args.gradient_accumulation_steps}")
    print(f"  Total optimisation steps = {args.max_train_steps}")

    global_step, first_epoch = resume_from_latest(accelerator, args, num_update_steps_per_epoch)
    if run.before_training is not None:
        run.before_training(global_step)

    progress_bar = tqdm(range(args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process)
    transformer = run.transformer
    transformer.train()

    for epoch in range(first_epoch, num_train_epochs):
        for batch in run.train_dataloader:
            with accelerator.accumulate(transformer):
                step_type = draw_step_type()
                is_analora_step = step_type in ("base_partial", "base_total")
                set_exlora_requires_grad(run.unwrapped, not is_analora_step)
                set_analora_requires_grad(run.unwrapped, is_analora_step)

                prompts, suppress_texts = step_prompts(batch, step_type)

                anchor_loss = None

                def anchor_fn():
                    nonlocal anchor_loss
                    anchor_loss = analora_anchor_loss(run.unwrapped, run.analora_anchor)
                    return analora_anchor_weight * anchor_loss

                use_anchor = is_analora_step and analora_anchor_weight > 0
                loss = run.train_step(select_images(batch, step_type), prompts, suppress_texts,
                                      anchor_fn if use_anchor else None)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                run.optimizer.step()
                run.lr_scheduler.step()
                run.optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                log_metrics(accelerator, loss, run.lr_scheduler.get_last_lr()[0], epoch, global_step)
                if accelerator.is_main_process:
                    save_checkpoint(accelerator, args, global_step)
                    run.visualize(global_step)

            logs = {"loss": loss, "lr": run.lr_scheduler.get_last_lr()[0], "step_type": step_type}
            if anchor_loss is not None:
                logs["anchor_loss"] = (analora_anchor_weight * anchor_loss).detach().item()
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    end_run(accelerator)


if __name__ == "__main__":
    main()
