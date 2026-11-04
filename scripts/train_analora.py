"""AnaLoRA training (Stage 1 of SIA): a PEFT LoRA that learns the full analogy (A, A', B) -> B'.

The backbone (``backbone: flux2_klein | qwen_image21``, inferred from ``pretrained_model_name_or_path``
when absent) comes from the config. The analogy prompt lists the edits of the A -> A' demonstration;
the dataset drops it 15 % of the time. The checkpoint is a diffusers LoRA file
(``pytorch_lora_weights.safetensors``), so it also loads with the pipeline's ``load_lora_weights``;
Stage 2 (``scripts/train_exlora.py``) reads it through ``analora_path``.

On Qwen-Image-2.1 the three images are separate condition images and each sample goes through the
transformer on its own (see ``sia/backbones/qwen_image21.py``).

Usage:
  accelerate launch --mixed_precision bf16 scripts/train_analora.py --config configs/analora_flux2_klein.yaml
"""
import copy
import math

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from peft import LoraConfig

from sia.adapters import ANALORA_ADAPTER_NAME
from sia.data import DATASET, AnalogyPairDataset, collate_pairs
from sia.inference import backbone_name
from sia.training import (
    TrainingRun,
    due,
    end_run,
    get_analora_load_hook,
    get_analora_save_hook,
    log_metrics,
    parse_args,
    resume_from_latest,
    save_checkpoint,
    start_run,
)


# ---------------------------------------------------------------------------
# Backbone-specific setup
# ---------------------------------------------------------------------------

def add_lora(transformer, bb, args):
    """Add the trainable PEFT AnaLoRA (Gaussian init) to the backbone's AnaLoRA target modules."""
    transformer.add_adapter(
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=bb.ANALORA_TARGET_MODULES,
        ),
        adapter_name=ANALORA_ADAPTER_NAME,
    )


def build_training(transformer, args, accelerator, harmonize_fn=None):
    """Optimizer over the LoRA parameters, the pair dataset and the constant LR schedule, all
    passed through ``accelerator.prepare``. Returns the prepared objects and the dataset."""
    lora_parameters = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": lora_parameters, "lr": args.learning_rate}],
        betas=(0.9, 0.999), weight_decay=1e-04, eps=1e-08,
    )

    train_dataset = AnalogyPairDataset(args.get("dataset", DATASET), args.get("dataset_revision"),
                                       harmonize_fn=harmonize_fn)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                  collate_fn=collate_pairs, num_workers=2, shuffle=True)

    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler)
    return transformer, optimizer, train_dataloader, lr_scheduler, train_dataset


def setup_flux2_klein(args, accelerator):
    from diffusers import Flux2KleinPipeline
    from diffusers.models import Flux2Transformer2DModel

    from sia.backbones import flux2_klein as bb

    tokenizer, text_encoder, vae, transformer, noise_scheduler = bb.load_models(args)
    # Only train the LoRA layers
    for m in [text_encoder, vae, transformer]:
        m.requires_grad_(False)
    for m in [text_encoder, vae, transformer]:
        m.to(accelerator.device, dtype=torch.bfloat16)

    if args.get("gradient_checkpointing", True):
        transformer.enable_gradient_checkpointing()
    add_lora(transformer, bb, args)
    accelerator.register_save_state_pre_hook(
        get_analora_save_hook(accelerator, Flux2KleinPipeline, Flux2Transformer2DModel))
    accelerator.register_load_state_pre_hook(get_analora_load_hook(accelerator, Flux2Transformer2DModel))
    # Make sure the trainable params are in float32. [only upcast trainable parameters (LoRA) into fp32]
    cast_training_params([transformer], dtype=torch.float32)

    transformer, optimizer, train_dataloader, lr_scheduler, train_dataset = build_training(
        transformer, args, accelerator)

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

    def train_step(batch):
        # Validation also selects the adapter; select it again before every training forward.
        transformer.set_adapters([ANALORA_ADAPTER_NAME], [1])
        return bb.train_step(accelerator, pipeline, transformer, noise_scheduler, batch, batch["prompts"], args)

    def visualize(global_step):
        if due(global_step, args.validation_steps, args):
            pipeline.transformer = accelerator.unwrap_model(transformer, keep_fp32_wrapper=False)
            bb.log_analora_validation(pipeline, train_dataset, tokenizer, text_encoder,
                                      args, accelerator, global_step)
            transformer.train()

    return TrainingRun(transformer, optimizer, lr_scheduler, train_dataloader, train_dataset,
                       train_step, visualize)


def setup_qwen_image21(args, accelerator):
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

    from sia.backbones import qwen_image21 as bb

    args.resolution = args.get("resolution", 512)
    pipe = bb.load_pipeline_qwen21(args.pretrained_model_name_or_path)
    for m in [pipe.text_encoder, pipe.vae, pipe.transformer]:
        m.requires_grad_(False)
    pipe.to(accelerator.device)
    transformer = pipe.transformer

    if args.get("gradient_checkpointing", True):
        transformer.enable_gradient_checkpointing()
    add_lora(transformer, bb, args)
    accelerator.register_save_state_pre_hook(
        get_analora_save_hook(accelerator, QwenImage21Pipeline, QwenImage21Transformer2DModel))
    accelerator.register_load_state_pre_hook(get_analora_load_hook(accelerator, QwenImage21Transformer2DModel))
    cast_training_params([transformer], dtype=torch.float32)

    transformer, optimizer, train_dataloader, lr_scheduler, train_dataset = build_training(
        transformer, args, accelerator, harmonize_fn=bb.harmonize_images_qwen21)
    pipe.transformer = accelerator.unwrap_model(transformer)

    def train_step(batch):
        return bb.train_step(accelerator, pipe, transformer, batch, batch["prompts"], args)

    def visualize(global_step):
        if due(global_step, args.validation_steps, args):
            transformer.eval()
            bb.log_analora_validation(pipe, train_dataset, args, accelerator, global_step)
            transformer.train()

    return TrainingRun(transformer, optimizer, lr_scheduler, train_dataloader, train_dataset,
                       train_step, visualize)


SETUP = {"flux2_klein": setup_flux2_klein, "qwen_image21": setup_qwen_image21}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args(print_args=True)
    backbone = backbone_name(args)
    accelerator = start_run(args)
    run = SETUP[backbone](args, accelerator)

    num_update_steps_per_epoch = math.ceil(
        math.ceil(len(run.train_dataset) / args.train_batch_size) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    print(f"***** Running AnaLoRA training ({backbone}) *****")
    print(f"  Num examples             = {len(run.train_dataset)}")
    print(f"  Num epochs               = {num_train_epochs}")
    print(f"  Batch size per device    = {args.train_batch_size}")
    print(f"  Gradient accumulation    = {args.gradient_accumulation_steps}")
    print(f"  Total optimisation steps = {args.max_train_steps}")

    global_step, first_epoch = resume_from_latest(accelerator, args, num_update_steps_per_epoch)
    progress_bar = tqdm(range(args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process)
    transformer = run.transformer
    transformer.train()

    for epoch in range(first_epoch, num_train_epochs):
        for batch in run.train_dataloader:
            with accelerator.accumulate(transformer):
                loss = run.train_step(batch)
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

            progress_bar.set_postfix(loss=loss, lr=run.lr_scheduler.get_last_lr()[0])
            if global_step >= args.max_train_steps:
                break

    end_run(accelerator)


if __name__ == "__main__":
    main()
