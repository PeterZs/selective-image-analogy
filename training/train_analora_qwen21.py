"""
train_analora_qwen21.py

AnaLoRA (Stage 1, full-transfer analogy) trainer on Qwen-Image-2.1.

Same recipe as train_analora_flux2_klein.py — a PEFT LoRA learning
(A, A', B) -> B' from the analogy prompt, prompt dropped 15 % of the time —
with the Qwen-Image-2.1 conditioning and sampler from sia/backbones/qwen_image21.py.
Only the concat formulation is supported (Qwen-Image-2.1 takes the three
images as separate condition images natively).

The checkpoint is a diffusers LoRA file (pytorch_lora_weights.safetensors),
so it also loads with ``QwenImage21Pipeline.load_lora_weights``, and Stage 2
reads it through ``analora_path``.

Usage:
  python training/train_analora_qwen21.py \
      --config training/configs/train_analora_qwen21.yaml
"""

import math
import os
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.train_exlora_qwen21 import _USE_WANDB, parse_args, render_rows  # noqa: E402
from training.train_analora_flux2_klein import ANALORA_ADAPTER_NAME  # noqa: E402
from sia.backbones.qwen_image21 import (  # noqa: E402
    ANALORA_TARGET_MODULES,
    backward_per_sample,
    encode_prompt,
    generate_qwen21,
    harmonize_images_qwen21,
    load_pipeline_qwen21,
    prepare_conditions,
    sample_flow_loss,
)
from sia.data import DATASET, AnalogyPairDataset, collate_pairs  # noqa: E402

if _USE_WANDB:
    import wandb

from accelerate import Accelerator  # noqa: E402
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, set_seed  # noqa: E402
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel  # noqa: E402
from diffusers.optimization import get_scheduler  # noqa: E402
from diffusers.training_utils import cast_training_params  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from peft import LoraConfig  # noqa: E402
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def get_save_model_hook(accelerator):
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                unwrapped = accelerator.unwrap_model(model)
                if not isinstance(unwrapped, QwenImage21Transformer2DModel):
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                QwenImage21Pipeline.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=get_peft_model_state_dict(unwrapped, adapter_name=ANALORA_ADAPTER_NAME),
                )
                weights.pop()  # make sure the model is not saved again
    return save_model_hook


def get_load_model_hook(accelerator):
    def load_model_hook(models, input_dir):
        lora_weights_path = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
        while models:
            model = models.pop()
            unwrapped = accelerator.unwrap_model(model)
            if not isinstance(unwrapped, QwenImage21Transformer2DModel):
                raise ValueError(f"Unexpected model type in load_model_hook: {type(model)=}.")
            # diffusers key format: "transformer.<peft_key>"
            peft_state_dict = {
                k[len("transformer."):]: v
                for k, v in load_safetensors_file(lora_weights_path).items()
                if k.startswith("transformer.")
            }
            incompatible = set_peft_model_state_dict(unwrapped, peft_state_dict, adapter_name=ANALORA_ADAPTER_NAME)
            # Frozen base weights are always reported missing (strict=False); only LoRA keys matter.
            missing = [k for k in getattr(incompatible, "missing_keys", []) if "lora_" in k]
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            if missing or unexpected:
                raise RuntimeError(f"[load_model_hook] missing LoRA keys: {missing[:5]}, "
                                   f"unexpected keys: {unexpected[:5]}")
            print(f"[load_model_hook] Restored {len(peft_state_dict)} LoRA tensors from {lora_weights_path}")
            cast_training_params([unwrapped], dtype=torch.float32)
    return load_model_hook


@torch.no_grad()
def log_analogy_validation(pipe, train_dataset, args, accelerator, global_step):
    """Strip A | A' | B | B' GT | B' pred for a random matched pair."""
    print(f"Running analogy validation at step {global_step}...")
    device = accelerator.device
    meta = train_dataset.sample_val_pair()
    imgs = {k: meta[k] for k in ("a", "a_prime", "b", "b_prime")}

    cond = prepare_conditions(pipe, [imgs["a"], imgs["a_prime"], imgs["b"]], args.resolution, device)
    prompt_embeds, image_pad_mask = encode_prompt(pipe, meta["prompt"], cond, device)
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    pred = generate_qwen21(pipe, cond, prompt_embeds, image_pad_mask, args.num_val_inference_steps,
                           generator=generator, use_kv_cache=args.get("use_kv_cache", True))

    if accelerator.is_main_process:
        vis_dir = Path(args.output_dir) / "validation_images"
        vis_dir.mkdir(parents=True, exist_ok=True)
        strip = render_rows(
            [(meta["edit_name"], [(imgs["a"], "A"), (imgs["a_prime"], "A'"), (imgs["b"], "B"),
                                  (imgs["b_prime"], "B' GT"), (pred, "B' pred")])],
            title=meta["prompt"])
        strip_path = vis_dir / f"step_{global_step:06d}.png"
        strip.save(strip_path)
        print(f"Saved validation strip to {strip_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args(print_args=True)
    args.resolution = args.get("resolution", 512)

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

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    set_seed(args.seed)

    pipe = load_pipeline_qwen21(args.pretrained_model_name_or_path)
    for m in [pipe.text_encoder, pipe.vae, pipe.transformer]:
        m.requires_grad_(False)
    pipe.to(accelerator.device)
    transformer = pipe.transformer
    if args.get("gradient_checkpointing", True):
        transformer.enable_gradient_checkpointing()

    transformer.add_adapter(
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=ANALORA_TARGET_MODULES,
        ),
        adapter_name=ANALORA_ADAPTER_NAME,
    )
    accelerator.register_save_state_pre_hook(get_save_model_hook(accelerator))
    accelerator.register_load_state_pre_hook(get_load_model_hook(accelerator))
    cast_training_params([transformer], dtype=torch.float32)

    lora_parameters = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": lora_parameters, "lr": args.learning_rate}],
        betas=(0.9, 0.999), weight_decay=1e-04, eps=1e-08,
    )

    train_dataset = AnalogyPairDataset(args.get("dataset", DATASET), args.get("dataset_revision"),
                                       harmonize_fn=harmonize_images_qwen21)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                  collate_fn=collate_pairs, num_workers=2, shuffle=True)

    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler)
    pipe.transformer = accelerator.unwrap_model(transformer)

    num_update_steps_per_epoch = math.ceil(
        math.ceil(len(train_dataset) / args.train_batch_size) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    print("***** Running AnaLoRA training (Qwen-Image-2.1) *****")
    print(f"  Num examples             = {len(train_dataset)}")
    print(f"  Batch size per device    = {args.train_batch_size} (one transformer pass per sample)")
    print(f"  Total optimisation steps = {args.max_train_steps}")
    print(f"  Resolution (area)        = {args.resolution}^2")

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        existing = os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else []
        dirs = sorted([d for d in existing if d.startswith("checkpoint")], key=lambda x: int(x.split("-")[1]))
        if dirs:
            print(f"Resuming from checkpoint {dirs[-1]}")
            accelerator.load_state(os.path.join(args.output_dir, dirs[-1]))
            global_step = int(dirs[-1].split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            print(f"No checkpoint found in {args.output_dir}. Starting a new training run.")

    progress_bar = tqdm(range(args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process)
    transformer.train()
    device = accelerator.device

    for epoch in range(first_epoch, num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                def losses_fn(i):
                    images = [batch['A'][i], batch['A_prime'][i], batch['B'][i]]
                    return sample_flow_loss(pipe, transformer, images, batch['B_prime'][i],
                                            batch['prompts'][i], args.resolution, device, args)

                loss = backward_per_sample(accelerator, transformer, losses_fn, len(batch['A']))
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

            progress_bar.set_postfix(loss=loss, lr=lr_scheduler.get_last_lr()[0])
            if global_step >= args.max_train_steps:
                break

    if accelerator.is_main_process and _USE_WANDB:
        wandb.finish()
    accelerator.end_training()


if __name__ == "__main__":
    main()
