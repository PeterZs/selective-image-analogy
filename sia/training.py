"""Training glue shared by ``scripts/train_analora.py`` and ``scripts/train_exlora.py``.

Config loading, the accelerator / Weights & Biases setup, the accelerate checkpoint hooks of both
adapter types, resuming, the AnaLoRA L2 anchor of Stage 2 and the hand-off between a trainer's
backbone-specific setup and its training loop (``TrainingRun``). Nothing here depends on a
backbone: model classes are passed in by the caller.

Importing this module also patches ``accelerate``'s ``set_module_tensor_to_device`` (see below).
"""
import argparse
import inspect as _inspect
import os
import pickle
import re
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, set_seed
from diffusers.training_utils import cast_training_params
from omegaconf import OmegaConf
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

from sia.adapters import (
    ANALORA_ADAPTER_NAME,
    find_latest_checkpoint,
    load_analora_state_dict,
    load_exlora_state_dict,
    save_analora_state_dict,
    save_exlora_state_dict,
)
from sia.lora import AnaLoRALinear, ExLoRALinear

# WandB is disabled when WANDB_MODE=disabled or WANDB_DISABLED=true.
# Some GPU nodes have no wandb-core installed, so we skip
# all wandb calls entirely rather than just switching to offline mode.
_WANDB_MODE = os.environ.get("WANDB_MODE", "online").lower()
USE_WANDB = (
    _WANDB_MODE not in ("disabled", "dryrun")
    and os.environ.get("WANDB_DISABLED", "false").lower() != "true"
)
if USE_WANDB:
    import wandb

# ── Compatibility shim: strip unknown kwargs from set_module_tensor_to_device
try:
    from accelerate.utils import modeling as _accelerate_modeling
    _orig_smttd = _accelerate_modeling.set_module_tensor_to_device
    _known_smttd_params = set(_inspect.signature(_orig_smttd).parameters)
    _VAR_KEYWORD = _inspect.Parameter.VAR_KEYWORD
    _has_var_kw = any(
        p.kind == _VAR_KEYWORD
        for p in _inspect.signature(_orig_smttd).parameters.values()
    )
    if not _has_var_kw:
        def _smttd_compat(*args, **kwargs):
            filtered = {k: v for k, v in kwargs.items()
                        if k in _known_smttd_params}
            return _orig_smttd(*args, **filtered)
        _accelerate_modeling.set_module_tensor_to_device = _smttd_compat
        try:
            import diffusers.models.model_loading_utils as _dml
            _dml.set_module_tensor_to_device = _smttd_compat
        except Exception:
            pass
except Exception:
    pass
# ───────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Config and run setup
# ---------------------------------------------------------------------------

def parse_args(print_args=False):
    """Load the YAML config given by ``--config`` (an ``OmegaConf`` object)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = OmegaConf.load(parser.parse_args().config)
    if print_args:
        print("=" * 40, "Arguments", "=" * 40)
        for k in args:
            print(f"  {k}: {args[k]}")
        print("=" * 91)
    return args


def start_run(args):
    """Create the bf16 ``Accelerator``, start Weights & Biases on the main process, create
    ``output_dir`` and seed every RNG. Seeding comes last, as in the original trainers, so the
    random streams of a run depend only on ``args.seed``."""
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
    )

    if accelerator.is_main_process and USE_WANDB:
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if wandb_key:
            wandb.login(key=wandb_key)
        wandb.init(
            project=args.tracker_name,
            name=args.wandb_run_name,
            config=OmegaConf.to_container(args, resolve=True),
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="train/step")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    set_seed(args.seed)
    return accelerator


@dataclass
class TrainingRun:
    """What a trainer's backbone-specific setup hands to its training loop."""
    transformer: torch.nn.Module       # as returned by ``accelerator.prepare`` (forward, grad clipping)
    optimizer: torch.optim.Optimizer
    lr_scheduler: object
    train_dataloader: object
    train_dataset: object
    # Forward, loss and backward for one batch; returns the loss value that is logged.
    train_step: Callable
    # Writes the periodic validation (and benchmark) images after an optimizer step; main process only.
    visualize: Callable[[int], None]
    # Called once before the first step with the (possibly resumed) global step.
    before_training: Optional[Callable[[int], None]] = None
    unwrapped: Optional[torch.nn.Module] = None   # ExLoRA: bare model (trainability toggles, anchor)
    analora_anchor: Optional[dict] = None         # ExLoRA: AnaLoRA weights at the start of Stage 2


def resume_from_latest(accelerator, args, num_update_steps_per_epoch):
    """With ``resume_from_checkpoint``, load the latest ``checkpoint-<N>`` of ``output_dir``.

    Returns ``(global_step, first_epoch)``; ``(0, 0)`` when there is nothing to resume.
    """
    if not args.resume_from_checkpoint:
        return 0, 0
    # A fresh run has no checkpoint yet; treat that as "nothing to resume".
    path = find_latest_checkpoint(args.output_dir)
    if path is None:
        print(f"No checkpoint found in {args.output_dir}. Starting a new training run.")
        return 0, 0
    print(f"Resuming from checkpoint {path}")
    accelerator.load_state(path)
    global_step = int(os.path.basename(path).split("-")[1])
    return global_step, global_step // num_update_steps_per_epoch


def due(global_step, every, args):
    """True every ``every`` steps and at the last step."""
    return global_step % every == 0 or global_step == args.max_train_steps


def save_checkpoint(accelerator, args, global_step):
    """``accelerator.save_state`` to ``output_dir/checkpoint-<step>`` every ``checkpointing_steps``."""
    if due(global_step, args.checkpointing_steps, args):
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        print(f"Saved state to {save_path}")


def log_metrics(accelerator, loss, lr, epoch, global_step):
    """Log the step's training loss and learning rate to Weights & Biases (main process)."""
    if accelerator.is_main_process and USE_WANDB:
        wandb.log({"train/loss": loss, "train/lr": lr, "train/epoch": epoch, "train/step": global_step})


def end_run(accelerator):
    """Close Weights & Biases and the accelerator at the end of training."""
    if accelerator.is_main_process and USE_WANDB:
        wandb.finish()
    accelerator.end_training()


# ---------------------------------------------------------------------------
# Accelerate save / load hooks
# ---------------------------------------------------------------------------

def get_analora_save_hook(accelerator, pipeline_cls, transformer_cls):
    """Save the PEFT AnaLoRA as ``pytorch_lora_weights.safetensors`` (diffusers LoRA format, so it
    also loads with ``pipeline_cls.load_lora_weights``)."""
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                unwrapped = accelerator.unwrap_model(model)
                if not isinstance(unwrapped, transformer_cls):
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                pipeline_cls.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=get_peft_model_state_dict(unwrapped, adapter_name=ANALORA_ADAPTER_NAME),
                )
                weights.pop()  # make sure the model is not saved again
    return save_model_hook


def get_analora_load_hook(accelerator, transformer_cls):
    """Restore the PEFT AnaLoRA written by ``get_analora_save_hook``."""
    def load_model_hook(models, input_dir):
        lora_weights_path = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
        if not os.path.exists(lora_weights_path):
            raise FileNotFoundError(
                f"LoRA checkpoint not found at {lora_weights_path}. "
                "Expected 'pytorch_lora_weights.safetensors' saved by the save hook."
            )
        while models:
            model = models.pop()
            unwrapped = accelerator.unwrap_model(model)
            if not isinstance(unwrapped, transformer_cls):
                raise ValueError(f"Unexpected model type in load_model_hook: {type(model)=}.")
            # diffusers key format: "transformer.<peft_key>"
            peft_state_dict = {
                k[len("transformer."):]: v
                for k, v in load_file(lora_weights_path).items()
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


def get_exlora_save_hook(accelerator, transformer_cls):
    """Save ExLoRA as ``selective_lora.pt`` and AnaLoRA as ``base_lora.pt`` (plain tensor dicts)."""
    def hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                unwrapped = accelerator.unwrap_model(model)
                if isinstance(unwrapped, transformer_cls):
                    save_exlora_state_dict(unwrapped, os.path.join(output_dir, "selective_lora.pt"))
                    save_analora_state_dict(unwrapped, os.path.join(output_dir, "base_lora.pt"))
                    weights.pop()
    return hook


def get_exlora_load_hook(accelerator):
    """Restore the files written by ``get_exlora_save_hook``."""
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
# Checkpoint initialisation
# ---------------------------------------------------------------------------

def load_peft_ckpt_into_exlora(transformer, ckpt_dir, device):
    """Initialize ExLoRA A/B weights from a PEFT safetensors checkpoint."""
    ckpt_path = os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors")
    if not os.path.exists(ckpt_path):
        print(f"[PEFT Init] No file at {ckpt_path}, skipping checkpoint init.")
        return

    print(f"[PEFT Init] Loading PEFT weights from {ckpt_path}")
    peft_sd = load_file(ckpt_path)
    print(f"[PEFT Init] Found {len(peft_sd)} keys in PEFT checkpoint.")

    exlora_map = {
        name: mod
        for name, mod in transformer.named_modules()
        if isinstance(mod, ExLoRALinear)
    }
    print(f"[PEFT Init] ExLoRA has {len(exlora_map)} injected modules.")

    matched, skipped, mismatched = [], [], []

    for key, weight in peft_sd.items():
        norm = key
        for pfx in ("transformer.", "base_model.model."):
            if norm.startswith(pfx):
                norm = norm[len(pfx):]

        norm = re.sub(r'\.lora_([AB])\.[\w\-]+\.weight$', r'.lora_\1.weight', norm)

        if ".lora_A.weight" in norm:
            mod_name = norm.replace(".lora_A.weight", "")
            ab = "lora_A"
        elif ".lora_B.weight" in norm:
            mod_name = norm.replace(".lora_B.weight", "")
            ab = "lora_B"
        else:
            skipped.append(key)
            continue

        if mod_name not in exlora_map:
            skipped.append(key)
            continue

        target = getattr(exlora_map[mod_name], ab)
        if target.weight.shape != weight.shape:
            mismatched.append(f"{key}: peft={weight.shape} exlora={target.weight.shape}")
            continue

        with torch.no_grad():
            target.weight.copy_(weight.to(device))
        matched.append(key)

    print(
        f"[PEFT Init] matched={len(matched)}, "
        f"skipped={len(skipped)}, shape_mismatch={len(mismatched)}"
    )
    if mismatched:
        print("[PEFT Init] Shape mismatches:")
        for m in mismatched:
            print(f"  {m}")


def restore_training_state(optimizer, lr_scheduler, ckpt_dir, device):
    """Restore optimizer, lr_scheduler, and RNG state from an Accelerate checkpoint dir."""
    opt_path = os.path.join(ckpt_dir, "optimizer.bin")
    sch_path = os.path.join(ckpt_dir, "scheduler.bin")
    rng_path = os.path.join(ckpt_dir, "random_states_0.pkl")

    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        print(f"[Ckpt] Restored optimizer from {opt_path}")
    if os.path.exists(sch_path):
        lr_scheduler.load_state_dict(torch.load(sch_path, map_location=device))
        print(f"[Ckpt] Restored lr_scheduler from {sch_path}")
    if os.path.exists(rng_path):
        with open(rng_path, "rb") as f:
            rng_state = pickle.load(f)
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        print(f"[Ckpt] Restored RNG state from {rng_path}")


# ---------------------------------------------------------------------------
# AnaLoRA L2 anchor (Stage 2)
# ---------------------------------------------------------------------------

def snapshot_analora(transformer, device):
    """Copy every ``AnaLoRALinear`` A/B weight to ``device``, keyed by ``(module name, "A"|"B")``."""
    return {(name, ab): weight.detach().clone().to(device)
            for name, mod in transformer.named_modules() if isinstance(mod, AnaLoRALinear)
            for ab, weight in (("A", mod.lora_A.weight), ("B", mod.lora_B.weight))}


def analora_anchor_loss(transformer, anchor):
    """Squared L2 distance of the AnaLoRA weights from ``anchor`` (``snapshot_analora``); keeps the
    Stage 1 AnaLoRA from drifting on the AnaLoRA steps of Stage 2."""
    return sum(
        torch.sum((p.float() - anchor[(n, ab)].float()) ** 2)
        for n, mod in transformer.named_modules() if isinstance(mod, AnaLoRALinear)
        for ab, p in (('A', mod.lora_A.weight), ('B', mod.lora_B.weight))
    )
