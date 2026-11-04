"""Injection, scaling, token masking, saving and loading of the SIA adapters.

Checkpoint layouts:
  AnaLoRA (Stage 1)  pytorch_lora_weights.safetensors (PEFT/diffusers keys)
  ExLoRA  (Stage 2)  analora.safetensors + exlora.safetensors (released weights) or
                     base_lora.pt + selective_lora.pt (written by the trainers); same tensor names
"""
import json
import os
import re
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from sia.lora import AnaLoRALinear, ExLoRALinear

# PEFT adapter name of the AnaLoRA trainers (diffusers ``add_adapter`` / ``set_adapters``).
ANALORA_ADAPTER_NAME = "visual-analogy-analora"


@contextmanager
def exlora_token_mask_ctx(model, mask: torch.Tensor, disable_mask_after=True):
    """Set the token mask of every ``ExLoRALinear`` in ``model`` for the duration of the block.

    Args:
        model: module containing ``ExLoRALinear`` layers.
        mask: ``(B, S)`` boolean tensor; ExLoRA is applied where it is true.
        disable_mask_after: clear the mask (ExLoRA off) when the block exits.
    """
    loras = [m for m in model.modules() if isinstance(m, ExLoRALinear)]
    for m in loras:
        m.set_token_mask(mask)
    try:
        yield
    finally:
        if disable_mask_after:
            for m in loras:
                m.set_token_mask(None)


@contextmanager
def lora_mode_ctx(transformer, analora_scale: float = 1.0, exlora_scale: float = 1.0):
    """Scale AnaLoRA and ExLoRA by the given multipliers inside the block (0 = off, 1 = full).

    ``type(m) is AnaLoRALinear`` selects AnaLoRA layers only; the original scalings are
    restored on exit.
    """
    base_mods = [m for m in transformer.modules() if type(m) is AnaLoRALinear]
    st_mods   = [m for m in transformer.modules() if isinstance(m, ExLoRALinear)]
    orig_base = [m.scaling for m in base_mods]
    for m, s in zip(base_mods, orig_base):
        m.scaling = analora_scale * s
    for m in st_mods:
        m.set_scaling(exlora_scale * m.alpha / m.r)
    try:
        yield
    finally:
        for m, s in zip(base_mods, orig_base):
            m.scaling = s
        for m in st_mods:
            m.reset_scaling()


def _get_parent(model, dotted_name):
    """Return ``(parent_module, attribute_name)`` for a dotted submodule name."""
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_exlora_modules(model, target_linear_name_suffixes, r, alpha, dropout):
    """Wrap every target ``nn.Linear`` / ``AnaLoRALinear`` in an ``ExLoRALinear``.

    Args:
        model: module to modify in place.
        target_linear_name_suffixes: a module is wrapped when its qualified name ends with one
            of these suffixes.
        r, alpha, dropout: ExLoRA hyperparameters (see ``ExLoRALinear``).

    Returns:
        Qualified names of the wrapped modules.
    """
    replaced = []
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == "ExLoRALinear":
            module = module.base

        is_target = isinstance(module, (nn.Linear, AnaLoRALinear))
        if is_target and any(name.endswith(sfx) for sfx in target_linear_name_suffixes):
            parent, attr = _get_parent(model, name)
            wrapped = ExLoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, wrapped)
            replaced.append(name)

    return replaced


def save_exlora_state_dict(unwraped_model, output_file_path):
    """Save the ExLoRA weights with ``torch.save`` under keys ``<module>.lora_{A,B}.weight``."""
    ckpt = OrderedDict()
    for name, module in unwraped_model.named_modules():
        if isinstance(module, ExLoRALinear):
            ckpt[f"{name}.lora_A.weight"] = module.lora_A.weight.cpu()
            ckpt[f"{name}.lora_B.weight"] = module.lora_B.weight.cpu()

    torch.save(ckpt, output_file_path)


def inject_analora_modules(model, target_linear_name_suffixes, r, alpha, dropout=0.0):
    """Wrap every target ``nn.Linear`` in an ``AnaLoRALinear``; returns the wrapped names.

    Targets are selected by name suffix, as in ``inject_exlora_modules``.
    """
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(name.endswith(sfx) for sfx in target_linear_name_suffixes):
            parent, attr = _get_parent(model, name)
            wrapped = AnaLoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, wrapped)
            replaced.append(name)
    return replaced


def collect_exlora_params(model):
    """Return the ExLoRA ``lora_A`` / ``lora_B`` weights (excluding any wrapped AnaLoRA)."""
    params = []
    for module in model.modules():
        if isinstance(module, ExLoRALinear):
            params.append(module.lora_A.weight)
            params.append(module.lora_B.weight)
    return params


def collect_analora_params(model):
    """Return the AnaLoRA ``lora_A`` / ``lora_B`` weights, each parameter once."""
    params = []
    seen = set()
    for module in model.modules():
        if isinstance(module, AnaLoRALinear):
            for p in [module.lora_A.weight, module.lora_B.weight]:
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
    return params


def set_exlora_requires_grad(model, requires_grad: bool):
    """Toggle requires_grad on all ExLoRA parameters."""
    for p in collect_exlora_params(model):
        p.requires_grad_(requires_grad)


def set_analora_requires_grad(model, requires_grad: bool):
    """Toggle requires_grad on all AnaLoRA parameters."""
    for p in collect_analora_params(model):
        p.requires_grad_(requires_grad)


def save_analora_state_dict(model, output_file_path):
    """Save the AnaLoRA weights with ``torch.save``.

    An AnaLoRA wrapped by an ExLoRA is named ``<module>.base``; the suffix is stripped so the
    keys are ``<module>.lora_{A,B}.weight`` whether or not ExLoRA has been injected.
    """
    ckpt = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, AnaLoRALinear):
            save_name = name[:-5] if name.endswith(".base") else name
            ckpt[f"{save_name}.lora_A.weight"] = module.lora_A.weight.cpu()
            ckpt[f"{save_name}.lora_B.weight"] = module.lora_B.weight.cpu()
    torch.save(ckpt, output_file_path)


def load_tensors(path, device="cpu"):
    """Load a tensor dict from a ``.safetensors`` file or a ``torch.save`` file (``.pt``)."""
    if str(path).endswith(".safetensors"):
        return load_file(path, device=str(device))
    return torch.load(path, map_location=device, weights_only=True)


def checkpoint_files(ckpt_dir):
    """Return ``(analora_file, exlora_file)`` of a Stage-2 checkpoint directory.

    The release names ``analora.safetensors`` / ``exlora.safetensors`` are preferred; otherwise
    the trainer names ``base_lora.pt`` / ``selective_lora.pt`` are returned. The returned paths
    may not exist.
    """
    ckpt_dir = Path(ckpt_dir)
    analora, exlora = ckpt_dir / "analora.safetensors", ckpt_dir / "exlora.safetensors"
    return (analora if analora.exists() else ckpt_dir / "base_lora.pt",
            exlora if exlora.exists() else ckpt_dir / "selective_lora.pt")


def checkpoint_lora_rank(ckpt_dir, state):
    """Return the LoRA rank of a checkpoint, read from the ``lora_A`` shapes in ``state``.

    The adapters use ``alpha = rank``. If a ``config.json`` sits next to the weights, its
    ``lora_rank`` and ``lora_alpha`` must agree with that; otherwise ``ValueError`` is raised.
    """
    rank = next(w.shape[0] for k, w in state.items() if k.endswith("lora_A.weight"))
    config = Path(ckpt_dir) / "config.json"
    if config.exists():
        cfg = json.loads(config.read_text())
        declared = (cfg.get("lora_rank", rank), cfg.get("lora_alpha", rank))
        if declared != (rank, rank):
            raise ValueError(f"{config}: lora_rank/lora_alpha {declared} disagree with the weights "
                             f"(rank {rank}, alpha = rank)")
    return rank


def load_exlora_state_dict(model, path):
    """Load ExLoRA weights (``exlora.safetensors`` or ``selective_lora.pt``) into ``model``.

    Non-strict load: every key in the file must exist in the model and at least one must be an
    ExLoRA weight; otherwise an error is raised.
    """
    state = load_tensors(path)
    _, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys when loading ExLoRA from {path}: {unexpected[:5]}")
    exlora_keys = {f"{name}.{ab}.weight" for name, m in model.named_modules()
                   if isinstance(m, ExLoRALinear) for ab in ("lora_A", "lora_B")}
    matched = sum(1 for k in state if k in exlora_keys)
    print(f"[ExLoRA Load] {matched}/{len(state)} keys matched ({len(exlora_keys)} ExLoRA params in model)")
    if matched == 0:
        raise RuntimeError(f"No ExLoRA keys in {path} matched the model.")


def load_analora_state_dict(model, ckpt_path, device="cpu"):
    """Load AnaLoRA weights written by ``save_analora_state_dict`` (or the release file).

    Raises ``RuntimeError`` unless every AnaLoRA ``lora_A`` / ``lora_B`` in the model is loaded.
    """
    raw_sd = load_tensors(ckpt_path, device)

    analora_map = {}
    for name, module in model.named_modules():
        if isinstance(module, AnaLoRALinear):
            save_name = name[:-5] if name.endswith(".base") else name
            analora_map[save_name] = module

    loaded = 0
    for key, weight in raw_sd.items():
        if ".lora_A.weight" in key:
            mod_name = key.replace(".lora_A.weight", "")
            ab = "lora_A"
        elif ".lora_B.weight" in key:
            mod_name = key.replace(".lora_B.weight", "")
            ab = "lora_B"
        else:
            continue

        if mod_name in analora_map:
            target = getattr(analora_map[mod_name], ab)
            with torch.no_grad():
                target.weight.copy_(weight.to(target.weight.device))
            loaded += 1

    print(f"[AnaLoRA Resume] Loaded {loaded} params from {ckpt_path}")
    if loaded != 2 * len(analora_map):
        raise RuntimeError(
            f"{ckpt_path} covered {loaded} of {2 * len(analora_map)} AnaLoRA weights.")


def load_analora_from_peft_checkpoint(model, ckpt_path, device="cpu"):
    """Load a PEFT/diffusers LoRA checkpoint directory into the ``AnaLoRALinear`` layers.

    Reads ``pytorch_lora_weights.safetensors`` (or ``.bin``) from ``ckpt_path``. Keys may carry a
    ``transformer.`` or ``base_model.model.`` prefix and an adapter name
    (``<module>.lora_A.<adapter>.weight``). Raises ``RuntimeError`` on any shape mismatch or when
    no key matches.
    """
    ckpt_dir = Path(ckpt_path)
    sf_file = ckpt_dir / "pytorch_lora_weights.safetensors"
    pt_file = ckpt_dir / "pytorch_lora_weights.bin"

    if sf_file.exists():
        raw_sd = load_file(sf_file, device=str(device))
    elif pt_file.exists():
        raw_sd = torch.load(pt_file, map_location=device, weights_only=True)
    else:
        raise FileNotFoundError(
            f"No LoRA weights found in {ckpt_path}. "
            "Expected 'pytorch_lora_weights.safetensors' or '.bin'."
        )

    analora_map = {}
    for name, module in model.named_modules():
        if isinstance(module, AnaLoRALinear):
            analora_map[name] = module

    matched, skipped, mismatched = [], [], []

    for key, weight in raw_sd.items():
        norm = key
        for pfx in ("transformer.", "base_model.model."):
            if norm.startswith(pfx):
                norm = norm[len(pfx):]

        # "<module>.lora_A.<adapter>.weight" -> "<module>.lora_A.weight"
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

        if mod_name not in analora_map:
            skipped.append(key)
            continue

        target = getattr(analora_map[mod_name], ab)
        if target.weight.shape != weight.shape:
            mismatched.append(f"{key}: ckpt={weight.shape} model={target.weight.shape}")
            continue

        with torch.no_grad():
            target.weight.copy_(weight.to(target.weight.device))
        matched.append(key)

    print(
        f"[AnaLoRA Load] matched={len(matched)}, "
        f"skipped={len(skipped)}, shape_mismatch={len(mismatched)}"
    )
    # Fail loudly: a partial load (e.g. a lora_rank that differs from the checkpoint's)
    # would otherwise leave AnaLoRA silently zero-initialized.
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} AnaLoRA weights have the wrong shape in {ckpt_path} "
            f"(is lora_rank set to the checkpoint's rank?). First: {mismatched[0]}")
    if not matched:
        raise RuntimeError(f"No AnaLoRA weights in {ckpt_path} matched the model.")

    for name, mod in analora_map.items():
        if mod.lora_B.weight.data.abs().sum().item() > 0:
            print(f"[AnaLoRA Load] Weights OK (sample: {name}, "
                  f"B norm={mod.lora_B.weight.data.norm().item():.4f})")
            break
    else:
        print("[AnaLoRA Load] WARNING — all lora_B weights are zero!")


def find_latest_checkpoint(output_dir):
    """Return the highest-step ``checkpoint-<N>`` directory in ``output_dir``, or ``None``.

    Entries whose name is not ``checkpoint-`` followed by digits are ignored.
    """
    if not os.path.isdir(output_dir):
        return None
    dirs = [d for d in os.listdir(output_dir) if re.fullmatch(r"checkpoint-\d+", d)]
    if not dirs:
        return None
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, dirs[-1])
