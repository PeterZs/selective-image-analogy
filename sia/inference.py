"""Inference core shared by the scripts, the evaluation and the demo.

- ``load_backbone`` / ``load_pipeline``: pick the backbone module by name (or from the model id) and
  load its pipeline with the adapters of a checkpoint. Backbone modules are imported lazily, so the
  FLUX.2-Klein environment never imports Qwen-Image-2.1 code and vice versa.
- Qwen-Image-2.1 selective analogy: ``load_qwen_pipeline_with_exlora`` (base model + AnaLoRA + ExLoRA
  from a local directory or the Hub), ``encode_analogy`` (condition images and full prompt, once per
  case), ``generate_selective`` (one B' for one set of kept edits, optionally with CSV-steered
  prompt embeddings) and ``run_selective_analogy`` (both, the demo's entry point). The benchmark
  evaluation calls the same two functions, so its regression check covers the demo's code path.

FLUX.2-Klein per-image generation is ``sia.backbones.flux2_klein.run_inference``.

Importing this module changes no Hugging Face setting: weights are downloaded as usual unless
``HF_HUB_OFFLINE=1`` is set by the user.
"""
import importlib
from dataclasses import dataclass

import torch

from sia.adapters import (
    checkpoint_files,
    checkpoint_lora_rank,
    inject_analora_modules,
    inject_exlora_modules,
    load_analora_state_dict,
    load_exlora_state_dict,
    load_tensors,
)
from sia.hf import resolve_checkpoint
from sia.prompt import build_analogy_prompt

BACKBONES = ("flux2_klein", "qwen_image21")
QWEN_BASE_MODEL = "Qwen/Qwen-Image-2.1"


def load_backbone(name):
    """Import and return the backbone module ``sia.backbones.<name>`` (one of ``BACKBONES``)."""
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; expected one of {BACKBONES}")
    return importlib.import_module(f"sia.backbones.{name}")


def backbone_name(cfg):
    """The config's ``backbone`` key, else inferred from ``pretrained_model_name_or_path``."""
    if cfg.get("backbone"):
        return cfg.backbone
    model = str(cfg.pretrained_model_name_or_path).lower()
    if "qwen-image" in model:
        return "qwen_image21"
    if "flux.2-klein" in model:
        return "flux2_klein"
    raise ValueError(f"cannot infer the backbone from {cfg.pretrained_model_name_or_path!r}; "
                     f"set 'backbone' to one of {BACKBONES}")


def load_pipeline(cfg, device, backbone=None):
    """Load the base model with AnaLoRA + ExLoRA for inference (``<backbone>.load_pipeline``).

    Args:
        cfg: config with ``pretrained_model_name_or_path``, ``lora_rank`` and the checkpoint keys
            ``checkpoint_dir`` (local dir or Hub path), ``output_dir`` (latest checkpoint is used
            when ``checkpoint_dir`` is unset) and ``analora_path`` (Stage-1 AnaLoRA, read only
            without a Stage-2 checkpoint).
        device: target device.
        backbone: backbone name; ``None`` uses ``backbone_name(cfg)``.

    Returns:
        What the backbone returns: ``(pipe, tokenizer, text_encoder)`` for FLUX.2-Klein, ``pipe``
        for Qwen-Image-2.1.
    """
    return load_backbone(backbone or backbone_name(cfg)).load_pipeline(cfg, device)


def attach_adapters(backbone, transformer, checkpoint, rank, analora=True, exlora=True):
    """Inject the adapter layers of ``backbone`` into ``transformer`` and load a Stage-2 checkpoint.

    Args:
        backbone: backbone module (its ``ANALORA_TARGET_MODULES`` / ``EXLORA_TARGET_MODULES``).
        transformer: the base transformer.
        checkpoint: local checkpoint directory (see ``sia.hf.resolve_checkpoint``).
        rank: LoRA rank (alpha = rank).
        analora, exlora: load that component. AnaLoRA layers are injected whenever either component
            is on: with ``analora=False`` they stay zero (``lora_B = 0``), which is the model
            without AnaLoRA. ExLoRA layers exist only when ``exlora=True``.
    """
    analora_file, exlora_file = checkpoint_files(checkpoint)
    if analora or exlora:
        inject_analora_modules(transformer, backbone.ANALORA_TARGET_MODULES, r=rank, alpha=rank, dropout=0.0)
    if analora:
        load_analora_state_dict(transformer, analora_file, "cpu")
    if exlora:
        inject_exlora_modules(transformer, backbone.EXLORA_TARGET_MODULES, r=rank, alpha=rank, dropout=0.0)
        load_exlora_state_dict(transformer, exlora_file)


# ---------------------------------------------------------------------------
# Qwen-Image-2.1 selective analogy
# ---------------------------------------------------------------------------

def load_qwen_pipeline_with_exlora(weights, device="cuda", *, subfolder=None, revision=None,
                                   base=QWEN_BASE_MODEL, base_revision=None, token=None):
    """Qwen-Image-2.1 pipeline with the SIA Stage-2 adapters (AnaLoRA + ExLoRA) attached.

    Args:
        weights: checkpoint directory, or a Hub path ``<org>/<repo>[/<subfolder>][@revision]``
            (e.g. ``selective-image-analogy/SIA-weights/qwen_image21/exlora``).
        device: device the pipeline is moved to.
        subfolder: appended to ``weights`` (``<weights>/<subfolder>``).
        revision: Hub revision of the weights when ``weights`` names none.
        base, base_revision: base model id and its revision.
        token: Hugging Face token for private repos (default: the logged-in token / ``HF_TOKEN``).

    The LoRA rank is read from the checkpoint. Returns the pipeline in eval mode.
    """
    qwen = load_backbone("qwen_image21")
    checkpoint = resolve_checkpoint(f"{weights.rstrip('/')}/{subfolder}" if subfolder else weights,
                                    revision=revision, token=token)
    pipe = qwen.load_pipeline_qwen21(base, revision=base_revision, token=token)
    rank = checkpoint_lora_rank(checkpoint, load_tensors(checkpoint_files(checkpoint)[0]))
    attach_adapters(qwen, pipe.transformer, checkpoint, rank)
    pipe.to(device)
    pipe.transformer.eval()
    return pipe


@dataclass
class QwenAnalogy:
    """One (A, A', B, edits) case prepared for Qwen-Image-2.1: shared by all its kept-edit subsets."""
    cond: object                  # sia.backbones.qwen_image21.Qwen21Conditions
    edits: list
    prompt: str                   # the full analogy prompt (every edit listed)
    prompt_embeds: torch.Tensor   # (1, S, D) encoding of ``prompt`` with the condition images
    image_pad_mask: torch.Tensor  # (1, S)


def encode_analogy(pipe, a, a_prime, b, edits, resolution=512, device=None):
    """Prepare the condition images A, A', B and encode the full analogy prompt.

    ``device`` defaults to the pipeline's device. Images are snapped to a common size per pair
    (``harmonize_images_qwen21``) and resized to ``resolution**2`` area as in training.
    """
    qwen = load_backbone("qwen_image21")
    device = device or pipe.device
    a, a_prime, b = qwen.harmonize_images_qwen21(a, a_prime, b, b)[:3]
    cond = qwen.prepare_conditions(pipe, [a, a_prime, b], resolution, device)
    prompt = build_analogy_prompt(edits, edits)
    prompt_embeds, image_pad_mask = qwen.encode_prompt(pipe, prompt, cond, device)
    return QwenAnalogy(cond, list(edits), prompt, prompt_embeds, image_pad_mask)


@torch.no_grad()
def generate_selective(pipe, analogy: QwenAnalogy, kept, steps=40, seed=0, exlora=True,
                       prompt_embeds=None, use_kv_cache=True):
    """Generate B' keeping the edits ``kept`` (0-based indices) and suppressing the others.

    Args:
        pipe: Qwen-Image-2.1 pipeline with adapters attached.
        analogy: output of ``encode_analogy``.
        kept: indices into ``analogy.edits`` of the edits to apply.
        steps: sampling steps (the pipeline's sigma grid is used when the checkpoint has one).
        seed: seed of the initial noise.
        exlora: gate ExLoRA on the suppressed edits' prompt tokens. The edits are located by text
            in the full prompt (``qwen21_text_token_mask``).
        prompt_embeds: replacement for ``analogy.prompt_embeds`` (e.g. CSV-steered, see
            ``sia.concept_suppression.apply_csv``); the full prompt is kept.
        use_kv_cache: reuse the condition prefix's keys and values across steps.

    Without ExLoRA and without ``prompt_embeds`` the suppressed edits are deleted from the prompt
    instead (the AnaLoRA-only baseline). Returns the image at the generated size.
    """
    qwen = load_backbone("qwen_image21")
    cond, edits = analogy.cond, analogy.edits
    device = analogy.prompt_embeds.device
    suppressed = [p for p in range(len(edits)) if p not in kept]
    mask = None
    if exlora or prompt_embeds is not None:
        embeds = analogy.prompt_embeds if prompt_embeds is None else prompt_embeds
        image_pad_mask = analogy.image_pad_mask
        if exlora:
            mask = qwen.qwen21_text_token_mask(pipe, analogy.prompt, [edits[p] for p in suppressed],
                                               cond.image_sizes, analogy.prompt_embeds.shape[1], device)
    else:
        prompt = build_analogy_prompt(edits, edits, suppress_indices=suppressed)
        embeds, image_pad_mask = qwen.encode_prompt(pipe, prompt, cond, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    return qwen.generate_qwen21(pipe, cond, embeds, image_pad_mask, steps, generator=generator,
                                exlora_text_mask=mask, use_kv_cache=use_kv_cache)


def run_selective_analogy(pipe, a, a_prime, b, edits, kept, seed=0, steps=40, resolution=512):
    """Apply to ``b`` only the edits ``kept`` (0-based indices into ``edits``) of the A -> A' pair.

    ``pipe`` comes from ``load_qwen_pipeline_with_exlora``; the other edits are suppressed by
    ExLoRA. Returns the generated PIL image (its size follows B's aspect ratio at
    ``resolution**2`` area, not B's exact size).
    """
    if not kept or any(not 0 <= k < len(edits) for k in kept):
        raise ValueError(f"kept must be non-empty indices into the {len(edits)} edits, got {kept}")
    analogy = encode_analogy(pipe, a, a_prime, b, edits, resolution)
    return generate_selective(pipe, analogy, kept, steps=steps, seed=seed)
