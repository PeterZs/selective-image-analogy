#!/usr/bin/env python3
"""Precompute the concept suppression vectors (CSV) of every distinct edit of an evaluation split.

One direction per edit text, built from ``--pairs`` contrastive prompt pairs that the backbone's own
text encoder writes (paper: 64), and cached in ``--csv_cache`` (default ``~/.cache/sia/csv/<backbone>``)
where ``scripts/eval.py generate --csv 1`` and ``scripts/infer.py`` read them. Only the text encoder
is loaded, so several shards fit on one GPU. Edits already cached are only read. Per-edit statistics
go to ``<out>/csv_build_<shard>.json``.

Pairs are sampled without a fixed seed: a rebuilt direction differs from a cached one, which is why
existing caches should be kept.

Usage:
  python scripts/build_csv.py --config configs/exlora_qwen_image21.yaml --num_shards 2 --shard_index 0
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from sia.concept_suppression import get_or_build_steering_vector
from sia.data import DATASET, SPLITS, distinct_edits
from sia.inference import backbone_name, load_backbone

DEFAULT_OUT = {"qwen_image21": "experiments/eval_qwen21", "flux2_klein": "experiments/eval_flux2_klein"}


def load_text_encoder(cfg, backbone, device):
    """The part of the pipeline a direction needs: text encoder and tokenizer / processor."""
    if backbone == "qwen_image21":
        from diffusers import QwenImage21Pipeline
        # Skip the 14 GB transformer and the VAE.
        pipe = QwenImage21Pipeline.from_pretrained(
            cfg.pretrained_model_name_or_path, transformer=None, vae=None, dtype=torch.bfloat16)
        pipe.text_encoder.to(device)
        return pipe
    from huggingface_hub.constants import HF_HUB_OFFLINE
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
    path = cfg.pretrained_model_name_or_path
    # Same loading as sia.backbones.flux2_klein.load_models.
    tokenizer = Qwen2TokenizerFast.from_pretrained(path, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(path, subfolder="text_encoder", torch_dtype=torch.bfloat16,
                                                    local_files_only=HF_HUB_OFFLINE).to(device)
    return SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/exlora_qwen_image21.yaml",
                    help="model config (backbone and base model)")
    ap.add_argument("--split", default="analogy_benchmark", choices=SPLITS, help="split whose edits are built")
    ap.add_argument("--dataset", default=DATASET, help="Hugging Face dataset id")
    ap.add_argument("--pairs", type=int, default=64, help="contrastive pairs per edit")
    ap.add_argument("--csv_cache", default=None, help="CSV cache directory (default ~/.cache/sia/csv/<backbone>)")
    ap.add_argument("--out", default=None,
                    help="where csv_build_<shard>.json goes (default experiments/eval_<backbone>)")
    ap.add_argument("--limit", type=int, help="first N edits only")
    ap.add_argument("--shard_index", type=int, default=0, help="this worker's slice of the edits (parallel workers)")
    ap.add_argument("--num_shards", type=int, default=1)
    return ap


def main():
    args = build_parser().parse_args()
    cfg = OmegaConf.load(args.config)
    backbone = backbone_name(cfg)
    module = load_backbone(backbone)
    device = torch.device("cuda")
    pipe = load_text_encoder(cfg, backbone, device)
    edits = distinct_edits(args.split, args.dataset)
    edits = edits[:args.limit] if args.limit else edits
    edits = edits[args.shard_index::args.num_shards]
    t0 = time.time()
    stats = []
    for i, e in enumerate(edits):
        v, meta = get_or_build_steering_vector(module, e, args.pairs, pipe, device, args.csv_cache)
        stats.append(dict(edit=e, parsed=meta.get("n_parsed_from_llm"), requested=meta.get("n_requested"),
                          fallback=meta.get("used_fallback_padding"), usable=meta.get("n_usable_after_pooling"),
                          source=meta.get("source")))
        print(f"[csv] {i + 1}/{len(edits)} {e!r}: parsed {stats[-1]['parsed']}/{stats[-1]['requested']} "
              f"usable {stats[-1]['usable']} fallback {stats[-1]['fallback']} ({stats[-1]['source']})  "
              f"{(time.time() - t0) / (i + 1):.1f} s/edit", flush=True)
    out = Path(args.out or DEFAULT_OUT[backbone])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"csv_build_{args.shard_index}.json").write_text(json.dumps(stats, indent=2))
    n_fb = sum(1 for x in stats if x["fallback"])
    print(f"[csv] DONE {len(stats)} directions, {n_fb} used fallback padding", flush=True)


if __name__ == "__main__":
    main()
