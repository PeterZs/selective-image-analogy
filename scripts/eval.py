#!/usr/bin/env python3
"""Benchmark evaluation of SIA checkpoints (paper Sec. 5.2), for FLUX.2-Klein and Qwen-Image-2.1.

The `analogy_benchmark` test split has 239 (A, A', B) cases without ground truth. For every case
and every non-empty proper subset of its edits (2^N - 2 per case) the method generates B' keeping
that subset; a VLM judge sees (A, A', B, output) with the retained / suppressed edit lists and scores
Retained Accuracy, Preservation, Suppression and Edit Recall; CLIP directional scores are computed
too. The `full_transfer` split (75 cases with ground truth B', nothing suppressed) is scored with
LPIPS, DINOv2 distance and CLIP-I.

Subcommands (each is resumable: existing outputs are skipped):

  generate  --name N | --arm NAME:METHOD[:VALUE] ...   write predicted images
  score     --names N1 N2 ... [--identity]             CLIP + VLM-judge scores
  report    --names N1 N2 ...                          tables, strata, ablation, paired CIs
  compare   --method N1 --baseline N2                  per-sample paired comparison, sign test

Layout under --out: <name>/<case id with / -> __>/keepXY.png (XY = 1-based kept edits),
<name>/full_transfer/<case>.png, judge.jsonl, clip.jsonl, full_transfer.jsonl, report*.md.

Weights (generate)
  --ckpt DIR      Stage 2 checkpoint: AnaLoRA + ExLoRA, ExLoRA gated on the suppressed edits' tokens
  --analora DIR   Stage 1 AnaLoRA only (no ExLoRA)
  Both accept a local directory or a Hub path <org>/<repo>/<subfolder>[@revision]. The backbone
  comes from --config (`backbone`, or inferred from `pretrained_model_name_or_path`).

Methods (generate). One process loads the weights once and writes one or more methods ("arms"):
  --name N        one arm, set by the flags: --csv 1 -> CSV (--csv_mode, --csv_k, --csv_alpha,
                  --csv_lambda); otherwise ExLoRA suppression with ExLoRA, prompt deletion without.
                  Qwen-Image-2.1 only: --ana 0 / --ex 0 leave that component of --ckpt out.
  --arm NAME:METHOD[:VALUE]   (repeatable)
     full         full prompt; the suppressed edits are gated by ExLoRA if it is loaded
     del          suppressed edits deleted from the prompt (without ExLoRA only)
     add:K        CSV additive form: x - K * gap * v at each suppressed edit's tokens
     abl:LAMBDA   CSV ablation: x - LAMBDA (x.v) v at each suppressed edit's tokens (1 = projection)
  Kept edits are never touched. Every image uses the same seed, so arms are paired.

Data (every subcommand)
  --split analogy_benchmark   the test benchmark (default)
  --split dev_train           development cases from training subjects (sia.data.dev_train_cases),
                              for choosing method settings without touching the test set

Usage:
  python scripts/eval.py generate --name s2_5000 --ckpt selective-image-analogy/SIA-weights/qwen_image21/exlora
  python scripts/eval.py generate --config configs/exlora_flux2_klein.yaml --split dev_train \\
      --ckpt selective-image-analogy/SIA-weights/flux2_klein/exlora --arm full:full --arm abl:abl:1
  python scripts/eval.py score  --names s2_5000 s2_5000_csv --identity
  python scripts/eval.py report --names s2_5000 s2_5000_csv
  python scripts/eval.py compare --method s2_5000_csv --baseline s2_5000
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from sia.adapters import inject_analora_modules, load_analora_from_peft_checkpoint
from sia.concept_suppression import (
    apply_csv,
    build_steering_vector_from_pairs,
    default_cache_dir,
    get_or_build_steering_vector,
    reusable_pairs,
)
from sia.data import DATASET, SPLITS, load_split, subsets_for
from sia.eval.judge import score_judge
from sia.eval.metrics import score_clip, score_full_transfer
from sia.eval.report import compare_runs, write_report
from sia.eval.results import full_path, keep_tag, pred_path
from sia.hf import resolve_checkpoint
from sia.inference import attach_adapters, backbone_name, encode_analogy, generate_selective, load_backbone
from sia.prompt import build_analogy_prompt

DEFAULT_CONFIG = "configs/exlora_qwen_image21.yaml"
DEFAULT_OUT = {"qwen_image21": "experiments/eval_qwen21", "flux2_klein": "experiments/eval_flux2_klein"}
DEFAULT_STEPS = {"qwen_image21": 40, "flux2_klein": 28}
ARM_METHODS = ("full", "del", "add", "abl")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def parse_arm(spec):
    """``NAME:full|del|add:K|abl:LAMBDA`` -> arm dict (see ``csv_alphas`` for the CSV fields)."""
    parts = spec.split(":")
    if len(parts) < 2 or parts[1] not in ARM_METHODS:
        raise argparse.ArgumentTypeError(f"bad --arm {spec!r}: expected NAME:full|del|add:K|abl:LAMBDA")
    name, method = parts[0], parts[1]
    if method in ("add", "abl") and len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--arm {spec!r}: {method} needs a value")
    arm = dict(name=name, method=method)
    if method == "add":
        arm.update(csv_k=float(parts[2]), csv_alpha=0.0, csv_lambda=1.0)
    elif method == "abl":
        arm.update(csv_k=None, csv_alpha=0.0, csv_lambda=float(parts[2]))
    return arm


def generation_arms(args, exlora):
    """The arms to generate: ``--arm`` as given, or the single arm ``--name`` described by the flags."""
    if bool(args.arm) == bool(args.name):
        raise SystemExit("give either --name (one arm set by the flags) or one or more --arm")
    if args.arm:
        arms = args.arm
    elif args.csv:
        arms = [dict(name=args.name, method="add" if args.csv_mode == "add" else "abl",
                     csv_k=args.csv_k, csv_alpha=args.csv_alpha, csv_lambda=args.csv_lambda)]
    else:
        # with neither ExLoRA nor CSV the suppressed edits are left out of the prompt
        arms = [dict(name=args.name, method="full" if exlora else "del")]
    names = [a["name"] for a in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate arm names: {names}")
    if exlora and any(a["method"] == "del" for a in arms):
        raise SystemExit("'del' arms need ExLoRA off (the ExLoRA mask needs the edit in the prompt)")
    return arms


def csv_alphas(arm, suppressed, gaps):
    """Per-edit CSV strengths of one arm (``None`` = edit untouched, i.e. every kept edit).

    With ``csv_k`` the additive (Diffusion Sliders) form pushes ``k`` mean-pole gaps toward the
    "absent" pole (alpha = -k * gap); otherwise the push after ablation is ``csv_alpha``.
    """
    return [(-arm["csv_k"] * gaps[p] if arm["csv_k"] else arm["csv_alpha"]) if p in suppressed else None
            for p in range(len(gaps))]


class QwenGenerator:
    """Qwen-Image-2.1 generation through ``sia.inference`` (the demo's code path).

    ``ana`` / ``ex`` select the components of the Stage-2 ``ckpt``; ``analora`` loads a Stage-1 PEFT
    AnaLoRA instead (ExLoRA off).
    """

    def __init__(self, cfg, args, device):
        self.qwen = load_backbone("qwen_image21")
        self.device = device
        self.resolution = cfg.get("resolution", 512)
        self.use_kv_cache = cfg.get("use_kv_cache", True)
        self.seed = args.seed
        rank = cfg.lora_rank
        if args.analora:
            self.ana, self.exlora = 1, False
        else:
            self.ana, self.exlora = args.ana, bool(args.ex)
            if (self.ana or self.exlora) and not args.ckpt:
                raise SystemExit("--ckpt is required when AnaLoRA or ExLoRA is on")
        self.pipe = self.qwen.load_pipeline_qwen21(cfg.pretrained_model_name_or_path)
        transformer = self.pipe.transformer
        if args.analora:
            inject_analora_modules(transformer, self.qwen.ANALORA_TARGET_MODULES, r=rank, alpha=rank, dropout=0.0)
            load_analora_from_peft_checkpoint(transformer, resolve_checkpoint(args.analora), device="cpu")
        elif self.ana or self.exlora:
            attach_adapters(self.qwen, transformer, resolve_checkpoint(args.ckpt), rank,
                            analora=self.ana, exlora=self.exlora)
        self.pipe.to(device)
        transformer.eval()

    def prepare(self, case, with_positions):
        """Condition images and full-prompt embeddings of a case (+ the edits' token rows for CSV)."""
        analogy = encode_analogy(self.pipe, case["a"], case["a_prime"], case["b"], case["edits"],
                                 self.resolution, self.device)
        positions = None
        if with_positions:
            positions = [self.qwen.qwen21_find_substring_token_indices(
                self.pipe, analogy.prompt, e, analogy.cond.image_sizes) for e in case["edits"]]
        return analogy, analogy.prompt_embeds, positions

    def generate(self, analogy, kept, arm, steered, steps):
        """One B' for an arm; ``steered`` are the CSV-steered embeddings of a CSV arm."""
        if arm["method"] == "full" and not self.exlora:
            steered = analogy.prompt_embeds          # full prompt, nothing suppressed
        return generate_selective(self.pipe, analogy, kept, steps, self.seed, exlora=self.exlora,
                                  prompt_embeds=steered, use_kv_cache=self.use_kv_cache)

    def generate_full(self, case, steps):
        """Full transfer: every edit applied, ExLoRA inactive (no mask)."""
        a, a_prime, b = self.qwen.harmonize_images_qwen21(case["a"], case["a_prime"], case["b"], case["b"])[:3]
        cond = self.qwen.prepare_conditions(self.pipe, [a, a_prime, b], self.resolution, self.device)
        prompt = build_analogy_prompt(case["edits"], case["edits"])
        embeds, pad = self.qwen.encode_prompt(self.pipe, prompt, cond, self.device)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        return self.qwen.generate_qwen21(self.pipe, cond, embeds, pad, steps, generator=generator,
                                         exlora_text_mask=None, use_kv_cache=self.use_kv_cache)


class KleinGenerator:
    """FLUX.2-Klein generation through ``sia.backbones.flux2_klein.run_inference``: no CFG, Euler
    steps, ExLoRA gated on the suppressed edits' templated-prompt tokens.

    ``ckpt`` loads a Stage-2 checkpoint (ExLoRA on); ``analora`` a Stage-1 AnaLoRA (ExLoRA off).
    """

    def __init__(self, cfg, args, device):
        self.klein = load_backbone("flux2_klein")
        self.device = device
        self.seed = args.seed
        if args.ana != 1 or args.ex != 1:
            raise SystemExit("--ana / --ex are Qwen-Image-2.1 only; use --analora for AnaLoRA without ExLoRA")
        if not (args.ckpt or args.analora):
            raise SystemExit("FLUX.2-Klein needs --ckpt or --analora")
        self.exlora = args.ckpt is not None
        if self.exlora:
            cfg.checkpoint_dir = args.ckpt
        else:
            cfg.analora_path = args.analora
            cfg.checkpoint_dir = None
            # load_pipeline falls back to the latest checkpoint in output_dir when checkpoint_dir is
            # unset: point it at a directory that does not exist so only the AnaLoRA is loaded.
            cfg.output_dir = "/nonexistent/sia_eval_no_checkpoint"
        self.pipe, self.tokenizer, self.text_encoder = self.klein.load_pipeline(cfg, device)

    def prepare(self, case, with_positions):
        """Full prompt and its embeddings (+ the edits' token rows for CSV)."""
        prompt = build_analogy_prompt(case["edits"], case["edits"])
        embeds, _ = self.klein.compute_text_embeddings([prompt], self.tokenizer, self.text_encoder)
        positions = None
        if with_positions:
            positions = [self.klein.klein_find_substring_token_indices(prompt, e, self.tokenizer,
                                                                       max_length=embeds.shape[1])
                         for e in case["edits"]]
        return (case, prompt, embeds), embeds, positions

    def generate(self, state, kept, arm, steered, steps):
        """One B' for an arm; ``steered`` are the CSV-steered embeddings of a CSV arm."""
        case, prompt, embeds = state
        edits = case["edits"]
        suppressed = [p for p in range(len(edits)) if p not in kept]
        if arm["method"] == "del":
            prompt = build_analogy_prompt(edits, edits, suppress_indices=suppressed)
            embeds = None                                # run_inference encodes the shorter prompt
        elif arm["method"] in ("add", "abl"):
            embeds = steered
        return self._run(case, prompt, [edits[p] for p in suppressed] if self.exlora else [], embeds, steps)

    def generate_full(self, case, steps):
        """Full transfer: every edit applied, nothing suppressed."""
        return self._run(case, build_analogy_prompt(case["edits"], case["edits"]), [], None, steps)

    def _run(self, case, prompt, suppress, embeds, steps):
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        return self.klein.run_inference(self.pipe, self.tokenizer, self.text_encoder,
                                        case["a"], case["a_prime"], case["b"], prompt, suppress, steps,
                                        self.device, analora_scale=1.0, exlora_scale=1.0 if self.exlora else 0.0,
                                        prompt_embeds_override=embeds, generator=generator)


def save_png(img, path):
    """Write ``img`` atomically, so a killed run never leaves a truncated PNG that would be skipped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def cmd_generate(args):
    cfg = OmegaConf.load(args.config)
    backbone = backbone_name(cfg)
    device = torch.device("cuda")
    steps = args.num_steps or cfg.get("num_val_inference_steps", DEFAULT_STEPS[backbone])
    out = Path(args.out or DEFAULT_OUT[backbone])
    if args.analora and args.ckpt:
        raise SystemExit("give --ckpt or --analora, not both")

    allowed = None
    if args.items_file:
        allowed = {(r["case"], r["keep"]) for r in json.loads(Path(args.items_file).read_text())}
        args.skip_full = True
        print(f"restricted to {len(allowed)} items from {args.items_file}", flush=True)
    cases = load_split(args.split, args.limit, args.dataset)[args.shard_index::args.num_shards]

    # ExLoRA is on with a Stage-2 checkpoint unless --ex 0 (Qwen-Image-2.1) leaves it out.
    arms = generation_arms(args, exlora=bool(args.ckpt and args.ex))
    csv_arms = [a for a in arms if a["method"] in ("add", "abl")]
    model = (QwenGenerator if backbone == "qwen_image21" else KleinGenerator)(cfg, args, device)
    weights = args.ckpt or args.analora
    print(f"[{backbone}] {len(cases)} {args.split} cases, weights {weights}, "
          f"ExLoRA {'on' if model.exlora else 'off'}, arms {[a['name'] for a in arms]}", flush=True)
    for arm in arms:
        (out / arm["name"]).mkdir(parents=True, exist_ok=True)
        (out / arm["name"] / "arm.json").write_text(json.dumps(
            dict(arm, backbone=backbone, weights=weights, exlora=model.exlora, seed=args.seed, num_steps=steps,
                 csv_pairs_from=args.csv_pairs_from if arm in csv_arms else None), indent=1))

    # CSV directions, built (or read from the cache) the first time an edit needs one.
    module = load_backbone(backbone)
    cache = Path(args.csv_cache).expanduser() if args.csv_cache else default_cache_dir(module)
    pairs_from = reusable_pairs(args.csv_pairs_from) if args.csv_pairs_from else {}
    directions = {}

    def direction(edit):
        if edit not in directions:
            if edit in pairs_from and not (cache / module.cache_key(edit) / "vector.pt").exists():
                v, meta = build_steering_vector_from_pairs(module, edit, pairs_from[edit], model.pipe, device, cache)
            else:
                v, meta = get_or_build_steering_vector(module, edit, args.csv_pairs, model.pipe, device, cache)
            # pre_normalization_norm = |mean(pos) - mean(neg)|: the concept's own scale (the gap of csv_k)
            directions[edit] = (v, meta["pre_normalization_norm"])
            print(f"[csv] {edit!r}: gap {directions[edit][1]:.2f} usable {meta.get('n_usable_after_pooling')} "
                  f"({meta.get('method', meta.get('source'))})", flush=True)
        return directions[edit]

    # ---- the split's cases: every retained subset ------------------------------------------
    t0, n_done = time.time(), 0
    for ci, case in enumerate(cases):
        edits = case["edits"]
        n = len(edits)
        jobs = [(k, a) for k in subsets_for(n) for a in arms if not pred_path(out, a["name"], case["id"], k).exists()]
        if allowed is not None:
            jobs = [(k, a) for k, a in jobs if (case["id"], keep_tag(k)) in allowed]
        if not jobs:
            continue
        state, embeds, positions = model.prepare(case, with_positions=bool(csv_arms))
        if csv_arms:
            vecs = [direction(e)[0].to(device=device, dtype=embeds.dtype) for e in edits]
            gaps = [direction(e)[1] for e in edits]
        for kept, arm in jobs:
            steered = None
            if arm in csv_arms:
                suppressed = [p for p in range(n) if p not in kept]
                steered = apply_csv(embeds, positions, vecs, csv_alphas(arm, suppressed, gaps), scope="tokens",
                                    mode="add" if arm["method"] == "add" else "ablate",
                                    ablate_scale=arm["csv_lambda"])
            save_png(model.generate(state, kept, arm, steered, steps), pred_path(out, arm["name"], case["id"], kept))
            n_done += 1
        print(f"case {ci + 1}/{len(cases)} {case['id']}  {n_done} new images, "
              f"{(time.time() - t0) / max(n_done, 1):.1f} s/img", flush=True)

    # ---- full_transfer: nothing suppressed (CSV never acts), scored against ground truth ----
    full_arms = [a for a in arms if a not in csv_arms]
    if not args.skip_full and full_arms:
        for case in load_split("full_transfer", args.limit, args.dataset)[args.shard_index::args.num_shards]:
            for arm in full_arms:
                path = full_path(out, arm["name"], case["id"])
                if not path.exists():
                    save_png(model.generate_full(case, steps), path)
        print("full_transfer done", flush=True)
    print(f"DONE {n_done} new images in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def cmd_score(args):
    cases = load_split(args.split, args.limit, args.dataset)[args.shard_index::args.num_shards]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if "clip" in args.stages:
        score_clip(args, cases)
        if args.names:
            score_full_transfer(args)
    if "judge" in args.stages:
        score_judge(args, cases)


# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--split", default="analogy_benchmark", choices=SPLITS,
                      help="cases to evaluate: the test benchmark, or dev_train (training subjects, "
                           "for choosing method settings without touching the test set)")
    data.add_argument("--dataset", default=DATASET, help="Hugging Face dataset id")
    shards = argparse.ArgumentParser(add_help=False)
    shards.add_argument("--shard_index", type=int, default=0, help="this worker's slice of the cases (parallel workers)")
    shards.add_argument("--num_shards", type=int, default=1)

    g = sub.add_parser("generate", parents=[data, shards], help="write predicted images")
    g.add_argument("--name", help="method name of a single arm set by the flags below")
    g.add_argument("--arm", action="append", type=parse_arm, default=[],
                   help="NAME:full|del|add:K|abl:LAMBDA (repeatable; instead of --name)")
    g.add_argument("--ckpt", help="Stage 2 checkpoint (weights for AnaLoRA / ExLoRA): local dir or Hub path "
                                  "<org>/<repo>/<subfolder>[@revision]")
    g.add_argument("--analora", help="Stage 1 PEFT AnaLoRA (local dir or Hub path), ExLoRA off")
    g.add_argument("--ana", type=int, default=1, choices=[0, 1], help="AnaLoRA from --ckpt (Qwen-Image-2.1)")
    g.add_argument("--ex", type=int, default=1, choices=[0, 1], help="ExLoRA from --ckpt (Qwen-Image-2.1)")
    g.add_argument("--csv", type=int, default=0, choices=[0, 1], help="CSV on the suppressed edits' tokens")
    g.add_argument("--csv_mode", default="ablate", choices=["add", "ablate"])
    g.add_argument("--csv_lambda", type=float, default=1.0,
                   help="paper's lambda: x - lambda (x.v) v at suppressed tokens; 1 = orthogonal projection")
    g.add_argument("--csv_k", type=float, default=0.0,
                   help="push the suppressed tokens by -k times the edit's mean-pole gap along d")
    g.add_argument("--csv_alpha", type=float, default=0.0, help="push after ablation; 0 = pure concept removal")
    g.add_argument("--csv_pairs", type=int, default=64, help="contrast pairs per edit when a direction is built")
    g.add_argument("--csv_cache", default=None, help="CSV cache directory (default ~/.cache/sia/csv/<backbone>)")
    g.add_argument("--csv_pairs_from", default=None,
                   help="CSV cache of another backbone whose pairs.jsonl (matched by edit text) are re-encoded "
                        "with this backbone's encoder instead of sampling new pairs")
    g.add_argument("--items_file", help="JSON list of {case, keep}: generate only these items")
    g.add_argument("--config", default=DEFAULT_CONFIG, help="model config (backbone, base model, lora_rank)")
    g.add_argument("--out", default=None, help="output root (default experiments/eval_<backbone>)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--num_steps", type=int, default=None,
                   help="sampling steps (default: num_val_inference_steps of the config; 40 Qwen, 28 Klein)")
    g.add_argument("--limit", type=int, help="first N cases only (smoke test)")
    g.add_argument("--skip_full", action="store_true", help="skip the full_transfer split")

    s = sub.add_parser("score", parents=[data, shards], help="CLIP and VLM-judge scores")
    s.add_argument("--names", nargs="*", default=[])
    s.add_argument("--identity", action="store_true", help="also score the no-edit control (output = B)")
    s.add_argument("--stages", nargs="+", default=["clip", "judge"], choices=["clip", "judge"])
    s.add_argument("--out", default=DEFAULT_OUT["qwen_image21"])
    s.add_argument("--limit", type=int)
    s.add_argument("--limit_items", type=int)
    s.add_argument("--judge_batch", type=int, default=8)

    r = sub.add_parser("report", parents=[data, shards], help="markdown report of scored runs")
    r.add_argument("--names", nargs="+", required=True)
    r.add_argument("--identity", action="store_true")
    r.add_argument("--out", default=DEFAULT_OUT["qwen_image21"])
    r.add_argument("--report_name", default="report.md")
    r.add_argument("--items_file", help="JSON list of {case, keep}: report only these items")
    r.add_argument("--note", default="", help="text added under the report title")

    c = sub.add_parser("compare", parents=[data], help="per-sample paired comparison of two scored runs")
    c.add_argument("--method", required=True, help="run to evaluate (e.g. a CSV run)")
    c.add_argument("--baseline", required=True, help="run it is compared with (same items, same seed)")
    c.add_argument("--out", default=DEFAULT_OUT["qwen_image21"])
    c.add_argument("--items_out", default="better_items.json",
                   help="file (under --out) listing the samples where --method scored higher")
    c.add_argument("--report_name", default="compare.md")
    return ap


def main():
    args = build_parser().parse_args()
    {"generate": cmd_generate, "score": cmd_score, "report": write_report, "compare": compare_runs}[args.cmd](args)


if __name__ == "__main__":
    main()
