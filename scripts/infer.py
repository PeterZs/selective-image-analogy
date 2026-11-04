#!/usr/bin/env python3
"""Single-demonstration selective analogy with every suppression subset, on FLUX.2-Klein or Qwen-Image-2.1.

Given A, A', B and the edits [e1, ..., eN] of A -> A', runs
  - AnaLoRA only (ExLoRA off): the reference strip's "analora" image;
  - ExLoRA suppressing every non-empty subset of the edits (rows), for each ExLoRA scale
    (--scale_values, columns) and, with CSV, each concept-suppression strength (--embedding_scales,
    or one time-varying --embedding_schedule).

Writes, next to --output: analora.png, one folder per row with every annotated cell and a row grid
(grid.png), and the combined grid at --output (one per CSV scope under scope_<scope>/ when several
scopes are swept). Every cell starts from the same seeded noise, so cells differ only by the method.

The backbone comes from --backbone, else from the config (`backbone`, or inferred from
`pretrained_model_name_or_path`). Defaults follow each backbone's inference recipe: FLUX.2-Klein 28
Euler steps without CFG; Qwen-Image-2.1 40 steps, no CFG, prefix KV cache on (a time-varying
--embedding_schedule changes the prompt embeddings every step, so it runs without the KV cache).

Usage:
  python scripts/infer.py \\
      --config configs/exlora_qwen_image21.yaml \\
      --checkpoint_dir selective-image-analogy/SIA-weights/qwen_image21/exlora \\
      --image_a A.png --image_a_prime A_prime.png --image_b B.png \\
      --edits "add sweater" "add santa hat" "add street background" \\
      --output results/result.png
"""
import argparse
import itertools
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from sia.concept_suppression import (
    STEERING_MODES,
    STEERING_SCOPES,
    alpha_at_step,
    apply_csv,
    format_schedule_tag,
    get_or_build_steering_vector,
    parse_embedding_schedule,
)
from sia.inference import BACKBONES, backbone_name, load_backbone
from sia.prompt import build_analogy_prompt
from sia.viz import annotate_image, build_combined_grid, build_joint_grid, build_row_grid

SCALE_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]  # overridden by --scale_values
DEFAULT_STEPS = {"flux2_klein": 28, "qwen_image21": 40}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",        type=str, required=True)
    parser.add_argument("--backbone", type=str, default=None, choices=BACKBONES,
                        help="Model family (default: from the config).")
    parser.add_argument("--image_a",       type=str, required=True)
    parser.add_argument("--image_a_prime", type=str, required=True)
    parser.add_argument("--image_b",       type=str, required=True)
    parser.add_argument("--edits", type=str, nargs="+", required=True,
                        help="Raw edit texts in order (e.g. 'change expression to laugh'). "
                             "The prompt is auto-built from these using the same format as training.")
    parser.add_argument("--scale_values", type=float, nargs="+",
                        default=None,
                        help="LoRA scale sweep values (default: 0 0.25 0.5 0.75 1).")
    parser.add_argument("--embedding_scales", type=float, nargs="+",
                        default=None,
                        help="Text-embedding steering alpha values, applied at suppressed "
                             "edits' token positions (Diffusion-Sliders-style). "
                             "alpha>0 amplifies the edit; alpha<0 suppresses. "
                             "Default [0.0] = pure LoRA interpolation (no embedding steering).")
    parser.add_argument("--embedding_schedule", type=str, nargs="+", default=None,
                        help="Piecewise-constant schedule for the embedding alpha "
                             "across denoising steps. Each entry is "
                             "'step_boundary:alpha'; for step index t the first "
                             "entry with t<boundary wins. Example: "
                             "'10:-40 20:-30 28:-20' uses alpha=-40 for steps 0-9, "
                             "-30 for 10-19, -20 for 20-27. "
                             "When set, overrides --embedding_scales and runs one "
                             "inference per (suppress-mode × exlora-scale). "
                             "The schedule can also be set in the YAML config as "
                             "'embedding_schedule: [[10, -40], [20, -30], [28, -20]]'.")
    parser.add_argument("--embedding_scale_factor", type=float, default=1.0,
                        help="Absolute magnitude multiplier on the unit-norm steering "
                             "direction. Since edit tokens in prompt_embeds can have norm "
                             "~10-50, alpha=1 corresponds to a unit-length push. Raise this "
                             "to amplify steering effect (typical: 5-30).")
    parser.add_argument("--csv_scope", type=str, nargs="+",
                        default=None, choices=list(STEERING_SCOPES),
                        help="Where to add the steering vector in the prompt "
                             "embedding: 'tokens' (only the suppressed edit's "
                             "own token positions — same footprint as ExLoRA), "
                             "or 'prompt' (every non-padding prompt token). "
                             "Pass multiple values to sweep both as an "
                             "ablation — each scope gets its own output "
                             "subdirectory. Default: 'tokens'. "
                             "Can also be set in YAML as "
                             "'csv_scope: \"tokens\"' or "
                             "'csv_scope: [\"tokens\", \"prompt\"]'.")
    parser.add_argument("--csv_mode", type=str, default=None,
                        choices=list(STEERING_MODES),
                        help="How the steering vector modifies the prompt "
                             "embedding at target positions. "
                             "'add' (default): x' = x + alpha*v_hat — pure "
                             "additive push (original behaviour). "
                             "'ablate': x' = x - (x.v_hat)*v_hat + alpha*v_hat "
                             "— directional ablation (project the concept "
                             "component OUT of x) then optional push. Usually "
                             "much stronger for suppression at the same |alpha|. "
                             "With alpha=0, 'ablate' performs pure concept "
                             "removal. Can also be set in YAML as "
                             "'csv_mode: \"ablate\"'.")
    parser.add_argument("--num_steering_pairs", type=int, default=32,
                        help="Number of contrastive pos/neg sentence pairs per edit that "
                             "the text encoder generates (Diffusion-Sliders paper uses 100 via GPT-4o; "
                             "we default to 32 with the local Qwen to keep runtime modest).")
    parser.add_argument("--steering_cache_dir", type=str, default=None,
                        help="Directory in which to cache generated pairs.jsonl and "
                             "vector.pt per edit (keyed by sha1(edit)); "
                             "default ~/.cache/sia/csv/<backbone>.")
    parser.add_argument("--regen_steering", action="store_true",
                        help="Regenerate the contrastive pairs + DoM vector even if a cached "
                             "vector.pt exists for the edit.")
    parser.add_argument("--max_push_norm_ratio", type=float, default=0.0,
                        help="Clamp the steering push magnitude to this multiple of the "
                             "per-token embedding norm (before modification). Prevents "
                             "extreme out-of-distribution embeddings. 0 = no clamp (default). "
                             "Typical safe value: 1.0-3.0.")
    parser.add_argument("--incontext_steering", action="store_true",
                        help="Compute the DoM steering direction inside the actual inference "
                             "prompt (in-context) instead of from isolated short sentences. "
                             "Fixes the causal-LM context mismatch: the direction is extracted "
                             "in the same representation space where it is applied. Requires "
                             "1+8 text-encoder forward passes per edit (fast) and no Qwen "
                             "generation.")
    parser.add_argument("--output",         type=str, default="result.png")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Stage-2 checkpoint to load ExLoRA/AnaLoRA from: a local dir or a Hub path "
                             "<org>/<repo>/<subfolder>[@revision]. Overrides the auto-discovery "
                             "from output_dir in the config.")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Sampling steps (default: 28 for FLUX.2-Klein, 40 for Qwen-Image-2.1).")
    parser.add_argument("--cell_size", type=int, default=512)
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--resolution", type=int, default=None,
                        help="Qwen-Image-2.1: target-area side (default: 'resolution' from the config, else 512).")
    parser.add_argument("--no_kv_cache", action="store_true",
                        help="Qwen-Image-2.1: disable the prefix KV cache (QwenImage21Pipeline use_kv_cache=False).")
    return parser


def parse_args(cli=None):
    """Build the run config from ``cli`` (parsed ``build_parser()`` args;
    parsed from sys.argv when omitted) merged into the YAML config."""
    if cli is None:
        cli = build_parser().parse_args()

    cfg = OmegaConf.load(cli.config)
    cfg.backbone = cli.backbone or backbone_name(cfg)
    cfg.num_steps  = cli.num_steps or DEFAULT_STEPS[cfg.backbone]
    cfg.cell_size  = cli.cell_size
    cfg.seed       = cli.seed
    cfg.image_a         = cli.image_a
    cfg.image_a_prime   = cli.image_a_prime
    cfg.image_b         = cli.image_b
    cfg.edits           = cli.edits
    # Build the prompt the same way training does — guarantees edits are exact substrings
    cfg.prompt          = build_analogy_prompt(cli.edits, cli.edits)
    cfg.output          = cli.output
    cfg.checkpoint_dir  = cli.checkpoint_dir
    cfg.scale_values    = cli.scale_values if cli.scale_values is not None else SCALE_VALUES
    cfg.embedding_scales = (
        cli.embedding_scales if cli.embedding_scales is not None else [0.0])
    cfg.embedding_scale_factor = cli.embedding_scale_factor
    cfg.num_steering_pairs     = cli.num_steering_pairs
    cfg.steering_cache_dir     = cli.steering_cache_dir
    cfg.regen_steering         = cli.regen_steering
    cfg.incontext_steering     = cli.incontext_steering
    cfg.max_push_norm_ratio    = cli.max_push_norm_ratio
    if cfg.backbone == "qwen_image21":
        cfg.resolution = cli.resolution or cfg.get("resolution", 512)
        cfg.use_kv_cache = not cli.no_kv_cache
    elif cli.resolution is not None or cli.no_kv_cache:
        raise SystemExit("--resolution and --no_kv_cache are Qwen-Image-2.1 options "
                         "(FLUX.2-Klein encodes the images at its training size)")

    # Schedule: CLI overrides YAML. Store as a plain list of (boundary, alpha)
    # tuples on cfg.embedding_schedule_parsed; leave the raw form on
    # cfg.embedding_schedule for traceability.
    raw_sched = cli.embedding_schedule
    if raw_sched is None:
        raw_sched = OmegaConf.to_container(cfg.get("embedding_schedule"), resolve=True) \
            if "embedding_schedule" in cfg else None
    cfg.embedding_schedule_parsed = parse_embedding_schedule(raw_sched)

    # Steering scope: CLI overrides YAML. Normalize to a list of valid choices.
    if cli.csv_scope is not None:
        raw_scopes = cli.csv_scope
    elif "csv_scope" in cfg:
        raw_scopes = OmegaConf.to_container(cfg.csv_scope, resolve=True)
    else:
        raw_scopes = ["tokens"]
    if isinstance(raw_scopes, str):
        raw_scopes = [raw_scopes]
    scopes = []
    for s in raw_scopes:
        s = str(s).strip().lower()
        if s not in STEERING_SCOPES:
            raise ValueError(
                f"csv_scope entry {s!r} invalid; "
                f"expected one of {STEERING_SCOPES}.")
        if s not in scopes:   # dedupe, preserve order
            scopes.append(s)
    cfg.csv_scopes = scopes

    # Steering mode (single value): CLI overrides YAML, default "add".
    if cli.csv_mode is not None:
        mode = cli.csv_mode
    elif "csv_mode" in cfg:
        mode = str(cfg.csv_mode)
    else:
        mode = "add"
    mode = str(mode).strip().lower()
    if mode not in STEERING_MODES:
        raise ValueError(f"csv_mode={mode!r} invalid; "
                         f"expected one of {STEERING_MODES}.")
    cfg.csv_mode = mode
    return cfg


# ---------------------------------------------------------------------------
# Backbone-specific setup: load the model, encode the prompt, render the AnaLoRA reference and
# locate the edits. Each returns the pieces the shared sweep needs.
# ---------------------------------------------------------------------------

def setup_flux2_klein(args, device, image_a, image_a_prime, image_b, out_dir):
    """FLUX.2-Klein: images are encoded at the training size; token positions index the
    chat-templated prompt that the Qwen3 text encoder sees."""
    klein = load_backbone("flux2_klein")
    pipe, tokenizer, text_encoder = klein.load_pipeline(args, device)
    base_prompt_embeds, _ = klein.compute_text_embeddings([args.prompt], tokenizer, text_encoder)
    max_seq_len = base_prompt_embeds.shape[1]

    def generate(prompt_embeds, prompt_embeds_schedule, suppress, exlora_scale):
        # Freshly seeded generator per cell so the initial noise is identical across the sweep.
        generator = torch.Generator(device=device).manual_seed(int(args.seed))
        pred = klein.run_inference(
            pipe, tokenizer, text_encoder, image_a, image_a_prime, image_b, args.prompt, suppress,
            args.num_steps, device, analora_scale=1.0, exlora_scale=exlora_scale,
            prompt_embeds_override=prompt_embeds, prompt_embeds_schedule=prompt_embeds_schedule,
            generator=generator)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return pred

    print("[analora] Running AnaLoRA-only inference for the reference strip …")
    analora_image = generate(base_prompt_embeds, None, [], 0.0)
    analora_image.save(out_dir / "analora.png")

    # Positions index ``base_prompt_embeds`` (computed on the chat-templated sequence).
    edit_positions = []
    templated_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    templated_ids = tokenizer(templated_prompt, max_length=max_seq_len, padding="max_length",
                              truncation=True, return_tensors="pt")["input_ids"][0].tolist()
    for i, edit_text in enumerate(args.edits):
        pos = klein.templated_substring_indices(args.prompt, edit_text, tokenizer, max_length=max_seq_len)
        edit_positions.append(pos)
        if pos:
            # Decode the tokens at those positions: a mismatch means the alignment silently failed.
            decoded = tokenizer.decode([templated_ids[p] for p in pos], skip_special_tokens=False)
            print(f"[Steering] edit_{i+1} positions[{len(pos)}]: {pos[0]}..{pos[-1]}  decoded={decoded!r}")
        else:
            print(f"[Steering] edit_{i+1} ({edit_text[:50]}): "
                  "NO tokens aligned in templated prompt — steering disabled for this edit.")

    def prompt_positions():
        return klein.klein_templated_prompt_token_positions(args.prompt, tokenizer, max_length=max_seq_len)

    return dict(backbone=klein, pipe=pipe, generate=generate, base_prompt_embeds=base_prompt_embeds,
                edit_positions=edit_positions, prompt_positions=prompt_positions, vl_images=None,
                n_schedule_steps=int(args.num_steps), analora_image=analora_image)


def setup_qwen_image21(args, device, image_a, image_a_prime, image_b, out_dir):
    """Qwen-Image-2.1: the condition images and the unsteered embeddings are shared by every cell;
    token rows index ``prompt_embeds`` (the condition images' vision tokens included)."""
    qwen = load_backbone("qwen_image21")
    pipe = qwen.load_pipeline(args, device)
    cond = qwen.prepare_conditions(pipe, [image_a, image_a_prime, image_b], args.resolution, device)
    base_prompt_embeds, image_pad_mask = qwen.encode_prompt(pipe, args.prompt, cond, device)
    print(f"[Qwen-Image-2.1] conditions {cond.image_sizes}, output {cond.target_size}, "
          f"prompt_embeds {tuple(base_prompt_embeds.shape)}")

    def generate(prompt_embeds, prompt_embeds_schedule, suppress, exlora_scale):
        # Freshly seeded generator per cell so the initial noise is identical across the sweep.
        generator = torch.Generator(device=device).manual_seed(int(args.seed))
        return qwen.run_inference(
            pipe, cond, image_b, args.prompt,
            prompt_embeds if prompt_embeds is not None else base_prompt_embeds, image_pad_mask,
            suppress, args.num_steps, exlora_scale=exlora_scale,
            prompt_embeds_schedule=prompt_embeds_schedule, generator=generator,
            use_kv_cache=args.use_kv_cache)

    print("[analora] Running AnaLoRA-only inference for the reference strip …")
    analora_image = generate(base_prompt_embeds, None, [], 0.0)
    analora_image.save(out_dir / "analora.png")

    edit_positions = []
    for i, edit_text in enumerate(args.edits):
        pos = qwen.qwen21_find_substring_token_indices(pipe, args.prompt, edit_text, cond.image_sizes)
        edit_positions.append(pos)
        print(f"[Steering] edit_{i+1} rows {pos[0]}..{pos[-1]} ({len(pos)} tokens)")

    def prompt_positions():
        return qwen.qwen21_prompt_token_positions(pipe, args.prompt, cond.image_sizes)

    # The schedule has one entry per sampling step of the pipeline's sigma grid.
    n_schedule_steps = len(pipe.config.sample_sigmas or range(int(args.num_steps)))
    return dict(backbone=qwen, pipe=pipe, generate=generate, base_prompt_embeds=base_prompt_embeds,
                edit_positions=edit_positions, prompt_positions=prompt_positions, vl_images=cond.vl_images,
                n_schedule_steps=n_schedule_steps, analora_image=analora_image)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def steering_vectors(args, run, device, any_alpha):
    """CSV direction of every edit (``None`` for an edit without token positions or without CSV)."""
    base_prompt_embeds = run["base_prompt_embeds"]
    steering_vecs = [None] * len(args.edits)
    if not any_alpha:
        print("[Steering DoM] All embedding_scales == 0 and mode='add' — "
              "skipping DoM vector computation (pure LoRA mode).")
        return steering_vecs
    if args.get("incontext_steering", False):
        print("[Steering DoM] In-context mode enabled — DoM will be computed "
              "inside the full inference prompt (causal-LM-aware).")
    hidden_dim = base_prompt_embeds.shape[-1]
    for i, edit_text in enumerate(args.edits):
        if not run["edit_positions"][i]:
            print(f"[Steering DoM] Skipping edit_{i+1} — no positions in prompt.")
            continue
        incontext = {}
        if args.get("incontext_steering", False):
            incontext = dict(incontext_prompt=args.prompt, all_edits=list(args.edits), edit_index=i,
                             build_prompt_fn=build_analogy_prompt)
            if run["vl_images"] is not None:
                incontext["vl_images"] = run["vl_images"]
        v, meta = get_or_build_steering_vector(
            run["backbone"], edit_text, int(args.num_steering_pairs), run["pipe"], device,
            cache_dir=args.steering_cache_dir, regen=bool(args.regen_steering), **incontext)
        if v.shape[-1] != hidden_dim:
            raise RuntimeError(f"Steering vector dim {v.shape[-1]} != prompt_embeds dim {hidden_dim} for "
                               f"edit_{i+1}. Cache from a different encoder config? Pass --regen_steering.")
        steering_vecs[i] = v.to(device=device, dtype=base_prompt_embeds.dtype)
        print(f"[Steering DoM] edit_{i+1} ({edit_text[:50]}): source={meta.get('source', '?')} "
              f"|v|_pre={meta.get('pre_normalization_norm')} pooled_pairs={meta.get('n_usable_after_pooling', '?')} "
              f"dim={v.shape[-1]}")
    return steering_vecs


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    image_a = Image.open(args.image_a).convert("RGB")
    image_a_prime = Image.open(args.image_a_prime).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading models ({args.backbone}) …")
    setup = setup_flux2_klein if args.backbone == "flux2_klein" else setup_qwen_image21
    run = setup(args, device, image_a, image_a_prime, image_b, out_dir)
    base_prompt_embeds, edit_positions, analora_image = (
        run["base_prompt_embeds"], run["edit_positions"], run["analora_image"])

    schedule = args.get("embedding_schedule_parsed", None)
    if schedule:
        print(f"[Steering schedule] Using time-varying alpha schedule: {[(b, a) for b, a in schedule]} "
              f"(overrides --embedding_scales)")
    # Directions are needed whenever steering happens: in additive mode only for a non-zero alpha,
    # in ablate mode always (alpha = 0 is pure concept removal).
    mode = str(args.csv_mode)
    any_alpha = (mode == "ablate"
                 or any(abs(float(a)) > 0 for a in args.embedding_scales)
                 or (schedule is not None and any(abs(a) > 0 for _, a in schedule)))
    steering_vecs = steering_vectors(args, run, device, any_alpha)

    # Rows: every non-empty suppression subset ("no supp" does not demonstrate selective suppression).
    n = len(args.edits)
    row_descs = []
    for r in range(1, n + 1):
        for indices in itertools.combinations(range(n), r):
            label = ("exlora (supp all)" if r == n and n > 1
                     else "exlora (supp " + "+".join(f"e{i+1}" for i in indices) + ")")
            row_descs.append((label, [args.edits[i] for i in indices], list(indices)))

    # With a schedule the per-step alpha comes from the schedule; the embedding axis collapses to one
    # placeholder value so the grid layout (a scalar embedding axis) still works.
    emb_scales = [0.0] if schedule else list(args.embedding_scales)
    sched_tag = format_schedule_tag(schedule) if schedule else None
    st_scales = list(args.scale_values)

    # "tokens" touches only the suppressed edit's own tokens (the ExLoRA footprint); "prompt" every
    # non-padding prompt token (against concept residue spread by the encoder's self-attention).
    scopes = list(args.csv_scopes)
    all_prompt_positions = run["prompt_positions"]() if "prompt" in scopes else None
    print(f"[Steering] Scope ablation: {scopes}  Mode: {mode}")

    output_path = Path(args.output)
    total_runs = len(scopes) * len(row_descs) * len(st_scales) * len(emb_scales)
    run_idx = 0
    for scope in scopes:
        scope_out_dir = out_dir / f"scope_{scope}" if len(scopes) > 1 else out_dir
        scope_out_dir.mkdir(parents=True, exist_ok=True)
        scope_output = scope_out_dir / output_path.name
        all_preds = {}   # (row_label, st_s, emb_s) -> PIL.Image

        def steer(alpha, suppress_idx):
            # None for non-suppressed edits: leave their concept untouched even in 'ablate'.
            alphas = [(alpha * args.embedding_scale_factor) if i in suppress_idx else None for i in range(n)]
            return apply_csv(base_prompt_embeds, edit_positions, steering_vecs, alphas,
                             scope=scope, all_prompt_positions=all_prompt_positions, mode=mode,
                             max_push_norm_ratio=float(args.max_push_norm_ratio))

        for row_label, suppress, suppress_idx in row_descs:
            row_dir = scope_out_dir / row_label.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
            row_dir.mkdir(parents=True, exist_ok=True)
            row_preds = {}
            for emb_s in emb_scales:
                if schedule:
                    embeds, embeds_schedule = None, [steer(alpha_at_step(schedule, s), suppress_idx)
                                                     for s in range(run["n_schedule_steps"])]
                    step_tag = sched_tag
                else:
                    embeds, embeds_schedule = steer(emb_s, suppress_idx), None
                    step_tag = f"emb={emb_s:.2g}"

                for st_s in st_scales:
                    run_idx += 1
                    print(f"[{run_idx}/{total_runs}] scope={scope}  {row_label}  suppress={suppress}  "
                          f"st={st_s}  {step_tag}")
                    pred = run["generate"](embeds, embeds_schedule, suppress, st_s)
                    all_preds[(row_label, st_s, emb_s)] = pred
                    row_preds[(st_s, emb_s)] = pred
                    fname = f"st{st_s:.2f}".replace(".", "p") + (
                        f"_{sched_tag}.png" if schedule else f"_emb{emb_s:.2f}".replace(".", "p") + ".png")
                    annotate_image(pred, f"{row_label}  st={st_s:.2g}  {step_tag}  scope={scope}",
                                   suppress).save(row_dir / fname)

            build_row_grid(image_a, image_a_prime, image_b, f"{row_label}  [scope={scope}]", suppress,
                           row_preds, st_scales, emb_scales, cell_size=args.cell_size,
                           analora_image=analora_image).save(row_dir / "grid.png")
            print(f"  Saved row grid → {row_dir / 'grid.png'}")

        if len(emb_scales) == 1:
            legacy = {(rl, s): all_preds[(rl, s, emb_scales[0])] for rl, _, _ in row_descs for s in st_scales}
            combined = build_combined_grid(image_a, image_a_prime, image_b,
                                           [(rl, sup) for rl, sup, _ in row_descs], legacy, st_scales,
                                           cell_size=args.cell_size, analora_image=analora_image)
        else:
            combined = build_joint_grid(image_a, image_a_prime, image_b, row_descs, all_preds,
                                        st_scales, emb_scales, cell_size=args.cell_size,
                                        analora_image=analora_image)
        combined.save(scope_output)
        print(f"[scope={scope}] Saved combined grid → {scope_output}")


if __name__ == "__main__":
    main()
