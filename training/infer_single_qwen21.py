#!/usr/bin/env python3
"""
infer_single_qwen21.py

Single-image selective analogy on Qwen-Image-2.1, with every suppression mode.
Same CLI, sweeps and outputs as infer_single.py (FLUX.2-Klein), plus:

  --resolution     target area side, as in training (QwenImage21Pipeline rule)
  --no_kv_cache    disable the prefix KV cache (on by default, as in the pipeline)

Defaults follow QwenImage21Pipeline: 40 steps, no CFG, KV cache on.
A time-varying --embedding_schedule changes the prompt embeddings every step,
so it runs without the KV cache.

Usage:
  python training/infer_single_qwen21.py \\
      --config        training/configs/train_exlora_qwen21.yaml \\
      --image_a       /path/to/A.png \\
      --image_a_prime /path/to/A_prime.png \\
      --image_b       /path/to/B.png \\
      --edits "X" "Y" "Z" \\
      --output results/result.png
"""

import itertools
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.train_exlora_qwen21 import (  # noqa: E402
    ANALORA_TARGET_MODULES,
    EXLORA_TARGET_MODULES,
    encode_prompt,
    generate_qwen21,
    load_pipeline_qwen21,
    prepare_conditions,
)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from training.infer_single import (  # noqa: E402
    alpha_at_step,
    annotate_image,
    apply_csv,
    build_combined_grid,
    build_joint_grid,
    build_parser,
    build_row_grid,
    format_schedule_tag,
    parse_args as parse_args_klein,
)
from training.steering_dom_qwen21 import get_or_build_steering_vector_qwen21  # noqa: E402
from sia.adapters import (  # noqa: E402
    checkpoint_files,
    checkpoint_lora_rank,
    find_latest_checkpoint,
    inject_analora_modules,
    inject_exlora_modules,
    load_analora_from_peft_checkpoint,
    load_analora_state_dict,
    load_exlora_state_dict,
    load_tensors,
    lora_mode_ctx,
)
from sia.hf import resolve_checkpoint  # noqa: E402
from sia.prompt import build_analogy_prompt  # noqa: E402
from sia.utils.qwen21 import (  # noqa: E402
    qwen21_find_substring_token_indices,
    qwen21_prompt_token_positions,
    qwen21_text_token_mask,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pipeline(args, device):
    """Qwen-Image-2.1 pipeline with AnaLoRA + ExLoRA loaded — same checkpoint
    resolution rules as infer_single.load_pipeline."""
    pipe = load_pipeline_qwen21(args.pretrained_model_name_or_path)
    transformer = pipe.transformer

    explicit_ckpt = getattr(args, "checkpoint_dir", None)
    ckpt_dir = resolve_checkpoint(explicit_ckpt) if explicit_ckpt else find_latest_checkpoint(args.output_dir)

    # A Stage-2 checkpoint carries its own AnaLoRA; take the rank from it.
    analora_file, exlora_file = checkpoint_files(ckpt_dir) if ckpt_dir else (None, None)
    has_ckpt_base = analora_file is not None and analora_file.exists()
    if has_ckpt_base:
        rank = checkpoint_lora_rank(ckpt_dir, load_tensors(analora_file))
        if rank != args.lora_rank:
            print(f"[Checkpoint] lora_rank {args.lora_rank} in config -> {rank} from checkpoint")
            args.lora_rank = rank

    n_base = inject_analora_modules(transformer, ANALORA_TARGET_MODULES,
                                    r=args.lora_rank, alpha=args.lora_rank, dropout=0.0)
    print(f"[AnaLoRA] Injected {len(n_base)} modules")
    if not has_ckpt_base:
        load_analora_from_peft_checkpoint(transformer, resolve_checkpoint(args.analora_path), device="cpu")

    n_st = inject_exlora_modules(transformer, EXLORA_TARGET_MODULES,
                                 r=args.lora_rank, alpha=args.lora_rank, dropout=0.0)
    print(f"[ExLoRA] Injected {len(n_st)} modules")

    if ckpt_dir:
        print(f"[Checkpoint] Loading from {ckpt_dir}")
        if exlora_file.exists():
            load_exlora_state_dict(transformer, exlora_file)
        elif explicit_ckpt:
            raise FileNotFoundError(
                f"{exlora_file.name} missing from {ckpt_dir} — without it ExLoRA is "
                f"zero-initialised and has no effect.")
        if has_ckpt_base:
            load_analora_state_dict(transformer, analora_file, "cpu")
            print("[Checkpoint] AnaLoRA loaded from checkpoint")
    else:
        print("[Checkpoint] No checkpoint found — using freshly initialised ExLoRA")

    pipe.to(device)
    transformer.eval()
    return pipe


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(pipe, cond, image_b, prompt, prompt_embeds, image_pad_mask,
                  texts_to_suppress, num_steps,
                  analora_scale=1.0, exlora_scale=1.0,
                  prompt_embeds_schedule=None, generator=None, use_kv_cache=True):
    """One B' sample. ``texts_to_suppress`` gate ExLoRA (mask built even at
    scale 0, as in infer_single.run_inference); ``prompt_embeds`` may be
    CSV-steered. The output is resized back to B's original size."""
    token_mask = None
    if texts_to_suppress:
        token_mask = qwen21_text_token_mask(pipe, prompt, texts_to_suppress, cond.image_sizes,
                                            prompt_embeds.shape[1], prompt_embeds.device)
    with lora_mode_ctx(pipe.transformer, analora_scale, exlora_scale):
        out = generate_qwen21(pipe, cond, prompt_embeds, image_pad_mask, num_steps,
                              generator=generator, exlora_text_mask=token_mask,
                              use_kv_cache=use_kv_cache,
                              prompt_embeds_schedule=prompt_embeds_schedule)
    return out.resize(image_b.size, Image.LANCZOS)


def parse_args():
    parser = build_parser()
    parser.set_defaults(num_steps=40, steering_cache_dir="./steering_cache_qwen21")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Target-area side (default: 'resolution' from the config, else 512).")
    parser.add_argument("--no_kv_cache", action="store_true",
                        help="Disable the prefix KV cache (QwenImage21Pipeline use_kv_cache=False).")
    cli = parser.parse_args()
    cfg = parse_args_klein(cli)
    cfg.resolution = cli.resolution or cfg.get("resolution", 512)
    cfg.use_kv_cache = not cli.no_kv_cache
    return cfg


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    image_a = Image.open(args.image_a).convert("RGB")
    image_a_prime = Image.open(args.image_a_prime).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")

    print("Loading models …")
    pipe = load_pipeline(args, device)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Conditions and the unsteered embeddings are shared by every cell.
    cond = prepare_conditions(pipe, [image_a, image_a_prime, image_b], args.resolution, device)
    base_prompt_embeds, image_pad_mask = encode_prompt(pipe, args.prompt, cond, device)
    print(f"[Qwen-Image-2.1] conditions {cond.image_sizes}, output {cond.target_size}, "
          f"prompt_embeds {tuple(base_prompt_embeds.shape)}")

    def seeded():
        return torch.Generator(device=device).manual_seed(int(args.seed))

    print("[analora] Running AnaLoRA-only inference for the reference strip …")
    analora_image = run_inference(
        pipe, cond, image_b, args.prompt, base_prompt_embeds, image_pad_mask, [],
        args.num_steps, exlora_scale=0.0, generator=seeded(), use_kv_cache=args.use_kv_cache)
    analora_image.save(out_dir / "analora.png")

    schedule = args.get("embedding_schedule_parsed", None)
    if schedule:
        print(f"[Steering schedule] {schedule} (overrides --embedding_scales; KV cache off)")

    # ---- Per-edit rows of prompt_embeds (shared by ExLoRA mask and CSV) ----
    edit_positions = []
    for i, edit_text in enumerate(args.edits):
        pos = qwen21_find_substring_token_indices(pipe, args.prompt, edit_text, cond.image_sizes)
        edit_positions.append(pos)
        print(f"[Steering] edit_{i+1} rows {pos[0]}..{pos[-1]} ({len(pos)} tokens)")

    mode = str(args.csv_mode)
    any_alpha = (mode == "ablate"
                 or any(abs(float(a)) > 0 for a in args.embedding_scales)
                 or (schedule is not None and any(abs(a) > 0 for _, a in schedule)))
    steering_vecs = [None] * len(args.edits)
    if any_alpha:
        cache_root = Path(args.steering_cache_dir).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        for i, edit_text in enumerate(args.edits):
            incontext = {}
            if args.get("incontext_steering", False):
                incontext = dict(incontext_prompt=args.prompt, all_edits=list(args.edits),
                                 edit_index=i, build_prompt_fn=build_analogy_prompt,
                                 vl_images=cond.vl_images)
            v, meta = get_or_build_steering_vector_qwen21(
                edit_text, int(args.num_steering_pairs), pipe, device, cache_root,
                regen=bool(args.regen_steering), **incontext)
            if v.shape[-1] != base_prompt_embeds.shape[-1]:
                raise RuntimeError(f"Steering vector dim {v.shape[-1]} != prompt_embeds dim "
                                   f"{base_prompt_embeds.shape[-1]}. Pass --regen_steering.")
            steering_vecs[i] = v.to(device=device, dtype=base_prompt_embeds.dtype)
            print(f"[Steering DoM] edit_{i+1}: source={meta.get('source')} "
                  f"pooled={meta.get('n_usable_after_pooling')}")
    else:
        print("[Steering DoM] All embedding_scales == 0 and mode='add' — pure LoRA mode.")

    # ---- Rows: every non-empty suppression subset ----
    n = len(args.edits)
    row_descs = []
    for r in range(1, n + 1):
        for indices in itertools.combinations(range(n), r):
            label = ("exlora (supp all)" if r == n and n > 1
                     else "exlora (supp " + "+".join(f"e{i+1}" for i in indices) + ")")
            row_descs.append((label, [args.edits[i] for i in indices], list(indices)))

    emb_scales = [0.0] if schedule else list(args.embedding_scales)
    sched_tag = format_schedule_tag(schedule) if schedule else None
    st_scales = list(args.scale_values)
    scopes = list(args.csv_scopes)
    all_prompt_positions = None
    if "prompt" in scopes:
        all_prompt_positions = qwen21_prompt_token_positions(pipe, args.prompt, cond.image_sizes)

    output_path = Path(args.output)
    total_runs = len(scopes) * len(row_descs) * len(st_scales) * len(emb_scales)
    run_idx = 0
    for scope in scopes:
        scope_out_dir = out_dir / f"scope_{scope}" if len(scopes) > 1 else out_dir
        scope_out_dir.mkdir(parents=True, exist_ok=True)
        scope_output = scope_out_dir / output_path.name
        all_preds = {}

        def steer(alpha, suppress_idx):
            # None for non-suppressed edits: leave their concept untouched even in 'ablate'.
            alphas = [(alpha * args.embedding_scale_factor) if i in suppress_idx else None for i in range(n)]
            return apply_csv(base_prompt_embeds, edit_positions, steering_vecs, alphas,
                             scope=scope, all_prompt_positions=all_prompt_positions, mode=mode,
                             max_push_norm_ratio=float(args.max_push_norm_ratio))

        for row_label, suppress, suppress_idx in row_descs:
            row_dir = scope_out_dir / row_label.replace(" ", "_").replace("(", "").replace(")", "")
            row_dir.mkdir(parents=True, exist_ok=True)
            row_preds = {}
            for emb_s in emb_scales:
                n_sched = len(pipe.config.sample_sigmas or range(int(args.num_steps)))
                if schedule:
                    embeds, embeds_schedule = None, [steer(alpha_at_step(schedule, s), suppress_idx)
                                                     for s in range(n_sched)]
                    step_tag = sched_tag
                else:
                    embeds, embeds_schedule = steer(emb_s, suppress_idx), None
                    step_tag = f"emb={emb_s:.2g}"

                for st_s in st_scales:
                    run_idx += 1
                    print(f"[{run_idx}/{total_runs}] scope={scope}  {row_label}  st={st_s}  {step_tag}")
                    pred = run_inference(
                        pipe, cond, image_b, args.prompt,
                        embeds if embeds is not None else base_prompt_embeds, image_pad_mask,
                        suppress, args.num_steps, exlora_scale=st_s,
                        prompt_embeds_schedule=embeds_schedule, generator=seeded(),
                        use_kv_cache=args.use_kv_cache)
                    all_preds[(row_label, st_s, emb_s)] = pred
                    row_preds[(st_s, emb_s)] = pred
                    fname = f"st{st_s:.2f}".replace(".", "p") + (
                        f"_{sched_tag}.png" if schedule else f"_emb{emb_s:.2f}".replace(".", "p") + ".png")
                    annotate_image(pred, f"{row_label}  st={st_s:.2g}  {step_tag}  scope={scope}",
                                   suppress).save(row_dir / fname)

            build_row_grid(image_a, image_a_prime, image_b, f"{row_label}  [scope={scope}]", suppress,
                           row_preds, st_scales, emb_scales, cell_size=args.cell_size,
                           analora_image=analora_image).save(row_dir / "grid.png")

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
