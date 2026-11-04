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

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from training.infer_single import (  # noqa: E402
    annotate_image,
    build_combined_grid,
    build_joint_grid,
    build_parser,
    build_row_grid,
    parse_args as parse_args_klein,
)
from sia.backbones import qwen_image21  # noqa: E402
from sia.backbones.qwen_image21 import (  # noqa: E402
    encode_prompt,
    load_pipeline,
    prepare_conditions,
    qwen21_find_substring_token_indices,
    qwen21_prompt_token_positions,
    run_inference,
)
from sia.concept_suppression import (  # noqa: E402
    alpha_at_step,
    apply_csv,
    format_schedule_tag,
    get_or_build_steering_vector,
)
from sia.prompt import build_analogy_prompt  # noqa: E402


def parse_args():
    parser = build_parser()
    parser.set_defaults(num_steps=40)
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
        for i, edit_text in enumerate(args.edits):
            incontext = {}
            if args.get("incontext_steering", False):
                incontext = dict(incontext_prompt=args.prompt, all_edits=list(args.edits),
                                 edit_index=i, build_prompt_fn=build_analogy_prompt,
                                 vl_images=cond.vl_images)
            v, meta = get_or_build_steering_vector(
                qwen_image21, edit_text, int(args.num_steering_pairs), pipe, device,
                cache_dir=args.steering_cache_dir, regen=bool(args.regen_steering), **incontext)
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
