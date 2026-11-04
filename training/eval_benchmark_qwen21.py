#!/usr/bin/env python3
"""
eval_benchmark_qwen21.py

Full evaluation of Stage 2 (ExLoRA) checkpoints on the unseen `analogy_benchmark`
test split of the SIA dataset (239 (A, A', B) cases, no ground truth), following
the paper's protocol (Sec. 5.2): for each case and each retained subset of its
edits, a VLM judge sees (A, A', B, output) plus the retained / suppressed edit
lists and scores Retained Accuracy, Preservation and Edit Recall (and, as in the
paper's ablation, Suppression); CLIP directional metrics are reported too.
The `full_transfer` split (75 cases with ground truth B') scores full transfer.

Every non-empty proper subset of the edits is evaluated: 2^N - 2 per case.

Subcommands (each is resumable; existing outputs are skipped):

  generate  --name N (--ckpt DIR | --analora DIR)   write predicted images
  score     --names N1 N2 ... [--identity]          CLIP + VLM-judge scores
  report    --names N1 N2 ...                       tables, strata, paired CIs

Methods
  --ckpt DIR      ExLoRA: full prompt, ExLoRA gated on the suppressed edit tokens
  --analora DIR   baseline: AnaLoRA only, suppressed edits removed from the prompt

Data (every subcommand)
  --split analogy_benchmark   the test benchmark (default)
  --split dev_train           development cases from training subjects (sia.data.dev_train_cases),
                              for choosing method settings without touching the test set

Usage:
  python training/eval_benchmark_qwen21.py generate --name s2_5000 \\
      --ckpt experiments/exlora_qwen21/checkpoint-5000
  python training/eval_benchmark_qwen21.py score  --names s2_5000 s2_6000 --identity
  python training/eval_benchmark_qwen21.py report --names s2_5000 s2_6000
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from sia.data import DATASET, SPLITS, covered_by_training, distinct_edits, load_split, subsets_for  # noqa: E402

JUDGE_REPO = "Qwen/Qwen3-VL-32B-Instruct"
CLIP_REPO = "openai/clip-vit-large-patch14"
DINO_REPO = "facebook/dinov2-base"
DEFAULT_OUT = "experiments/eval_qwen21"


def keep_tag(kept):
    return "keep" + "".join(str(p + 1) for p in kept)


def safe_id(case_id):
    return case_id.replace("/", "__")


def pred_path(out, name, case_id, kept):
    return Path(out) / name / safe_id(case_id) / f"{keep_tag(kept)}.png"


def full_path(out, name, case_id):
    return Path(out) / name / "full_transfer" / f"{safe_id(case_id)}.png"


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

# Component grid of the ablation: AnaLoRA (visual analogy), ExLoRA (token-gated
# suppression) and CSV (concept suppression vector on the edit-token embeddings).
# Every row uses the weights of one Stage 2 checkpoint; a component that is off
# is simply not loaded, which is exactly the model without it (LoRA B starts at 0).
CONFIGS = {
    "base":         dict(ana=0, ex=0, csv=0),
    "base_csv":     dict(ana=0, ex=0, csv=1),
    "ex_noana":     dict(ana=0, ex=1, csv=0),
    "ex_noana_csv": dict(ana=0, ex=1, csv=1),
    "ana_prompt":   dict(ana=1, ex=0, csv=0),
    "ana_csv":      dict(ana=1, ex=0, csv=1),
    "s2_5000":      dict(ana=1, ex=1, csv=0),
    "s2_5000_csv":  dict(ana=1, ex=1, csv=1),
}
CSV_CACHE = "steering_cache_qwen21"


def cmd_csvbuild(args):
    """Build (and cache) one CSV direction per distinct benchmark edit text,
    from ``--pairs`` Qwen-generated contrastive prompt pairs each (paper: 64)."""
    from training.train_exlora_qwen21 import QwenImage21Pipeline
    from training.steering_dom_qwen21 import get_or_build_steering_vector_qwen21
    from omegaconf import OmegaConf
    import torch

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda")
    # Directions only need the text encoder and processor: skip the 14 GB transformer and the VAE
    # so several shards fit next to each other on one GPU.
    pipe = QwenImage21Pipeline.from_pretrained(
        cfg.pretrained_model_name_or_path, transformer=None, vae=None, dtype=torch.bfloat16)
    pipe.text_encoder.to(device)
    edits = distinct_edits(args.split, args.dataset)
    edits = edits[:args.limit] if args.limit else edits
    edits = edits[args.shard_index::args.num_shards]
    t0 = time.time()
    stats = []
    for i, e in enumerate(edits):
        v, meta = get_or_build_steering_vector_qwen21(e, args.pairs, pipe, device, Path(args.cache))
        stats.append(dict(edit=e, parsed=meta.get("n_parsed_from_llm"), requested=meta.get("n_requested"),
                          fallback=meta.get("used_fallback_padding"), usable=meta.get("n_usable_after_pooling"),
                          source=meta.get("source")))
        print(f"[csv] {i + 1}/{len(edits)} {e!r}: parsed {stats[-1]['parsed']}/{stats[-1]['requested']} "
              f"usable {stats[-1]['usable']} fallback {stats[-1]['fallback']} ({stats[-1]['source']})  "
              f"{(time.time() - t0) / (i + 1):.1f} s/edit", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"csv_build_{args.shard_index}.json").write_text(json.dumps(stats, indent=2))
    n_fb = sum(1 for x in stats if x["fallback"])
    print(f"[csv] DONE {len(stats)} directions, {n_fb} used fallback padding", flush=True)


def cmd_generate(args):
    from omegaconf import OmegaConf

    from training.train_exlora_qwen21 import (
        ANALORA_TARGET_MODULES,
        EXLORA_TARGET_MODULES,
        build_analogy_prompt,
        encode_prompt,
        generate_qwen21,
        harmonize_images_qwen21,
        load_pipeline_qwen21,
        prepare_conditions,
    )
    import torch
    from training.infer_single import apply_csv
    from training.steering_dom_qwen21 import get_or_build_steering_vector_qwen21
    from sia.adapters import (
        checkpoint_files,
        inject_analora_modules,
        inject_exlora_modules,
        load_analora_from_peft_checkpoint,
        load_analora_state_dict,
        load_exlora_state_dict,
    )
    from sia.hf import resolve_checkpoint
    from sia.utils.qwen21 import qwen21_find_substring_token_indices, qwen21_text_token_mask

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda")
    resolution = cfg.get("resolution", 512)
    steps = cfg.get("num_val_inference_steps", 40)
    use_cache = cfg.get("use_kv_cache", True)
    rank = cfg.lora_rank

    if args.analora:                      # legacy: Stage-1 PEFT AnaLoRA, prompt suppression
        ana, ex, csv = 1, 0, 0
    else:
        ana, ex, csv = args.ana, args.ex, args.csv
        if (ana or ex) and not args.ckpt:
            raise SystemExit("--ckpt is required when AnaLoRA or ExLoRA is on")

    pipe = load_pipeline_qwen21(cfg.pretrained_model_name_or_path)
    transformer = pipe.transformer
    if ana or ex:
        inject_analora_modules(transformer, ANALORA_TARGET_MODULES, r=rank, alpha=rank, dropout=0.0)
    if args.analora:
        load_analora_from_peft_checkpoint(transformer, resolve_checkpoint(args.analora), device="cpu")
    elif ana or ex:
        analora_file, exlora_file = checkpoint_files(resolve_checkpoint(args.ckpt))
        if ana:
            load_analora_state_dict(transformer, analora_file, "cpu")
    if ex:
        inject_exlora_modules(transformer, EXLORA_TARGET_MODULES, r=rank, alpha=rank, dropout=0.0)
        load_exlora_state_dict(transformer, exlora_file)
    print(f"[{args.name}] AnaLoRA={ana} ExLoRA={ex} CSV={csv}  ckpt={args.ckpt or args.analora}", flush=True)
    pipe.to(device)
    transformer.eval()

    def run(cond, embeds, pad, mask):
        generator = torch.Generator(device=device).manual_seed(args.seed)
        return generate_qwen21(pipe, cond, embeds, pad, steps, generator=generator,
                               exlora_text_mask=mask, use_kv_cache=use_cache)

    out = Path(args.out)
    t0, n_done = time.time(), 0
    allowed = None
    if args.items_file:
        allowed = {(r["case"], r["keep"]) for r in json.loads(Path(args.items_file).read_text())}
        args.skip_full = True
        print(f"[{args.name}] restricted to {len(allowed)} items from {args.items_file}", flush=True)

    # ---- analogy_benchmark: every retained subset -------------------------
    cases = load_split(args.split, args.limit, args.dataset)[args.shard_index::args.num_shards]
    for ci, case in enumerate(cases):
        edits = case["edits"]
        n = len(edits)
        todo = [k for k in subsets_for(n) if not pred_path(out, args.name, case["id"], k).exists()]
        if allowed is not None:
            todo = [k for k in todo if (case["id"], keep_tag(k)) in allowed]
        if not todo:
            continue
        a, a_prime, b = harmonize_images_qwen21(case["a"], case["a_prime"], case["b"], case["b"])[:3]
        cond = prepare_conditions(pipe, [a, a_prime, b], resolution, device)
        full_prompt = build_analogy_prompt(edits, edits)
        if ex or csv:                     # these keep the full prompt; suppression happens elsewhere
            embeds, pad = encode_prompt(pipe, full_prompt, cond, device)
        if csv:
            positions = [qwen21_find_substring_token_indices(pipe, full_prompt, e, cond.image_sizes) for e in edits]
            built = [get_or_build_steering_vector_qwen21(e, args.csv_pairs, pipe, device, Path(args.csv_cache))
                     for e in edits]
            vecs = [v.to(device=device, dtype=embeds.dtype) for v, _ in built]
            gaps = [m["pre_normalization_norm"] for _, m in built]   # |mean(pos) - mean(neg)|: the concept's own scale
        for kept in todo:
            supp_pos = [p for p in range(n) if p not in kept]
            if ex or csv:
                e_k = embeds
                if csv:                   # ablate the concept direction at each suppressed edit's tokens
                    if args.csv_k:        # Diffusion Sliders form: x - k * gap * d, k gaps toward the "absent" pole
                        alphas = [-args.csv_k * gaps[p] if p in supp_pos else None for p in range(n)]
                    else:
                        alphas = [args.csv_alpha if p in supp_pos else None for p in range(n)]
                    e_k = apply_csv(embeds, positions, vecs, alphas,
                                    scope="tokens", mode=args.csv_mode, ablate_scale=args.csv_lambda)
                mask = (qwen21_text_token_mask(pipe, full_prompt, [edits[p] for p in supp_pos],
                                               cond.image_sizes, embeds.shape[1], device) if ex else None)
                img = run(cond, e_k, pad, mask)
            else:                         # suppressed edits are left out of the prompt
                prompt = build_analogy_prompt(edits, edits, suppress_indices=supp_pos)
                e_k, p_k = encode_prompt(pipe, prompt, cond, device)
                img = run(cond, e_k, p_k, None)
            path = pred_path(out, args.name, case["id"], kept)
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path)
            n_done += 1
        el = time.time() - t0
        print(f"[{args.name}] case {ci + 1}/{len(cases)} {case['id']}  "
              f"{n_done} new images, {el / max(n_done, 1):.1f} s/img", flush=True)

    # ---- full_transfer: nothing suppressed (CSV never acts), scored against ground truth ----
    if not args.skip_full and not csv:
        for case in load_split("full_transfer", args.limit, args.dataset)[args.shard_index::args.num_shards]:
            path = full_path(out, args.name, case["id"])
            if path.exists():
                continue
            a, a_prime, b = harmonize_images_qwen21(case["a"], case["a_prime"], case["b"], case["b"])[:3]
            cond = prepare_conditions(pipe, [a, a_prime, b], resolution, device)
            prompt = build_analogy_prompt(case["edits"], case["edits"])
            embeds, pad = encode_prompt(pipe, prompt, cond, device)
            path.parent.mkdir(parents=True, exist_ok=True)
            run(cond, embeds, pad, None).save(path)
        print(f"[{args.name}] full_transfer done", flush=True)
    print(f"[{args.name}] DONE {n_done} new images in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are a strict, careful judge of image-editing results.

You see four images:
  Image 1 (A): an original image.
  Image 2 (A'): the same image after SEVERAL edits were applied to it.
  Image 3 (B): a different query image.
  Image 4 (OUTPUT): a result that is supposed to apply only SOME of the edits shown in A -> A' to B.

The edits shown in A -> A' are:
{all_edits}

RETAINED edits (the user wants these applied to B, so they must appear in OUTPUT):
{retained}

SUPPRESSED edits (the user does not want these, so they must NOT appear in OUTPUT):
{suppressed}

Judge OUTPUT only against B and the lists above. Be strict and use the full range of scores:
10 = flawless, 7-8 = good with small defects, 4-6 = partly right or visibly flawed, 1-3 = mostly wrong, 0 = completely wrong or absent.
- "retained_accuracy": how accurately and completely ALL retained edits are applied to B, as demonstrated in A -> A'. Deduct for every retained edit that is missing, weak or wrong.
- "preservation": how well OUTPUT keeps the identity, pose, layout and every part of B that the retained edits do not change. Deduct for any other change.
- "suppression": how completely the suppressed edits are absent from OUTPUT. Deduct for every suppressed edit that is visible, even partly. If there are no suppressed edits give 10.
- "retained_detected": one true/false per RETAINED edit (R1, R2, ... in order): true only if that edit is clearly visible in OUTPUT.
- "suppressed_visible": one true/false per SUPPRESSED edit (S1, S2, ... in order): true if any trace of that edit is visible in OUTPUT. Use [] if there are no suppressed edits.

Reply with one JSON object and nothing else, in exactly this form (replace the placeholders with your own judgement):
{{"retained_accuracy": <0-10>, "preservation": <0-10>, "suppression": <0-10>, "retained_detected": [<true or false for R1>, ...], "suppressed_visible": [<true or false for S1>, ...]}}"""


def build_judge_prompt(edits, kept):
    supp = [p for p in range(len(edits)) if p not in kept]
    return JUDGE_PROMPT.format(
        all_edits="\n".join(f"  {p + 1}. {e}" for p, e in enumerate(edits)),
        retained="\n".join(f"  R{i + 1}: {edits[p]}" for i, p in enumerate(kept)) or "  (none)",
        suppressed="\n".join(f"  S{i + 1}: {edits[p]}" for i, p in enumerate(supp)) or "  (none)")


def parse_judge(text, n_retained, n_suppressed=None):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        det = d.get("retained_detected", [])
        if not isinstance(det, list):
            det = []
        vis = d.get("suppressed_visible")
        vis = vis if isinstance(vis, list) else None
        return {
            "ret_acc": float(np.clip(float(d["retained_accuracy"]), 0, 10)),
            "pres": float(np.clip(float(d["preservation"]), 0, 10)),
            "supp": float(np.clip(float(d["suppression"]), 0, 10)),
            "recall": float(np.mean([bool(x) for x in det])) if len(det) == n_retained else None,
            # fraction of suppressed edits the judge found absent (1 = all suppressed)
            "supp_rate": (1.0 - float(np.mean([bool(x) for x in vis]))) if vis and len(vis) == n_suppressed else None,
        }
    except Exception:
        return None


def _jsonl_read(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _items(out, names, cases, include_identity):
    """(method, case, kept, image-or-path) for every item to score."""
    for name in names:
        for case in cases:
            for kept in subsets_for(len(case["edits"])):
                p = pred_path(out, name, case["id"], kept)
                if p.exists():
                    yield name, case, kept, p
    if include_identity:
        for case in cases:
            for kept in subsets_for(len(case["edits"])):
                yield "identity", case, kept, None   # output = B, the no-edit control


def _load_pred(item, case):
    return case["b"].copy() if item is None else Image.open(item).convert("RGB")


def _clip_feats(out):
    """Projected CLIP embedding from get_{image,text}_features: a tensor in
    older transformers, a BaseModelOutputWithPooling (pooler_output) in v5."""
    return out if hasattr(out, "float") else out.pooler_output


def score_clip(args, cases):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda"
    model = CLIPModel.from_pretrained(CLIP_REPO).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_REPO)

    @torch.no_grad()
    def img_emb(images):
        feats = []
        for i in range(0, len(images), 32):
            x = proc(images=images[i:i + 32], return_tensors="pt").to(device)
            feats.append(torch.nn.functional.normalize(_clip_feats(model.get_image_features(**x)).float(), dim=-1))
        return torch.cat(feats)

    @torch.no_grad()
    def txt_emb(texts):
        x = proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        return torch.nn.functional.normalize(_clip_feats(model.get_text_features(**x)).float(), dim=-1)

    cos = torch.nn.functional.cosine_similarity
    base_txt = txt_emb(["a photo"])[0]
    path = Path(args.out) / "clip.jsonl"
    done = {(r["method"], r["case"], r["keep"]) for r in _jsonl_read(path)}
    by_case = {c["id"]: c for c in cases}
    emb_cache = {}
    rows = []
    items = [it for it in _items(args.out, args.names, cases, args.identity)
             if (it[0], it[1]["id"], keep_tag(it[2])) not in done]
    # group by case so A, A', B embeddings are computed once
    for cid in sorted({it[1]["id"] for it in items}):
        case = by_case[cid]
        if cid not in emb_cache:
            emb_cache[cid] = img_emb([case["a"], case["a_prime"], case["b"]])
        ea, eap, eb = emb_cache[cid]
        group = [it for it in items if it[1]["id"] == cid]
        e_out = img_emb([_load_pred(it[3], case) for it in group])
        for it, eo in zip(group, e_out):
            kept = it[2]
            retained = " ".join(case["edits"][p] for p in kept)
            d_out = eo - eb
            d_demo = eap - ea
            d_txt = txt_emb(["a photo. " + retained])[0] - base_txt
            rows.append({"method": it[0], "case": cid, "keep": keep_tag(kept),
                         "clip_dir_img": float(cos(d_out, d_demo, dim=0)),
                         "clip_dir_ret": float(cos(d_out, d_txt, dim=0))})
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[clip] {len(rows)} new items", flush=True)
    del model
    torch.cuda.empty_cache()


def score_full_transfer(args):
    """Full-transfer quality against ground truth: LPIPS, DINOv2 distance, CLIP-I."""
    import lpips
    import torch
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    device = "cuda"
    cases = load_split("full_transfer", args.limit, args.dataset)[args.shard_index::args.num_shards]
    lp = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    dino = AutoModel.from_pretrained(DINO_REPO).to(device).eval()
    dproc = AutoImageProcessor.from_pretrained(DINO_REPO)
    clip = CLIPModel.from_pretrained(CLIP_REPO).to(device).eval()
    cproc = CLIPProcessor.from_pretrained(CLIP_REPO)

    def t(img):
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(device)

    rows = []
    for name in args.names:
        for case in cases:
            p = full_path(args.out, name, case["id"])
            if not p.exists():
                continue
            gt = case["b_prime"]
            pred = Image.open(p).convert("RGB").resize(gt.size, Image.LANCZOS)
            with torch.no_grad():
                l = float(lp(t(pred), t(gt)))
                ed = dino(**dproc(images=[pred, gt], return_tensors="pt").to(device)).last_hidden_state[:, 0]
                dd = 1 - float(torch.nn.functional.cosine_similarity(ed[0], ed[1], dim=0))
                ec = torch.nn.functional.normalize(_clip_feats(clip.get_image_features(
                    **cproc(images=[pred, gt], return_tensors="pt").to(device))).float(), dim=-1)
                ci = float(ec[0] @ ec[1])
            rows.append({"method": name, "case": case["id"], "lpips": l, "dino_dist": dd, "clip_i": ci})
    path = Path(args.out) / "full_transfer.jsonl"
    seen = {(r["method"], r["case"]) for r in rows}
    keep = [r for r in _jsonl_read(path) if (r["method"], r["case"]) not in seen]   # keep other shards' rows
    path.write_text("".join(json.dumps(r) + "\n" for r in keep + rows))
    print(f"[full_transfer] {len(rows)} items", flush=True)


def score_judge(args, cases):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(JUDGE_REPO)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(JUDGE_REPO, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"[judge] loaded {JUDGE_REPO}", flush=True)

    out_path = Path(args.out) / "judge.jsonl"
    done = {(r["method"], r["case"], r["keep"]) for r in _jsonl_read(out_path)}
    todo = [it for it in _items(args.out, args.names, cases, args.identity)
            if (it[0], it[1]["id"], keep_tag(it[2])) not in done]
    if args.limit_items:
        todo = todo[:args.limit_items]
    print(f"[judge] {len(todo)} items to score ({len(done)} done)", flush=True)

    def messages(it):
        method, case, kept, p = it
        content = []
        for label, im in (("Image 1 (A):", case["a"]), ("Image 2 (A'):", case["a_prime"]),
                          ("Image 3 (B):", case["b"]), ("Image 4 (OUTPUT):", _load_pred(p, case))):
            content += [{"type": "text", "text": label}, {"type": "image", "image": im}]
        content.append({"type": "text", "text": build_judge_prompt(case["edits"], kept)})
        return [{"role": "user", "content": content}]

    t0 = time.time()
    n_fail = 0
    with open(out_path, "a") as f:
        for i in range(0, len(todo), args.judge_batch):
            batch = todo[i:i + args.judge_batch]
            inputs = proc.apply_chat_template(
                [messages(it) for it in batch], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            texts = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for it, text in zip(batch, texts):
                method, case, kept, _ = it
                s = parse_judge(text, len(kept), len(case["edits"]) - len(kept))
                if s is None:
                    n_fail += 1
                    s = {"ret_acc": None, "pres": None, "supp": None, "recall": None, "supp_rate": None}
                f.write(json.dumps({"method": method, "case": case["id"], "keep": keep_tag(kept),
                                    **s, "raw": text}) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"[judge] {min(i + args.judge_batch, len(todo))}/{len(todo)}  "
                  f"{el / (i + len(batch)):.2f} s/item  parse failures {n_fail}", flush=True)


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
# report
# ---------------------------------------------------------------------------



def cmd_report(args):
    all_cases = load_split(args.split, dataset=args.dataset)
    cases = {c["id"]: c for c in all_cases}
    selected = {c["id"] for c in all_cases[args.shard_index::args.num_shards]}
    ft_selected = {c["id"] for c in load_split("full_transfer", dataset=args.dataset)[args.shard_index::args.num_shards]}
    judge = {(r["method"], r["case"], r["keep"]): r for r in _jsonl_read(Path(args.out) / "judge.jsonl")}
    clip = {(r["method"], r["case"], r["keep"]): r for r in _jsonl_read(Path(args.out) / "clip.jsonl")}
    names = list(args.names) + (["identity"] if args.identity else [])

    only = None
    if args.items_file:
        only = {(r["case"], r["keep"]) for r in json.loads(Path(args.items_file).read_text())}
    rows = {n: [] for n in names}
    for (method, cid, keep), j in judge.items():
        if method not in rows or j["ret_acc"] is None or cid not in selected:
            continue
        if only is not None and (cid, keep) not in only:
            continue
        c = cases[cid]
        kept = tuple(int(ch) - 1 for ch in keep[4:])
        r = {"case": cid, "keep": keep, "n_edits": len(c["edits"]), "unseen_recipe": not c["recipe_in_train"],
             "covered": covered_by_training(kept, len(c["edits"]), c["subject_type"]),
             "subject_type": c["subject_type"],
             "ret_acc": j["ret_acc"], "pres": j["pres"], "supp": j["supp"], "recall": j["recall"],
             "supp_rate": j.get("supp_rate")}
        r["avg"] = (r["ret_acc"] + r["supp"] + r["pres"]) / 3
        cl = clip.get((method, cid, keep))
        if cl:
            r["clip_dir_img"], r["clip_dir_ret"] = cl["clip_dir_img"], cl["clip_dir_ret"]
        rows[method].append(r)

    rng = np.random.default_rng(0)

    def mean_ci(rs, key, reps=1000):
        """Mean and 95% bootstrap CI, resampling cases (items of one case are correlated)."""
        rs = [r for r in rs if r.get(key) is not None]
        if not rs:
            return None
        by = {}
        for r in rs:
            by.setdefault(r["case"], []).append(r[key])
        cs = list(by.values())
        mean = float(np.mean([v for c in cs for v in c]))
        boots = []
        for _ in range(reps):
            pick = rng.integers(0, len(cs), len(cs))
            boots.append(np.mean([v for i in pick for v in cs[i]]))
        return mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    def fmt(v):
        return "–" if v is None else f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"

    part = "all cases" if args.num_shards == 1 else f"interleaved subset {args.shard_index + 1} of {args.num_shards}: {len(selected)} of {len(cases)} cases"
    lines = [f"# Qwen-Image-2.1 SIA — benchmark evaluation ({part})", ""]
    n_items = {n: len(rows[n]) for n in names}
    lines += [f"Judge: {JUDGE_REPO}; items scored per method: {n_items}", ""]
    if args.note:
        lines += [args.note, ""]

    def table(title, filt):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| method | n | Avg | Ret. Acc | Supp. | Pres. | Recall | Supp-rate | CLIP-Dir Img | CLIP-Dir Ret |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for n in names:
            rs = [r for r in rows[n] if filt(r)]
            cells = [fmt(mean_ci(rs, k)) for k in ["avg", "ret_acc", "supp", "pres", "recall", "supp_rate", "clip_dir_img", "clip_dir_ret"]]
            lines.append(f"| {n} | {len(rs)} | " + " | ".join(cells) + " |")
        lines.append("")

    def paired(n_on, n_off, key, filt=lambda r: True):
        """Mean of (n_on - n_off) over the items both have, 95% CI resampling cases."""
        m = {(r["case"], r["keep"]): r for r in rows[n_off]}
        ds = [dict(case=r["case"], d=r[key] - m[(r["case"], r["keep"])][key]) for r in rows[n_on]
              if (r["case"], r["keep"]) in m and filt(r) and r.get(key) is not None
              and m[(r["case"], r["keep"])].get(key) is not None]
        return mean_ci(ds, "d")

    cfg_names = [n for n in names if n in CONFIGS and rows[n]]
    if cfg_names:
        lines += ["## Ablation: AnaLoRA / ExLoRA / CSV", "",
                  "AnaLoRA = visual-analogy LoRA, ExLoRA = token-gated suppression LoRA, CSV = concept suppression "
                  "vector (ablate mode, alpha 0, on the suppressed edits' tokens). Rows without ExLoRA and without CSV "
                  "suppress by deleting the edit from the prompt. All rows use one checkpoint's weights.", "",
                  "| AnaLoRA | ExLoRA | CSV | run | n | Avg | Ret. Acc | Supp. | Pres. | Recall | Supp-rate |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        mark = lambda v: "✓" if v else "–"
        for n in sorted(cfg_names, key=lambda n: (CONFIGS[n]["ana"], CONFIGS[n]["ex"], CONFIGS[n]["csv"])):
            c = CONFIGS[n]
            cells = [fmt(mean_ci(rows[n], k)) for k in ["avg", "ret_acc", "supp", "pres", "recall", "supp_rate"]]
            lines.append(f"| {mark(c['ana'])} | {mark(c['ex'])} | {mark(c['csv'])} | {n} | {len(rows[n])} | "
                         + " | ".join(cells) + " |")
        lines.append("")

        lines += ["## Paired effect of switching one component on (with − without; 95% CI over cases)", "",
                  "| component | rest of the model | n pairs | Δ Avg | Δ Ret. Acc | Δ Supp. | Δ Pres. | Δ Recall |",
                  "|---|---|---|---|---|---|---|---|"]
        label = {"ana": "AnaLoRA", "ex": "ExLoRA", "csv": "CSV"}
        for comp in ("csv", "ex", "ana"):
            for n_on in cfg_names:
                if not CONFIGS[n_on][comp]:
                    continue
                want = dict(CONFIGS[n_on], **{comp: 0})
                n_off = next((m for m in cfg_names if CONFIGS[m] == want), None)
                if n_off is None:
                    continue
                rest = ", ".join(f"{label[k]}={'on' if CONFIGS[n_on][k] else 'off'}" for k in ("ana", "ex", "csv") if k != comp)
                n_pairs = len({(r["case"], r["keep"]) for r in rows[n_on]} & {(r["case"], r["keep"]) for r in rows[n_off]})
                cells = [fmt(paired(n_on, n_off, k)) for k in ("avg", "ret_acc", "supp", "pres", "recall")]
                lines.append(f"| {label[comp]} | {rest} | {n_pairs} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("A confidence interval that excludes 0 means the component reliably changes that score.")
        lines.append("")

    table("All items", lambda r: True)
    table("Edit subset covered by training data (prefix subsets)", lambda r: r["covered"])
    table("Edit subset NOT covered by training data (e.g. keep only a later edit)", lambda r: not r["covered"])
    table("Unseen recipes (edit lists absent from training)", lambda r: r["unseen_recipe"])
    table("Seen recipes, new subjects", lambda r: not r["unseen_recipe"])
    for n_e in (2, 3, 4):
        table(f"{n_e}-edit demonstrations", lambda r, n_e=n_e: r["n_edits"] == n_e)

    # full-transfer ground-truth metrics
    ft = [r for r in _jsonl_read(Path(args.out) / "full_transfer.jsonl") if r["case"] in ft_selected]
    if ft:
        lines += ["## Full transfer vs ground truth (75 cases, nothing suppressed)", "",
                  "| method | n | LPIPS ↓ | DINOv2 dist ↓ | CLIP-I ↑ |", "|---|---|---|---|---|"]
        for n in args.names:
            rs = [r for r in ft if r["method"] == n]
            if rs:
                lines.append(f"| {n} | {len(rs)} | " + " | ".join(
                    f"{np.mean([r[k] for r in rs]):.4f}" for k in ("lpips", "dino_dist", "clip_i")) + " |")
        lines.append("")

    # paired comparison against the best method by Avg
    real = [n for n in args.names if rows[n]]
    if len(real) > 1:
        best = max(real, key=lambda n: np.mean([r["avg"] for r in rows[n]]))
        lines += [f"## Paired difference in Avg vs best ({best}), 95% CI over cases", "",
                  "| method | Δ Avg | covered Δ | uncovered Δ |", "|---|---|---|---|"]
        key = lambda r: (r["case"], r["keep"])
        b_map = {key(r): r for r in rows[best]}
        for n in real:
            if n == best:
                continue
            def diffs(filt):
                return [dict(case=r["case"], d=b_map[key(r)]["avg"] - r["avg"]) for r in rows[n]
                        if key(r) in b_map and filt(r)]
            cells = [fmt(mean_ci(diffs(f), "d")) for f in (lambda r: True, lambda r: r["covered"],
                                                          lambda r: not r["covered"])]
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Δ = best − method; a CI above 0 means the method is reliably worse than the best.")
        lines.append("")

    text = "\n".join(lines)
    (Path(args.out) / args.report_name).write_text(text)
    print(text)


# ---------------------------------------------------------------------------
# csv_effect: in how many samples does CSV help?
# ---------------------------------------------------------------------------

def _sign_test_p(better, worse):
    """Two-sided exact sign test on the non-tied items."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def cmd_csv_effect(args):
    cases = {c["id"]: c for c in load_split(args.split, dataset=args.dataset)}
    judge = {}
    for r in _jsonl_read(Path(args.out) / "judge.jsonl"):
        if r["method"] in (args.with_csv, args.without) and r["ret_acc"] is not None:
            judge.setdefault((r["case"], r["keep"]), {})[r["method"]] = r
    items = []
    for (cid, keep), d in sorted(judge.items()):
        if args.with_csv not in d or args.without not in d:
            continue
        w, o = d[args.with_csv], d[args.without]
        c = cases[cid]
        kept = tuple(int(ch) - 1 for ch in keep[4:])
        avg = lambda r: (r["ret_acc"] + r["supp"] + r["pres"]) / 3
        items.append(dict(case=cid, keep=keep, n_edits=len(c["edits"]), subject_type=c["subject_type"],
                          unseen=not c["recipe_in_train"],
                          covered=covered_by_training(kept, len(c["edits"]), c["subject_type"]),
                          d_avg=avg(w) - avg(o), d_ret=w["ret_acc"] - o["ret_acc"],
                          d_supp=w["supp"] - o["supp"], d_pres=w["pres"] - o["pres"]))
    if not items:
        raise SystemExit("no items scored for both runs yet")

    def counts(rs, key):
        b = sum(1 for r in rs if r[key] > 1e-9)
        w = sum(1 for r in rs if r[key] < -1e-9)
        return b, len(rs) - b - w, w

    def row(label, rs):
        out = [label, str(len(rs))]
        for key in ("d_avg", "d_ret", "d_supp", "d_pres"):
            b, t, w = counts(rs, key)
            out.append(f"{b} / {t} / {w}")
        out.append(f"{_sign_test_p(*[counts(rs, 'd_avg')[i] for i in (0, 2)]):.3g}")
        out.append(f"{np.mean([r['d_avg'] for r in rs]):+.3f}")
        return "| " + " | ".join(out) + " |"

    head = ["| subset | items | Avg: better / same / worse | Ret. Acc | Supp. | Pres. | sign-test p (Avg) | mean Δ Avg |",
            "|---|---|---|---|---|---|---|---|"]
    L = [f"# Does CSV help? `{args.with_csv}` vs `{args.without}`", "",
         "A sample is one (case, kept-edits subset) output pair generated with the same seed. "
         "Columns count samples where the CSV run scored higher / equal / lower. Avg = (Ret. Acc + Supp. + Pres.) / 3.", ""]
    L += head
    L.append(row("all samples", items))
    for lab, f in (("subset covered by training data", lambda r: r["covered"]),
                   ("subset NOT covered by training data", lambda r: not r["covered"]),
                   ("unseen recipes", lambda r: r["unseen"]), ("seen recipes", lambda r: not r["unseen"]),
                   ("2-edit demos", lambda r: r["n_edits"] == 2), ("3-edit demos", lambda r: r["n_edits"] == 3),
                   ("4-edit demos", lambda r: r["n_edits"] == 4),
                   ("creature", lambda r: r["subject_type"] == "creature"),
                   ("non-creature", lambda r: r["subject_type"] != "creature")):
        rs = [r for r in items if f(r)]
        if rs:
            L.append(row(lab, rs))
    L.append("")
    big = [r for r in items if abs(r["d_avg"]) >= 1]
    L.append(f"Clear differences (|Δ Avg| ≥ 1 point): {sum(r['d_avg'] >= 1 for r in big)} samples better, "
             f"{sum(r['d_avg'] <= -1 for r in big)} worse.")
    by_case = {}
    for r in items:
        by_case.setdefault(r["case"], []).append(r["d_avg"])
    cm = {k: float(np.mean(v)) for k, v in by_case.items()}
    L.append(f"Per case (mean over its subsets): {sum(v > 1e-9 for v in cm.values())} better, "
             f"{sum(abs(v) <= 1e-9 for v in cm.values())} same, {sum(v < -1e-9 for v in cm.values())} worse "
             f"of {len(cm)} cases.")
    L.append("")
    L.append("Both runs use the same seed, but CSV changes the prompt embeddings, so the two sampling trajectories "
             "diverge: part of every per-sample difference is sampling noise, which is why the better and worse "
             "counts and the sign test matter more than any single sample.")
    better = [dict(case=r["case"], keep=r["keep"]) for r in items if r["d_avg"] > 1e-9]
    Path(args.out, args.items_out).write_text(json.dumps(better, indent=1))
    L.append("")
    L.append(f"{len(better)} samples where CSV scored higher were written to `{args.items_out}`.")
    text = "\n".join(L)
    Path(args.out, args.report_name).write_text(text)
    print(text)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--split", default="analogy_benchmark", choices=SPLITS,
                      help="cases to evaluate: the test benchmark, or dev_train (training subjects, "
                           "for choosing method settings without touching the test set)")
    data.add_argument("--dataset", default=DATASET, help="Hugging Face dataset id")

    g = sub.add_parser("generate", parents=[data])
    g.add_argument("--name", required=True)
    g.add_argument("--ckpt", help="Stage 2 checkpoint (weights for AnaLoRA / ExLoRA): local dir or Hub path "
                                  "<org>/<repo>/<subfolder>[@revision]")
    g.add_argument("--analora", help="legacy: Stage 1 PEFT AnaLoRA (local dir or Hub path), prompt suppression")
    g.add_argument("--ana", type=int, default=1, choices=[0, 1], help="AnaLoRA from --ckpt")
    g.add_argument("--ex", type=int, default=1, choices=[0, 1], help="ExLoRA from --ckpt")
    g.add_argument("--csv", type=int, default=0, choices=[0, 1], help="CSV on the suppressed edits' tokens")
    g.add_argument("--csv_mode", default="ablate", choices=["add", "ablate"])
    g.add_argument("--csv_lambda", type=float, default=1.0,
                   help="paper's lambda: x - lambda (x.v) v at suppressed tokens; 1 = orthogonal projection")
    g.add_argument("--csv_k", type=float, default=0.0,
                   help="with --csv_mode add: push the suppressed tokens by -k times the edit's mean-pole gap along d")
    g.add_argument("--csv_alpha", type=float, default=0.0, help="push after ablation; 0 = pure concept removal")
    g.add_argument("--csv_pairs", type=int, default=64)
    g.add_argument("--csv_cache", default=CSV_CACHE)
    g.add_argument("--items_file", help="JSON list of {case, keep}: generate only these items")
    g.add_argument("--config", default="training/configs/train_exlora_qwen21.yaml")
    g.add_argument("--out", default=DEFAULT_OUT)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--limit", type=int, help="first N cases only (smoke test)")
    g.add_argument("--skip_full", action="store_true")
    g.add_argument("--shard_index", type=int, default=0, help="this worker's slice of the cases (parallel workers)")
    g.add_argument("--num_shards", type=int, default=1)

    c = sub.add_parser("csvbuild", parents=[data])
    c.add_argument("--config", default="training/configs/train_exlora_qwen21.yaml")
    c.add_argument("--pairs", type=int, default=64)
    c.add_argument("--cache", default=CSV_CACHE)
    c.add_argument("--out", default=DEFAULT_OUT)
    c.add_argument("--limit", type=int)
    c.add_argument("--shard_index", type=int, default=0)
    c.add_argument("--num_shards", type=int, default=1)

    s = sub.add_parser("score", parents=[data])
    s.add_argument("--names", nargs="*", default=[])
    s.add_argument("--identity", action="store_true", help="also score the no-edit control (output = B)")
    s.add_argument("--stages", nargs="+", default=["clip", "judge"], choices=["clip", "judge"])
    s.add_argument("--out", default=DEFAULT_OUT)
    s.add_argument("--limit", type=int)
    s.add_argument("--limit_items", type=int)
    s.add_argument("--judge_batch", type=int, default=8)
    s.add_argument("--shard_index", type=int, default=0)
    s.add_argument("--num_shards", type=int, default=1)

    e = sub.add_parser("csv_effect", parents=[data])
    e.add_argument("--with_csv", default="s2_5000_csv")
    e.add_argument("--without", default="s2_5000")
    e.add_argument("--out", default=DEFAULT_OUT)
    e.add_argument("--items_out", default="csv_better_items.json")
    e.add_argument("--report_name", default="csv_effect.md")

    r = sub.add_parser("report", parents=[data])
    r.add_argument("--names", nargs="+", required=True)
    r.add_argument("--identity", action="store_true")
    r.add_argument("--out", default=DEFAULT_OUT)
    r.add_argument("--shard_index", type=int, default=0)
    r.add_argument("--num_shards", type=int, default=1)
    r.add_argument("--report_name", default="report.md")
    r.add_argument("--items_file", help="JSON list of {case, keep}: report only these items")
    r.add_argument("--note", default="", help="text added under the report title")

    args = ap.parse_args()
    if args.cmd == "csvbuild":
        cmd_csvbuild(args)
    elif args.cmd == "csv_effect":
        cmd_csv_effect(args)
    elif args.cmd == "generate":
        if args.analora and args.ckpt:
            ap.error("give --ckpt or --analora, not both")
        cmd_generate(args)
    elif args.cmd == "score":
        cmd_score(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
