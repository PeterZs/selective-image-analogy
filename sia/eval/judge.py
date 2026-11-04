"""VLM judge of selective analogy outputs.

The judge sees (A, A', B, output) with the retained and suppressed edit lists and returns Retained
Accuracy, Preservation and Suppression (0-10) plus per-edit detections, from which Edit Recall and
the suppression rate are computed.
"""
import json
import re
import time
from pathlib import Path

import numpy as np

from sia.eval.results import iter_items, jsonl_read, keep_tag, load_pred

JUDGE_REPO = "Qwen/Qwen3-VL-32B-Instruct"


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
    """``JUDGE_PROMPT`` filled with all edits, the retained ones (``kept``, 0-based) and the suppressed rest."""
    supp = [p for p in range(len(edits)) if p not in kept]
    return JUDGE_PROMPT.format(
        all_edits="\n".join(f"  {p + 1}. {e}" for p, e in enumerate(edits)),
        retained="\n".join(f"  R{i + 1}: {edits[p]}" for i, p in enumerate(kept)) or "  (none)",
        suppressed="\n".join(f"  S{i + 1}: {edits[p]}" for i, p in enumerate(supp)) or "  (none)")


def parse_judge(text, n_retained, n_suppressed=None):
    """Scores from the judge's reply, or ``None`` if it holds no valid JSON object.

    Returns ``ret_acc``, ``pres``, ``supp`` (clipped to 0-10), ``recall`` (fraction of retained edits
    detected; ``None`` unless one flag per retained edit) and ``supp_rate`` (fraction of suppressed
    edits absent; ``None`` unless one flag per suppressed edit).
    """
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


def score_judge(args, cases):
    """Judge every unscored item of ``args.names`` (and the identity control with ``args.identity``)
    on ``cases``, appending rows to ``<args.out>/judge.jsonl`` in batches of ``args.judge_batch``
    (at most ``args.limit_items`` items). Resumable: items already in the file are skipped."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(JUDGE_REPO)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(JUDGE_REPO, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"[judge] loaded {JUDGE_REPO}", flush=True)

    out_path = Path(args.out) / "judge.jsonl"
    done = {(r["method"], r["case"], r["keep"]) for r in jsonl_read(out_path)}
    todo = [it for it in iter_items(args.out, args.names, cases, args.identity)
            if (it[0], it[1]["id"], keep_tag(it[2])) not in done]
    if args.limit_items:
        todo = todo[:args.limit_items]
    print(f"[judge] {len(todo)} items to score ({len(done)} done)", flush=True)

    def messages(it):
        method, case, kept, p = it
        content = []
        for label, im in (("Image 1 (A):", case["a"]), ("Image 2 (A'):", case["a_prime"]),
                          ("Image 3 (B):", case["b"]), ("Image 4 (OUTPUT):", load_pred(p, case))):
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
