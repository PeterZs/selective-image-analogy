"""Markdown reports of scored runs: per-method tables with bootstrap CIs over cases, the AnaLoRA /
ExLoRA / CSV ablation with paired effects, and a per-sample paired comparison of two runs with a
sign test."""
import json
import math
from pathlib import Path

import numpy as np

from sia.data import covered_by_training, load_split
from sia.eval.judge import JUDGE_REPO
from sia.eval.results import _jsonl_read

# Component grid of the ablation: AnaLoRA (visual analogy), ExLoRA (token-gated suppression) and CSV
# (concept suppression vector on the edit-token embeddings). Every row uses the weights of one Stage 2
# checkpoint; a component that is off is simply not loaded. Runs with these names form the ablation tables.
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


def write_report(args):
    """Write ``<out>/<report_name>``: tables of judge (and CLIP) scores per method with 95% bootstrap
    CIs over cases, strata (training coverage, seen/unseen recipes, number of edits), the ablation
    of the ``CONFIGS`` runs present, full-transfer metrics and paired differences to the best method.

    ``args``: ``out``, ``names``, ``identity``, ``split``, ``dataset``, ``shard_index``, ``num_shards``,
    ``items_file`` (restrict to these items), ``note`` and ``report_name``.
    """
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
    lines = [f"# SIA — benchmark evaluation ({part})", ""]
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


def _sign_test_p(better, worse):
    """Two-sided exact sign test on the non-tied items."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare_runs(args):
    """Per-sample paired comparison of run ``args.method`` against ``args.baseline``.

    A sample is one (case, kept-edits subset) scored by the judge for both runs. Writes
    ``<out>/<report_name>`` (better / same / worse counts per score and stratum, sign test on Avg)
    and ``<out>/<items_out>`` (the samples where ``method`` scored higher, usable as ``--items_file``).
    """
    cases = {c["id"]: c for c in load_split(args.split, dataset=args.dataset)}
    judge = {}
    for r in _jsonl_read(Path(args.out) / "judge.jsonl"):
        if r["method"] in (args.method, args.baseline) and r["ret_acc"] is not None:
            judge.setdefault((r["case"], r["keep"]), {})[r["method"]] = r
    items = []
    for (cid, keep), d in sorted(judge.items()):
        if args.method not in d or args.baseline not in d:
            continue
        w, o = d[args.method], d[args.baseline]
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
    L = [f"# Paired comparison: `{args.method}` vs `{args.baseline}`", "",
         "A sample is one (case, kept-edits subset) output pair generated with the same seed. "
         f"Columns count samples where `{args.method}` scored higher / equal / lower than `{args.baseline}`. "
         "Avg = (Ret. Acc + Supp. + Pres.) / 3.", ""]
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
    L.append("Both runs use the same seed, but the methods condition the model differently, so the two sampling "
             "trajectories diverge: part of every per-sample difference is sampling noise, which is why the better "
             "and worse counts and the sign test matter more than any single sample.")
    better = [dict(case=r["case"], keep=r["keep"]) for r in items if r["d_avg"] > 1e-9]
    Path(args.out, args.items_out).write_text(json.dumps(better, indent=1))
    L.append("")
    L.append(f"{len(better)} samples where `{args.method}` scored higher were written to `{args.items_out}`.")
    text = "\n".join(L)
    Path(args.out, args.report_name).write_text(text)
    print(text)
