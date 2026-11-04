"""SIA: Selective Image Analogy — Hugging Face Space demo.

Given an example pair A -> A' showing several edits and a new image B, SIA
applies only the edits you keep to B (the rest are suppressed by ExLoRA).
Backbone: Qwen-Image-2.1 with the SIA stage-2 adapters (AnaLoRA + ExLoRA,
step 5000). No concept suppression vector (CSV) is used.

Set SIA_DEMO_STUB=1 to run the UI without a GPU or model: the "result" is B
unchanged, which is enough to test layout, validation and examples.
"""

import json
import os
from pathlib import Path

import gradio as gr

try:
    import spaces  # ZeroGPU; a no-op decorator on other hardware
except ImportError:  # local CPU testing without the `spaces` package
    spaces = None

STUB = os.getenv("SIA_DEMO_STUB") == "1"
HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Model adapter: the only place that touches the SIA inference API.
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen-Image-2.1"
BASE_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
SIA_WEIGHTS = "selective-image-analogy/SIA-weights"
SIA_SUBFOLDER = "qwen_image21/exlora"
SIA_REVISION = "27aa16012f76a911112e4906139641fa902846be"

DEFAULT_STEPS = 40
DEFAULT_RESOLUTION = 512


def load_pipeline():
    """Load Qwen-Image-2.1 + SIA adapters onto CUDA (module level, as ZeroGPU requires)."""
    if STUB:
        return None
    from sia.inference import load_qwen_pipeline_with_exlora  # lazy: heavy imports

    return load_qwen_pipeline_with_exlora(
        base=BASE_MODEL, base_revision=BASE_REVISION,
        weights=SIA_WEIGHTS, subfolder=SIA_SUBFOLDER, revision=SIA_REVISION,
        token=os.getenv("HF_TOKEN"), device="cuda",
    )


def generate(pipe, a, a_prime, b, edits, kept, seed, steps, resolution):
    """One B' image. ``kept`` are 0-based indices into ``edits``."""
    if STUB:
        return b.copy()
    from sia.inference import run_selective_analogy

    return run_selective_analogy(pipe, a, a_prime, b, edits, kept,
                                 seed=seed, steps=steps, resolution=resolution)


PIPE = load_pipeline()

# ---------------------------------------------------------------------------
# GPU entry point
# ---------------------------------------------------------------------------


def _gpu_duration(a, a_prime, b, edits, kept, seed, steps, resolution):
    # Measured ~10 s for 40 steps at 512 on a 48 GB GPU; leave headroom.
    return int(20 + 0.5 * steps * (resolution / 512) ** 2)


def _gpu(fn):
    return spaces.GPU(duration=_gpu_duration)(fn) if spaces is not None else fn


@_gpu
def run_on_gpu(a, a_prime, b, edits, kept, seed, steps, resolution):
    out = generate(PIPE, a, a_prime, b, edits, kept, seed, steps, resolution)
    return out if out.size == b.size else out.resize(b.size)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

MAX_EDITS = 4
MIN_EDITS = 2
MAX_EDIT_CHARS = 200


class InputError(ValueError):
    pass


def clean_edits(raw):
    """Strip each box; drop trailing periods (the prompt adds its own)."""
    return [(t or "").strip().rstrip(".").strip() for t in raw]


def validate(a, a_prime, b, raw_edits, kept_labels):
    """Return (edits, kept_indices, note) or raise InputError."""
    missing = [name for name, img in (("A", a), ("A'", a_prime), ("B", b)) if img is None]
    if missing:
        raise InputError(f"Please provide image(s): {', '.join(missing)}.")

    boxes = clean_edits(raw_edits)
    n = sum(1 for t in boxes if t)
    if any(boxes[n:]) or not all(boxes[:n]):
        raise InputError("Fill the edit boxes in order, starting with Edit 1, without gaps.")
    edits = boxes[:n]
    if not MIN_EDITS <= n <= MAX_EDITS:
        raise InputError(f"Describe between {MIN_EDITS} and {MAX_EDITS} edits (got {n}).")
    for i, e in enumerate(edits):
        if len(e) > MAX_EDIT_CHARS:
            raise InputError(f"Edit {i + 1} is too long (max {MAX_EDIT_CHARS} characters).")
    low = [e.lower() for e in edits]
    for i in range(n):
        for j in range(n):
            if i != j and low[i] in low[j]:
                what = "is the same as" if low[i] == low[j] else "is contained in"
                raise InputError(f"Edit {i + 1} {what} Edit {j + 1}; each edit must be distinct.")

    kept = sorted({int(k) - 1 for k in (kept_labels or [])})
    if not kept:
        raise InputError("Tick at least one edit to keep.")
    bad = [k + 1 for k in kept if k >= n]
    if bad:
        raise InputError(f"Edit {', '.join(map(str, bad))} is ticked but empty.")

    if len(kept) == n:
        note = "All edits are kept: this is full transfer, nothing is suppressed."
    else:
        sup = [f"Edit {i + 1} ({edits[i]})" for i in range(n) if i not in kept]
        note = "Suppressed: " + "; ".join(sup) + "."
    return edits, kept, note


def run(a, a_prime, b, e1, e2, e3, e4, kept_labels, seed, steps, resolution):
    try:
        edits, kept, note = validate(a, a_prime, b, [e1, e2, e3, e4], kept_labels)
    except InputError as exc:
        raise gr.Error(str(exc), print_exception=False)
    out = run_on_gpu(a, a_prime, b, edits, kept, int(seed), int(steps), int(resolution))
    kept_txt = "; ".join(f"Edit {i + 1} ({edits[i]})" for i in kept)
    return out, f"Kept: {kept_txt}.\n\n{note}"


def kept_choices(*raw):
    labels = [(f"Edit {i + 1}: {t}" if t else f"Edit {i + 1} (empty)", str(i + 1))
              for i, t in enumerate(clean_edits(raw))]
    return gr.update(choices=labels)


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

EXAMPLES_DIR = HERE / "examples"
EXAMPLES_META = json.loads((EXAMPLES_DIR / "examples.json").read_text())


def example_rows():
    rows, labels = [], []
    for ex in EXAMPLES_META["examples"]:
        d = EXAMPLES_DIR / ex["folder"]
        edits = ex["edits"] + [""] * (MAX_EDITS - len(ex["edits"]))
        rows.append([str(d / "a.png"), str(d / "a_prime.png"), str(d / "b.png"), *edits,
                     [str(k + 1) for k in ex["kept"]]])
        keep = ", ".join(ex["edits"][k] for k in ex["kept"])
        labels.append(f"{ex['id']} — keep: {keep}")
    return rows, labels


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

HEADER = """
# SIA: Selective Image Analogy
**SIGGRAPH Asia 2026** · Give an example pair **A → A'** that shows several edits, list those edits,
pick which ones to **keep**, and SIA applies only those to **B**. The other edits are suppressed.

**Built with Qwen.** This demo runs the SIA adapters on Qwen-Image-2.1.
Non-commercial research demo.
"""

FOOTER = """
---
**Built with Qwen.** SIA (Selective Image Analogy) is fine-tuned from
[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) and distributed under the
Qwen RESEARCH LICENSE AGREEMENT (non-commercial research and evaluation only; see `LICENSE` and `NOTICE`).
Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory
Technology Co., Ltd. All Rights Reserved. This demo is not affiliated with or endorsed by the Qwen team,
Alibaba, or Black Forest Labs.

**Privacy.** Uploaded images are processed on Hugging Face infrastructure to produce the result; this app
does not store them or log them. Gradio's temporary files are deleted after about an hour. Only upload
images you have the right to use, and do not upload images of people without their consent.

**Use.** Research demo: results can be wrong or unexpected and there is no content filter. Do not use it
to deceive, harass, or create non-consensual edits of real people.

**Example images** come from the [SIA Multi-Edit Analogy dataset](https://huggingface.co/datasets/selective-image-analogy/SIA-Multi-Edit-Analogy)
(`analogy_benchmark` split, unchanged), licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Edited images were generated with [FLUX.2 [klein]](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)
(FLUX Non-Commercial License); 2-edit recipes build on
[Relation252K](https://huggingface.co/datasets/handsomeWilliam/Relation252K) (Apache-2.0).
"""


def build_demo():
    with gr.Blocks(title="SIA: Selective Image Analogy", delete_cache=(3600, 3600)) as demo:
        gr.Markdown(HEADER)
        if STUB:
            gr.Markdown("> **Stub mode** (`SIA_DEMO_STUB=1`): no model loaded, the result is B unchanged.")
        with gr.Row():
            a = gr.Image(label="A (original)", type="pil", height=320)
            a_prime = gr.Image(label="A' (edited)", type="pil", height=320)
            b = gr.Image(label="B (your image)", type="pil", height=320)
            out = gr.Image(label="B' (result)", type="pil", height=320, interactive=False,
                           format="png")  # Gradio's default (webp) is lossy
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("**Edits between A and A'** (2 to 4, one per box, in order)")
                boxes = [gr.Textbox(label=f"Edit {i + 1}", max_lines=1,
                                    placeholder="e.g. add a santa hat" if i == 0 else "")
                         for i in range(MAX_EDITS)]
            with gr.Column(scale=2):
                kept = gr.CheckboxGroup(
                    choices=[(f"Edit {i + 1} (empty)", str(i + 1)) for i in range(MAX_EDITS)],
                    label="Edits to keep (unticked edits are suppressed)")
                with gr.Accordion("Advanced", open=False):
                    seed = gr.Number(label="Seed", value=0, precision=0, minimum=0, maximum=2**31 - 1)
                    steps = gr.Slider(label="Steps", minimum=20, maximum=50, step=1, value=DEFAULT_STEPS)
                    resolution = gr.Radio(label="Resolution (target area side; 640 is untested)",
                                          choices=[512, 640], value=DEFAULT_RESOLUTION)
                btn = gr.Button("Run", variant="primary")
                info = gr.Markdown()

        for box in boxes:
            box.change(kept_choices, inputs=boxes, outputs=kept, api_visibility="private")
        btn.click(run, inputs=[a, a_prime, b, *boxes, kept, seed, steps, resolution],
                  outputs=[out, info], api_name="run")

        rows, labels = example_rows()
        gr.Examples(examples=rows, inputs=[a, a_prime, b, *boxes, kept], example_labels=labels,
                    cache_examples=False, label="Examples from the SIA benchmark",
                    api_visibility="public")
        gr.Markdown(FOOTER)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
