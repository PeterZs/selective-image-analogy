"""SIA: Selective Image Analogy — Hugging Face Space demo.

Given an example pair A -> A' showing several edits and a new image B, SIA
applies only the edits you keep to B (the rest are suppressed by ExLoRA).
Backbone: Qwen-Image-2.1 with the SIA stage-2 adapters (AnaLoRA + ExLoRA,
step 5000). No concept suppression vector (CSV) is used.

Set SIA_DEMO_STUB=1 to run the UI without a GPU or model: Run returns B
unchanged, while the example gallery and its precomputed results work as usual.
"""

import html
import json
import os
import tempfile
import threading
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
        SIA_WEIGHTS, device="cuda", subfolder=SIA_SUBFOLDER, revision=SIA_REVISION,
        base=BASE_MODEL, base_revision=BASE_REVISION,
        token=os.getenv("HF_TOKEN") or None,  # public repos: no token needed
    )


def generate(pipe, a, a_prime, b, edits, kept, seed, steps, resolution):
    """One B' image at the generated size. ``kept`` are 0-based indices into ``edits``."""
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
    """Generate B' on the GPU and return it at B's size."""
    out = generate(PIPE, a, a_prime, b, edits, kept, seed, steps, resolution)
    return out if out.size == b.size else out.resize(b.size)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

MAX_EDITS = 4
MIN_EDITS = 2
MAX_EDIT_CHARS = 200
MAX_IMAGE_SIDE = 4096
MAX_IMAGE_PIXELS = 20_000_000


class InputError(ValueError):
    """Invalid user input; shown to the user as a Gradio error."""


def clean_edits(raw):
    """Strip each box; drop trailing periods (the prompt adds its own)."""
    return [(t or "").strip().rstrip(".").strip() for t in raw]


def validate(a, a_prime, b, raw_edits, kept_labels):
    """Return (edits, kept_indices, note) or raise InputError."""
    missing = [name for name, img in (("A", a), ("A'", a_prime), ("B", b)) if img is None]
    if missing:
        raise InputError(f"Please provide image(s): {', '.join(missing)}.")
    for name, img in (("A", a), ("A'", a_prime), ("B", b)):
        w, h = img.size
        if max(w, h) > MAX_IMAGE_SIDE or w * h > MAX_IMAGE_PIXELS:
            raise InputError(f"Image {name} is too large ({w}x{h}); the limit is {MAX_IMAGE_SIDE} px per side "
                             f"and {MAX_IMAGE_PIXELS // 1_000_000} megapixels.")

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
    """Validate the inputs, generate B' and describe what was kept and suppressed."""
    try:
        edits, kept, note = validate(a, a_prime, b, [e1, e2, e3, e4], kept_labels)
    except InputError as exc:
        raise gr.Error(str(exc), print_exception=False) from None
    out = run_on_gpu(a, a_prime, b, edits, kept, int(seed), int(steps), int(resolution))
    kept_txt = "; ".join(f"Edit {i + 1} ({edits[i]})" for i in kept)
    return out, f"Kept: {kept_txt}.\n\n{note}"


def kept_choices(*raw):
    """Labels of the keep checkboxes from the current edit boxes."""
    labels = [(f"Edit {i + 1}: {t}" if t else f"Edit {i + 1} (empty)", str(i + 1))
              for i, t in enumerate(clean_edits(raw))]
    return gr.update(choices=labels)




# ---------------------------------------------------------------------------
# Examples: cards and precomputed results are bundled; the full-resolution
# inputs are the original PNGs of the public dataset, fetched by case id.
# ---------------------------------------------------------------------------

EXAMPLES_DIR = HERE / "examples"
EXAMPLES_META = json.loads((EXAMPLES_DIR / "examples.json").read_text())
EXAMPLES = EXAMPLES_META["examples"]
EXAMPLES_BY_ID = {ex["id"]: ex for ex in EXAMPLES}
DATASET = EXAMPLES_META["source"]["dataset"]
DATASET_REVISION = EXAMPLES_META["source"]["revision"]
DATASET_PARQUET = EXAMPLES_META["source"]["parquet"]
INPUT_CACHE = Path(tempfile.gettempdir()) / "sia_examples"
_download_lock = threading.Lock()


def _dataset_parquet():
    """Local path of the benchmark parquet (about 90 MB, downloaded once into the HF cache)."""
    from huggingface_hub import hf_hub_download

    with _download_lock:
        return hf_hub_download(DATASET, DATASET_PARQUET, repo_type="dataset", revision=DATASET_REVISION)


def example_inputs(ex):
    """Paths of the original A, A', B PNGs of an example (cached after the first load)."""
    folder = INPUT_CACHE / ex["folder"]
    paths = [folder / f"{k}.png" for k in ("a", "a_prime", "b")]
    if all(p.exists() for p in paths):
        return paths
    import pyarrow.parquet as pq

    table = pq.read_table(_dataset_parquet(), columns=["id", "a", "a_prime", "b"],
                          filters=[("id", "==", ex["id"])])
    if table.num_rows != 1:
        raise RuntimeError(f"case {ex['id']!r} not found in {DATASET}@{DATASET_REVISION[:7]}")
    row = table.to_pylist()[0]
    folder.mkdir(parents=True, exist_ok=True)
    for key, path in zip(("a", "a_prime", "b"), paths):
        with tempfile.NamedTemporaryFile(dir=folder, delete=False, suffix=".part") as f:
            f.write(row[key]["bytes"])           # the dataset's own bytes: inputs are reproduced exactly
        os.replace(f.name, path)
    return paths


def _prefetch_dataset():
    try:
        _dataset_parquet()
    except Exception as exc:  # the app still works; example inputs report the error when clicked
        print(f"[examples] could not prefetch {DATASET}: {exc}", flush=True)


def edit_chips(edits, kept):
    """HTML chips: kept edits in green, suppressed edits in red and struck through."""
    chips = []
    for i, e in enumerate(edits):
        cls, verb = ("sia-keep", "keep") if i in kept else ("sia-drop", "suppress")
        chips.append(f'<span class="sia-chip {cls}">{verb}: {html.escape(e)}</span>')
    return '<div class="sia-chips">' + "".join(chips) + "</div>"


def example_caption(ex):
    n, k = ex["num_edits"], len(ex["kept"])
    return f"{n} edits, keep {k}: " + "; ".join(ex["edits"][i] for i in ex["kept"])


EDIT_FILTERS = ["All", "2 edits", "3 edits", "4 edits"]


def gallery_items(choice="All"):
    """(gallery value, case ids in gallery order) for a filter choice."""
    n = None if choice == "All" else int(choice.split()[0])
    shown = [ex for ex in EXAMPLES if n is None or ex["num_edits"] == n]
    return [(str(EXAMPLES_DIR / ex["card"]), example_caption(ex)) for ex in shown], [ex["id"] for ex in shown]


def load_example(case_id):
    """Values for A, A', B, the four edit boxes, the kept checkboxes, B' and the info text."""
    ex = EXAMPLES_BY_ID[case_id]
    edits = ex["edits"] + [""] * (MAX_EDITS - len(ex["edits"]))
    kept = [str(k + 1) for k in ex["kept"]]
    info = (f"**Example `{ex['id']}`** · precomputed result (seed 0, {DEFAULT_STEPS} steps, "
            f"{DEFAULT_RESOLUTION}). Press **Run** to recompute it live.\n\n"
            + edit_chips(ex["edits"], ex["kept"]))
    if ex["seen_in_training"]:
        info += ("\n\n*This example's edit recipe or images were also part of the training data. "
                 "Examples without this note are unseen by the model.*")
    try:
        a, a_prime, b = (str(p) for p in example_inputs(ex))
    except Exception as exc:
        a = a_prime = b = None
        msg = (f"Could not load this example's input images from the dataset {DATASET} ({type(exc).__name__}). "
               "The precomputed result is shown; to run SIA, check the connection or upload your own images.")
        gr.Warning(msg)
        info += f"\n\n**{msg}**"
    return [a, a_prime, b, *edits, kept, str(EXAMPLES_DIR / ex["result"]), info]


def on_gallery_select(shown_ids, evt: gr.SelectData):
    return load_example(shown_ids[evt.index])


def on_filter(choice):
    items, ids = gallery_items(choice)
    return gr.update(value=items, selected_index=None), ids


def example_api(case_id: str) -> dict:
    """API view of ``load_example`` (used by the tests): edits, kept and the file paths."""
    a, a_prime, b, e1, e2, e3, e4, kept, result, info = load_example(case_id)
    return {"a": a, "a_prime": a_prime, "b": b, "edits": [e1, e2, e3, e4], "kept": kept,
            "result": result, "info": info}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

HEADER = """
# SIA: Selective Image Analogy
**SIGGRAPH Asia 2026** · Give an example pair **A → A'** that shows several edits, list those edits,
pick which ones to **keep**, and SIA applies only those to **B**. The other edits are suppressed.

**Built with Qwen.** This demo runs the SIA adapters on Qwen-Image-2.1.
Non-commercial research demo. Pick an example below or upload your own images.
"""

EXAMPLES_NOTE = (
    f"**{len(EXAMPLES)} examples from the SIA benchmark.** Click a card to load it with its precomputed result. "
    "These examples were selected because the automatic judge scored them highly and they look right to us; "
    "results on your own images vary. "
    f"{sum(ex['seen_in_training'] for ex in EXAMPLES)} of them use an edit recipe or images that were also part "
    "of the training data (marked when you open them); the others are unseen."
)

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
(`analogy_benchmark` split; inputs are loaded unchanged from the dataset, the cards are downscaled composites),
licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Edited images were generated with [FLUX.2 [klein]](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)
(FLUX Non-Commercial License); 2-edit recipes build on
[Relation252K](https://huggingface.co/datasets/handsomeWilliam/Relation252K) (Apache-2.0).
The example results are outputs of this demo's model.
"""

CSS = """
.sia-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.sia-chip { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.9em; line-height: 1.5; }
.sia-keep { background: #dcfce7; color: #14532d; border: 1px solid #86efac; }
.sia-drop { background: #fee2e2; color: #7f1d1d; border: 1px solid #fca5a5; text-decoration: line-through; }
/* Gallery: show every card at its own aspect ratio, without an inner scroll box. */
#sia-gallery, #sia-gallery .gallery-container { height: auto !important; }
#sia-gallery .grid-wrap { height: auto !important; max-height: none !important; overflow: visible !important; }
#sia-gallery .thumbnail-item { aspect-ratio: 724 / 328 !important; height: auto !important; }
#sia-gallery .caption-label { display: none; }  /* the edits are drawn on the card itself */
@media (max-width: 760px) {
  #sia-gallery .grid-container { grid-template-columns: 1fr !important; }
}
"""


# After a card is clicked, bring the loaded inputs and the result into view.
SCROLL_TO_INPUTS = "() => document.getElementById('sia-inputs').scrollIntoView({behavior: 'smooth', block: 'start'})"


def build_demo():
    """Build the Gradio interface."""
    with gr.Blocks(title="SIA: Selective Image Analogy", delete_cache=(3600, 3600)) as demo:
        gr.Markdown(HEADER)
        if STUB:
            gr.Markdown("> **Stub mode** (`SIA_DEMO_STUB=1`): no model loaded, Run returns B unchanged.")
        with gr.Row(elem_id="sia-inputs"):
            a = gr.Image(label="A (original)", type="pil", height=320, min_width=150)
            a_prime = gr.Image(label="A' (edited)", type="pil", height=320, min_width=150)
            b = gr.Image(label="B (your image)", type="pil", height=320, min_width=150)
            out = gr.Image(label="B' (result)", type="pil", height=320, min_width=150, interactive=False,
                           format="png")  # Gradio's default (webp) is lossy
        with gr.Row():
            with gr.Column(scale=3, min_width=280):
                gr.Markdown("**Edits between A and A'** (2 to 4, one per box, in order)")
                boxes = [gr.Textbox(label=f"Edit {i + 1}", max_lines=1,
                                    placeholder="e.g. add a santa hat" if i == 0 else "")
                         for i in range(MAX_EDITS)]
            with gr.Column(scale=2, min_width=280):
                kept = gr.CheckboxGroup(
                    choices=[(f"Edit {i + 1} (empty)", str(i + 1)) for i in range(MAX_EDITS)],
                    label="Edits to keep (unticked edits are suppressed)")
                with gr.Accordion("Advanced", open=False):
                    seed = gr.Number(label="Seed", value=0, precision=0, minimum=0, maximum=2**31 - 1)
                    steps = gr.Slider(label="Steps", minimum=20, maximum=50, step=1, value=DEFAULT_STEPS)
                    resolution = gr.Radio(label="Resolution (target area side)",
                                          choices=[str(DEFAULT_RESOLUTION)], value=str(DEFAULT_RESOLUTION))
                btn = gr.Button("Run", variant="primary")
                info = gr.Markdown()

        gr.Markdown("## Examples")
        gr.Markdown(EXAMPLES_NOTE)
        items, ids = gallery_items()
        shown_ids = gr.State(ids)
        edit_filter = gr.Radio(EDIT_FILTERS, value="All", label="Number of edits", show_label=False)
        gallery = gr.Gallery(value=items, columns=2, height="auto", object_fit="contain",
                             allow_preview=False, show_label=False, buttons=[], elem_id="sia-gallery")
        gr.Markdown(FOOTER)

        for box in boxes:
            box.change(kept_choices, inputs=boxes, outputs=kept, api_visibility="private")
        btn.click(run, inputs=[a, a_prime, b, *boxes, kept, seed, steps, resolution],
                  outputs=[out, info], api_name="run")
        edit_filter.change(on_filter, inputs=edit_filter, outputs=[gallery, shown_ids], api_visibility="private")
        gallery.select(on_gallery_select, inputs=shown_ids, outputs=[a, a_prime, b, *boxes, kept, out, info],
                       api_visibility="private",
                       ).then(None, js=SCROLL_TO_INPUTS)
        gr.api(example_api, api_name="example")
    return demo


demo = build_demo()
threading.Thread(target=_prefetch_dataset, daemon=True).start()

if __name__ == "__main__":
    demo.queue(max_size=20).launch(css=CSS)
