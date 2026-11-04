---
title: "SIA: Selective Image Analogy"
emoji: 🎨
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: other
license_name: qwen-research
license_link: LICENSE
models:
  - Qwen/Qwen-Image-2.1
  - selective-image-analogy/SIA-weights
datasets:
  - selective-image-analogy/SIA-Multi-Edit-Analogy
short_description: Selective Image Analogy (SIGGRAPH Asia 2026)
tags:
  - image-analogy
  - image-editing
  - siggraph-asia-2026
---

# SIA: Selective Image Analogy

**Built with Qwen.**

This is a non-commercial research demo of **SIA** (SIGGRAPH Asia 2026). You give an example pair
**A → A'** showing several edits, list those edits (2 to 4), and tick the ones to **keep**. SIA applies
only the kept edits to a new image **B**; the unticked edits are suppressed.

- **Model:** the SIA stage-2 adapters (AnaLoRA + ExLoRA, step 5000) from
  [`selective-image-analogy/SIA-weights`](https://huggingface.co/selective-image-analogy/SIA-weights)
  (`qwen_image21/exlora`), fine-tuned from [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1).
  This demo does not use the concept suppression vector (CSV).
- **Defaults:** 40 steps, 512 target resolution, seed 0, no classifier-free guidance.
- **Inputs:** images up to 4096 px per side and 20 megapixels.
- **Code:** <https://github.com/alimohammadiamirhossein/selective-image-analogy>

## Examples

The demo has a gallery of **20 examples** from the `analogy_benchmark` split of the SIA dataset, with
2, 3 and 4 edits, people, animals and objects, and different sets of kept edits. Clicking a card loads
A, A', B, the edits and the kept edits, and shows the precomputed result (seed 0, 40 steps, resolution 512)
straight away; **Run** recomputes it live. These examples were selected because the automatic judge scored
them highly and they look right to us; results on your own images vary. Eleven of the 20 use an edit recipe
or images that were also part of the training data (all 2-edit and all non-creature examples, plus one more;
the app marks them when you open them); the other nine are unseen. The selection criteria and the judge
scores are in `examples/examples.json`.

The repository ships only the gallery cards and the precomputed results (`examples/cards`,
`examples/results`). The full-resolution inputs are the dataset's original PNGs: the app downloads the
benchmark parquet file (about 90 MB, pinned revision) from the public dataset once and reads each
example by its case id.

## License and attribution

- **Qwen.** The SIA adapters are derived from Qwen-Image-2.1 and distributed under the
  **Qwen RESEARCH LICENSE AGREEMENT**, which allows non-commercial research and evaluation only. A copy is in
  [`LICENSE`](LICENSE) and the notice is in [`NOTICE`](NOTICE). Qwen is licensed under the Qwen RESEARCH
  LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.
- **No endorsement.** This demo is made by the SIA authors. It is not affiliated with or endorsed by the
  Qwen team, Alibaba, or Black Forest Labs.
- **Example images.** The 20 examples use images from the `analogy_benchmark` split of the
  [SIA Multi-Edit Analogy dataset](https://huggingface.co/datasets/selective-image-analogy/SIA-Multi-Edit-Analogy),
  licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The inputs are loaded unchanged
  from the dataset; the gallery cards in `examples/cards` are downscaled composites of those images and the
  model's result, and `examples/results` holds the model's outputs (JPEG). Edited images were generated with
  [FLUX.2 [klein]](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) and are subject to the
  [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/blob/main/LICENSE.md).
  The 2-edit recipes build on [Relation252K](https://huggingface.co/datasets/handsomeWilliam/Relation252K)
  (Apache-2.0). Per-example sources are listed in `examples/examples.json`.

## Privacy and use

- Uploaded images are processed on Hugging Face infrastructure to produce the result. The app does not
  store or log them, and Gradio's temporary files are deleted after about an hour.
- Only upload images you have the right to use. Do not upload images of people without their consent.
- Results can be wrong or unexpected, and there is no content filter. Do not use the demo to deceive, to
  harass, or to make non-consensual edits of real people.

## Citation

```bibtex
@inproceedings{alimohammadi2026sia,
  title     = {SIA: Selective Image Analogy},
  author    = {Alimohammadi, Amirhossein and Izadpanah, Kian and Kariminia, Bardiya and Chrysanthou, Yiorgos and Mahdavi-Amiri, Ali},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026},
  doi       = {10.1145/3829340.3842205}
}
```

## Running locally

```bash
pip install -r requirements.txt
python app.py                  # needs a GPU with ~48 GB
SIA_DEMO_STUB=1 python app.py  # UI only, no model; the result is B unchanged
```
