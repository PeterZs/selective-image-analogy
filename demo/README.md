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

## License and attribution

- **Qwen.** The SIA adapters are derived from Qwen-Image-2.1 and distributed under the
  **Qwen RESEARCH LICENSE AGREEMENT**, which allows non-commercial research and evaluation only. A copy is in
  [`LICENSE`](LICENSE) and the notice is in [`NOTICE`](NOTICE). Qwen is licensed under the Qwen RESEARCH
  LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.
- **No endorsement.** This demo is made by the SIA authors. It is not affiliated with or endorsed by the
  Qwen team, Alibaba, or Black Forest Labs.
- **Example images** (`examples/`) are unchanged images from the `analogy_benchmark` split of the
  [SIA Multi-Edit Analogy dataset](https://huggingface.co/datasets/selective-image-analogy/SIA-Multi-Edit-Analogy),
  licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Edited images were generated with
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
