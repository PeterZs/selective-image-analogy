"""Image metrics: CLIP directional similarity of selective outputs, and full-transfer quality
against ground truth (LPIPS, DINOv2 distance, CLIP-I)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sia.data import load_split
from sia.eval.results import _items, _jsonl_read, _load_pred, full_path, keep_tag

CLIP_REPO = "openai/clip-vit-large-patch14"


DINO_REPO = "facebook/dinov2-base"


def _clip_feats(out):
    """Projected CLIP embedding from get_{image,text}_features: a tensor in
    older transformers, a BaseModelOutputWithPooling (pooler_output) in v5."""
    return out if hasattr(out, "float") else out.pooler_output


def score_clip(args, cases):
    """CLIP directional scores of every unscored item, appended to ``<args.out>/clip.jsonl``.

    ``clip_dir_img`` = cos(E(out) - E(B), E(A') - E(A)) (the output moves B as the demo moves A);
    ``clip_dir_ret`` = cos(E(out) - E(B), T("a photo. <retained edits>") - T("a photo")).
    """
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
    """Full-transfer quality of ``args.names`` against ground truth B' (LPIPS, DINOv2 distance, CLIP-I),
    written to ``<args.out>/full_transfer.jsonl`` (rows of other shards are kept)."""
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
