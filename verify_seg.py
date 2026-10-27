import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import cv2
import numpy as np
import torch
from transformers import (SegformerImageProcessor,
                          SegformerForSemanticSegmentation)
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
name = "nvidia/segformer-b0-finetuned-ade-512-512"

proc = SegformerImageProcessor.from_pretrained(name)
model = SegformerForSemanticSegmentation.from_pretrained(
    name).to(DEVICE).eval()

for fid in ["000000", "000001", "000100"]:
    img = cv2.imread(rf"E:\Project\Source Code\training\image_2\{fid}.png")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    inputs = proc(images=img_rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = F.interpolate(logits, size=(H, W),
                         mode="bilinear").argmax(1)[0].cpu().numpy()

    unique_ids, counts = np.unique(pred, return_counts=True)
    order = np.argsort(-counts)

    print(f"\n=== {fid} 前 15 大类 ===")
    for i in order[:15]:
        uid = int(unique_ids[i])
        ratio = float(counts[i]) / (H * W)
        label = model.config.id2label.get(uid, "unknown")
        print(f"  id={uid:3d}  ratio={ratio:.3f}  {label}")

    # 可视化
    vis = img_rgb.copy()
    rng = np.random.RandomState(42)
    colors = {int(uid): rng.randint(0, 255, 3) for uid in unique_ids[:15]}
    for uid in unique_ids[:15]:
        vis[pred == uid] = vis[pred == uid] * 0.3 + colors[int(uid)] * 0.7

    out = rf"E:\Project\Source Code\figures\seg_vis_{fid}.png"
    cv2.imwrite(out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"  Saved {out}")