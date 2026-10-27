import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pickle
import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torchvision.transforms as T

from config import *
from calib_utils import read_kitti_full_calib, load_velodyne, lidar_to_image, valid_lidar_mask
from semantic_seg import SemanticSegmenter
from depth_utils import compute_stat_feature, compute_S_gt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_NORM = T.Compose([
    T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_holdout():
    print(f"[HOLDOUT] device={DEVICE}")
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_MODEL_NAME).to(DEVICE).eval()
    seg_model = SemanticSegmenter(device=DEVICE)

    img_files = sorted([f for f in os.listdir(TRAIN_IMG) if f.endswith(".png")])
    # ★ 核心改动：只取 4000 之后的数据
    holdout_files = img_files[4000:]
    print(f"[HOLDOUT] 留出集共计 {len(holdout_files)} 帧 (从 {holdout_files[0]} 到 {holdout_files[-1]})")

    partial_path = os.path.join(CACHE_DIR, "train_cache_holdout_partial.pkl")
    done_ids, cache = set(), []
    if os.path.exists(partial_path):
        with open(partial_path, "rb") as f:
            cache = pickle.load(f)
            done_ids = set(s["file_id"] for s in cache)
        print(f"[RESUME] 已有 {len(cache)} 条")

    for fname in tqdm(holdout_files, desc="Building Holdout Cache"):
        fid = os.path.splitext(fname)[0]
        if fid in done_ids:
            continue
        img_path = os.path.join(TRAIN_IMG, fname)
        velo_path = os.path.join(TRAIN_VELO, fid + ".bin")
        calib_path = os.path.join(TRAIN_CALIB, fid + ".txt")
        if not all(os.path.exists(p) for p in [img_path, velo_path, calib_path]):
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        H, W = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            # 深度
            inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                depth = depth_model(**inputs).predicted_depth.squeeze().cpu().numpy()
            depth = cv2.resize(depth, (W, H))
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

            # 语义
            semantic_valid, _ = seg_model.get_valid_mask(img_rgb)

            # LiDAR
            P2, R0, Tr = read_kitti_full_calib(calib_path)
            pts = load_velodyne(velo_path)
            u, v, z = lidar_to_image(pts[:, :3], P2, R0, Tr)
            mask = valid_lidar_mask(u, v, z, H, W, LIDAR_MIN_DEPTH, LIDAR_MAX_DEPTH)

            u_v, v_v, z_v = u[mask].astype(int), v[mask].astype(int), z[mask].astype(np.float32)
            valid_at_lidar = semantic_valid[v_v, u_v]
            if valid_at_lidar.sum() < LIDAR_MIN_POINTS: continue
            u_v, v_v, z_v = u_v[valid_at_lidar], v_v[valid_at_lidar], z_v[valid_at_lidar]

            dn_at_lidar = depth_norm[v_v, u_v].astype(np.float32)
            S_gt = compute_S_gt(dn_at_lidar, z_v)
            if S_gt is None: continue

            feat_6d = compute_stat_feature(depth_norm, valid_mask=semantic_valid)
            img_224 = IMG_NORM(img_rgb).numpy().astype(np.float32)

            cache.append({
                "file_id": fid,
                "feat_8d": feat_6d,
                "img_224": img_224,
                "dn_at_lidar": dn_at_lidar,
                "z_lidar": z_v,
                "S_gt": float(S_gt),
            })

            if len(cache) % 100 == 0:
                with open(partial_path, "wb") as f:
                    pickle.dump(cache, f, protocol=4)
        except Exception as e:
            continue

    out_path = os.path.join(CACHE_DIR, "train_cache_holdout.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"\n[HOLDOUT] 留出集缓存完成，成功保存 {len(cache)} 帧 → {out_path}")

if __name__ == "__main__":
    build_holdout()