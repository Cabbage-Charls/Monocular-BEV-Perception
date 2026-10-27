import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from models import ScalePredictor
from dataset import load_and_split, collate_fn
from depth_utils import depth_metrics, scale_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def evaluate_model(model, loader, mode="mlp", anchors=None):
    if model is not None: model.eval()
    S_pred_all, S_gt_all = [], []
    Z_pred_all, Z_gt_all = [], []

    for batch in tqdm(loader, desc=f"评估 [{mode}]", leave=False):
        feat, img = batch["feat_6d"].to(DEVICE), batch["img_224"].to(DEVICE)
        S_gt = batch["S_gt"].to(DEVICE)

        if mode == "mlp":
            S_pred, _, _ = model(feat, img)
        elif mode == "fixed_30":
            S_pred = torch.full_like(S_gt, 30.0)
        elif mode == "fixed_20":
            S_pred = torch.full_like(S_gt, 20.0)
        elif mode == "oracle":
            S_pred = S_gt

        S_pred_all.append(S_pred.cpu().numpy())
        S_gt_all.append(S_gt.cpu().numpy())

        for i in range(len(S_pred)):
            dn = batch["dn_at_lidar"][i].to(DEVICE)
            z = batch["z_lidar"][i].to(DEVICE)
            Z_pred = (S_pred[i] * (1.0 - dn)).cpu().numpy()
            Z_pred_all.append(Z_pred)
            Z_gt_all.append(z.cpu().numpy())

    S_pred_all = np.concatenate(S_pred_all)
    S_gt_all = np.concatenate(S_gt_all)
    Z_pred_all = np.concatenate(Z_pred_all)
    Z_gt_all = np.concatenate(Z_gt_all)

    out = {"depth": depth_metrics(Z_pred_all, Z_gt_all),
           "scale": scale_metrics(S_pred_all, S_gt_all)}
    return out

if __name__ == "__main__":
    _, val_ds = load_and_split()
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # 计算动态 anchors
    train_ds, _ = load_and_split()
    S_gt_all = np.array([s["S_gt"] for s in train_ds.samples])
    anchors = np.percentile(S_gt_all, [10, 25, 40, 60, 75, 90]).tolist()

    # 加载你的模型
    model = ScalePredictor(anchors=anchors).to(DEVICE)
    ckpt_path = os.path.join(CKPT_DIR, "scale_predictor_main.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print("模型加载成功！\n")

    print(f"{'方案':<12} | {'AbsRel↓':<8} | {'RMSE↓':<8} | {'δ1↑':<8} | {'MAE_S↓':<8}")
    print("-" * 60)

    for mode, name in [("fixed_30", "固定 S=30"), ("fixed_20", "固定 S=20"),
                       ("mlp", "MLP预测(本文)"), ("oracle", "Oracle(上限)")]:
        res = evaluate_model(model if mode == "mlp" else None, val_loader, mode)
        d, s = res["depth"], res["scale"]
        mae_s = s['MAE_S'] if mode not in ["fixed_30", "fixed_20"] else abs((30 if mode=="fixed_30" else 20) - np.mean(S_gt_all))
        print(f"{name:<12} | {d['AbsRel']:<8.3f} | {d['RMSE']:<8.3f} | {d['delta1']:<8.3f} | {mae_s:<8.3f}")
    print("-" * 60)
    print("注：AbsRel 和 RMSE 越低越好，δ1（1.25倍准确率）越高越好。")