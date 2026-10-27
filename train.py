import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from models import ScalePredictor
from dataset import load_and_split, collate_fn

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 支持在命令行传参命名实验，方便以后跑消融实验
EXP_NAME = sys.argv[1] if len(sys.argv) > 1 else "main"


def lidar_loss_batch(S_pred, dn_list, z_list, beta=0.5):
    """按距离分层加权的 Smooth L1 损失（远处的点少，权重加大）"""
    losses = []
    for i in range(len(dn_list)):
        dn, z = dn_list[i].to(DEVICE), z_list[i].to(DEVICE)
        Z_pred = S_pred[i] * (1.0 - dn)
        err_smooth = F.smooth_l1_loss(Z_pred, z, beta=beta, reduction="none")

        err_log = F.l1_loss(torch.log(Z_pred + 1e-6), torch.log(z + 1e-6), reduction="none")

        err = 0.8 * err_smooth + 0.2 * err_log

        w = torch.ones_like(z)
        w[(z >= 20) & (z < 40)] = 1.5
        w[z >= 40] = 2.5
        losses.append((err * w).mean())
    return torch.stack(losses).mean()


def total_loss(model_out, batch, anchors):
    S_pred, prob, cls_logits = model_out
    S_gt = batch["S_gt"].to(DEVICE)

    # 1. LiDAR 点的深度损失
    L_lidar = lidar_loss_batch(S_pred, batch["dn_at_lidar"], batch["z_lidar"])

    # 2. 尺度 S 的直接回归损失
    L_S = F.smooth_l1_loss(S_pred, S_gt, beta=2.0)

    # 3. 方差约束（防止同一个 batch 里的 S 塌缩成同一个值）
    if S_pred.numel() > 1:
        L_var = torch.relu(TARGET_MIN_STD - torch.std(S_pred))
    else:
        L_var = torch.tensor(0.0, device=DEVICE)

    # 4. Anchor 分类损失
    anchors_t = torch.tensor(anchors, device=DEVICE)
    cls_target = torch.argmin(torch.abs(S_gt.unsqueeze(1) - anchors_t.unsqueeze(0)), dim=1)
    L_anchor = F.cross_entropy(cls_logits, cls_target)

    # 总损失
    L = L_lidar + LAMBDA_S * L_S + LAMBDA_VAR * L_var + LAMBDA_ANCHOR * L_anchor
    return L, {"lidar": L_lidar.item(), "S": L_S.item(), "var": L_var.item()}


def train():
    train_ds, val_ds = load_and_split()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn)

    # ★ 从训练集动态计算 anchors（10%, 25%, 40%, 60%, 75%, 90% 分位数）
    S_gt_all = np.array([s["S_gt"] for s in train_ds.samples])
    anchors = np.percentile(S_gt_all, [10, 25, 40, 60, 75, 90]).tolist()
    print(f"[ANCHORS] 动态计算出的锚点: {[round(a, 2) for a in anchors]}")

    model = ScalePredictor(anchors=anchors).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH)

    best_val, patience_cnt = float("inf"), 0
    best_path = os.path.join(CKPT_DIR, f"scale_predictor_{EXP_NAME}.pth")
    print(f"开始训练，模型将保存至: {best_path}")

    history = {"train_loss": [], "val_mae": []}

    for epoch in range(MAX_EPOCH):
        model.train()
        epoch_loss = 0.0

        # 训练循环
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            feat, img = batch["feat_6d"].to(DEVICE), batch["img_224"].to(DEVICE)
            out = model(feat, img)
            L, info = total_loss(out, batch, anchors)

            optimizer.zero_grad()
            L.backward()
            # 梯度裁剪，防止偶尔的异常点导致梯度爆炸
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            epoch_loss += L.item()
            pbar.set_postfix({"loss": f"{L.item():.3f}"})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # 验证循环
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                feat, img = batch["feat_6d"].to(DEVICE), batch["img_224"].to(DEVICE)
                S_pred, _, _ = model(feat, img)
                val_mae += F.l1_loss(S_pred, batch["S_gt"].to(DEVICE)).item()
        val_mae /= len(val_loader)

        print(f"Epoch {epoch:3d} | TrainLoss={avg_loss:.4f} | Val MAE_S={val_mae:.4f}")

        history["train_loss"].append(avg_loss)
        history["val_mae"].append(val_mae)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label='Train Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(history["val_mae"], label='Val MAE_S (m)', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error (m)')
        plt.title('Validation Metric')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        # 保存到 figures 文件夹
        plt.savefig(os.path.join(FIG_DIR, f"loss_curve_{EXP_NAME}.png"), dpi=150)
        plt.close()

        # 保存最好模型 & 早停
        if val_mae < best_val:
            best_val = val_mae
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
            print(f"  --> 发现更好的模型，已保存 (Val MAE_S={val_mae:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"[EARLY STOP] 验证集连续 {PATIENCE} 轮未提升，提前结束训练！")
                break

    print(f"\n[DONE] 训练完成！最佳验证集误差: {best_val:.4f}")


if __name__ == "__main__":
    train()