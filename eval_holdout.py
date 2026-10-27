import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from models import ScalePredictor
from dataset import ScaleDataset, collate_fn
from eval import evaluate_model # 复用你之前写好的评估函数

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print("加载留出集 (Hold-out Set)...")
    with open(os.path.join(CACHE_DIR, "train_cache_holdout.pkl"), "rb") as f:
        holdout_data = pickle.load(f)
    holdout_ds = ScaleDataset(holdout_data)
    holdout_loader = DataLoader(holdout_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"留出集总帧数: {len(holdout_ds)}")

    # ★ 注意：必须使用和训练时一模一样的 Anchors！
    # 如果你之前没保存，就用 4000 训练集的分布来算，保证一致性。
    # 这里直接用你之前训练跑出来的 Anchor 逻辑：
    with open(os.path.join(CACHE_DIR, f"train_cache_{NUM_TRAIN_FRAMES}.pkl"), "rb") as f:
        train_data = pickle.load(f)
    S_gt_train = np.array([s["S_gt"] for s in train_data])
    anchors = np.percentile(S_gt_train, [10, 25, 40, 60, 75, 90]).tolist()

    # 加载你之前训练好的模型
    model = ScalePredictor(anchors=anchors).to(DEVICE)
    ckpt_path = os.path.join(CKPT_DIR, "scale_predictor_main.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"✅ 成功加载模型: {ckpt_path}\n")

    # 跑测试
    res_mlp = evaluate_model(model, holdout_loader, "mlp")
    res_oracle = evaluate_model(None, holdout_loader, "oracle")

    print(f"留出集盲测结果 [MLP预测]:")
    print(
        f"  AbsRel={res_mlp['depth']['AbsRel']:.4f} | RMSE={res_mlp['depth']['RMSE']:.3f} | δ1={res_mlp['depth']['delta1']:.3f} | MAE_S={res_mlp['scale']['MAE_S']:.3f}")

    print(f"留出集盲测结果 [Oracle上限]:")
    print(
        f"  AbsRel={res_oracle['depth']['AbsRel']:.4f} | RMSE={res_oracle['depth']['RMSE']:.3f} | δ1={res_oracle['depth']['delta1']:.3f} | MAE_S={res_oracle['scale']['MAE_S']:.3f}")
    # 保存结果
    out_json = os.path.join(CKPT_DIR, "eval_holdout_results.json")
    with open(out_json, "w") as f:
        json.dump({"mlp": res_mlp, "oracle": res_oracle}, f, indent=4)
    print(f"\n✅ 盲测结果已保存至 {out_json}")