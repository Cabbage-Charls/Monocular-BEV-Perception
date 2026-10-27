import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# 注意：请确认这里的路径和你的实际路径一致
pkl_path = r"E:\Project\Source Code\cache\train_cache_4000.pkl"

with open(pkl_path, "rb") as f:
    cache = pickle.load(f)

print(f"总共成功保存了: {len(cache)} 帧")
S_gt_all = [s["S_gt"] for s in cache]
print(f"S_gt 中位数: {np.median(S_gt_all):.2f}")
print(f"S_gt 标准差: {np.std(S_gt_all):.2f}")
print(f"特征维度: {cache[0]['feat_8d'].shape}")  # 必须是 (6,)

# 画图
plt.figure(figsize=(8, 4))
plt.hist(S_gt_all, bins=40, edgecolor='k')
plt.xlabel("S_gt (真实尺度)")
plt.ylabel("频数")
plt.title("S_gt 分布直方图")
plt.savefig(r"E:\Project\Source Code\figures\S_gt_hist.png", dpi=150)
print("图表已保存到 figures 文件夹")