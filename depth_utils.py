import numpy as np


def compute_stat_feature(depth_norm, valid_mask=None):
    """8 维统计特征（去掉了 fx 和 cx，提升跨相机迁移能力）"""
    if valid_mask is not None:
        d_in = depth_norm[valid_mask]
        d_in = d_in[(d_in > 0.10) & (d_in < 0.92)]
    else:
        d_in = depth_norm[(depth_norm > 0.15) & (depth_norm < 0.85)]

    if len(d_in) < 100:
        d_in = depth_norm.flatten()
        d_in = d_in[(d_in > 0.05) & (d_in < 0.95)]
    if len(d_in) < 10:
        d_in = depth_norm.flatten()

    mu = float(np.mean(d_in))
    sigma = float(np.std(d_in))
    d_max = float(np.max(d_in))
    d_min = float(np.min(d_in))
    near_ratio = float(np.sum(d_in < 0.40) / len(d_in))
    mid_ratio = float(np.sum((d_in >= 0.40) & (d_in < 0.70)) / len(d_in))

    return np.array([mu, sigma, d_max, d_min, near_ratio, mid_ratio], dtype=np.float32)


def compute_S_gt(dn_at_lidar, z_lidar, eps=1e-3):
    """从 LiDAR 反推 S_gt，按距离分层加权中位数"""
    denom = 1.0 - dn_at_lidar
    valid = denom > eps
    if valid.sum() < 20:
        return None
    S_per_pt = z_lidar[valid] / denom[valid]
    z_valid = z_lidar[valid]

    weights = np.ones_like(z_valid)
    weights[(z_valid >= 20) & (z_valid < 40)] = 0.7
    weights[z_valid >= 40] = 0.3

    order = np.argsort(S_per_pt)
    S_sorted, w_sorted = S_per_pt[order], weights[order]
    cumsum = np.cumsum(w_sorted)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    S_gt = float(S_sorted[median_idx])

    return S_gt if 5.0 < S_gt < 45.0 else None


def depth_metrics(z_pred, z_gt, eps=1e-6):
    """深度估计标准指标：AbsRel / RMSE / δ1/δ2/δ3"""
    z_pred, z_gt = np.asarray(z_pred), np.asarray(z_gt)
    if len(z_pred) < 5:
        return None
    abs_rel = np.mean(np.abs(z_pred - z_gt) / (z_gt + eps))
    rmse = np.sqrt(np.mean((z_pred - z_gt) ** 2))
    ratio = np.maximum(z_pred / (z_gt + eps), z_gt / (z_pred + eps))
    return {"AbsRel": float(abs_rel), "RMSE": float(rmse),
            "delta1": float(np.mean(ratio < 1.25)),
            "delta2": float(np.mean(ratio < 1.25**2)),
            "delta3": float(np.mean(ratio < 1.25**3)),
            "n_points": len(z_pred)}


def scale_metrics(S_pred, S_gt):
    """评估 S 的准确度"""
    S_pred, S_gt = np.asarray(S_pred), np.asarray(S_gt)
    return {"MAE_S": float(np.mean(np.abs(S_pred - S_gt))),
            "RelErr_S": float(np.mean(np.abs(S_pred - S_gt) / (S_gt + 1e-6)))}