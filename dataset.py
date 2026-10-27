import pickle
import random
import torch
from torch.utils.data import Dataset
from config import CACHE_DIR, NUM_TRAIN_FRAMES, TRAIN_RATIO


class ScaleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "feat_6d": torch.from_numpy(s["feat_8d"]),  # 注意这里改名成6d了
            "img_224": torch.from_numpy(s["img_224"]),
            "dn_at_lidar": torch.from_numpy(s["dn_at_lidar"]),
            "z_lidar": torch.from_numpy(s["z_lidar"]),
            "S_gt": torch.tensor(s["S_gt"], dtype=torch.float32),
            "file_id": s["file_id"],
        }


def collate_fn(batch):
    return {
        "feat_6d": torch.stack([b["feat_6d"] for b in batch]),
        "img_224": torch.stack([b["img_224"] for b in batch]),
        "dn_at_lidar": [b["dn_at_lidar"] for b in batch],  # 变长，用list装
        "z_lidar": [b["z_lidar"] for b in batch],
        "S_gt": torch.stack([b["S_gt"] for b in batch]),
        "file_id": [b["file_id"] for b in batch],
    }


def load_and_split(seed=42):
    """按 KITTI sequence 划分，避免相邻帧泄漏"""
    cache_path = f"{CACHE_DIR}/train_cache_{NUM_TRAIN_FRAMES}.pkl"
    with open(cache_path, "rb") as f:
        all_samples = pickle.load(f)

    # 按 file_id 前 4 位分组（KITTI sequence id）
    seq_groups = {}
    for s in all_samples:
        seq_id = s["file_id"][:4]
        seq_groups.setdefault(seq_id, []).append(s)

    seq_ids = sorted(seq_groups.keys())
    print(f"[SPLIT] {len(seq_ids)} 个 sequence: {seq_ids}")

    random.seed(seed)
    # 取 20% 的 sequence 做验证集（至少 2 个）
    n_val = max(2, int(round(len(seq_ids) * TRAIN_RATIO)))
    val_seqs = sorted(random.sample(seq_ids, n_val))
    train_seqs = [s for s in seq_ids if s not in val_seqs]

    train = [s for sid in train_seqs for s in seq_groups[sid]]
    val = [s for sid in val_seqs for s in seq_groups[sid]]

    print(f"[SPLIT] 训练集 seqs={train_seqs}, 共 {len(train)} 帧")
    print(f"[SPLIT] 验证集 seqs={val_seqs}, 共 {len(val)} 帧")

    return ScaleDataset(train), ScaleDataset(val)


