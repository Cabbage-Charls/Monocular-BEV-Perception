import os

ROOT        = r"E:\Project\Source Code"
TRAIN_IMG   = os.path.join(ROOT, "training", "image_2")
TRAIN_VELO  = os.path.join(ROOT, "training", "velodyne")
TRAIN_CALIB = os.path.join(ROOT, "training", "calib")
TRAIN_LABEL = os.path.join(ROOT, "training", "label_2")

TEST_IMG    = os.path.join(ROOT, "testing", "image_2")
TEST_VELO   = os.path.join(ROOT, "testing", "velodyne")
TEST_CALIB  = os.path.join(ROOT, "testing", "calib")

SAVE_DIR    = os.path.join(ROOT, "bev_output_v2")
CACHE_DIR   = os.path.join(ROOT, "cache")
CKPT_DIR    = os.path.join(ROOT, "ckpt")
FIG_DIR     = os.path.join(ROOT, "figures")

for d in [SAVE_DIR, CACHE_DIR, CKPT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ===== 数据规模 =====
NUM_TRAIN_FRAMES = 4000
TRAIN_RATIO      = 0.20

# ===== 深度模型 =====
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

# ===== MLP =====
STAT_FEAT_DIM    = 6
IMG_EMBED_DIM    = 512

# ===== 训练超参 =====
BATCH_SIZE       = 8
MAX_EPOCH        = 80
LR               = 3e-4
WEIGHT_DECAY     = 1e-4
PATIENCE         = 10

LAMBDA_S         = 0.3
LAMBDA_VAR       = 0.3
LAMBDA_ANCHOR    = 0.2
TARGET_MIN_STD   = 0.45

# ===== LiDAR 过滤 =====
LIDAR_MIN_DEPTH  = 2.0
LIDAR_MAX_DEPTH  = 70.0
LIDAR_MIN_POINTS = 30