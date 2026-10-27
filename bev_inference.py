import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torchvision.transforms as T
from ultralytics import YOLO

from config import *
from calib_utils import read_kitti_full_calib, load_velodyne, lidar_to_image, valid_lidar_mask
from semantic_seg import SemanticSegmenter
from depth_utils import compute_stat_feature
from models import ScalePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_NORM = T.Compose([
    T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def read_kitti_calib_p2(calib_file):
    with open(calib_file, "r") as f:
        for line in f:
            if line.startswith("P2:"):
                v = list(map(float, line.strip().split()[1:]))
                return v[0], v[5], v[2], v[6]
    return 707.0, 707.0, 604.0, 180.0


def main():
    print("Loading Models...")
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
    depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME).to(DEVICE).eval()
    seg_model = SemanticSegmenter(device=DEVICE)
    yolo = YOLO("yolov8n.pt")

    anchors = [10, 14, 18, 22, 26, 30]
    scale_model = ScalePredictor(anchors=anchors).to(DEVICE)
    ckpt_path = os.path.join(CKPT_DIR, "scale_predictor_main.pth")

    if not os.path.exists(ckpt_path):
        print(f"错误：找不到模型文件 {ckpt_path}")
        return

    scale_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    scale_model.eval()

    # 跑测试集的前 20 张，你想多跑可以改数字
    img_files = sorted([f for f in os.listdir(TEST_IMG) if f.endswith(".png")])[:20]
    print(f"开始生成 {len(img_files)} 张 BEV 可视化...")

    for fname in img_files:
        fid = os.path.splitext(fname)[0]
        img_path = os.path.join(TEST_IMG, fname)
        calib_path = os.path.join(TEST_CALIB, fid + ".txt")

        if not os.path.exists(calib_path): continue

        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fx, fy, cx, cy = read_kitti_calib_p2(calib_path)

        with torch.no_grad():
            # 深度
            inputs = processor(images=img_rgb, return_tensors="pt").to(DEVICE)
            depth = depth_model(**inputs).predicted_depth.squeeze().cpu().numpy()
            depth = cv2.resize(depth, (W, H))
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

            # 语义
            seg_valid, _ = seg_model.get_valid_mask(img_rgb)

            # 尺度
            feat_6d = compute_stat_feature(depth_norm, valid_mask=seg_valid)
            feat_t = torch.from_numpy(feat_6d).unsqueeze(0).to(DEVICE)
            img_t = IMG_NORM(img_rgb).unsqueeze(0).to(DEVICE)
            S_pred, _, _ = scale_model(feat_t, img_t)
            S = float(S_pred.item())

        depth_meters = (1.0 - depth_norm) * S

        # YOLO 检测
        res = yolo(img, conf=0.3, verbose=False)
        boxes = res[0].boxes
        KEEP_COCO = {0, 1, 2, 3, 5, 6, 7}

        # 初始化 BEV 画布
        bev_size = 800
        bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
        cx_bev, cy_bev = bev_size // 2, bev_size - 50
        cv2.line(bev, (cx_bev, 0), (cx_bev, bev_size), (0, 255, 255), 2)
        bev_scale = 10

        # 画距离网格线
        for z_dist in range(10, 80, 10):
            y_line = int(cy_bev - z_dist * bev_scale)
            if 0 <= y_line < bev_size:
                for x_line in range(0, bev_size, 20):
                    cv2.line(bev, (x_line, y_line), (x_line + 10, y_line), (50, 50, 50), 1)
                cv2.putText(bev, f"{z_dist}m", (10, y_line - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # 叠加雷达真值
        velo_path = os.path.join(TEST_VELO, fid + ".bin")
        if os.path.exists(velo_path):
            P2, R0, Tr = read_kitti_full_calib(calib_path)
            pts = load_velodyne(velo_path)
            u_l, v_l, z_l = lidar_to_image(pts[:, :3], P2, R0, Tr)
            mask_l = valid_lidar_mask(u_l, v_l, z_l, H, W)
            for ui, zi in zip(u_l[mask_l].astype(int), z_l[mask_l]):
                if zi > 60: continue
                X_l = (ui - cx) * zi / fx
                bx_l = int(cx_bev + X_l * bev_scale)
                by_l = int(cy_bev - zi * bev_scale)
                if 0 <= bx_l < bev_size and 0 <= by_l < bev_size:
                    cv2.circle(bev, (bx_l, by_l), 1, (100, 100, 100), -1)

        # ==================== ★ 终极：基于真实深度的贴地投影 ====================
        for box in boxes:
            cls = int(box.cls[0])
            if cls not in KEEP_COCO: continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            u, v = (x1 + x2) / 2, (y1 + y2) / 2

            # 取中心深度用来算中心点距离
            y1_half = int(y1 + (y2 - y1) * 0.5)
            patch_depth = depth_meters[y1_half:int(y2), int(x1):int(x2)]
            if patch_depth.size == 0: continue
            Z = float(np.median(patch_depth))
            if Z > 60: continue

            X = (u - cx) * Z / fx
            bx = int(cx_bev + X * bev_scale)
            by = int(cy_bev - Z * bev_scale)

            if 0 <= bx < bev_size and 0 <= by < bev_size:
                if cls == 0:
                    color = (255, 0, 0)  # 行人：蓝
                    thickness = 2
                elif cls in (2, 5, 6, 7):
                    color = (0, 255, 0)  # 车辆：绿
                    thickness = 4
                else:
                    color = (0, 165, 255)  # 两轮车：橙
                    thickness = 3


                cv2.circle(bev, (bx, by), thickness + 2, (255, 255, 255), 1)
                cv2.circle(bev, (bx, by), thickness, color, -1)

                cv2.putText(bev, f"{Z:.0f}m", (bx - 15, by - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # ==================== 拼接排版 ====================
        yolo_img = res[0].plot()
        depth_vis = cv2.normalize(depth_norm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        depth_colored = cv2.resize(depth_colored, (yolo_img.shape[1], yolo_img.shape[0]))

        top = np.hstack([yolo_img, depth_colored])

        bottom_h = 600
        bottom_canvas = np.zeros((bottom_h, top.shape[1], 3), dtype=np.uint8)

        bev_display = cv2.resize(bev, (550, 550))
        bottom_canvas[25:575, 50:600] = bev_display

        text_x = 800
        cv2.putText(bottom_canvas, f"Predicted Global Scale S = {S:.2f}",
                    (text_x, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        cv2.putText(bottom_canvas, "Legend:",
                    (text_x, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(bottom_canvas, "Gray Dots : LiDAR Ground Truth",
                    (text_x + 20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        cv2.putText(bottom_canvas, "Green Box : Predicted Car (3D)",
                    (text_x + 20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(bottom_canvas, "Blue Box : Predicted Pedestrian (3D)",
                    (text_x + 20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(bottom_canvas, "Orange Box : Predicted Cyclist (3D)",
                    (text_x + 20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        final = np.vstack([top, bottom_canvas])

        save_path = os.path.join(SAVE_DIR, f"bev_{fname}")
        cv2.imwrite(save_path, final)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()