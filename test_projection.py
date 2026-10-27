import cv2
import numpy as np

from calib_utils import (read_kitti_full_calib, load_velodyne,
                         lidar_to_image, valid_lidar_mask)


def check_one(fid):
    img_path   = rf"E:\Project\Source Code\training\image_2\{fid}.png"
    calib_path = rf"E:\Project\Source Code\training\calib\{fid}.txt"
    velo_path  = rf"E:\Project\Source Code\training\velodyne\{fid}.bin"

    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    P2, R0, Tr = read_kitti_full_calib(calib_path)
    pts = load_velodyne(velo_path)

    u, v, z = lidar_to_image(pts[:, :3], P2, R0, Tr)
    mask = valid_lidar_mask(u, v, z, H, W)

    vis = img.copy()
    for ui, vi, zi in zip(u[mask].astype(int),
                          v[mask].astype(int),
                          z[mask]):
        if zi < 15:
            color = (0, 255, 0)      # 绿：近处 <15m
        elif zi < 30:
            color = (0, 255, 255)    # 黄：中距 15-30m
        else:
            color = (0, 0, 255)      # 红：远处 >30m
        cv2.circle(vis, (ui, vi), 1, color, -1)

    out_path = rf"E:\Project\Source Code\figures\projection_check_{fid}.png"
    cv2.imwrite(out_path, vis)
    print(f"[{fid}] valid={mask.sum()}/{len(u)}  saved={out_path}")


if __name__ == "__main__":
    for fid in ["000000", "000001", "000002", "000100", "001000"]:
        try:
            check_one(fid)
        except Exception as e:
            print(f"[{fid}] ERROR: {e}")