import numpy as np


def read_kitti_full_calib(calib_path):
    """
    读 KITTI 标定文件，返回三个矩阵：
      P2 (3x4) : 相机投影矩阵
      R0 (4x4) : 相机校正矩阵
      Tr (4x4) : LiDAR 到相机的变换
    """
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip()] = np.array(list(map(float, v.split())))

    P2 = data["P2"].reshape(3, 4)

    R0 = np.eye(4, dtype=np.float64)
    R0[:3, :3] = data["R0_rect"].reshape(3, 3)

    Tr = np.eye(4, dtype=np.float64)
    Tr[:3, :] = data["Tr_velo_to_cam"].reshape(3, 4)

    return P2, R0, Tr


def load_velodyne(bin_path):
    """读 KITTI 点云 .bin，返回 (N, 4)：x, y, z, 反射率"""
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)


def lidar_to_image(pts_velo, P2, R0, Tr):
    """
    把 LiDAR 点投影到图像：
      pts_velo (N, 3)  LiDAR 坐标系
    返回：
      u (N,)  像素 x
      v (N,)  像素 y
      z (N,)  相机坐标系下的深度（米）
    """
    if pts_velo.shape[1] == 3:
        pts_velo = np.hstack([pts_velo, np.ones((pts_velo.shape[0], 1))])

    # LiDAR -> 相机 -> 校正 -> 图像
    pts_cam = (R0 @ Tr @ pts_velo.T).T
    pts_img = (P2 @ pts_cam.T).T

    z = pts_img[:, 2]
    u = pts_img[:, 0] / (z + 1e-9)
    v = pts_img[:, 1] / (z + 1e-9)
    return u, v, z


def valid_lidar_mask(u, v, z, H, W, min_d=2.0, max_d=70.0):
    """筛选：落在图像内 + 深度合理"""
    return ((u >= 0) & (u < W) & (v >= 0) & (v < H)
            & (z > min_d) & (z < max_d))

if __name__ == "__main__":
    P2, R0, Tr = read_kitti_full_calib(r"E:\Project\Source Code\training\calib\000000.txt")
    print("P2 shape:", P2.shape)
    print("R0 shape:", R0.shape)
    print("Tr shape:", Tr.shape)
    print("fx:", P2[0, 0], "  cx:", P2[0, 2])

    pts = load_velodyne(r"E:\Project\Source Code\training\velodyne\000000.bin")
    print("pts shape:", pts.shape)

    u, v, z = lidar_to_image(pts[:, :3], P2, R0, Tr)
    mask = valid_lidar_mask(u, v, z, 375, 1242)
    print("valid points:", mask.sum(), "/", len(u))