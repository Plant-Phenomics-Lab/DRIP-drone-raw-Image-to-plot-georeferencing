# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 16:22:15 2025

@author: 13527
"""

import json
import numpy as np
import cv2
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps
from pymap3d import enu2geodetic
from pyproj import Transformer
from PIL import Image
import os
from sklearn.neighbors import KDTree

wd = 'C:/Users/czheng23/Documents/Raw Image Analysis/opendronemap_test/Strawberry_Imagery'
os.chdir(wd)

# === 参数设置 ===
recon_path = "opensfm/reconstruction.topocentric.json"  # ODM 的 SfM 文件路径
image_dir = "images"  # 原始图像路径
output_dir = "georeferenced"  # 输出 GeoTIFF 路径

os.makedirs(output_dir, exist_ok=True)


with open(recon_path) as f:
    recon_data = json.load(f)[0]  # 取第一个模型
    
ref = recon_data["reference_lla"]
ref_lat, ref_lon, ref_alt = ref["latitude"], ref["longitude"], ref["altitude"]
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)  # always_xy: lon, lat 顺序

point_xyz = np.array([
    v["coordinates"] for v in recon_data["points"].values()
])

point_ids = list(recon_data["points"].keys())

# AGL = 12


def estimate_ground_height(C_world, point_xyz, point_ids, k=3):
    """
    输入：
        C_world: 相机中心世界坐标，shape = (3,)
        recon_data: reconstruction.json 中的模型数据（通常是 json.load()[0]）
        k: 最近点数量（默认 3 个）
    返回：
        ground_z: 最近 k 个点的平均地面高度（Z）
        cam_height: 相机离地高度（C_z - ground_z）
    """

    # 用 X, Y 建立 KDTree
    point_xy = point_xyz[:, :2]
    tree = KDTree(point_xy)

    # 查询最近 k 个点（只用 X,Y）
    dist, idx = tree.query([C_world[:2]], k=k)

    # 平均地面 Z 值
    ground_z = point_xyz[idx[0], 2].mean()
    AGL = C_world[2] - ground_z

    return AGL

# === 主循环，每张图像处理 ===
for shot_name, shot in recon_data["shots"].items():
    base_name = shot_name.replace(".JPG.tif", ".JPG")
    img_path = os.path.join(image_dir, base_name)
    if not os.path.exists(img_path):
        print(f"Skipping missing: {base_name}")
        continue

    cam = recon_data["cameras"][shot["camera"]]
    with Image.open(img_path) as im:
        width, height = im.size

    # 内参矩阵 K
    f_x = cam["focal_x"] * width
    f_y = cam["focal_y"] * width #height
    c_x = cam["c_x"] * width + width / 2
    c_y = cam["c_y"] * height + height / 2
    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0,   0,   1]])

    dist_coeffs = np.array([
        cam.get("k1", 0),
        cam.get("k2", 0),
        cam.get("p1", 0),
        cam.get("p2", 0),
        cam.get("k3", 0)
    ])

    # 外参：R 和 T（world-to-camera）
    rot_vec = np.array(shot["rotation"])  # shape (3,)
    R_wc, _ = cv2.Rodrigues(rot_vec)      # shape (3,3)
    T_wc = np.array(shot["translation"])  # shape (3,)

    # 相机中心在世界坐标：X_cam = -R.T @ T
    C_world = -R_wc.T @ T_wc
    
    AGL = estimate_ground_height(C_world, point_xyz, point_ids, k=3)
    print(AGL)

    # 图像角点 + 中心像素
    points = [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1), (width//2, height//2)]
    gcps = []
    
    ray_points = []
    ray_cam_all1 = []
    ray_cam_all2 = []

    for col, row in points:
        u = col
        v = row
        # 去畸变并归一化成射线
        print([u,v])
        distorted = np.array([[[u, v]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(distorted, K, dist_coeffs)
        
        print(undistorted[0, 0])
        pixel_homog = np.array([u, v, 1.0])  # 齐次像素坐标
        #ray_cam = np.linalg.inv(K) @ pixel_homog  # 等价于归一化坐标
        #ray_cam_all2.append(ray_cam)
        
        ray_cam = np.append(undistorted[0, 0], 1.0)
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        # ray_cam_all1.append(ray_cam)

        # 转为世界坐标射线方向
        ray_world = R_wc.T @ ray_cam
        t = AGL / ray_world[2]
        P_world = C_world - t * ray_world
        # 转换为经纬度
        lat, lon, alt = enu2geodetic(P_world[0], P_world[1], P_world[2], ref_lat, ref_lon, ref_alt)
        x_utm, y_utm = transformer.transform(lon, lat)


        # 创建 GCP（x,y 是地面坐标，这里用的是 UTM）      
        gcp = GroundControlPoint(row=v, col=u, x=x_utm, y=y_utm, z=alt)
        # 添加 ENU 坐标信息到 gcp 对象（动态添加属性）
        gcp.x_local = P_world[0]  # East
        gcp.y_local = P_world[1]  # North
        
        gcp.x_utm = x_utm
        gcp.y_utm = y_utm
        
        gcps.append(gcp)
        ray_points.append(ray_world)

        print(f"Pixel ({P_world[0]},{P_world[1]}) ➜ Lon={lon:.6f}, Lat={lat:.6f}")


    # 写入GeoTIFF
    img_cv = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    transform = from_gcps(gcps)
    output_path = os.path.join(output_dir, base_name.replace(".JPG", "_georef.tif"))
    
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=np.uint8,
        crs="EPSG:32617",  # ✅ 使用 UTM
        transform=transform,
        compress="JPEG"
    ) as dst:
        for i in range(3):
            dst.write(img_rgb[:, :, i], i + 1)
    
    print(f"✅ Saved: {output_path}")

            
        
