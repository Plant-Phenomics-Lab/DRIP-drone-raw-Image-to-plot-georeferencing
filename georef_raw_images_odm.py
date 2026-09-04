# -*- coding: utf-8 -*-
"""
DRIP: Drone Raw Image-to-Plot Georeferencing Workflow
Created on Thu Jul 17 15:22:15 2025
Author: Caiwang Zheng

This script implements the georeferencing of individual raw drone images 
by leveraging SfM (Structure from Motion) outputs from OpenSfM / OpenDroneMap (ODM).
It directly projects raw image pixels onto an estimated ground plane (DEM-free) 
using camera intrinsic/extrinsic parameters and exports them as UTM GeoTIFFs.
"""

import os
import json
import numpy as np
import cv2
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps
from pymap3d import enu2geodetic
from pyproj import Transformer
from PIL import Image
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation as R

# =============================================================================
# 1. CONFIGURATION & PATHS
# =============================================================================
# Working directory containing raw imagery and SfM reconstruction outputs
WORKING_DIR = '/Raw_DJI_2023_Final'

# Input files and directories (relative to WORKING_DIR)
RECON_PATH = "opensfm/reconstruction.topocentric.json"  # OpenSfM reconstruction output
IMAGE_DIR = "images"                                    # Directory containing raw JPG images
OUTPUT_DIR = "georeferenced"                            # Directory to save georeferenced GeoTIFFs

# Spatial Reference System (SRS) configurations
# Target UTM projection (Example: EPSG:32617 - UTM Zone 17N)
TARGET_CRS = "EPSG:32617" 


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def estimate_ground_height(C_world, point_xyz, k=3):
    """
    Estimates the local ground height (Z-coordinate) and calculates the 
    Camera Altitude Above Ground Level (AGL) using a spatial KDTree search.

    Parameters:
    -----------
    C_world : numpy.ndarray
        Camera center coordinate in the local world frame (ENU), shape (3,)
    point_xyz : numpy.ndarray
        3D sparse point cloud array from SfM reconstruction, shape (N, 3)
    k : int, optional
        Number of nearest neighbor points to average (default is 3)

    Returns:
    --------
    float
        Estimated Altitude Above Ground Level (AGL) in meters.
    """
    # Build a 2D KDTree using only the horizontal coordinates (X, Y)
    point_xy = point_xyz[:, :2]
    tree = KDTree(point_xy)

    # Query the k-nearest points in the horizontal plane to the camera's location
    _, idx = tree.query([C_world[:2]], k=k)

    # Calculate average ground elevation (Z) from the nearest neighbors
    ground_z = point_xyz[idx[0], 2].mean()
    
    # Altitude Above Ground Level (AGL)
    agl = C_world[2] - ground_z
    return agl


def georeference_workflow():
    """
    Main workflow for georeferencing raw individual drone images 
    and outputting orthorectified UTM GeoTIFFs.
    """
    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load OpenSfM reconstruction JSON data
    if not os.path.exists(RECON_PATH):
        raise FileNotFoundError(f"Reconstruction file not found at: {RECON_PATH}")
        
    with open(RECON_PATH, 'r') as f:
        recon_data = json.load(f)[0]  # Extract the primary reconstruction model
        
    # Get reference coordinate system (Latitude, Longitude, Altitude) from SfM local frame
    ref = recon_data["reference_lla"]
    ref_lat, ref_lon, ref_alt = ref["latitude"], ref["longitude"], ref["altitude"]
    
    # Initialize projection transformer (WGS84 Lat/Lon to Target UTM)
    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)

    # Extract 3D point cloud coordinates (SfM local coordinates)
    point_xyz = np.array([v["coordinates"] for v in recon_data["points"].values()])

    print(f"Loaded {len(point_xyz)} point cloud references from SfM.")
    print(f"Reference Location LLA: {ref_lat:.6f}, {ref_lon:.6f}, {ref_alt:.2f}")

    # Process each camera shot (image) in the SfM reconstruction
    for shot_name, shot in recon_data["shots"].items():
        base_name = shot_name.replace(".JPG.tif", ".JPG")
        img_path = os.path.join(IMAGE_DIR, base_name)
        
        if not os.path.exists(img_path):
            print(f"[-] Skipping missing raw image: {base_name}")
            continue

        print(f"\n[+] Processing: {base_name}")

        # Get image dimensions
        with Image.open(img_path) as im:
            width, height = im.size

        # Extract Camera Intrinsic Parameters
        cam = recon_data["cameras"][shot["camera"]]
        f_x = cam["focal_x"] * width
        f_y = cam["focal_y"] * width 
        c_x = cam["c_x"] * width + width / 2
        c_y = cam["c_y"] * height + height / 2
        
        # Camera Intrinsic Matrix K
        K = np.array([[f_x, 0,   c_x],
                      [0,   f_y, c_y],
                      [0,   0,   1]])

        # Lens Distortion Coefficients
        dist_coeffs = np.array([
            cam.get("k1", 0),
            cam.get("k2", 0),
            cam.get("p1", 0),
            cam.get("p2", 0),
            cam.get("k3", 0)
        ])

        # Extract Camera Extrinsic Parameters (World-to-Camera)
        rot_vec = np.array(shot["rotation"])      # Rotation vector (Rodrigues format)
        R_wc, _ = cv2.Rodrigues(rot_vec)          # 3x3 Rotation matrix
        T_wc = np.array(shot["translation"])      # Translation vector
        
        # Calculate Camera Orientation Angles (Yaw, Pitch, Roll) for logging
        rot_obj = R.from_matrix(R_wc)
        yaw, pitch, roll = rot_obj.as_euler('zyx', degrees=True)
        print(f"    Orientation -> Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°")
        
        # Calculate Camera Optical Center in Local World Coordinates (ENU): C = -R^T * T
        C_world = -R_wc.T @ T_wc
        
        # Dynamically estimate AGL at current camera horizontal location
        agl = estimate_ground_height(C_world, point_xyz, k=3)
        print(f"    Estimated AGL: {agl:.2f} meters")

        # Define key image coordinates (4 corners + center pixel) for Ground Control Points (GCPs)
        pixel_points = [
            (0, 0),                  # Top-Left
            (width - 1, 0),          # Top-Right
            (width - 1, height - 1),  # Bottom-Right
            (0, height - 1),          # Bottom-Left
            (width // 2, height // 2) # Center
        ]
        
        gcps = []

        # Project image pixels to the local ground plane (World coordinate frame)
        for col, row in pixel_points:
            # Undistort pixel coordinates and normalize them to camera ray frame
            distorted = np.array([[[col, row]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(distorted, K, dist_coeffs)
            
            # Formulate the 3D ray in camera coordinates (depth = 1.0)
            ray_cam = np.append(undistorted[0, 0], 1.0)
            ray_cam = ray_cam / np.linalg.norm(ray_cam)  # Normalize ray vector

            # Transform ray direction to World frame: ray_world = R_wc^T * ray_cam
            ray_world = R_wc.T @ ray_cam
            
            # Intersection with ground plane: P_world = C_world - (AGL / ray_world_z) * ray_world
            t = agl / ray_world[2]
            P_world = C_world - t * ray_world
            
            # Convert local ENU coordinates to geodetic coordinates (Lat, Lon, Alt)
            lat, lon, alt = enu2geodetic(
                P_world[0], P_world[1], P_world[2], 
                ref_lat, ref_lon, ref_alt
            )
            
            # Convert geodetic coordinates to target projection (UTM)
            x_utm, y_utm = transformer.transform(lon, lat)

            # Create Rasterio Ground Control Point (GCP)
            gcp = GroundControlPoint(row=row, col=col, x=x_utm, y=y_utm, z=alt)
            
            # Attach local coordinates for record-keeping
            gcp.x_local = P_world[0]  # East
            gcp.y_local = P_world[1]  # North
            gcp.x_utm = x_utm
            gcp.y_utm = y_utm
            
            gcps.append(gcp)

        # ---------------------------------------------------------------------
        # 3. EXPORT TO GEOTIFF
        # ---------------------------------------------------------------------
        # Read the raw image and convert color channel configuration
        img_cv = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Calculate spatial transform matrix using the generated GCPs
        transform = from_gcps(gcps)
        output_path = os.path.join(OUTPUT_DIR, base_name.replace(".JPG", "_georef.tif"))
        
        # Write image layers into GeoTIFF format with spatial references
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype=np.uint8,
            crs=TARGET_CRS,
            transform=transform,
            compress="JPEG"
        ) as dst:
            for i in range(3):
                dst.write(img_rgb[:, :, i], i + 1)
        
        print(f"    [✔] Saved Georeferenced TIF to: {output_path}")


# =============================================================================
# 4. EXECUTION
# =============================================================================
if __name__ == "__main__":
    georeference_workflow()
