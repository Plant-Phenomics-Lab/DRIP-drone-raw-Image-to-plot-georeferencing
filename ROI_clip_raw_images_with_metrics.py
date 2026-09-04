# -*- coding: utf-8 -*-
"""
DRIP: Automated Plot-Level ROI Extractor & Quality Evaluator
Created on Thu Apr 23 2026
Author: Caiwang Zheng

This module automates the extraction (clipping) of individual plot-level 
Regions of Interest (ROIs) from the Metashape orthomosaic and raw, 
georeferenced individual drone images. 

It evaluates each extracted plot-level clip using:
  1. Image Quality Metrics: Laplacian Variance (Blur), Mean Intensity (Brightness), 
     and Shannon Entropy (Information density).
  2. Image-Relative Geometric Metrics: Physical distance, yaw/heading rotation, 
     and local sensor-frame relative offsets (X, Y) between camera and plot centers.
"""

import os
import math
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from pyproj import Transformer
import cv2
from skimage.measure import shannon_entropy

# =============================================================================
# 1. GLOBAL CONFIGURATIONS
# =============================================================================
# Absolute path parameters for the pipeline execution
IMAGE_FOLDER = "/georeferenced"
META_ORTHO   = "/DJI_202306141508_028_G2F2023_20240220T1605_ortho_dsm_UTM.tif"
POLYGON_PATH = "/G2F2023_plots_manual.shp"
OUTPUT_ROOT  = "/ROI_clipped_results_RE"

# Pipeline Parameters
OVERLAP_THRESHOLD = 0.8  # Minimum overlap ratio (0.0 to 1.0) required to clip raw images


# =============================================================================
# 2. IMAGE INVENTORY BUILDER
# =============================================================================
def build_image_inventory(image_folder):
    """
    Scans the directory of georeferenced TIF files to index spatial boundaries
    and build a spatial inventory catalog for rapid overlap queries.

    Parameters:
    -----------
    image_folder : str
        Directory containing the georeferenced GeoTIFF images (*.tif).

    Returns:
    --------
    geopandas.GeoDataFrame or None
        An indexed spatial database of all available raw georeferenced footprints.
    """
    print("🔍 Building image inventory... Indexing TIF boundaries.")
    inventory_data = []
    
    for file in sorted(os.listdir(image_folder)):
        if not file.endswith(".tif"): 
            continue
        path = os.path.join(image_folder, file)
        try:
            with rasterio.open(path) as src:
                inventory_data.append({
                    "image_file": file,
                    "image_path": path,
                    "geometry": box(*src.bounds),
                    "crs": src.crs
                })
        except Exception as e:
            print(f"⚠️ Could not read spatial boundaries for {file}: {e}")
            
    if not inventory_data: 
        return None
        
    return gpd.GeoDataFrame(inventory_data, crs=inventory_data[0]['crs'])


# =============================================================================
# 3. QUALITY METRICS COMPUTATION
# =============================================================================
def compute_quality_metrics(image_array):
    """
    Calculates computer-vision descriptors of image quality for the clipped ROI.

    Calculated Metrics:
      - Laplacian Variance: Indicator of image focus/blur (higher = sharper).
      - Mean Intensity: Overall brightness of the cropped plot.
      - Shannon Entropy: Information density and texture complexity (higher = more detailed).

    Parameters:
    -----------
    image_array : numpy.ndarray
        Multi-channel or single-channel image array (C, H, W).

    Returns:
    --------
    tuple of (float, float, float)
        (Laplacian Variance, Mean Intensity, Shannon Entropy) rounded to 4 decimals.
    """
    # Convert multi-channel (C, H, W) to OpenCV standard (H, W, C)
    if len(image_array.shape) == 3:
        img_rgb = np.moveaxis(image_array[:3, :, :], 0, -1)
        # Downsample 16-bit imagery to standard 8-bit for OpenCV functions if necessary
        if img_rgb.dtype == np.uint16:
            img_rgb = (img_rgb / 256).astype(np.uint8)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array[0]

    # Calculate metrics
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_int = np.mean(gray)
    entropy_val = shannon_entropy(gray)
    
    return round(lap_var, 4), round(mean_int, 4), round(entropy_val, 4)


# =============================================================================
# 4. GEOMETRIC METRICS & CAMERA ROTATION
# =============================================================================
def compute_geometry_metrics(poly_geom, poly_crs, image_path):
    """
    Calculates spatial relation metrics between the camera optical center and the 
    centroid of the target plot polygon.

    This function rotates the geographic (UTM) displacement vector onto the 
    camera's local coordinate system (sensor-relative frame) using the aircraft's 
    heading calculated directly from the GeoTIFF's affine transformation matrix.

    Parameters:
    -----------
    poly_geom : shapely.geometry.Polygon
        Vector boundary of the target crop plot.
    poly_crs : pyproj.crs.CRS
        Spatial CRS of the input polygon shapefile.
    image_path : str
        File path to the georeferenced drone GeoTIFF.

    Returns:
    --------
    dict
        Contains distance, local relative angle, camera/plot centers in UTM, 
        rotated local offset coordinates (rel_x, rel_y), and calculated heading.
    """
    with rasterio.open(image_path) as src:
        img_crs = src.crs
        b = src.bounds
        # Compute image center coordinates in map frame
        img_cx = (b.left + b.right) / 2.0
        img_cy = (b.top + b.bottom) / 2.0
        
        # --- Heading extraction from GeoTIFF Affine Transform Matrix ---
        # transform[0] is pixel width (resolution x), transform[3] is skew y
        transform = src.transform
        heading_rad = math.atan2(transform[3], transform[0])
        heading_deg = math.degrees(heading_rad) % 360.0

    # Project plot polygon to match raw image coordinate system
    poly_proj = gpd.GeoSeries([poly_geom], crs=poly_crs).to_crs(img_crs).iloc[0]
    plot_cx, plot_cy = poly_proj.centroid.x, poly_proj.centroid.y

    # Determine the local UTM projection zone to perform distance math in meters
    utm_crs = gpd.GeoSeries([gpd.points_from_xy([img_cx], [img_cy])[0]], crs=img_crs).estimate_utm_crs()
    tf = Transformer.from_crs(img_crs, utm_crs, always_xy=True)
    
    # Project both centers to UTM meters
    img_cx_m, img_cy_m = tf.transform(img_cx, img_cy)
    plot_cx_m, plot_cy_m = tf.transform(plot_cx, plot_cy)

    # 1. Base translation offsets in the projected Map Coordinates (East, North)
    dx_map = plot_cx_m - img_cx_m
    dy_map = plot_cy_m - img_cy_m
    
    # 2. Coordinate rotation to local camera sensor-relative frame
    # Standard transformation matrix applied to compensate for aircraft heading angle (yaw)
    cos_a = math.cos(heading_rad)
    sin_a = math.sin(heading_rad)
    
    rel_x = dx_map * cos_a + dy_map * sin_a
    rel_y = -dx_map * sin_a + dy_map * cos_a
    
    # Calculate observation angle relative to drone sensor heading orientation
    local_observation_angle = math.degrees(math.atan2(rel_x, rel_y)) % 360.0
    
    return {
        "distance_m": round(math.hypot(dx_map, dy_map), 3),
        "observation_angle_deg": round(local_observation_angle, 4), 
        "img_cx_m": img_cx_m,
        "img_cy_m": img_cy_m,
        "plot_cx_m": plot_cx_m,
        "plot_cy_m": plot_cy_m,
        "rel_x_m": round(rel_x, 4),    # Transformed local X coordinate on camera frame
        "rel_y_m": round(rel_y, 4),    # Transformed local Y coordinate on camera frame
        "img_heading": round(heading_deg, 2) # Calculated camera heading angle
    }


# =============================================================================
# 5. BATCH PROCESSING PIPELINE
# =============================================================================
def process_plots_optimized(image_folder, meta_path, polygon_path, output_root, overlap_threshold=0.8):
    """
    Executes the spatial extraction pipeline. Loops through all plot polygons to:
      1. Extract reference plot crops from the Metashape Orthomosaic.
      2. Intersect, filter, and extract overlapping raw georeferenced individual images.
      3. Compute geometric and quality scores, and save the crops using a metric-based naming format.

    Parameters:
    -----------
    image_folder : str
        Directory holding individual georeferenced TIFs.
    meta_path : str
        Path to the Metashape Orthomosaic.
    polygon_path : str
        Path to the GIS Shapefile containing plot polygons.
    output_root : str
        Directory where outputs will be stored.
    overlap_threshold : float, optional
        Minimum coverage percentage needed to qualify for a crop (default is 0.8).

    Returns:
    --------
    list of dict
        A list of record dictionaries summarizing metrics for every successfully clipped image.
    """
    inventory_gdf = build_image_inventory(image_folder)
    if inventory_gdf is None:
        print("❌ Error: No valid georeferenced raw images found in IMAGE_FOLDER.")
        return []
    
    print(f"📖 Opening Metashape Ortho: {os.path.basename(meta_path)}")
    metashape_src = rasterio.open(meta_path)

    polygons = gpd.read_file(polygon_path)
    records = []

    # Iterate over each crop plot in the vector layer
    for idx, poly in polygons.iterrows():
        poly_id = poly.get("PlotId", idx)
        poly_geom = poly.geometry
        print(f"\n🚀 Processing Plot: {poly_id}")

        poly_folder = os.path.join(output_root, f"plot_{poly_id}")
        os.makedirs(poly_folder, exist_ok=True)

        # Inner helper function to crop and evaluate orthomosaics
        def clip_ortho(src, name_prefix):
            try:
                poly_proj = gpd.GeoSeries([poly_geom], crs=polygons.crs).to_crs(src.crs).iloc[0]
                clip, trans = mask(src, [poly_proj], crop=True)
                lap, mi, ent = compute_quality_metrics(clip)
                
                # Format metrics for the filename
                l_int = int(round(lap))
                m_int = int(round(mi))
                e_int = int(round(ent * 100))
                
                out_name = f"{name_prefix}_L{l_int}_M{m_int}_E{e_int}.tif"
                out_path = os.path.join(poly_folder, out_name)
                
                meta = src.meta.copy()
                meta.update({"height": clip.shape[1], "width": clip.shape[2], "transform": trans})
                with rasterio.open(out_path, "w", **meta) as dest:
                    dest.write(clip)
                
                records.append({
                    "PlotId": poly_id, "Source_Image": name_prefix, "Distance_m": np.nan, 
                    "Observation_Angle_deg": np.nan, 
                    "img_cx_m": np.nan, "img_cy_m": np.nan, 
                    "plot_cx_m": np.nan, "plot_cy_m": np.nan,
                    "Laplacian_Variance": lap, "Mean_Intensity": mi, 
                    "Shannon_Entropy": ent, "Clipped_Path": out_path
                })
                print(f"  🖼️ {name_prefix} clip generated: {out_name}")
            except Exception as e:
                print(f"  ⚠️ {name_prefix} clipping failed for plot {poly_id}: {e}")

        # --- PART 1: Clip Metashape Orthomosaic for baseline reference ---
        clip_ortho(metashape_src, "META")

        # --- PART 2: Query and Clip raw individual drone images ---
        poly_series = gpd.GeoSeries([poly_geom], crs=polygons.crs).to_crs(inventory_gdf.crs)
        matches = inventory_gdf[inventory_gdf.intersects(poly_series.iloc[0])]
        
        for _, row in matches.iterrows():
            try:
                with rasterio.open(row['image_path']) as src:
                    # 1. Project the plot boundary polygon to current image CRS
                    p_in_img = poly_series.to_crs(src.crs).iloc[0]
                    
                    # 2. Check strict intersection topology
                    if not p_in_img.intersects(row['geometry']):
                        continue
                        
                    # 3. Assess if overlap meets the minimum research threshold
                    intersection_area = p_in_img.intersection(row['geometry']).area
                    if (intersection_area / p_in_img.area) < overlap_threshold:
                        continue
        
                    # 4. Perform raster cropping
                    try:
                        out_img, out_trans = mask(src, [p_in_img], crop=True)
                    except ValueError:
                        # Catch marginal coordinates clipping index errors gracefully
                        print(f"  ⚠️ Skipping {row['image_file']} due to edge/marginal boundary overlaps.")
                        continue
        
                    # 5. Calculate Metrics and construct output attributes
                    lap, mi, ent = compute_quality_metrics(out_img)
                    geo = compute_geometry_metrics(poly_geom, polygons.crs, row['image_path'])
        
                    l_int = int(round(lap))
                    m_int = int(round(mi))
                    e_int = int(round(ent * 100))
                    
                    img_stem = os.path.splitext(row['image_file'])[0]
                    save_name = f"{img_stem}_L{l_int}_M{m_int}_E{e_int}.tif"
                    save_path = os.path.join(poly_folder, save_name)
                    
                    meta = src.meta.copy()
                    meta.update({"height": out_img.shape[1], "width": out_img.shape[2], "transform": out_trans})
                    with rasterio.open(save_path, "w", **meta) as dest:
                        dest.write(out_img)
        
                    # Append all spatial geometry metrics and texture analytics to central report
                    records.append({
                        "PlotId": poly_id, 
                        "Source_Image": row['image_file'], 
                        "Distance_m": geo["distance_m"],
                        "Observation_Angle_deg": geo["observation_angle_deg"], 
                        "rel_x_m": geo["rel_x_m"],   
                        "rel_y_m": geo["rel_y_m"],   
                        "img_heading": geo["img_heading"], 
                        "img_cx_m": geo["img_cx_m"], 
                        "img_cy_m": geo["img_cy_m"], 
                        "plot_cx_m": geo["plot_cx_m"], 
                        "plot_cy_m": geo["plot_cy_m"],
                        "Laplacian_Variance": lap, 
                        "Mean_Intensity": mi, 
                        "Shannon_Entropy": ent, 
                        "Clipped_Path": save_path
                    })
                    print(f"  ✅ Raw Match Processed: {save_name}")
                    
            except Exception as e:
                print(f"  ❌ Error processing raw imagery {row['image_file']}: {e}")

    metashape_src.close()
    return records


# =============================================================================
# 6. PIPELINE EXECUTION ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    # Launch pipeline
    results = process_plots_optimized(
        IMAGE_FOLDER, 
        META_ORTHO, 
        POLYGON_PATH, 
        OUTPUT_ROOT,
        overlap_threshold=OVERLAP_THRESHOLD
    )

    # Save output summaries
    if results:
        df = pd.DataFrame(results)
        excel_path = os.path.join(OUTPUT_ROOT, "plot_analysis_report_full.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"\n📊 Process Complete! Total records: {len(df)}. Report exported to: {excel_path}")
    else:
        print("\nℹ️ No overlapping clips generated. Ensure input geometries align spatially.")
