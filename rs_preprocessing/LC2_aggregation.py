# This script is a modified version of LC_aggregation.py and handles all lccs classes individually,
# not the reclassified cropland, forest, and urban classes.
# Additionally, water bodies (class 210) can be used to compute distance to lake metrics.
# Use output from this script for all 'lc2_' tagged models.

import os
import glob
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from shapely.geometry import mapping, box
from pyproj import Transformer
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import logging
import sys
import traceback

# CONFIGURATION
LC_RASTER_DIR = "/p/projects/impactee/Josh/geo_variables/land_cover/ESA_CCI_LC_data/data"  # using original rasters, not reclassified
REPROJECTED_DIR = "/p/projects/impactee/Josh/geo_variables/land_cover/reprojected_rasters_lc2"  # folder to cache lc2 reprojections
GADM_PATH = "/p/projects/impactee/Josh/LandScan/gadm_custom_merged.shp"
# OUTPUT_CSV = "/p/projects/impactee/Josh/geo_variables/land_cover/lc2_class_shares.csv" # note: lc2
OUTPUT_CSV = "/p/projects/impactee/Josh/geo_variables/land_cover/lc2_class_shares_gid0.csv" # note: lc2
MOLLWEIDE_EPSG = "ESRI:54009"
TARGET_RES = 300  # meters
NUM_WORKERS = min(8, multiprocessing.cpu_count())

# commenting out, since this was missing classes like 11, 12, etc.
# LCCS classes to process (from C3S/ESA definition)
# LCCS_CLASSES = list(range(10, 230, 10))  # [10, 20, ..., 220]

LCCS_GROUPS = {
    "lccs_ag_101112":    [10, 11, 12],
    "lccs_ag_20":        [20],
    "lccs_ag_30":        [30],
    "lccs_ag_40":        [40],
    "lccs_forest_50":    [50],
    "lccs_forest_606162":[60, 61, 62],
    "lccs_forest_707172":[70, 71, 72],
    "lccs_forest_808182":[80, 81, 82],
    "lccs_forest_90":    [90],
    "lccs_forest_100":   [100],
    "lccs_forest_160":   [160],
    "lccs_forest_170":   [170],
    "lccs_grass_110":    [110],
    "lccs_grass_130":    [130],
    "lccs_wet_180":      [180],
    "lccs_urban_190":    [190],
    "lccs_shrub_120121122": [120, 121, 122],
    "lccs_sparse_140":   [140],
    "lccs_sparse_150151152153": [150, 151, 152, 153],
    "lccs_bare_200201202": [200, 201, 202],
    "lccs_water_210":   [210],
    "lccs_snow_220":    [220]
}

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/lc2_class_aggregation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

import os
import logging
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.transform import from_origin

def reproject_raster(src_path, dst_path, dst_crs, resolution):
    """Reproject a raster (NetCDF subdataset or GeoTIFF) to dst_crs at target resolution,
    fixing ESA-CCI transforms, clipping inside bounds, and using safe nodata.
    """

    # if os.path.exists(dst_path):
    #     logging.info(f"Reprojected raster already exists: {dst_path}")
    #     return

    with rasterio.open(src_path) as src0:
        # Handle NetCDF container (ESA-CCI) with subdatasets
        if src0.count == 0 and src0.subdatasets:
            logging.info(f"{src_path} has no bands; opening first subdataset {src0.subdatasets[0]}")
            src = rasterio.open(src0.subdatasets[0])
        else:
            src = src0

        # --- CRS handling ---
        if src.crs is None:
            logging.warning(f"No CRS found in {src_path}. Assuming EPSG:4326.")
            src_crs = "EPSG:4326"
        else:
            src_crs = src.crs

        # --- Transform fix for ESA-CCI rasters ---
        # this step is critical for the reprojection to work correctly
        src_transform = src.transform
        if src_transform.a == 0 or src_transform.e == 0:
            logging.info("Fixing missing/invalid transform for ESA-CCI dataset.")
            xres = 360.0 / src.width
            yres = 180.0 / src.height
            src_transform = from_origin(-180.0, 90.0, xres, yres)

        # --- Clip slightly inside bounds to avoid reprojection edge issues ---
        # this step is critical for the reprojection to work correctly
        clipped_window = from_bounds(
            -179.999, -89.999, 179.999, 89.999,
            transform=src_transform
        )
        clipped_transform = src.window_transform(clipped_window)
        clipped_width = int(clipped_window.width)
        clipped_height = int(clipped_window.height)

        # --- Destination transform ---
        dst_transform, width, height = calculate_default_transform(
            src_crs, dst_crs,
            clipped_width, clipped_height,
            *rasterio.transform.array_bounds(clipped_height, clipped_width, clipped_transform),
            resolution=(resolution, resolution)
        )

        # --- Nodata handling ---
        # this step is critical for the reprojection to work correctly
        dst_nodata = src.nodata if src.nodata is not None else 255

        # --- Metadata ---
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": width,
            "height": height,
            "driver": "GTiff",
            "compress": "lzw",
            "tiled": True,
            "dtype": "uint8",     # ESA-CCI classes fit in uint8
            "nodata": dst_nodata,
            "count": src.count if src.count > 0 else 1,
        })

        # --- Reproject ---
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                src_data = src.read(i, window=clipped_window)

                reproject(
                    source=src_data,
                    destination=rasterio.band(dst, i),
                    src_transform=clipped_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    src_nodata=src.nodata,
                    dst_nodata=dst_nodata,
                    warp_mem_limit=512,
                    num_threads=4
                )

    logging.info(f"Saved reprojected raster: {dst_path}")

### Older version; not working ###
# def reproject_raster(src_path, dst_path, dst_crs, target_res):
#     """Reproject a raster (NetCDF subdataset or GeoTIFF) to target CRS/resolution,
#     unless the output already exists.
#     """
#     # if os.path.exists(dst_path):
#     #     logging.info(f"Reprojected raster already exists: {dst_path}. Skipping.")
#     #     return

#     # Open the NetCDF or raster
#     with rasterio.open(src_path) as src0:
#         # If this is a NetCDF container with no bands, open the first subdataset
#         # The ESA-CCI files are containers; the first subdataset is the lccs_class variable 
#         if src0.count == 0 and src0.subdatasets:
#             logging.info(f"{src_path} has no bands; opening first subdataset {src0.subdatasets[0]}")
#             src = rasterio.open(src0.subdatasets[0])
#         else:
#             src = src0

#         # If CRS is missing, assume EPSG:4326 (WGS84)
#         src_crs = src.crs or "EPSG:4326"

#         # Fix broken transform
#         if src.transform.a == 0 or src.transform.e == 0:
#             logging.warning(f"Invalid transform in {src_path}, rebuilding from bounds")
#             src_transform = from_bounds(*src.bounds, src.width, src.height)
#         else:
#             src_transform = src.transform

#         # Compute target transform and shape
#         dst_transform, width, height = calculate_default_transform(
#             src_crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_res
#         )

#         kwargs = src.meta.copy()
#         kwargs.update({
#             "crs": dst_crs,
#             "transform": dst_transform,
#             "width": width,
#             "height": height,
#             "driver": "GTiff",
#             "compress": "lzw",
#             "tiled": True,
#             "dtype": src.dtypes[0],
#             "count": src.count if src.count > 0 else 1,
#             "nodata": 0,   # <--- explicitly set
#         })

#         with rasterio.open(dst_path, "w", **kwargs) as dst:
#             for i in range(1, src.count + 1):
#                 reproject(
#                     source=rasterio.band(src, i),
#                     destination=rasterio.band(dst, i),
#                     src_transform=src.transform,
#                     src_crs=src_crs,
#                     dst_transform=dst_transform,
#                     dst_crs=dst_crs,
#                     resampling=Resampling.nearest,
#                     dst_nodata=0,  # <--- explicitly set
#                 )
        
#         # inspect output transform and unique values
#         with rasterio.open(dst_path) as dst_check:
#             logging.info(f"Output transform for {dst_path}: {dst_check.transform}")
#             unique_vals = np.unique(dst_check.read(1))
#             if len(unique_vals) > 50:
#                 logging.info(f"Output unique values (sample) for {dst_path}: {unique_vals[:50]} (total {len(unique_vals)})")
#             else:
#                 logging.info(f"Output unique values for {dst_path}: {unique_vals}")

#     logging.info(f"Saved reprojected raster: {dst_path}")

def tile_raster_to_latitudinal_strips(reproj_path, out_dir, num_strips=12):
    """Split a raster into 12 equal latitudinal strips."""
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(reproj_path) as src:
        bounds = src.bounds
        ymin, ymax = bounds.bottom, bounds.top
        xmin, xmax = bounds.left, bounds.right

        step = (ymax - ymin) / num_strips
        out_paths = []

        for i in range(num_strips):
            y0 = ymin + i * step
            y1 = y0 + step

            tile_fname = f"{os.path.splitext(os.path.basename(reproj_path))[0]}_strip_{i}.tif"
            tile_path = os.path.join(out_dir, tile_fname)

            # Skip if file already exists
            # if os.path.exists(tile_path):
            #     logging.info(f"Strip {i} already exists: {tile_path}. Skipping.")
            #     out_paths.append(tile_path)
            #     continue

            window = rasterio.windows.from_bounds(xmin, y0, xmax, y1, src.transform)
            transform = src.window_transform(window)

            out_meta = src.meta.copy()
            out_meta.update({
                "height": int(window.height),
                "width": int(window.width),
                "transform": transform
            })

            with rasterio.open(tile_path, "w", **out_meta) as dest:
                for i in range(1, src.count + 1):
                    data = src.read(i, window=window)
                    dest.write(data, i)

            out_paths.append(tile_path)

    logging.info(f"Tiled {reproj_path} into {len(out_paths)} latitudinal strips.")
    return out_paths

def preprocess_all_rasters():
    """Reproject all rasters once, save to REPROJECTED_DIR, return dict year->path."""
    os.makedirs(REPROJECTED_DIR, exist_ok=True)
    input_paths = sorted(glob.glob(os.path.join(LC_RASTER_DIR, "*.nc")))
    year_to_path = {}
    for p in input_paths:
        fname = os.path.basename(p)
        # Extract year from filename
        import re

        # Extract 4-digit year from filename (first occurrence of YYYY)
        match = re.search(r"\b(19|20)\d{2}\b", fname)
        if not match:
            logging.error(f"Could not extract year from filename: {fname}")
            continue
        
        year = match.group(0)
        dst_path = os.path.join(REPROJECTED_DIR, f"lc2_{year}_mollweide.tif")
        
        # Do not include skipping logic here
        reproject_raster(p, dst_path, MOLLWEIDE_EPSG, TARGET_RES)
        
        # Skip if already exists
        # if os.path.exists(dst_path):
        #     logging.info(f"Reprojected file already exists: {dst_path}. Skipping.")
        # else:
        #     reproject_raster(p, dst_path, MOLLWEIDE_EPSG, TARGET_RES)

        year_to_path[year] = dst_path
    return year_to_path

def get_tile_rasters(year_to_path):
    tile_dir = os.path.join(REPROJECTED_DIR, "tiles")
    os.makedirs(tile_dir, exist_ok=True)

    tile_year_paths = []

    for year, reproj_path in year_to_path.items():
        year_tile_dir = os.path.join(tile_dir, str(year))
        out_tile_paths = tile_raster_to_latitudinal_strips(reproj_path, year_tile_dir)
        for tp in out_tile_paths:
            tile_year_paths.append((tp, year))

    return tile_year_paths

def process_tile(args):
    tile_path, year, gadm_path, completed_pairs = args
    results = []
    try:
        with rasterio.open(tile_path) as src:
            logging.info(f"Processing strip: {tile_path}")
            tile_bounds = src.bounds
            tile_crs = src.crs
            transform = src.transform
            arr = src.read(1)
            nodata = src.nodata

            # Load full GADM dataset
            gadm = gpd.read_file(gadm_path)

            # Reproject GADM to match tile CRS
            gadm = gadm.to_crs(tile_crs)

            # Create polygon from tile bounds (in tile CRS)
            tile_box = box(*tile_bounds)
            tile_geom = gpd.GeoDataFrame(geometry=[tile_box], crs=tile_crs)

            # Spatial filter: keep only GADM features that intersect this tile
            gadm = gpd.sjoin(gadm, tile_geom, how="inner", predicate="intersects").drop(columns="index_right")

            # Logging
            if gadm.empty:
                logging.warning(f"No GADM features found intersecting tile: {tile_path}")
            else:
                logging.info(f"Found {len(gadm)} GADM features in tile: {tile_path}")

            for _, region in gadm.iterrows():
                gid = region["GID_1"]
                
                # commented out so script recomputes pixel counts and shares for all regions
                # if (gid, int(year)) in completed_pairs:
                #     logging.info(f"Skipping already processed region {gid} for year {year}")
                #     continue  # Skip already processed region-year
                
                geom = [mapping(region.geometry)]
                try:
                    out_image, _ = mask(src, shapes=geom, crop=True, filled=True, nodata=nodata)
                    masked = out_image[0].astype(float)

                    if nodata is not None:
                        masked[masked == nodata] = np.nan

                    total_pixels = int(np.sum(~np.isnan(masked)))

                    row = {
                        'GID_1': gid,
                        'year': int(year),
                        'total_pixels': total_pixels
                    }

                    # Count pixels per group
                    masked_int = masked.astype(np.int32)

                    for group_name, codes in LCCS_GROUPS.items():
                        count = np.isin(masked_int, codes).sum()
                        row[f"{group_name}_pixels"] = int(count)

                    # commenting out; old lccs class logic
                    # # Count pixels for each LCCS class
                    # for cls in LCCS_CLASSES:
                    #     row[f"lccs_{cls}_pixels"] = np.sum(masked == cls)

                    results.append(row)

                except ValueError as e:
                    if "Input shapes do not overlap raster" in str(e):
                        pass
                    else:
                        logging.warning(f"Tile {tile_path}, region {gid}: {e}")
                        traceback.print_exc()

    except Exception as e:
        logging.error(f"Error processing tile {tile_path}: {e}")
        traceback.print_exc()

    return results

# Load previously computed results if OUTPUT_CSV exists
completed_pairs = set()
if os.path.exists(OUTPUT_CSV):
    try:
        prev_df = pd.read_csv(OUTPUT_CSV, usecols=["GID_1", "year"])
        completed_pairs = set(tuple(x) for x in prev_df[["GID_1", "year"]].drop_duplicates().values)
        logging.info(f"Found {len(completed_pairs)} previously completed region-year pairs. Will skip them.")
    except Exception as e:
        logging.warning(f"Failed to read existing output file {OUTPUT_CSV}: {e}")

if __name__ == "__main__":
    logging.info("Starting raster preprocessing...")
    year_to_reprojected = preprocess_all_rasters()

    logging.info("Tiling rasters into 30-degree chunks...")
    tile_raster_year_pairs = get_tile_rasters(year_to_reprojected)

    logging.info(f"Processing {len(tile_raster_year_pairs)} tiles with {NUM_WORKERS} workers...")

    all_results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        args = [(tile_path, year, GADM_PATH, completed_pairs) for tile_path, year in tile_raster_year_pairs]
        for tile_results in executor.map(process_tile, args):
            all_results.extend(tile_results)

    logging.info(f"Total records collected: {len(all_results)}")
    if all_results:
        logging.info(f"Sample result: {all_results[0]}")


    logging.info("Merging tile-based regional results...")
    df = pd.DataFrame(all_results)
    if df.empty:
        logging.error("No results were returned — DataFrame is empty.")
        sys.exit("Aborting: no data to aggregate. Check if any regions were processed.")

    # Sum raw counts across all tiles for each GID_1/year
    # df_sum = df.groupby(['GID_1', 'year'], as_index=False).sum()
    
    # create GID_0 column by extracting from GID_1 (GID_0 is first 3 characters of GID_1)
    df['GID_0'] = df['GID_1'].str.slice(0, 3)
    
    # sum by GID_0 and year
    df_sum = df.groupby(['GID_0', 'year'], as_index=False).sum()

    # commenting out; old lccs class logic
    # # Compute shares for each LCCS class
    # for cls in LCCS_CLASSES:
    #     df_sum[f"lccs_{cls}_share"] = df_sum[f"lccs_{cls}_pixels"] / df_sum['total_pixels']

    # compute shares for each LCCS group
    for group_name in LCCS_GROUPS.keys():
        df_sum[f"{group_name}_share"] = (
        df_sum[f"{group_name}_pixels"] / df_sum["total_pixels"]
    )

    # Keep only GID_1, year, and share columns
    # share_cols = [f"lccs_{cls}_share" for cls in LCCS_CLASSES]
    share_cols = [f"{group_name}_share" for group_name in LCCS_GROUPS.keys()]
    
    # df_final = df_sum[['GID_1', 'year'] + share_cols]
    df_final = df_sum[['GID_0', 'year'] + share_cols]

    # df_final.sort_values(["GID_1", "year"], inplace=True)
    df_final.sort_values(["GID_0", "year"], inplace=True)

    df_final.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved LC class shares to {OUTPUT_CSV}")
