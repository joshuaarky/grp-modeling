"""
This script aggregates river distance statistics for each GID_1 region
for river features with ORD_STRA >= 7, corresponding to the raster
produced by 'rivers_dist_calc_v3.py'
"""

import os
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from rasterio.mask import mask
from concurrent.futures import ProcessPoolExecutor
import logging
import sys
import multiprocessing
import traceback

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/river_aggregate_ord_stra_ge7.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Starting ORD_STRA >= 7 raster statistics aggregation script...")

# Paths
GADM_PATH = "/p/projects/impactee/Josh/LandScan/gadm_custom_merged.shp"
RASTER_PATH = "/p/projects/impactee/Josh/geo_variables/rivers/river_stra_ge7_distance.tif"
OUTPUT_CSV = "/p/projects/impactee/Josh/geo_variables/rivers/major_river_dist_aggregated.csv"
NUM_WORKERS = min(16, multiprocessing.cpu_count())

# Load GADM shapefile
try:
    logging.info(f"Loading GADM shapefile from {GADM_PATH}")
    gadm = gpd.read_file(GADM_PATH)
    logging.info(f"Loaded {len(gadm)} GADM regions")
except Exception as e:
    logging.error(f"Failed to load GADM shapefile: {e}")
    sys.exit(1)

# Detect CRS from the raster
with rasterio.open(RASTER_PATH) as ref:
    raster_crs = ref.crs

# Reproject GADM if needed
if gadm.crs != raster_crs:
    logging.info("Reprojecting GADM to match raster CRS...")
    gadm = gadm.to_crs(raster_crs)

def process_region(region):
    gid = region.get("GID_1", None)
    if gid is None:
        logging.warning("Region missing GID_1, skipping")
        return None

    logging.info(f"Processing GID_1: {gid}")
    stats = {"GID_1": gid}
    geom = [region.geometry]

    try:
        with rasterio.open(RASTER_PATH) as src:
            nodata = src.nodata if src.nodata is not None else np.nan
            out_image, _ = mask(src, geom, crop=True, filled=True)
            data = out_image[0].astype(float)
            data[data == nodata] = np.nan

            prefix = "major_river_dist"
            stats[f"{prefix}_mean"] = np.nanmean(data)
            stats[f"{prefix}_std"] = np.nanstd(data)
            stats[f"{prefix}_min"] = np.nanmin(data)
            stats[f"{prefix}_max"] = np.nanmax(data)

        return stats

    except Exception as e:
        logging.error(f"Error processing GID_1={gid}: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    logging.info("Beginning parallel processing of regions...")

    args_list = [row for _, row in gadm.iterrows()]
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(process_region, args_list))

    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    logging.info(f"Saved statistics to {OUTPUT_CSV}")
    logging.info("Script finished.")
