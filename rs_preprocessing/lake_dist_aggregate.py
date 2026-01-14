import os
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from rasterio.mask import mask
from rasterio.crs import CRS
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from rtree import index
import sys

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),  # Save to file
        logging.StreamHandler(sys.stdout)  # Print to stdout (important for SLURM logs!)
    ]
)

logging.info("Script started...")
# Setup Logging
LOG_FILE = "logs/aggregate.log"
os.makedirs("logs", exist_ok=True)

# Paths
GADM_PATH = "/p/projects/impactee/Josh/LandScan/gadm_custom_merged.shp"
DATA_DIR = "/p/projects/impactee/Josh/geo_variables/lakes/"
LAKE_DIST = os.path.join(DATA_DIR, 'lake_dist_mollweide.tif')
# OUTPUT_CSV = os.path.join(DATA_DIR, 'lake_dist_aggregated.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'lake_dist_aggregated_gid0.csv')

# Detect available cores
NUM_WORKERS = min(16, multiprocessing.cpu_count())

# Load GADM shapefile
logging.info("Loading GADM shapefile...")
gadm = gpd.read_file(GADM_PATH)
logging.info(f"Loaded {len(gadm)} regions.")

# Ensure GADM projection matches raster projection
with rasterio.open(LAKE_DIST) as src:
    raster_crs = src.crs
    if gadm.crs != raster_crs:
        logging.info("Reprojecting GADM shapefile to match raster CRS...")
        gadm = gadm.to_crs(raster_crs)
    else:
        logging.info("GADM CRS matches raster CRS, no reprojection needed.")

# GID_0 version only: dissolve by GID_0 (merge all GID_1s per GID_0)
logging.info("Dissolving GADM regions by GID_0...")
gadm_gid0 = gadm.dissolve(by="GID_0", as_index=False)
logging.info(f"Created {len(gadm_gid0)} GID_0-level regions.")

# Build spatial index (optional for future optimization)
idx = index.Index()
for i, region in gadm_gid0.iterrows():
# for i, region in gadm.iterrows():
    idx.insert(i, region.geometry.bounds, obj=region.geometry)

def process_region(args):
    region, raster_path = args
    # gid = region["GID_1"]
    gid = region["GID_0"]
    logging.info(f"Processing region: {gid}")
    
    try:
        with rasterio.open(raster_path) as src:
            logging.info(f"Raster CRS: {src.crs}")
            logging.info(f"Raster descriptions: {src.descriptions}")
            
            nodata = src.nodata if src.nodata is not None else np.nan
            out_image, _ = mask(src, [region.geometry], crop=True, nodata=nodata)

        out_image = out_image[0]  # Convert from (1, H, W) to (H, W)
        valid_data = out_image.astype(float)
        valid_data[valid_data == nodata] = np.nan

        if np.isnan(valid_data).all():
            logging.info(f"Region {gid}: All values are nodata or masked.")
            return {
                # "GID_1": gid,
                "GID_0": gid,
                "avg_lake_dist": np.nan,
                "std_lake_dist": np.nan,
                "min_lake_dist": np.nan,
                "max_lake_dist": np.nan
            }
            
        stats = {
            # "GID_1": gid,
            "GID_0": gid,
            "avg_lake_dist": np.nanmean(valid_data),
            "std_lake_dist": np.nanstd(valid_data),
            "min_lake_dist": np.nanmin(valid_data),
            "max_lake_dist": np.nanmax(valid_data),
        }

        logging.info(f"Finished region {gid}, stats: {stats}")
        return stats

    except Exception as e:
        logging.error(f"Error processing region {gid}: {e}")
        return {
            # "GID_1": gid,
            "GID_0": gid,
            "avg_lake_dist": np.nan,
            "std_lake_dist": np.nan,
            "min_lake_dist": np.nan,
            "max_lake_dist": np.nan
        }

# Prepare args for multiprocessing
logging.info("Preparing arguments for multiprocessing...")
# args_list = [(region, LAKE_DIST) for _, region in gadm.iterrows()]
args_list = [(region, LAKE_DIST) for _, region in gadm_gid0.iterrows()]

# Run processing
logging.info("Starting region-wise processing for lake distance...")
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_region, args_list))

# Save results
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
logging.info(f"Results saved to {OUTPUT_CSV}")
