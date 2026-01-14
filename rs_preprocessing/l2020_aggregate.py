# Update Nov. 3rd, 2025: added alternate lines for national-level aggregation

import os
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import re
from shapely.geometry import Point
from rasterio.mask import mask
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
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths (modify as needed)
GADM_PATH = "/p/projects/impactee/Josh/LandScan/gadm_custom_merged.shp"
DATA_DIR = "/p/projects/impactee/Josh/L2020/data/"
# OUTPUT_CSV = "/p/projects/impactee/Josh/L2020/output/aggregated_results.csv"
OUTPUT_CSV = "/p/projects/impactee/Josh/L2020/output/aggregated_results_gid0_v2.csv"

# Detect available cores
NUM_WORKERS = min(16, multiprocessing.cpu_count())  # Adjust based on HPC resources

# Load GADM shapefile
logging.info("Loading GADM shapefile...")
gadm = gpd.read_file(GADM_PATH)
logging.info(f"Loaded {len(gadm)} regions.")

for handler in logging.getLogger().handlers:
    handler.flush()

# Ensure GADM projection matches raster projection
raster_sample = next((f for f in os.listdir(DATA_DIR) if f.endswith(".tif")), None)
if raster_sample:
    with rasterio.open(os.path.join(DATA_DIR, raster_sample)) as src:
        if gadm.crs != src.crs:
            logging.info("Reprojecting GADM shapefile to match raster CRS...")
            gadm = gadm.to_crs(src.crs)

for handler in logging.getLogger().handlers:
    handler.flush()

# GID_0 version only: dissolve by GID_0 (merge all GID_1s per GID_0)
logging.info("Dissolving GADM regions by GID_0...")
gadm_gid0 = gadm.dissolve(by="GID_0", as_index=False)
logging.info(f"Created {len(gadm_gid0)} GID_0-level regions.")

# Create an R-tree index for spatial lookup
logging.info("Creating R-tree index for region lookups...")
idx = index.Index()
for i, region in gadm_gid0.iterrows():
# for i, region in gadm.iterrows():
    idx.insert(i, region.geometry.bounds, obj=region.geometry)

def process_region_wrapper(args):
    return process_region(*args)  # Unpack tuple before calling

def process_region(region, raster_path, year):
    # gid = region["GID_1"]
    gid = region["GID_0"]
    logging.info(f"Processing region: {gid} for year: {year}")

    try:
        with rasterio.open(raster_path) as src:
            nodata_value = src.nodata if src.nodata is not None else 0
            out_image, out_transform = mask(src, [region.geometry], crop=True, nodata=nodata_value)

        out_image = out_image[0]  # Convert from (1, H, W) to (H, W)

        if out_image.size == 0 or np.all(out_image == nodata_value):
            logging.info(f"Region {gid}: Masking resulted in an empty raster.")
            # return {"GID_1": gid, "year": year, "NTL_total": 0, **{f"DN{i}": 0 for i in range(64)}}
            return {"GID_0": gid, "year": year, "NTL_total": 0, **{f"DN{i}": 0 for i in range(64)}}

        # Compute NTL_total (sum of all values > 0)
        NTL_total = np.sum(out_image[out_image > 0])

        # Count occurrences of values between 0-63
        value_counts = np.bincount(out_image.flatten(), minlength=64)[:64]  # Ensure array size 64
        dn_counts = {f"DN{i}": value_counts[i] for i in range(64)}

        logging.info(f"Finished processing region: {gid}, NTL_total: {NTL_total}")
        # return {"GID_1": gid, "year": year, "NTL_total": NTL_total, **dn_counts}
        return {"GID_0": gid, "year": year, "NTL_total": NTL_total, **dn_counts}

    except Exception as e:
        logging.error(f"Error processing region {gid} for year {year}: {e}")
        # return {"GID_1": gid, "year": year, "NTL_total": -1, **{f"DN{i}": 0 for i in range(64)}}
        return {"GID_0": gid, "year": year, "NTL_total": -1, **{f"DN{i}": 0 for i in range(64)}}

def process_all_rasters(gadm, data_dir):
    results = []

    for raster_filename in os.listdir(data_dir):
        # Match naming format with year in the middle (e.g., 'Harmonized_DN_NTL_2007_calDMSP.tif')
        year_match = re.search(r'Harmonized_DN_NTL_(\d{4})', raster_filename)
        if not year_match:
            continue  # Skip files that don't match the pattern

        year = int(year_match.group(1))  # Extract year from the filename
        raster_path = os.path.join(data_dir, raster_filename)
        logging.info(f"Processing raster: {raster_filename} for year {year}")

        # Prepare arguments for parallel processing of regions
        args_list = [(region, raster_path, year) for _, region in gadm.iterrows()]

        # Process regions using multiprocessing
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            region_results = list(executor.map(process_region_wrapper, args_list))

        results.extend(region_results)

    return results

# Run Processing
logging.info("Starting processing of all rasters...")
results = process_all_rasters(gadm, DATA_DIR)
logging.info("Finished processing all rasters.")

for handler in logging.getLogger().handlers:
    handler.flush()

# Save results
# Convert list of dictionaries to DataFrame
df = pd.DataFrame(results)

# Ensure all DN columns exist (fill missing ones with 0)
# expected_columns = ["GID_1", "year", "NTL_total"] + [f"DN{i}" for i in range(64)]
expected_columns = ["GID_0", "year", "NTL_total"] + [f"DN{i}" for i in range(64)]
df = df.reindex(columns=expected_columns, fill_value=0)

# # assuming your dataframe is called df
# df_agg = (
#     df.groupby(["GID_0", "year"], as_index=False)["NTL_total"]
#       .sum()
# )

# Save to CSV
df_agg.to_csv(OUTPUT_CSV, index=False)
logging.info(f"Results saved to {OUTPUT_CSV}")