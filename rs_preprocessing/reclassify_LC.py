import xarray as xr
import numpy as np
import os
import logging
from datetime import datetime
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import argparse

# ==================== Logging Setup ====================
logger = logging.getLogger("ReclassifyLCCS")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ==================== Reclassification Logic ====================
# Define reclassification flags globally
cropland_flags = [10, 11, 12, 20]
forest_flags = [50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 170]
urban_flags = [190]

def reclassify_chunk(input_path, lat_start, lat_end):
    try:
        logger.info(f"Processing chunk: lat {lat_start} to {lat_end}")
        ds = xr.open_dataset(input_path)
        lccs = ds['lccs_class'].isel(lat=slice(lat_start, lat_end))

        reclass = xr.zeros_like(lccs)
        reclass = reclass.where(~lccs.isin(cropland_flags), 1)
        reclass = reclass.where(~lccs.isin(forest_flags), 2)
        reclass = reclass.where(~lccs.isin(urban_flags), 3)
        return reclass
    except Exception as e:
        logger.exception(f"Failed to process chunk {lat_start}-{lat_end}: {e}")
        raise

# ==================== Main File Processing ====================
def reclassify_chunk_wrapper(args):
    return reclassify_chunk(*args)

'''
def process_file_parallel(input_path, output_path, workers):
    logger.info(f"Opening input file: {input_path}")
    ds = xr.open_dataset(input_path)
    total_lat = ds.sizes['lat']
    logger.info(f"lat size: {total_lat}")

    chunk_size = total_lat // workers
    chunk_ranges = []

    for i in range(workers):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < workers - 1 else total_lat
        chunk_ranges.append((start, end))

    tasks = [(input_path, start, end) for (start, end) in chunk_ranges]

    logger.info(f"Starting parallel processing with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(reclassify_chunk_wrapper, tasks))

    logger.info("Combining chunks")
    reclass_full = xr.concat(results, dim='lat')
    reclass_full.name = "lccs_class_reclass"

    reclass_ds = reclass_full.to_dataset()
    reclass_ds.to_netcdf(output_path, format="NETCDF4", engine="netcdf4")
    logger.info(f"Saved reclassified raster to: {output_path}")
'''

def process_file_parallel(input_path, output_path, num_workers):
    try:
        ds = xr.open_dataset(input_path)
        total_lat = ds.sizes["lat"] if "lat" in ds.sizes else ds.sizes["lat"]
        ds.close()

        # Define 12 chunks
        num_chunks = 12
        indices = np.linspace(0, total_lat, num_chunks + 1, dtype=int)
        chunk_ranges = [(indices[i], indices[i + 1]) for i in range(num_chunks)]

        tasks = [(input_path, start, end) for (start, end) in chunk_ranges]

        logging.info(f"Spawning {len(tasks)} tasks using {num_workers} workers")

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(reclassify_chunk_wrapper, tasks))

        logging.info("All chunks processed. Merging and saving...")

        # Merge all chunks along the 'lat' dimension
        reclass_full = xr.concat(results, dim='lat')
        reclass_full.name = "lccs_class_reclass"

        # Save the merged dataset to NetCDF
        reclass_ds = reclass_full.to_dataset()
        reclass_ds.to_netcdf(output_path, format="NETCDF4", engine="netcdf4")

        logging.info(f"Saved reclassified raster to: {output_path}")


    except Exception as e:
        logging.error(f"Error during reclassification: {e}", exc_info=True)

# ==================== Entry Point ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reclassify global land cover using parallel processing.")
    parser.add_argument("input", help="Path to input .nc file")
    parser.add_argument("output", help="Path to output .nc file")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes to use")

    args = parser.parse_args()

    logger.info("===== Land Cover Reclassification Started =====")
    logger.info(f"Start time: {datetime.now().isoformat()}")

    try:
        process_file_parallel(args.input, args.output, args.workers)
    except Exception as e:
        logger.exception(f"Error during reclassification: {e}")
        exit(1)

    logger.info("===== Land Cover Reclassification Finished =====")
    logger.info(f"End time: {datetime.now().isoformat()}")

'''
import xarray as xr
import numpy as np
import os
import logging
import re

# ==== USER SETTINGS ====
input_dir = "/p/projects/impactee/Josh/geo_variables/land_cover/ESA_CCI_LC_data/data"
output_dir = "/p/projects/impactee/Josh/geo_variables/land_cover/reclassed_intermediates"
# ========================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LandCoverReclass")

def extract_year_from_filename(filename):
    match = re.search(r'\b(19|20)\d{2}\b', filename)
    if match:
        return match.group(0)
    else:
        raise ValueError(f"Could not extract year from filename: {filename}")

def reclassify_lccs(ds):
    if 'lccs_class' not in ds:
        raise KeyError("Variable 'lccs_class' not found in dataset.")

    lccs = ds['lccs_class']

    cropland_flags = [10, 11, 12, 20]
    forest_flags = [50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 170]
    urban_flags = [190]

    reclass = xr.zeros_like(lccs)
    reclass = reclass.where(~lccs.isin(cropland_flags), 1)
    reclass = reclass.where(~lccs.isin(forest_flags), 2)
    reclass = reclass.where(~lccs.isin(urban_flags), 3)

    return reclass

def process_file(input_path, output_path):
    logger.info(f"Processing: {input_path}")
    ds = xr.open_dataset(input_path, chunks={})  # Load entire file unless it’s very large
    reclass = reclassify_lccs(ds)
    reclass.name = "lccs_class_reclass"
    reclass_ds = reclass.to_dataset()
    reclass_ds.to_netcdf(output_path, format="NETCDF4", engine="netcdf4")
    logger.info(f"Saved output to: {output_path}")

def batch_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    nc_files = [f for f in os.listdir(input_dir) if f.endswith('.nc')]

    for fname in nc_files:
        try:
            year = extract_year_from_filename(fname)
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, f"landcover_{year}_reclass.nc")
            process_file(in_path, out_path)
        except Exception as e:
            logger.error(f"Failed to process {fname}: {e}")

if __name__ == "__main__":
    logger.info("==== Starting land cover reclassification ====")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    batch_process(input_dir, output_dir)
    logger.info("==== All done! ====")
'''