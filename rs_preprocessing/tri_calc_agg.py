import os
import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.transform import from_origin
import xarray as xr
from scipy.ndimage import generic_filter
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from rtree import index
import sys
import gc

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),  # Save to file
        logging.StreamHandler(sys.stdout)  # Print to stdout (important for SLURM logs!)
    ]
)

logging.info("Script started...")
# # Setup Logging
# LOG_FILE = "logs/aggregate.log"
# os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Paths
nc_path = "/p/projects/impactee/Josh/geo_variables/elevation/srtm30_full.nc"
tif_path = "/p/projects/impactee/Josh/geo_variables/elevation/srtm30_full.tif"

# Choose the variable name in the NetCDF file
ELEVATION_VAR_NAME = 'Band1'

if os.path.exists(tif_path):
    logging.info(f"{tif_path} already exists. Skipping conversion.")
else:
    logging.info(f"Converting {nc_path} to {tif_path}")
    # Load NetCDF
    ds = xr.open_dataset(nc_path)
    if ELEVATION_VAR_NAME not in ds:
        raise ValueError(f"Variable '{ELEVATION_VAR_NAME}' not found in {nc_path}. Available variables: {list(ds.data_vars)}")

    elevation = ds[ELEVATION_VAR_NAME]

    # Ensure 2D and drop time or extra dimensions if needed
    if 'time' in elevation.dims:
        elevation = elevation.isel(time=0)

    # Extract data and coordinate info
    data = elevation.values.astype(np.float32)
    data = np.flipud(data)  # Flip if needed (depends on CRS and how data is stored)

    lon = elevation['lon'].values
    lat = elevation['lat'].values

    # Calculate transform
    res_x = abs(lon[1] - lon[0])
    res_y = abs(lat[1] - lat[0])
    transform = from_origin(lon.min(), lat.max(), res_x, res_y)

    # Define CRS
    crs = "EPSG:4326"  # Assuming geographic lat/lon

    # Write GeoTIFF
    with rasterio.open(
        tif_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype='float32',
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)

    logging.info(f"Converted {nc_path} to {tif_path}")

# Paths
GADM_PATH = "/p/projects/impactee/Josh/LandScan/gadm_custom_merged.shp"
DATA_DIR = "/p/projects/impactee/Josh/geo_variables/elevation/"
ELEV_PATH = os.path.join(DATA_DIR, 'srtm30_full.tif')
# OUTPUT_CSV = os.path.join(DATA_DIR, 'tri_aggregated.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'tri_aggregated_gid0.csv')
TRI_OUTPUT = os.path.join(DATA_DIR, 'tri.tif')

# Detect available cores
NUM_WORKERS = min(16, multiprocessing.cpu_count())

def process_block(args):
    window, ELEV_PATH = args
    with rasterio.open(ELEV_PATH) as src_local:
        dem = src_local.read(1, window=window, masked=True)
    padded = dem.filled(np.nan)

    def tri_function(window):
        center = window[4]
        if np.isnan(center):
            return np.nan
        diffs = window - center
        diffs = np.delete(diffs, 4)
        return np.sqrt(np.nanmean(diffs ** 2))

    tri_block = generic_filter(padded, tri_function, size=3, mode='constant', cval=np.nan)

    return (window.col_off, window.row_off, tri_block)

# def compute_tri_parallel(ELEV_PATH, TRI_OUTPUT='tri.tif', block_size=256):
#     if os.path.exists(TRI_OUTPUT):
#         logging.info(f"{TRI_OUTPUT} already exists. Skipping recomputation.")
#         with rasterio.open(TRI_OUTPUT) as src:
#             tri_profile = src.profile
#         return None, tri_profile

#     logging.info(f"Computing TRI from {ELEV_PATH}...")

#     with rasterio.open(ELEV_PATH) as src:
#         profile = src.profile
#         width = src.width
#         height = src.height
#         profile.update(dtype='float32', count=1, nodata=np.nan, compress='lzw')

#         block_windows = [
#             Window(x, y,
#                    min(block_size + 2, width - x),
#                    min(block_size + 2, height - y))
#             for y in range(0, height, block_size)
#             for x in range(0, width, block_size)
#         ]
        
#         total_blocks = len(block_windows)
#         logging.info(f"Total blocks to process: {total_blocks}")

#         from concurrent.futures import as_completed

#         max_workers = min(4, multiprocessing.cpu_count())  # use fewer workers
#         logging.info(f"Using {max_workers} workers for TRI computation (memory-safe mode)")

#         with rasterio.open(TRI_OUTPUT, 'w', **profile) as dst:
#             with ProcessPoolExecutor(max_workers=max_workers) as executor:
#                 futures = {executor.submit(process_block, (w, ELEV_PATH)): w for w in block_windows}
#                 for i, future in enumerate(as_completed(futures)):
#                     try:
#                         x_off, y_off, tri_block = future.result()
#                         h, w = tri_block.shape
#                         dst.write(tri_block, 1, window=Window(x_off, y_off, w, h))
#                         del tri_block
#                     except Exception as e:
#                         logging.error(f"Block failed: {e}")
#                     if i % 100 == 0:
#                         logging.info(f"Processed {i}/{total_blocks} blocks...")
#                         gc.collect()

#     # # Prepare argument list for multiprocessing
#     # args = [(window, ELEV_PATH) for window in block_windows]

#     # with ProcessPoolExecutor(max_workers=min(4, multiprocessing.cpu_count())) as executor:
#     #     results = list(executor.map(process_block, args))

#     # # # Prepare full output
#     # # full_tri = np.full((height, width), np.nan, dtype=np.float32)
#     # # for x_off, y_off, tri_block in results:
#     # #     h, w = tri_block.shape
#     # #     full_tri[y_off:y_off + h, x_off:x_off + w] = tri_block

#     # # profile.update(dtype='float32', count=1, nodata=np.nan, compress='lzw')

#     # with rasterio.open(TRI_OUTPUT, 'w', **profile) as dst:
#     #     dst.write(full_tri, 1)

#     logging.info(f"TRI saved to {TRI_OUTPUT}")
#     return None, profile

def compute_tri_parallel(ELEV_PATH, TRI_OUTPUT='tri.tif', block_size=128):
    """
    Memory-safe serial fallback version of TRI computation.
    Processes the raster one block at a time, writing directly to disk.
    """
    if os.path.exists(TRI_OUTPUT):
        logging.info(f"{TRI_OUTPUT} already exists. Skipping recomputation.")
        with rasterio.open(TRI_OUTPUT) as src:
            tri_profile = src.profile
        return None, tri_profile

    logging.info(f"Computing TRI from {ELEV_PATH} (serial mode)...")

    with rasterio.open(ELEV_PATH) as src:
        profile = src.profile
        width, height = src.width, src.height
        profile.update(dtype='float32', count=1, nodata=np.nan, compress='lzw')

        block_windows = [
            Window(x, y,
                   min(block_size + 2, width - x),
                   min(block_size + 2, height - y))
            for y in range(0, height, block_size)
            for x in range(0, width, block_size)
        ]
        total_blocks = len(block_windows)
        logging.info(f"Total blocks to process (serial): {total_blocks}")

        with rasterio.open(TRI_OUTPUT, 'w', **profile) as dst:
            for i, window in enumerate(block_windows):
                try:
                    x_off, y_off, tri_block = process_block((window, ELEV_PATH))
                    h, w = tri_block.shape
                    dst.write(tri_block, 1, window=Window(x_off, y_off, w, h))
                    del tri_block
                except Exception as e:
                    logging.error(f"Block {i} failed: {e}")
                if i % 100 == 0:
                    logging.info(f"Processed {i}/{total_blocks} blocks...")
                    gc.collect()

    logging.info(f"TRI saved to {TRI_OUTPUT}")
    return None, profile

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
                "avg_tri": np.nan,
                "std_tri": np.nan,
                "min_tri": np.nan,
                "max_tri": np.nan
            }

        stats = {
            # "GID_1": gid,
            "GID_0": gid,
            "avg_tri": np.nanmean(valid_data),
            "std_tri": np.nanstd(valid_data),
            "min_tri": np.nanmin(valid_data),
            "max_tri": np.nanmax(valid_data),
        }

        logging.info(f"Finished region {gid}, stats: {stats}")
        return stats

    except Exception as e:
        logging.error(f"Error processing region {gid}: {e}")
        return {
            # "GID_1": gid,
            "GID_0": gid,
            "avg_tri": np.nan,
            "std_tri": np.nan,
            "min_tri": np.nan,
            "max_tri": np.nan
        }

# Compute TRI if not already done
logging.info("Computing TRI...")
tri_array, profile = compute_tri_parallel(ELEV_PATH, TRI_OUTPUT)

# Load GADM shapefile
logging.info("Loading GADM shapefile...")
gadm = gpd.read_file(GADM_PATH)
logging.info(f"Loaded {len(gadm)} regions.")

# Ensure GADM projection matches raster projection
with rasterio.open(TRI_OUTPUT) as src:
    raster_crs = src.crs or CRS.from_epsg(4326)  # assigns 4326 if CRS is not defined; metadata shows EPSE:4326
    if gadm.crs != raster_crs:
        logging.info("Reprojecting GADM shapefile to match raster CRS...")
        gadm = gadm.to_crs(raster_crs)
    else:
        logging.info("GADM CRS matches raster CRS, no reprojection needed.")

# GID_0 version only: dissolve by GID_0 (merge all GID_1s per GID_0)
logging.info("Dissolving GADM regions by GID_0...")
gadm_gid0 = gadm.dissolve(by="GID_0", as_index=False)
logging.info(f"Created {len(gadm_gid0)} GID_0-level regions.")

# Prepare args for multiprocessing
logging.info("Preparing arguments for multiprocessing...")
args_list = [(region, TRI_OUTPUT) for _, region in gadm_gid0.iterrows()]
# args_list = [(region, TRI_OUTPUT) for _, region in gadm.iterrows()]

# # Run processing
# logging.info("Starting region-wise processing for coast distance...")
# with ProcessPoolExecutor() as executor:
#     results = list(executor.map(process_region, args_list))

# Run processing serially (memory-safe mode)
logging.info("Starting region-wise processing for TRI (serial mode)...")
results = []
for i, args in enumerate(args_list):
    try:
        res = process_region(args)
        results.append(res)
        if i % 10 == 0:
            logging.info(f"Processed {i}/{len(args_list)} regions...")
        gc.collect()
    except Exception as e:
        logging.error(f"Error processing region {i}: {e}")

# Save results
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
logging.info(f"Results saved to {OUTPUT_CSV}")
