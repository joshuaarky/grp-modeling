'''
This script calculates the distance from river features with ORD_STRA >= 7
The previous version (v2) calculates distance from river features for each ORD_STRA value
(7, 8, 9, 10) separately and writes them to individual rasters.
'''

import os
import sys
import logging
import rasterio
from rasterio import features, mask
from rasterio.transform import from_origin
from rasterio.crs import CRS
import geopandas as gpd
import numpy as np
from scipy.ndimage import distance_transform_edt
import psutil
import traceback

# Logging setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/river_distance_filtered.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info("Script started.")

# File paths and raster params
base_dir = '/p/projects/impactee/Josh/geo_variables/'
rivers_path = os.path.join(base_dir, 'rivers/rivers_stra7to10/rivers_stra7to10.gpkg')
gadm_path = "/p/projects/impactee/Josh/LandScan/gadm_custom_merged.shp"
output_dir = os.path.join(base_dir, 'rivers')
os.makedirs(output_dir, exist_ok=True)

pixel_size = 500
crs = CRS.from_string("ESRI:54009")  # Mollweide

# Load and reproject GADM
logging.info(f"Loading and reprojecting GADM shapefile from {gadm_path}")
gadm = gpd.read_file(gadm_path)
gadm = gadm.to_crs(crs)
logging.info(f"{len(gadm)} total GADM features loaded.")

# Get bounding box
xmin, ymin, xmax, ymax = gadm.total_bounds
logging.info(f"GADM bounding box: ({xmin}, {ymin}, {xmax}, {ymax})")

# Define raster grid
width = int((xmax - xmin) / pixel_size)
height = int((ymax - ymin) / pixel_size)
transform = from_origin(xmin, ymax, pixel_size, pixel_size)
raster_shape = (height, width)

# Rasterize GADM mask
gadm_mask = features.rasterize(
    [(geom, 1) for geom in gadm.geometry],
    out_shape=raster_shape,
    transform=transform,
    fill=0,
    dtype="uint8"
).astype(bool)

# Load and reproject rivers
logging.info(f"Loading rivers from {rivers_path}")
rivers = gpd.read_file(rivers_path).to_crs(crs)
logging.info(f"{len(rivers)} total river features loaded.")

# Filter river features where ORD_STRA >= 7
attribute = "ORD_STRA"
logging.info(f"Filtering rivers where {attribute} >= 7")
filtered_rivers = rivers[rivers[attribute] >= 7]
if filtered_rivers.empty:
    logging.error(f"No river features with {attribute} >= 7 found. Exiting.")
    sys.exit(1)
else:
    logging.info(f"{len(filtered_rivers)} river features retained after filtering.")

try:
    # Rasterize filtered features to binary mask
    logging.info(f"Rasterizing filtered river features...")
    mask_raster = features.rasterize(
        [(geom, 1) for geom in filtered_rivers.geometry],
        out_shape=raster_shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    ).astype(bool)

    # Compute distance transform
    logging.info(f"Computing distance transform...")
    dist = distance_transform_edt(~mask_raster) * pixel_size
    dist[~np.isfinite(dist)] = np.nan

    # Apply GADM mask
    dist[~gadm_mask] = np.nan

    # Write raster
    out_meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform
    }
    out_raster_path = os.path.join(output_dir, f"river_stra_ge7_distance.tif")
    logging.info(f"Writing output raster to {out_raster_path}")
    with rasterio.open(out_raster_path, 'w', **out_meta) as dst:
        dst.write(dist.astype("float32"), 1)
        dst.set_band_description(1, f"{attribute}_ge7")

    logging.info(f"Output written successfully to {out_raster_path}")

except Exception as e:
    logging.error(f"Error while processing ORD_STRA >= 7: {e}")
    logging.error(traceback.format_exc())
