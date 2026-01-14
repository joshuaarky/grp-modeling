import logging
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from rasterio.crs import CRS
import geopandas as gpd
import numpy as np
from scipy.ndimage import distance_transform_edt
import os
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Script started...")

# Create logs directory
LOG_FILE = "logs/aggregate.log"
os.makedirs("logs", exist_ok=True)

# Define paths
directory = '/p/projects/impactee/Josh/geo_variables/'
lake_shp_path = os.path.join(directory, 'lakes/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp')
output_path = os.path.join(directory, 'lakes/lake_dist_mollweide.tif')

# Define Mollweide raster parameters (global)
pixel_size = 1000  # meters per pixel
crs = CRS.from_string("ESRI:54009")  # Mollweide projection

# Global bounds in Mollweide (in meters)
xmin, ymin = -20037508.34, -10018754.17
xmax, ymax = 20037508.34, 10018754.17
width = int((xmax - xmin) / pixel_size)
height = int((ymax - ymin) / pixel_size)

transform = from_origin(xmin, ymax, pixel_size, pixel_size)
raster_shape = (height, width)
raster_meta = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "float32",
    "crs": crs,
    "transform": transform
}

logging.info(f"Creating empty global Mollweide raster: {raster_shape}, pixel size: {pixel_size}m")

# Load and reproject lake polygons
logging.info(f"Loading lake polygons from: {lake_shp_path}")
lake_gdf = gpd.read_file(lake_shp_path).to_crs(crs)
logging.info(f"Lake vector data loaded with {len(lake_gdf)} features")

# Rasterize the lake polygons: water = 1, land = 0
logging.info("Rasterizing lake polygons...")
water_mask = features.rasterize(
    [(geom, 1) for geom in lake_gdf.geometry],
    out_shape=raster_shape,
    transform=transform,
    fill=0,
    dtype=np.uint8
)
logging.info("Lake rasterization complete.")

# Compute distance transform
logging.info("Computing distance transform...")
inverted_mask = 1 - water_mask  # 1 = land, 0 = water
pixel_distances = distance_transform_edt(inverted_mask)

# Convert pixel distances to meters
spatial_distances = pixel_distances * pixel_size
logging.info("Distance computation complete.")

# Save distance raster
logging.info(f"Saving distance raster to: {output_path}")
raster_meta.update(dtype='float32', count=1)
with rasterio.open(output_path, 'w', **raster_meta) as dst:
    dst.write(spatial_distances.astype('float32'), 1)

logging.info("Distance raster written successfully.")