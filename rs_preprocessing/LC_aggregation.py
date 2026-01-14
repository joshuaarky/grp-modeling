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
LC_RASTER_DIR = "/p/projects/impactee/Josh/geo_variables/land_cover/reclassed_intermediates"  # original rasters folder (e.g. 'lc_1992.tif')
REPROJECTED_DIR = "/p/projects/impactee/Josh/geo_variables/land_cover/reprojected_rasters"  # folder to cache reprojections
GADM_PATH = "/p/projects/impactee/Josh/LandScan/gadm_custom_merged.shp"
OUTPUT_CSV = "/p/projects/impactee/Josh/geo_variables/land_cover/lc_class_shares.csv"
MOLLWEIDE_EPSG = "ESRI:54009"
TARGET_RES = 300  # meters
NUM_WORKERS = min(8, multiprocessing.cpu_count())

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/lc_class_aggregation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def reproject_raster(src_path, dst_path, dst_crs, resolution):
    """Reproject src raster to dst_crs at resolution, save to dst_path as GeoTIFF.
    Clips just inside poles and dateline to avoid reprojection edge issues."""

    if os.path.exists(dst_path):
        logging.info(f"Reprojected raster already exists: {dst_path}")
        return

    with rasterio.open(src_path) as src:
        if src.crs is None:
            logging.warning(f"No CRS found in source raster {src_path}. Assuming EPSG:4326.")
            src_crs = "EPSG:4326"
        else:
            src_crs = src.crs

        # Step 1: Define clipped window just inside global bounds
        clipped_window = from_bounds(
            -179.999, -89.999, 179.999, 89.999,
            transform=src.transform
        )
        clipped_transform = src.window_transform(clipped_window)
        clipped_width = clipped_window.width
        clipped_height = clipped_window.height

        # Step 2: Calculate destination transform
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs,
            int(clipped_width), int(clipped_height),
            *rasterio.transform.array_bounds(int(clipped_height), int(clipped_width), clipped_transform),
            resolution=(resolution, resolution)
        )

        # Step 3: Handle nodata safely
        dst_nodata = src.nodata
        if dst_nodata is None or dst_nodata in [0, 1, 2, 3]:
            dst_nodata = 255  # Or -9999 for float rasters

        # Step 4: Prepare metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'driver': 'GTiff',
            'compress': 'lzw',
            'tiled': True,
            'nodata': dst_nodata,
            'dtype': 'uint8',
            'count': src.count
        })

        # Step 5: Reproject
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                src_data = src.read(i, window=clipped_window)

                reproject(
                    source=src_data,
                    destination=rasterio.band(dst, i),
                    src_transform=clipped_transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    src_nodata=src.nodata,
                    dst_nodata=dst_nodata,
                    warp_mem_limit=512,  # Lower memory use
                    num_threads=4
                )

    logging.info(f"Saved reprojected raster: {dst_path}")

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
            if os.path.exists(tile_path):
                logging.info(f"Strip {i} already exists: {tile_path}. Skipping.")
                out_paths.append(tile_path)
                continue

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
        # Extract year from filename, e.g. "lc_1992.tif" -> "1992"
        year = ''.join(filter(str.isdigit, fname))
        dst_path = os.path.join(REPROJECTED_DIR, f"lc_{year}_mollweide.tif")
        reproject_raster(p, dst_path, MOLLWEIDE_EPSG, TARGET_RES)
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
    """2025-06-16: Replaces process_region for tiled rasters."""
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
                if (gid, int(year)) in completed_pairs:
                    logging.info(f"Skipping already processed region {gid} for year {year}")
                    continue  # Skip already processed region-year
                
                geom = [mapping(region.geometry)]
                try:
                    out_image, _ = mask(src, shapes=geom, crop=True, filled=True, nodata=nodata)
                    masked = out_image[0].astype(float)

                    if nodata is not None:
                        masked[masked == nodata] = np.nan

                    cropland_pixels = np.sum(masked == 1)
                    forest_pixels = np.sum(masked == 2)
                    urban_pixels = np.sum(masked == 3)
                    total_pixels = np.sum(~np.isnan(masked))

                    results.append({
                        'GID_1': gid,
                        'year': int(year),
                        'total_pixels': total_pixels,
                        'cropland_pixels': cropland_pixels,
                        'forest_pixels': forest_pixels,
                        'urban_pixels': urban_pixels
                    })

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
    df_sum = df.groupby(['GID_1', 'year'], as_index=False).sum()

    # Compute accurate pixel-weighted shares
    df_sum['cropland_share'] = df_sum['cropland_pixels'] / df_sum['total_pixels']
    df_sum['forest_share'] = df_sum['forest_pixels'] / df_sum['total_pixels']
    df_sum['urban_share']   = df_sum['urban_pixels']   / df_sum['total_pixels']

    # Keep only necessary columns
    df_final = df_sum[['GID_1', 'year', 'cropland_share', 'forest_share', 'urban_share']]

    df_final.sort_values(["GID_1", "year"], inplace=True)
    df_final.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved LC class shares to {OUTPUT_CSV}")
