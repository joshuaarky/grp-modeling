# this is a combined version of 'df_merge.py', 'var_corr_analysis.py', 'merge_WB_GDPpc.py', 'merge_WB_GDPpc_lcu.py',
# and pieces of 'analysis.py' from the geopy project using L2020 data

#######################################################################################
### Step 1: create 'merged_data_outer.csv' and 'merged_data_inner.csv' ################
#######################################################################################

import pandas as pd
from functools import reduce
import numpy as np

def outer_merge_on_gid1_year(dfs):
    """
    Perform an outer merge on multiple DataFrames with a multi-index of GID_1 and year.
    DataFrames without a 'year' column are broadcasted to all matching GID_1-year combos
    based on the union of 'GID_1' and 'year' values from other DataFrames.
    """
    all_gid1_year = pd.DataFrame(columns=['GID_1', 'year'])
    for df in dfs:
        if 'GID_1' in df.columns and 'year' in df.columns:
            all_gid1_year = pd.concat([all_gid1_year, df[['GID_1', 'year']]])
    all_gid1_year = all_gid1_year.drop_duplicates().reset_index(drop=True)

    expanded_dfs = []
    for df in dfs:
        if 'GID_1' not in df.columns:
            raise ValueError("Each DataFrame must contain a 'GID_1' column.")
        if 'year' in df.columns:
            expanded_dfs.append(df.copy())
        else:
            expanded = pd.merge(
                all_gid1_year[['GID_1', 'year']],
                df,
                on='GID_1',
                how='left'
            )
            expanded_dfs.append(expanded)

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['GID_1', 'year'], how='outer'), expanded_dfs)
    return merged_df.set_index(['GID_1', 'year']).sort_index()


def inner_merge_on_gid1_year(dfs):
    """
    Perform an inner merge on multiple DataFrames with a multi-index of GID_1 and year.
    Only (GID_1, year) combinations common to all DataFrames will be retained.
    DataFrames without a 'year' column are broadcasted across the intersection.
    """
    all_gid1_year_sets = [
        set(zip(df['GID_1'], df['year']))
        for df in dfs if 'year' in df.columns
    ]
    common_gid1_year = set.intersection(*all_gid1_year_sets)
    all_gid1_year_df = pd.DataFrame(common_gid1_year, columns=['GID_1', 'year'])

    expanded_dfs = []
    for df in dfs:
        if 'GID_1' not in df.columns:
            raise ValueError("Each DataFrame must contain a 'GID_1' column.")
        if 'year' in df.columns:
            filtered_df = df.merge(all_gid1_year_df, on=['GID_1', 'year'], how='inner')
            expanded_dfs.append(filtered_df)
        else:
            expanded = pd.merge(
                all_gid1_year_df,
                df,
                on='GID_1',
                how='left'
            )
            expanded_dfs.append(expanded)

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['GID_1', 'year'], how='inner'), expanded_dfs)
    return merged_df.set_index(['GID_1', 'year']).sort_index()

def check_complete_years(df, name=""):
    df = df.reset_index()
    print(f"\n--- {name} MERGE SUMMARY ---")
    print("Columns:", df.columns.tolist())
    unique_years = sorted(df['year'].dropna().unique())
    print("All unique years:", unique_years)

    value_columns = [col for col in df.columns if col not in ['GID_1', 'year']]
    years_with_full_data = (
        df.groupby('year')[value_columns]
        .apply(lambda x: x.notnull().all().all())
    )
    complete_years = years_with_full_data[years_with_full_data].index.tolist()
    print("Years with complete data across all variables and regions:", complete_years)

# --- Load DataFrames ---
coasts = pd.read_csv("/p/projects/impactee/Josh/geo_variables/coasts/coast_dist_aggregated.csv")
elevation = pd.read_csv("/p/projects/impactee/Josh/geo_variables/elevation/tri_aggregated.csv")
lakes = pd.read_csv("/p/projects/impactee/Josh/geo_variables/lakes/lake_dist_aggregated.csv")

# updated land cover shares with complete lccs classes
land_cover = pd.read_csv("/p/projects/impactee/Josh/geo_variables/land_cover/lc2_class_shares.csv")
# old land cover shares
land_cover_old = pd.read_csv("/p/projects/impactee/Josh/geo_variables/land_cover/lc_class_shares.csv")

rivers = pd.read_csv("/p/projects/impactee/Josh/geo_variables/rivers/major_river_dist_aggregated.csv")
dose = pd.read_csv("/p/projects/impactee/DOSE_creation/DoseV2Corrections/DOSE_V2.10.csv")
z2024 = pd.read_csv("/p/projects/impactee/Josh/Z2024/Z2024_grp_ntl.csv", dtype={1: str})
z2024 = z2024[['GID_1', 'year', 'NTL_sum', 'lNTL_sum', 'laglNTL_sum']]

# L2020: new in this version
l2020 = pd.read_csv("/p/projects/impactee/Josh/L2020/data/L2020_aggregated.csv")

# select DN > 7 to filter out background noise (after Li et al., 2020)
columns_dn_g_7 = [f'DN{i}' for i in range(8, 64)]

# Ensure the columns exist in the DataFrame
columns_dn_g_7 = [col for col in columns_dn_g_7 if col in l2020.columns]

def compute_mean_std(row, columns):
    # Extract counts
    counts = row[columns].values
    # Get the numeric part from column names (DN0 -> 0, DN1 -> 1, etc.)
    values = np.array([int(col.replace("DN", "")) for col in columns])
    
    # Total count
    n = counts.sum()
    if n == 0:
        return pd.Series({"mean": np.nan, "std": np.nan})
    
    # Weighted mean
    mean = (counts * values).sum() / n
    
    # Weighted variance
    var = (counts * (values - mean) ** 2).sum() / n
    std = np.sqrt(var)
    
    return pd.Series({"mean": mean, "std": std})

# Apply row-wise
l2020[['NTL_mean', 'NTL_std']] = l2020.apply(lambda row: compute_mean_std(row, columns_dn_g_7), axis=1)

l2020 = l2020[['GID_1', 'year', 'NTL_mean', 'NTL_std']]

# --- Merge Inputs ---
dfs = [coasts, elevation, lakes, land_cover, land_cover_old, rivers, dose, z2024, l2020]

# Outer merge
outer_df = outer_merge_on_gid1_year(dfs)
outer_df = outer_df.reset_index()
# Fill null GID_0 values using first 3 characters from GID_1
outer_df['GID_0'] = outer_df['GID_0'].fillna(outer_df['GID_1'].str[:3])
outer_df.to_csv("/p/projects/impactee/Josh/thesis_analysis/merged_data_outer.csv")
check_complete_years(outer_df, name="OUTER")

# # Inner merge
# inner_df = inner_merge_on_gid1_year(dfs)
# inner_df = inner_df.reset_index()
# # Fill null GID_0 values using first 3 characters from GID_1
# inner_df['GID_0'] = inner_df['GID_0'].fillna(inner_df['GID_1'].str[:3])
# inner_df.to_csv("/p/projects/impactee/Josh/thesis_analysis/merged_data_inner.csv")
# check_complete_years(inner_df, name="INNER")

# Print summary information for outer_df
print("\n--- OUTER MERGE SUMMARY ---")
# number of entries for each column
outer_df = outer_df.reset_index()
print(outer_df.count())

df = pd.read_csv("/p/projects/impactee/Josh/thesis_analysis/merged_data_outer.csv")

# take absolute value of coast variables (which are reported as negative numbers)
df['avg_coast_dist'] = df['avg_coast_dist'].abs()
df['std_coast_dist'] = df['std_coast_dist'].abs()
df['max_coast_dist'] = df['max_coast_dist'].abs()
df['min_coast_dist'] = df['min_coast_dist'].abs()

# rename Z2024 ntl variables
df.rename(columns={'NTL_sum': 'ntlZ_sum'}, inplace=True)
df.rename(columns={'lNTL_sum': 'log_ntlZ_sum'}, inplace=True)

df.rename(columns={'laglNTL_sum': 'log_lag_ntlZ_sum'}, inplace=True)

# add lag_ntlZ_sum column (undo log transformation)
df['lag_ntlZ_sum'] = np.exp(df['log_lag_ntlZ_sum'])

# --- Compute area-weighted mean of ntlZ_sum ---
import geopandas as gpd

# Shapefile (geometries)
regions = gpd.read_file("/p/projects/impactee/DOSE_creation/DoseV2Creation/shapefile/GADM shapefile level 1/gadm36_1.shp")

# Merge feature data into shapefile by region ID
gdf = regions.merge(df, on="GID_1", how="left")

# Reproject to equal-area CRS
if gdf.crs.is_geographic:
    gdf = gdf.to_crs("ESRI:54009")  # Mollweide projection (equal-area)

# Compute region area (in km²)
gdf["region_area_km2"] = gdf.geometry.area / 1e6  # convert from m² to km²

# Compute area-weighted mean
gdf["ntlZ_mean"] = gdf["ntlZ_sum"] / gdf["region_area_km2"]

# merge back to original dataframe
df = pd.merge(df, gdf[["GID_1", "year", "ntlZ_mean"]], on=["GID_1", "year"], how="left")

# create IPCC LC class share variables (lc3)
# IPCC ag = 10, 11, 12, 20, 30, 40
df['lccs_ag_ipcc_share'] = (
    df['lccs_ag_101112_share'] + df['lccs_ag_20_share'] +
    df['lccs_ag_30_share'] + df['lccs_ag_40_share']
)

# IPCC forest = 50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 160, 170
df['lccs_forest_ipcc_share'] = (
    df['lccs_forest_50_share'] + df['lccs_forest_606162_share'] +
    df['lccs_forest_707172_share'] + df['lccs_forest_808182_share'] +
    df['lccs_forest_90_share'] + df['lccs_forest_100_share'] +
    df['lccs_forest_160_share'] + df['lccs_forest_170_share']
)

# IPCC grass = 110, 130
df['lccs_grass_ipcc_share'] = (
    df['lccs_grass_110_share'] + df['lccs_grass_130_share']
)

# IPCC wetland = 180
df['lccs_wet_ipcc_share'] = df['lccs_wet_180_share']

# IPCC urban = 190
df['lccs_urban_ipcc_share'] = df['lccs_urban_190_share']

# IPCC shrub = 120, 121, 122
df['lccs_shrub_ipcc_share'] = (
    df['lccs_shrub_120121122_share']
)

# IPCC sparse = 140, 150, 151, 152, 153
df['lccs_sparse_ipcc_share'] = (
    df['lccs_sparse_140_share'] + df['lccs_sparse_150151152153_share']
)

# IPCC bare = 200, 201, 202
df['lccs_bare_ipcc_share'] = (
    df['lccs_bare_200201202_share']
)

# IPCC water = 210
df['lccs_water_ipcc_share'] = df['lccs_water_210_share']

#### take log of all relevant variables #################################
log_vars = [
    'avg_coast_dist', 'avg_tri', 'avg_lake_dist', 'major_river_dist_mean',
    'std_coast_dist', 'std_tri', 'std_lake_dist', 'major_river_dist_std',

    # lc1 variables
    'cropland_share', 'forest_share', 'urban_share',

    # lc2 variables
    'lccs_ag_101112_share', 'lccs_ag_20_share', 'lccs_ag_30_share', 'lccs_ag_40_share',
    'lccs_forest_50_share', 'lccs_forest_606162_share', 'lccs_forest_707172_share',
    'lccs_forest_808182_share', 'lccs_forest_90_share', 'lccs_forest_100_share',
    'lccs_forest_160_share', 'lccs_forest_170_share', 'lccs_grass_110_share',
    'lccs_grass_130_share', 'lccs_wet_180_share', 'lccs_urban_190_share',
    'lccs_shrub_120121122_share', 'lccs_sparse_140_share', 'lccs_sparse_150151152153_share',
    'lccs_bare_200201202_share', 'lccs_water_210_share', 'lccs_snow_220_share',

    # lc3 variables
    'lccs_ag_ipcc_share', 'lccs_forest_ipcc_share', 'lccs_grass_ipcc_share',
    'lccs_wet_ipcc_share', 'lccs_urban_ipcc_share', 'lccs_shrub_ipcc_share',
    'lccs_sparse_ipcc_share', 'lccs_bare_ipcc_share', 'lccs_water_ipcc_share',

    # ntlL variables
    'NTL_mean', 'NTL_std',

    # ntlZ variable
    'ntlZ_mean',
    
    'grp_pc_usd_2015', 'grp_pc_lcu2015_usd', 'grp_pc_lcu'
]

for var in log_vars:
    # Create new column 'log_{var}' with log-transformed values (log(var + 1) if var == 0)
    df[f'log_{var}'] = np.log(df[var].replace(0, 1))

# save the updated DataFrame with log variables
df.to_csv("/p/projects/impactee/Josh/thesis_analysis/merged_data_outer_updated.csv", index=False)

########################################################################################
##### Step 2: merge with WB GDP per capita USD and LCU data ############################
########################################################################################

wb = pd.read_csv('/p/projects/impactee/Josh/thesis_analysis/API_NY.GDP.PCAP.KD_DS2_en_csv_v2_80947.csv', skiprows=4)

# print(wb.columns)

# Select the year columns (column names that are four-digit numbers)
year_cols = [col for col in wb.columns if col.isdigit() and len(col) == 4]

# Melt the dataframe
gdp_df = wb.melt(
    id_vars='Country Code',         # column to keep
    value_vars=year_cols,           # year columns to melt
    var_name='year',                # name for the 'year' column
    value_name='gdp_pc_2015_usd'    # name for the GDP value column
)

# Rename 'Country Code' to 'GID_0'
gdp_df = gdp_df.rename(columns={'Country Code': 'GID_0'})

# Optional: convert 'year' column to int if needed
gdp_df['year'] = gdp_df['year'].astype(int)

# print(gdp_df.head(20))

# load the main dataset
main_df = pd.read_csv('/p/projects/impactee/Josh/thesis_analysis/merged_data_outer_updated.csv')

# print(main_df.columns)

# Merge the GDP data with the main dataset on 'GID_0' and 'year'
merged_df = pd.merge(
    main_df,
    gdp_df,
    on=['GID_0', 'year'],
    how='left'
)

# print(merged_df.columns)

# calculate the log of GDP per capita in 2015 USD
merged_df['log_gdp_pc_2015_usd'] = merged_df['gdp_pc_2015_usd'].apply(lambda x: np.log(x) if x > 0 else np.nan)

# calculate the residual of log GDP pc and log GRP pc
merged_df['log_pc_residual_2015_usd'] = merged_df['log_grp_pc_lcu2015_usd'] - merged_df['log_gdp_pc_2015_usd']

# Save the merged dataframe to a new CSV file
merged_df.to_csv('/p/projects/impactee/Josh/thesis_analysis/merged_data_with_wb.csv', index=False)

### LCU part

wb = pd.read_csv('/p/projects/impactee/Josh/thesis_analysis/API_NY.GDP.PCAP.CN_DS2_en_csv_v2_38384.csv', skiprows=4)

# print(wb.columns)

# Select the year columns (column names that are four-digit numbers)
year_cols = [col for col in wb.columns if col.isdigit() and len(col) == 4]

# Melt the dataframe
gdp_df = wb.melt(
    id_vars='Country Code',         # column to keep
    value_vars=year_cols,           # year columns to melt
    var_name='year',                # name for the 'year' column
    value_name='gdp_pc_lcu'    # name for the GDP value column
)

# Rename 'Country Code' to 'GID_0'
gdp_df = gdp_df.rename(columns={'Country Code': 'GID_0'})

# Optional: convert 'year' column to int if needed
gdp_df['year'] = gdp_df['year'].astype(int)

# print(gdp_df.head(20))

# load the main dataset
main_df = pd.read_csv('/p/projects/impactee/Josh/thesis_analysis/merged_data_with_wb.csv')

# print(main_df.columns)

# Merge the GDP data with the main dataset on 'GID_0' and 'year'
merged_df = pd.merge(
    main_df,
    gdp_df,
    on=['GID_0', 'year'],
    how='left'
)

wb_pop = pd.read_csv('/p/projects/impactee/Josh/thesis_analysis/WB_population_1960_2023.csv', skiprows=4)

# Select the year columns (column names that are four-digit numbers)
year_cols = [col for col in wb_pop.columns if col.isdigit() and len(col) == 4]

# Melt the dataframe
pop_df = wb_pop.melt(
    id_vars='Country Code',         # column to keep
    value_vars=year_cols,           # year columns to melt
    var_name='year',                # name for the 'year' column
    value_name='pop_wb'    # name for the GDP value column
)

# Rename 'Country Code' to 'GID_0'
pop_df = pop_df.rename(columns={'Country Code': 'GID_0'})

# Optional: convert 'year' column to int if needed
pop_df['year'] = pop_df['year'].astype(int)

# Merge the GDP data with the main dataset on 'GID_0' and 'year'
merged_df2 = pd.merge(
    merged_df,
    pop_df,
    on=['GID_0', 'year'],
    how='left'
)

# calculate the log of GDP per capita LCU
merged_df2['log_gdp_pc_lcu'] = merged_df2['gdp_pc_lcu'].apply(lambda x: np.log(x) if x > 0 else np.nan)

# calculate the diff of log GDP pc and log GRP pc
merged_df2['log_regional_pc_diff_lcu'] = (
    merged_df2['log_grp_pc_lcu'] - merged_df2['log_gdp_pc_lcu']
)

# Save the merged dataframe to a new CSV file
merged_df2.to_csv('/p/projects/impactee/Josh/thesis_analysis/merged_data_final.csv', index=False)