import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("/p/projects/impactee/Josh/thesis_analysis/merged_data_final.csv")

# -----------------------------
# Feature groups
# -----------------------------
ntl_features = [
    "NTL_mean", "NTL_std"
]

lc_features = [
    "cropland_share", "forest_share", "urban_share"
]

geo_features = [
    "avg_coast_dist", "avg_tri", "avg_lake_dist", "major_river_dist_mean",
    "std_coast_dist", "std_tri", "std_lake_dist", "major_river_dist_std"
]

all_features = ntl_features + lc_features + geo_features + ["gdp_pc_2015_usd"]

target = "grp_pc_lcu2015_usd"

predictor_sets = {
    "all": all_features,
    "geo": geo_features,
    "lc": lc_features,
    "ntl": ntl_features
}

output_dir = "/p/projects/impactee/Josh/thesis_analysis/plots"

# -----------------------------
# Plotting function
# -----------------------------
def plot_obs_per_region(df, predictors, suffix):
    df_model = df.dropna(subset=[target] + predictors)
    df_model = df_model[df_model[target] != 0]

    obs_per_region = (
        df_model
        .groupby("GID_1")
        .size()
        .rename("n_obs")
        .reset_index()
    )

    print(f"\n[{suffix.upper()}]")
    print(f"Number of regions: {len(obs_per_region):,}")
    print(obs_per_region["n_obs"].describe())

    bins = np.arange(0, obs_per_region["n_obs"].max() + 2, 2)
    bin_counts, _ = np.histogram(obs_per_region["n_obs"], bins=bins)

    print(f"Max count per bin: {bin_counts.max()}")
    print(f"Min count per bin: {bin_counts.min()}")

    median_obs = obs_per_region["n_obs"].median()
    print(f"Median number of observations per region: {median_obs}")

    # -------------------------
    # Histogram
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        obs_per_region["n_obs"],
        bins=bins,
        color="green",
        edgecolor="black",
        alpha=0.8
    )

    ax.axvline(
        median_obs,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {median_obs}"
    )

    ax.set_xlabel("Observations per sub-national region", fontsize=12)
    ax.set_ylabel("Number of sub-national regions", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/obs_per_region_histogram_{suffix}.png",
        dpi=300
    )
    plt.close()

    # -------------------------
    # CDF
    # -------------------------
    obs_sorted = np.sort(obs_per_region["n_obs"])
    cdf = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(obs_sorted, cdf)

    ax.set_xlabel("Number of observations per region")
    ax.set_ylabel("Cumulative share of regions")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/obs_per_region_cdf_{suffix}.png",
        dpi=300
    )
    plt.close()


# -----------------------------
# Run for all predictor sets
# -----------------------------
for suffix, predictors in predictor_sets.items():
    plot_obs_per_region(df, predictors, suffix)
