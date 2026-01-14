import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score, mean_squared_error
import geopandas as gpd
import json

# --------------------------------------------------------
# USER INPUTS
# --------------------------------------------------------
top_models = [
    # "ann_rmspe_ntlL_lc1_b",
    # "ann_rmspe_ntlL_lc3_e",
    # "ann_rmspe_ntlL_lc3_c",

    "lgbm_raw_gid0_agg_lc1_riv1_ntlZ2_a",
    "lgbm_raw_gid0_agg_lc1_riv1_ntlL_g",

    # "lgbm_rmspe_ntlL_lc1_b",
    # "lgbm_rmspe_ntlL_lc1_c",
    # "lgbm_rmspe_ntlL_lc1_g",

    "mlp_raw2_gid0_agg_lc1_riv1_ntlL_b",
    "mlp_raw2_gid0_agg_lc1_riv1_ntlL_f",
    "mlp_raw2_gid0_agg_lc3_riv1_ntlL_e"
]

name_mapping = {

    # "ann_rmspe_ntlL_lc1_b": "ann_rmspe_L1b",
    # "ann_rmspe_ntlL_lc3_e": "ann_rmspe_L3e",
    # "ann_rmspe_ntlL_lc3_c": "ann_rmspe_L3c",

    "lgbm_raw_gid0_agg_lc1_riv1_ntlZ2_a": "gbdt_4",
    "lgbm_raw_gid0_agg_lc1_riv1_ntlL_g": "gbdt_5",

    # "lgbm_rmspe_ntlL_lc1_b": "gbdt_rmspe_L1b",
    # "lgbm_rmspe_ntlL_lc1_c": "gbdt_rmspe_L1c",
    # "lgbm_rmspe_ntlL_lc1_g": "gbdt_rmspe_L1g",

    "mlp_raw2_gid0_agg_lc1_riv1_ntlL_b": "ann_1",
    "mlp_raw2_gid0_agg_lc1_riv1_ntlL_f": "ann_2",
    "mlp_raw2_gid0_agg_lc3_riv1_ntlL_e": "ann_3"
}

model_summaries = {}

# GADM GID_1 shapefile
shp_path = "/p/projects/impactee/Josh/LandScan/gadm_custom_merged.shp"

# Load shapefile once
gdf = gpd.read_file(shp_path)

# --------------------------------------------------------
# CUSTOM COLORMAPS
# --------------------------------------------------------
def make_r2_cmap(gdf):
    # Extract R² range
    min_r2 = gdf["r2_median"].min()
    max_r2 = gdf["r2_median"].max()

    # Ensure 0 is included
    max_pos = max(max_r2, 0)

    # ---- POSITIVE COLORMAP (GREENS) ----
    greens = LinearSegmentedColormap.from_list(
        "greens_segment",
        ["#e5f5e0", "#006d2c"]
    )

    n_pos = 200
    pos_colors = greens(np.linspace(0, 1, n_pos))

    # ---- NEGATIVE COLOR (SINGLE ORANGE) ----
    neg_color = np.array([[0.80, 0.30, 0.10, 1.0]])  # dark red-orange

    # Combine colors
    all_colors = np.vstack([neg_color, pos_colors])
    cmap = ListedColormap(all_colors)

    # ---- BOUNDARIES ----
    # One bin for all negatives, then linear bins from 0 → max_pos
    neg_bounds = [-1e6, 0.0]
    pos_bounds = np.linspace(0.0, max_pos, n_pos + 1)

    bounds = np.concatenate([neg_bounds, pos_bounds[1:]])
    norm = BoundaryNorm(bounds, ncolors=cmap.N)

    return cmap, norm, max_pos

def make_r2_cmap_with_mask(gdf, mask_color="#ffff99"):
    """
    Same as make_r2_cmap but adds a mask color (yellow) as the first bin.
    """
    # Base R² range
    min_r2 = gdf["r2_median"].min()
    max_r2 = gdf["r2_median"].max()

    # Blue gradient for positive R²
    blues = LinearSegmentedColormap.from_list(
        "blues_segment", ["#cce0ff", "#0000cc"]
    )

    # --- COLORS ---
    # 1. Mask bin = yellow
    mask_color_rgba = np.array(
        list(plt.cm.colors.to_rgba(mask_color))
    ).reshape(1, 4)

    # 2. Negative R² = single red
    neg_color = np.array([[1.0, 0.2, 0.2, 1]])

    # 3. Positive R² gradient
    n_pos = 200
    pos_colors = blues(np.linspace(0, 1, n_pos))

    # Combine
    all_colors = np.vstack([mask_color_rgba, neg_color, pos_colors])
    cmap = ListedColormap(all_colors)

    # --- BOUNDS ---
    # mask bin: (-2 → -1)
    # red bin: (-1 → 0)
    # blue bins: 0 → max_r2

    neg_bounds = [-2, -1, 0]  # mask range, negative range
    pos_bounds = np.linspace(0, max_r2, n_pos + 1)

    bounds = np.concatenate([neg_bounds, pos_bounds[1:]])
    norm = BoundaryNorm(bounds, ncolors=cmap.N)

    return cmap, norm

for model in top_models:
    region_metrics_csv = f"/p/projects/impactee/Josh/thesis_analysis/model_metrics/{model}_region_metrics.csv"
    map_out_dir = f"/p/projects/impactee/Josh/thesis_analysis/plots/model_output/maps/{name_mapping[model]}"
    os.makedirs(map_out_dir, exist_ok=True)

    # --------------------------------------------------------
    # LOAD REGION METRICS
    # --------------------------------------------------------
    df = pd.read_csv(region_metrics_csv)

    print(f"Loaded {len(df):,} rows from region-metrics CSV")

    # Should contain: GID_1, repeat, fold, n_obs, r2, rmse

    # --------------------------------------------------------
    # COMPUTE SUMMARY STATISTICS PER GID_1
    # --------------------------------------------------------
    def rmspe(y_true, y_pred, eps=1e-6):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sqrt(np.mean(((y_true - y_pred) / (y_true + eps))**2))

    # Compute RMSPE per row (if absent)
    if 'rmspe' not in df.columns:
        df['rmspe'] = df.apply(
            lambda row: rmspe(
                json.loads(row['y_true_values']),
                json.loads(row['y_pred_values'])
            ),
        axis=1
    )

    def compute_rmse(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sqrt(mean_squared_error(y_true, y_pred))

    # compute RMSE per row (if absent)
    if 'rmse' not in df.columns:
        df['rmse'] = df.apply(
            lambda row: compute_rmse(
                json.loads(row['y_true_values']),
                json.loads(row['y_pred_values']),
            ),
        axis=1
    )

    def mape(y_true, y_pred, eps=1e-6):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, eps)))
    
    # Compute MAPE per row (if absent)
    if 'mape' not in df.columns:
        df['mape'] = df.apply(
            lambda row: mape(
                json.loads(row['y_true_values']),
                json.loads(row['y_pred_values'])
            ),
            axis=1
        )

    summary = (
        df.groupby("GID_1")
        .agg(
            n_obs=("n_obs", "mean"),
            r2_median=("r2", "median"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            r2_q25=("r2", lambda x: np.nanpercentile(x, 25)),
            r2_q75=("r2", lambda x: np.nanpercentile(x, 75)),

            rmse_median=("rmse", "median"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            rmse_q25=("rmse", lambda x: np.nanpercentile(x, 25)),
            rmse_q75=("rmse", lambda x: np.nanpercentile(x, 75)),

            # RMSPE stats
            rmspe_median=("rmspe", "median"),
            rmspe_mean=("rmspe", "mean"),
            rmspe_std=("rmspe", "std"),
            rmspe_q25=("rmspe", lambda x: np.nanpercentile(x, 25)),
            rmspe_q75=("rmspe", lambda x: np.nanpercentile(x, 75)),

            # MAPE stats
            mape_median=("mape", "median"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            mape_q25=("mape", lambda x: np.nanpercentile(x, 25)),
            mape_q75=("mape", lambda x: np.nanpercentile(x, 75))
        )
        .reset_index()
    )

    summary["r2_median_masked"] = summary.apply(
        lambda row: -1.5 if row["n_obs"] < 25 else row["r2_median"],
        axis=1
    )

    summary["rmse_median_clipped"] = summary["rmse_median"].clip(upper=20000)

    model_summaries[model] = summary

    cmap_rmse = plt.cm.magma
    norm_rmse = Normalize(vmin=0, vmax=20000)

    # cmap_qcd = plt.cm.viridis
    # norm_qcd = Normalize(vmin=0, vmax=1)

    # cmap_cv = plt.cm.viridis
    # norm_cv = Normalize(vmin=0, vmax=2.0)

    norm_rmspe = Normalize(vmin=0.0, vmax=1.0)

    # Clip MAPE for visualization
    MAPE_MAX = 1.0  # 100% average absolute percentage error

    summary["mape_median_clipped"] = summary["mape_median"].clip(upper=MAPE_MAX)
    norm_mape = Normalize(vmin=0.0, vmax=MAPE_MAX)

    gdf_merged = gdf.merge(summary, on="GID_1", how="left")
    # summary.to_csv(summary_out_csv, index=False)
    # print(f"Saved summary CSV to: {summary_out_csv}")

    cmap, norm, max_pos = make_r2_cmap(gdf_merged)

    cmap_r2_masked, norm_r2_masked = make_r2_cmap_with_mask(gdf_merged)

    # --------------------------------------------------------
    # PLOT WORLD MAP — MEDIAN R²
    # --------------------------------------------------------
    # fig, ax = plt.subplots(figsize=(14, 8))

    # plot = gdf_merged.plot(
    #     column="r2_median",
    #     cmap=cmap,
    #     norm=norm,
    #     linewidth=0.2,
    #     edgecolor="black",
    #     missing_kwds={"color": "lightgray", "label": "No Data"},
    #     ax=ax
    # )

    # ax.axis("off")

    # from matplotlib.cm import ScalarMappable
    # sm = ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])

    # cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    # cbar.set_label("Median R²", fontsize=16)

    # # ---- COLORBAR TICKS ----
    # # Major ticks at 0.0, 0.1, 0.2, ...
    # tick_step = 0.1
    # # Positive ticks start just above zero to avoid overlap
    # pos_ticks = np.arange(tick_step, np.ceil(max_pos / tick_step) * tick_step + 1e-6, tick_step)

    # # Center of the negative bin (-1e6, 0)
    # neg_tick = -5e5

    # ticks = np.concatenate([[neg_tick], pos_ticks])
    # labels = ["< 0.0"] + [f"{t:.1f}" for t in pos_ticks]

    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(labels)
    # cbar.ax.tick_params(labelsize=16)

    # plt.tight_layout()
    # r2_map_path = os.path.join(
    #     map_out_dir,
    #     f"{name_mapping[model]}_median_r2_world_map_GID1.png"
    # )
    # plt.savefig(r2_map_path, dpi=300)
    # plt.close()
    # print(f"Saved R² map: {r2_map_path}")


    # # --------------------------------------------------------
    # # PLOT WORLD MAP — MEDIAN R² MASKED
    # # --------------------------------------------------------
    # fig, ax = plt.subplots(figsize=(14, 8))
    # plot = gdf_merged.plot(
    #     column="r2_median_masked",
    #     cmap=cmap_r2_masked,
    #     norm=norm_r2_masked,
    #     linewidth=0.2,
    #     edgecolor="black",
    #     missing_kwds={"color": "lightgray", "label": "No Data"},
    #     ax=ax
    # )

    # # ax.set_title("Median R² (GID_1)", fontsize=16)
    # ax.axis("off")

    # from matplotlib.cm import ScalarMappable
    # sm = ScalarMappable(cmap=cmap_r2_masked, norm=norm_r2_masked)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    # cbar.set_label("Median R²", fontsize=16)

    # plt.tight_layout()
    # r2_map_path = os.path.join(map_out_dir, f"{name_mapping[model]}_median_r2_world_map_GID1_masked.png")
    # plt.savefig(r2_map_path, dpi=300)
    # plt.close()
    # print(f"Saved R² map: {r2_map_path}")


    # --------------------------------------------------------
    # PLOT WORLD MAP — MEDIAN RMSE
    # --------------------------------------------------------
    # fig, ax = plt.subplots(figsize=(14, 8))
    # gdf_merged.plot(
    #     column="rmse_median_clipped",
    #     cmap=cmap_rmse,
    #     norm=norm_rmse,
    #     linewidth=0.2,
    #     edgecolor="black",
    #     missing_kwds={"color": "lightgray", "label": "No Data"},
    #     ax=ax
    # )

    # # ax.set_title("Median RMSE (GID_1)", fontsize=16)
    # ax.axis("off")

    # sm = ScalarMappable(cmap=cmap_rmse, norm=norm_rmse)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    # cbar.set_label("Median RMSE (2015 USD)", fontsize=16)
    # cbar.ax.tick_params(labelsize=16)

    # # ---- UPDATE LAST TICK LABEL ----
    # ticks = cbar.get_ticks()
    # labels = [f"{int(t):d}" if float(t).is_integer() else f"{t:g}" for t in ticks]
    # labels[-1] = ">20000"

    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(labels)

    # plt.tight_layout()
    # rmse_map_path = os.path.join(map_out_dir, f"{name_mapping[model]}_median_rmse_world_map_GID1.png")
    # plt.savefig(rmse_map_path, dpi=300)
    # plt.close()
    # print(f"Saved RMSE map: {rmse_map_path}")

    # --------------------------------------------------------
    # PLOT WORLD MAP — RMSPE
    # --------------------------------------------------------
    # fig, ax = plt.subplots(figsize=(14, 8))

    # gdf_merged.plot(
    #     column="rmspe_median",
    #     cmap="viridis",
    #     norm=norm_rmspe,
    #     linewidth=0.2,
    #     edgecolor="black",
    #     missing_kwds={"color": "lightgray", "label": "No Data"},
    #     ax=ax
    # )

    # ax.axis("off")

    # sm = ScalarMappable(cmap="viridis", norm=norm_rmspe)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    # cbar.set_label("Median RMSPE", fontsize=16)
    # cbar.ax.tick_params(labelsize=16)

    # # ---- UPDATE LAST TICK LABEL ----
    # ticks = cbar.get_ticks()
    # labels = [f"{int(t):d}" if float(t).is_integer() else f"{t:g}" for t in ticks]
    # labels[-1] = ">1.0"

    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(labels)

    # rmspe_path = os.path.join(map_out_dir, f"{name_mapping[model]}_rmspe_world_map_GID1.png")
    # plt.tight_layout()
    # plt.savefig(rmspe_path, dpi=300)
    # plt.close()
    # print("Saved RMSPE map:", rmspe_path)

    # --------------------------------------------------------
    # PLOT WORLD MAP — MEDIAN MAPE
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))

    gdf_merged.plot(
        column="mape_median_clipped",
        cmap="viridis",
        norm=norm_mape,
        linewidth=0.2,
        edgecolor="black",
        missing_kwds={"color": "lightgray", "label": "No Data"},
        ax=ax
    )

    ax.axis("off")

    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap="viridis", norm=norm_mape)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Median MAPE", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    # Update last tick label to indicate clipping
    ticks = cbar.get_ticks()
    labels = [f"{t:.2f}" for t in ticks]
    labels[-1] = ">1.00"
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)

    plt.tight_layout()

    mape_map_path = os.path.join(
        map_out_dir,
        f"{name_mapping[model]}_median_mape_world_map_GID1.png"
    )

    plt.savefig(mape_map_path, dpi=300)
    plt.close()

    print(f"Saved MAPE map: {mape_map_path}")


# # --------------------------------------------------------
# # PLOT WORLD MAP — RMSPE (TWO MODELS, ONE GDF)
# # --------------------------------------------------------

# gdf_a = gdf.merge(model_summaries["mlp_raw2_gid0_agg_lc1_riv1_ntlL_b"], on="GID_1", how="left")
# gdf_b = gdf.merge(model_summaries["lgbm_rmspe_ntlL_lc1_g"], on="GID_1", how="left")

# # -----------------------------
# # Determine map aspect ratio
# # -----------------------------
# xmin, ymin, xmax, ymax = gdf.total_bounds
# map_aspect = (ymax - ymin) / (xmax - xmin)  # height / width

# n_panels = 2
# fig_width = 14
# fig_height = fig_width * map_aspect * n_panels  # scale figure height to fit 2 maps

# fig = plt.figure(figsize=(fig_width, fig_height))

# # -----------------------------
# # GridSpec: maps + colorbar
# # -----------------------------
# gs = GridSpec(
#     nrows=n_panels,
#     ncols=2,
#     width_ratios=[1, 0.04],   # right column = colorbar
#     height_ratios=[1]*n_panels,
#     wspace=0.02,
#     hspace=0.02
# )

# ax_top = fig.add_subplot(gs[0, 0])
# ax_bot = fig.add_subplot(gs[1, 0])
# cax    = fig.add_subplot(gs[:, 1])  # colorbar spans both rows

# # -----------------------------
# # Panel (a)
# # -----------------------------
# gdf_a.plot(
#     column="rmspe_median",
#     cmap="viridis",
#     norm=norm_rmspe,
#     linewidth=0.2,
#     edgecolor="black",
#     missing_kwds={"color": "lightgray"},
#     ax=ax_top
# )

# ax_top.axis("off")
# ax_top.set_aspect("equal")
# ax_top.text(
#     0.01, 0.99, "a)",
#     transform=ax_top.transAxes,
#     fontsize=16,
#     fontweight="bold",
#     va="top",
#     ha="left"
# )

# # -----------------------------
# # Panel (b)
# # -----------------------------
# gdf_b.plot(
#     column="rmspe_median",
#     cmap="viridis",
#     norm=norm_rmspe,
#     linewidth=0.2,
#     edgecolor="black",
#     missing_kwds={"color": "lightgray"},
#     ax=ax_bot
# )

# ax_bot.axis("off")
# ax_bot.set_aspect("equal")
# ax_bot.text(
#     0.01, 0.99, "b)",
#     transform=ax_bot.transAxes,
#     fontsize=16,
#     fontweight="bold",
#     va="top",
#     ha="left"
# )

# # -----------------------------
# # Shared colorbar
# # -----------------------------
# sm = ScalarMappable(cmap="viridis", norm=norm_rmspe)
# sm.set_array([])

# cbar = fig.colorbar(sm, cax=cax)
# cbar.set_label("Median RMSPE", fontsize=16)
# cbar.ax.tick_params(labelsize=16)

# ticks = cbar.get_ticks()
# labels = [f"{t:g}" for t in ticks]
# labels[-1] = ">1.0"
# cbar.set_ticks(ticks)
# cbar.set_ticklabels(labels)

# # -----------------------------
# # Save figure
# # -----------------------------
# plt.savefig(
#     "/p/projects/impactee/Josh/thesis_analysis/plots/model_output/maps/"
#     "ann_1_gbdt_5b_compare_rmspe_world_map.png",
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.close()

# # --------------------------------------------------------
# # PLOT BOXPLOT — RMSPE DIFFERENCE (MODEL A − MODEL B)
# # --------------------------------------------------------

# gdf_diff = gdf_a.copy()
# gdf_diff["rmspe_diff"] = (
#     gdf_a["rmspe_median"] - gdf_b["rmspe_median"]
# )

# diff_vals = gdf_diff["rmspe_diff"].dropna().values

# diff_col = 'rmspe_diff'

# # Compute the 99th percentile
# p99 = np.nanpercentile(gdf_diff[diff_col], 99)
# max_val = np.nanmax(gdf_diff[diff_col])

# # compute the 1st percentile
# p1 = np.nanpercentile(gdf_diff[diff_col], 1)
# min_val = np.nanmin(gdf_diff[diff_col])

# # Select rows between 99th percentile and maximum
# extreme_pos_regions = gdf_diff[(gdf_diff[diff_col] >= p99) & (gdf_diff[diff_col] <= max_val)]

# # select rows between 1st percentile and minimum
# extreme_neg_regions = gdf_diff[(gdf_diff[diff_col] <= p1) & (gdf_diff[diff_col] >= min_val)]

# print(extreme_pos_regions[['GID_1', diff_col]])
# print(extreme_neg_regions[['GID_1', diff_col]])

# # Compute 99th percentile for clipping
# upper_clip = np.percentile(diff_vals, 99)
# lower_clip = np.percentile(diff_vals, 1)
# clipped_diff_vals = np.clip(diff_vals, lower_clip, upper_clip)

# fig, ax = plt.subplots(figsize=(10, 2.5))

# ax.boxplot(
#     clipped_diff_vals,
#     vert=False,
#     widths=0.6,
#     patch_artist=True,
#     boxprops=dict(facecolor="lightgray", edgecolor="black", linewidth=1.2),
#     medianprops=dict(color="black", linewidth=2),
#     whiskerprops=dict(color="black", linewidth=1.2),
#     capprops=dict(color="black", linewidth=1.2),
#     flierprops=dict(
#         marker="o",
#         markerfacecolor="gray",
#         markeredgecolor="none",
#         alpha=0.4,
#         markersize=4
#     ),
#     whis=1.5
# )

# # Reference line at zero
# ax.axvline(0, color="black", linestyle="--", linewidth=1)

# ax.set_xlabel("Change in per-region RMSPE (ann_1 − gbdt_5b)", fontsize=14)
# ax.set_yticks([])  # no categorical y-axis
# ax.tick_params(axis="x", labelsize=12)

# plt.tight_layout()
# plt.savefig(
#     "/p/projects/impactee/Josh/thesis_analysis/plots/model_output/maps/"
#     "ann_1_gbdt_5b_compare_rmspe_difference_boxplot.png",
#     dpi=300
# )
# plt.close()

# # --------------------------------------------------------
# # PLOT WORLD MAP — RMSPE DIFFERENCE (MODEL A − MODEL B)
# # --------------------------------------------------------

# # print percentiles of difference
# print("RMSPE Difference Percentiles (Model A - Model B):")
# for p in [0, 5, 25, 50, 75, 95, 96, 97, 98, 99, 100]:
#     val = np.nanpercentile(gdf_diff["rmspe_diff"], p)
#     print(f"  {p}th percentile: {val:.4f}")

# p = 95  # or 95 if you want more contrast
# max_abs_diff = np.nanpercentile(
#     np.abs(gdf_diff["rmspe_diff"]),
#     p
# )

# cap = 1.86
# norm_diff = TwoSlopeNorm(vmin=-cap, vcenter=0.0, vmax=cap)

# cmap_diff = plt.cm.PiYG

# actual_min = np.nanmin(gdf_diff["rmspe_diff"])
# actual_max = np.nanmax(gdf_diff["rmspe_diff"])

# xmin, ymin, xmax, ymax = gdf.total_bounds
# map_aspect = (ymax - ymin) / (xmax - xmin)

# fig_width = 14
# fig_height = fig_width * map_aspect

# fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# gdf_diff.plot(
#     column="rmspe_diff",
#     cmap=cmap_diff,
#     norm=norm_diff,
#     linewidth=0.2,
#     edgecolor="black",
#     ax=ax,
#     missing_kwds={
#         "color": "lightgray",
#         "label": "No Data"
#     }
# )


# ax.set_aspect("equal")
# ax.axis("off")

# sm = ScalarMappable(cmap=cmap_diff, norm=norm_diff)
# sm.set_array([])

# cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)

# # Restrict visible range of the colorbar
# cbar.ax.set_ylim(actual_min, cap)

# cbar.set_label(
#     f"Δ Median RMSPE (ann_1 − gbdt_5b)",
#     fontsize=16
# )

# # ---- FIVE TICKS TOTAL ----
# # 1 negative, 0, and 3 positive (including the cap)
# pos_ticks = np.linspace(0, cap, 4)   # [0, ~0.62, ~1.24, 1.86]
# ticks = np.concatenate([[actual_min], pos_ticks])

# cbar.set_ticks(ticks)

# labels = [f"{actual_min:.2f}", "0"]
# labels += [f"{t:.2f}" for t in pos_ticks[1:-1]]
# labels += [f">{cap:.2f}"]

# cbar.set_ticklabels(labels)
# cbar.ax.tick_params(labelsize=16)

# plt.savefig(
#     "/p/projects/impactee/Josh/thesis_analysis/plots/model_output/maps/"
#     "ann_1_gbdt_5b_compare_rmspe_difference_world_map.png",
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.close()
