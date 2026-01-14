import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
# MODEL_GROUPS = {
#     "lgbm_rmspe": [
#         "lgbm_rmspe_ntlL_lc1",
#         "lgbm_rmspe_ntlZ_lc1",
#         "lgbm_rmspe_ntlL_lc2",
#         "lgbm_rmspe_ntlL_lc3",
#     ],
#     "lgbm_raw": [
#         "lgbm_raw_gid0_agg_lc1_riv1_ntlL",
#         "lgbm_raw_gid0_agg_lc1_riv1_ntlZ2",
#         "lgbm_raw_gid0_agg_lc2_riv1_ntlL",
#         "lgbm_raw_gid0_agg_lc3_riv1_ntlL",
#     ],
#     "mlp_raw2": [
#         "mlp_raw2_gid0_agg_lc1_riv1_ntlL",
#         "mlp_raw2_gid0_agg_lc1_riv1_ntlZ2",
#         "mlp_raw2_gid0_agg_lc2_riv1_ntlL",
#         "mlp_raw2_gid0_agg_lc3_riv1_ntlL",
#     ],
# }

# keys = ["a", "b", "c", "d", "e", "f", "g"]

# Explicit (model_id, key) selection — OPTION A
selected_models = [
    ("lgbm_rmspe_ntlL_lc1", "b"),
    ("lgbm_rmspe_ntlL_lc1", "c"),
    ("lgbm_rmspe_ntlL_lc1", "g"),

    ("ann_rmspe_ntlL_lc1", "b"),
    ("ann_rmspe_ntlL_lc3", "e"),
    ("ann_rmspe_ntlL_lc3", "c"),

    ("lgbm_raw_gid0_agg_lc1_riv1_ntlZ2", "a"),
    ("lgbm_raw_gid0_agg_lc1_riv1_ntlL", "g"),

    ("mlp_raw2_gid0_agg_lc1_riv1_ntlL", "b"),
    ("mlp_raw2_gid0_agg_lc1_riv1_ntlL", "f"),
    ("mlp_raw2_gid0_agg_lc3_riv1_ntlL", "e")
]

metrics_dir = "/p/projects/impactee/Josh/thesis_analysis/model_metrics"
out_dir = "/p/projects/impactee/Josh/thesis_analysis/plots/model_output/scatterplots"
os.makedirs(out_dir, exist_ok=True)

# Path to dataframe containing grp_pc_lcu2015_usd
df_income = pd.read_csv("/p/projects/impactee/Josh/thesis_analysis/merged_data_final.csv")

income_gid = (
    df_income
    .groupby("GID_1", as_index=False)
    .agg(grp_pc_lcu2015_usd=("grp_pc_lcu2015_usd", "mean"))
)

# --------------------------------------------------------
# PLOT NTL VARIABILITY VS GRP pc LCU 2015 USD
# --------------------------------------------------------

plot_df = df_income[["grp_pc_lcu2015_usd", "NTL_std", "log_NTL_mean", "log_avg_tri", "log_avg_coast_dist"]].copy()

# Drop NaNs
plot_df = plot_df.dropna()

# Remove non-finite values
plot_df = plot_df[
    np.isfinite(plot_df["grp_pc_lcu2015_usd"]) &
    np.isfinite(plot_df["NTL_std"]) &
    np.isfinite(plot_df["log_NTL_mean"])
]

# Remove non-positive income (quadratic + scale issues)
plot_df = plot_df[plot_df["grp_pc_lcu2015_usd"] > 0]

print(f"Observations used: {len(plot_df):,}")
print(plot_df.describe())

x = np.log10(plot_df["grp_pc_lcu2015_usd"].values)
y = plot_df["NTL_std"].values

coeffs = np.polyfit(x, y, deg=2)
poly_eq = np.poly1d(coeffs)

print("Quadratic fit (in log-income space):", coeffs)

# plot quadratic fit line
fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(
    plot_df["grp_pc_lcu2015_usd"],
    plot_df["NTL_std"],
    alpha=0.4,
    s=20
)

# Generate fit line
x_vals = np.logspace(
    np.log10(plot_df["grp_pc_lcu2015_usd"].min()),
    np.log10(plot_df["grp_pc_lcu2015_usd"].max()),
    200
)

y_vals = poly_eq(np.log10(x_vals))
ax.plot(x_vals, y_vals, color="red", linewidth=2, label="Quadratic fit (log income)")

ax.set_xscale("log")
ax.set_xlabel("GRP per capita (LCU, log scale)", fontsize=12)
ax.set_ylabel("NTL standard deviation", fontsize=12)
ax.set_title("NTL variability vs income", fontsize=14)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ntl_variability_vs_income.png"), dpi=300)
plt.close()

corr = np.corrcoef(
    np.log10(plot_df["grp_pc_lcu2015_usd"]),
    plot_df["NTL_std"]
)[0, 1]

print(f"Correlation (log income vs NTL_std): {corr:.2f}")

corr_mean = np.corrcoef(
    np.log10(plot_df["grp_pc_lcu2015_usd"]),
    plot_df["log_NTL_mean"]
)[0, 1]

print(f"Correlation (log income vs log_NTL_mean): {corr_mean:.2f}")

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(
    plot_df["grp_pc_lcu2015_usd"],
    plot_df["log_NTL_mean"],
    alpha=0.4,
    s=20
)

ax.set_xscale("log")
ax.set_xlabel("GRP per capita (LCU, log scale)", fontsize=12)
ax.set_ylabel("Log NTL mean", fontsize=12)
ax.set_title("Log NTL mean vs income", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "log_ntl_mean_vs_income.png"), dpi=300)
plt.close()

# plot scatterplot of log_avg_tri and log_grp_pc_lcu2015_usd; compute correlation
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(
    plot_df["grp_pc_lcu2015_usd"],
    plot_df["log_avg_tri"],
    alpha=0.4,
    s=20
)
ax.set_xscale("log")
ax.set_xlabel("GRP per capita (LCU, log scale)", fontsize=12)
ax.set_ylabel("Log average travel time to nearest city", fontsize=12)
ax.set_title("Log average travel time vs income", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "log_avg_tri_vs_income.png"), dpi=300
)
plt.close()

corr_tri = np.corrcoef(
    np.log10(plot_df["grp_pc_lcu2015_usd"]),
    plot_df["log_avg_tri"]
)[0, 1]
print(f"Correlation (log income vs log_avg_tri): {corr_tri:.2f}")

# plot correlation between log_avg_tri and log_avg_coast_dist
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(
    plot_df["log_avg_coast_dist"],
    plot_df["log_avg_tri"],
    alpha=0.4,
    s=20
)
ax.set_xscale("log")
ax.set_xlabel("Log average coastal distance", fontsize=12)
ax.set_ylabel("Log average travel time to nearest city", fontsize=12)
ax.set_title("Log average travel time vs coastal distance", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "log_avg_tri_vs_coast_dist.png"), dpi=300
)
plt.close()

corr_tri = np.corrcoef(
    plot_df["log_avg_coast_dist"],
    plot_df["log_avg_tri"]
)[0, 1]
print(f"Correlation (log coastal distance vs log_avg_tri): {corr_tri:.2f}")


# # --------------------------------------------------------
# # HELPERS
# # --------------------------------------------------------
# def rmspe(y_true, y_pred, eps=1e-6):
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     return np.sqrt(np.mean(((y_true - y_pred) / (y_true + eps)) ** 2))

# def model_group(model_id):
#     if model_id.startswith("lgbm_rmspe"):
#         return "lgbm_rmspe"
#     elif model_id.startswith("ann_rmspe"):
#         return "ann_rmspe"
#     elif model_id.startswith("lgbm_raw"):
#         return "lgbm_raw"
#     elif model_id.startswith("mlp_raw2"):
#         return "mlp_raw2"
#     else:
#         return None

# # --------------------------------------------------------
# # LOAD + PROCESS REGION METRICS
# # --------------------------------------------------------
# all_points = []

# for model_id, key in selected_models:
#     path = f"/p/projects/impactee/Josh/thesis_analysis/model_metrics/{model_id}_{key}_region_metrics.csv"

#     if not os.path.exists(path):
#         print(f"Skipping missing: {model_id}, {key}")
#         continue

#     df = pd.read_csv(path)

#     print(
#         f"[LOAD] {model_id}, key={key}: "
#         f"{len(df):,} rows "
#         f"({df['GID_1'].nunique()} unique GID_1)"
#     )

#     # Compute RMSPE if needed
#     if "rmspe" not in df.columns:
#         df["rmspe"] = df.apply(
#             lambda row: rmspe(
#                 json.loads(row["y_true_values"]),
#                 json.loads(row["y_pred_values"])
#             ),
#             axis=1
#         )

#     # ---- KEY STEP ----
#     # Median RMSPE per (model, GID_1)
#     per_model_region = (
#         df.groupby("GID_1", as_index=False)
#           .agg(median_rmspe=("rmspe", "median"))
#     )

#     print(
#         f"[AGG PER MODEL] {model_id}, key={key}: "
#         f"{len(per_model_region):,} rows "
#         f"(should equal unique GID_1 above)"
#     )

#     per_model_region["model_id"] = model_id
#     per_model_region["key"] = key

#     all_points.append(per_model_region)

# plot_df = pd.concat(all_points, ignore_index=True)

# income_df = (
#     df_income
#     .groupby("GID_1", as_index=False)
#     .agg(grp_pc_lcu2015_usd_mean=("grp_pc_lcu2015_usd", "mean"))
# )

# plot_df = plot_df.merge(income_df, on="GID_1", how="left")
# plot_df["model_group"] = plot_df["model_id"].apply(model_group)
# assert plot_df["model_group"].notna().all(), "Unclassified model_id detected"

# # --------------------------------------------------------
# # AGGREGATE ACROSS MODELS WITHIN GROUP
# # --------------------------------------------------------
# group_summary = (
#     plot_df
#     .groupby(["model_group", "GID_1"], as_index=False)
#     .agg(median_rmspe=("median_rmspe", "median"))
# )

# # Merge income
# group_summary = group_summary.merge(
#     income_gid,
#     on="GID_1",
#     how="inner"
# )

# print("\n[MODEL GROUP SUMMARY]")
# print(
#     group_summary
#     .groupby("model_group")
#     .agg(
#         n_rows=("GID_1", "size"),
#         n_regions=("GID_1", "nunique")
#     )
# )

# summary_table = (
#         plot_df
#         .groupby("model_group")
#         .agg(
#             # n_models=("model_id", "nunique"),
#             n_rows=("GID_1", "size"),
#             n_regions=("GID_1", "nunique")
#         )
#     )

# # --------------------------------------------------------
# # SCATTERPLOTS (ONE PER GROUP)
# # --------------------------------------------------------
# for grp, gdf in group_summary.groupby("model_group"):

#     print(
#         f"[PLOT] {grp}: "
#         f"{len(gdf):,} points "
#         f"({gdf['GID_1'].nunique()} unique GID_1)"
#     )

#     fig, ax = plt.subplots(figsize=(8, 6))

#     ax.scatter(
#         gdf["grp_pc_lcu2015_usd"],
#         gdf["median_rmspe"],
#         alpha=0.35,
#         s=18
#     )

#     ax.set_xscale("log")
#     ax.set_xlabel("GRP per capita (2015 USD)", fontsize=14)
#     ax.set_ylabel("Median RMSPE (fraction)", fontsize=14)

#     # ax.set_title("Median RMSPE vs GDP per capita\n(one point per model × region)", fontsize=14)

#     outpath = os.path.join(out_dir, f"{grp}_rmspe_vs_income_scatter.png")

#     print("\n[FINAL SANITY CHECK]")
#     print(summary_table)


#     plt.ylim(0, 20)
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=300)
#     plt.close()

#     print(f"Saved: {outpath}")