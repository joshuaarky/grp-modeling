import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import glob

# === Directories ===
metrics_dir = "/p/projects/impactee/Josh/thesis_analysis/model_metrics"
output_dir = "/p/projects/impactee/Josh/thesis_analysis/plots/model_output/"
os.makedirs(output_dir, exist_ok=True)

# def compute_boxplot_stats(series):
#     """Compute standard boxplot values."""
#     series = series.dropna()
#     q1 = series.quantile(0.25)
#     median = series.quantile(0.5)
#     q3 = series.quantile(0.75)
#     iqr = q3 - q1
#     whisker_low = series[series >= (q1 - 1.5 * iqr)].min()
#     whisker_high = series[series <= (q3 + 1.5 * iqr)].max()

#     return {
#         "count": len(series),
#         "median": median,
#         "q1": q1,
#         "q3": q3,
#         "iqr": iqr,
#         "whisker_low": whisker_low,
#         "whisker_high": whisker_high,
#     }

# def extract_model_and_key(filename):
#     """
#     Example filename:
#     lgbm_raw_gid0_agg_lc1_riv1_ntlL_a_test_metrics.csv

#     model_id = everything before the final '_a_test_metrics.csv'
#     key      = the 'a'
#     """
#     base = os.path.basename(filename).replace(".csv", "")
#     parts = base.split("_")

#     # parts[-1] = "metrics" (after stripping .csv, still "..._test_metrics")
#     # parts[-2] = "test"
#     # parts[-3] = key
#     key = parts[-3]

#     # model_id = join everything before the last 3 tokens
#     model_id = "_".join(parts[:-3])

#     return model_id, key

# def process_folder(folder, metric_col="r2", output_csv="all_boxplot_metrics.csv"):
#     files = glob.glob(os.path.join(folder, "*test_metrics.csv"))

#     records = []

#     for f in files:
#         model_id, key = extract_model_and_key(f)
#         df = pd.read_csv(f)

#         if metric_col not in df.columns:
#             print(f"WARNING: {metric_col} not found in {f}, skipping.")
#             continue

#         stats = compute_boxplot_stats(df[metric_col])

#         # ---- NEW: compute skewness ----
#         skew_val = df[metric_col].skew()  # Fisher moment skewness
#         stats["skew"] = skew_val

#         stats["model_id"] = model_id
#         stats["key"] = key

#         records.append(stats)

#     # Combine all into one dataframe
#     result_df = pd.DataFrame(records)

#     # Reorder columns
#     col_order = [
#         "model_id", "key", "count",
#         "median", "q1", "q3", "iqr",
#         "whisker_low", "whisker_high",
#         "skew"
#     ]
#     result_df = result_df[col_order]

#     # --- Sort by median ---
#     result_df = result_df.sort_values("median", ascending=True)

#     result_df.to_csv(os.path.join(metrics_dir, output_csv), index=False)
#     print(f"Saved: {output_csv}")

#     return result_df

# boxplot_metrics = process_folder(metrics_dir)

# === Model configurations ===
model_ids = [
    # "lgbm_raw_gid0_agg_lc1_riv1_ntlL",
    # "lgbm_raw_gid0_agg_lc1_riv1_ntlZ2",
    # "lgbm_raw_gid0_agg_lc2_riv1_ntlL",
    # "lgbm_raw_gid0_agg_lc3_riv1_ntlL",

    # "mlp_raw2_gid0_agg_lc1_riv1_ntlL",
    # "mlp_raw2_gid0_agg_lc1_riv1_ntlZ2",
    # "mlp_raw2_gid0_agg_lc2_riv1_ntlL",
    # "mlp_raw2_gid0_agg_lc3_riv1_ntlL",

    # "ann_adamw_raw2_gid0_agg_lc1_riv1_ntlL",
    # "ann_adamw_raw2_gid0_agg_lc1_riv1_ntlZ2",
    # "ann_adamw_raw2_gid0_agg_lc2_riv1_ntlL",
    # "ann_adamw_raw2_gid0_agg_lc3_riv1_ntlL",

    "lgbm_rmspe_ntlL_lc1",
    "lgbm_rmspe_ntlZ_lc1",
    "lgbm_rmspe_ntlL_lc2",
    "lgbm_rmspe_ntlL_lc3"
]

keys = ['a','b','c','d','e','f','g']

# Store all combined model data for the subplot figure
subplot_data = {}

# === Loop over models ===
for model_id in model_ids:
    print(f"\nProcessing model: {model_id}")
    all_test_metrics = []

    # --- Load test metrics for each key ---
    for key in keys:
        filepath = os.path.join(metrics_dir, f"{model_id}_{key}_test_metrics.csv")
        if not os.path.exists(filepath):
            print(f"  ⚠️ Missing file for {model_id}, key={key} — skipping.")
            continue

        df = pd.read_csv(filepath)
        df['key'] = key
        all_test_metrics.append(df)

    # --- Combine all submodels ---
    if not all_test_metrics:
        print(f"  ⚠️ No data available for {model_id}. Skipping model.")
        continue

    combined_test_metrics = pd.concat(all_test_metrics, ignore_index=True)

    # Save for subplot figure
    subplot_data[model_id] = combined_test_metrics

#     # === Boxplot of R² by key ===
#     plt.figure(figsize=(12,6))
#     sns.boxplot(x='key', y='r2', data=combined_test_metrics, order=keys)
#     plt.ylabel("Test R²")
#     plt.xlabel("Model Variant (key)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{model_id}_test_r2_boxplot.png"), dpi=300)
#     plt.close()

#     # === Boxplot of RMSE by key ===
#     plt.figure(figsize=(12,6))
#     sns.boxplot(x='key', y='rmse', data=combined_test_metrics, order=keys)
#     plt.ylabel("Test RMSE")
#     plt.xlabel("Model Variant (key)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{model_id}_test_rmse_boxplot.png"), dpi=300)
#     plt.close()

# # ---------------------------------------------------------------------
# # === Create combined subplot figure (one subplot per model_id) ===
# # ---------------------------------------------------------------------

# n_models = len(subplot_data)
# ncols = 4
# nrows = 2

# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 8))
# axes = axes.flatten()

# # --- Plot each model_id in its subplot ---
# for ax, (model_id, df) in zip(axes, subplot_data.items()):
#     sns.boxplot(x='key', y='r2', data=df, order=keys, ax=ax)
#     ax.set_title(model_id, fontsize=9)
#     ax.set_ylabel("Test R²")
#     ax.set_xlabel("")
#     ax.set_ylim(0, 0.9)   # <<< uniform y-axis limits

# # --- Turn off unused axes (if fewer than 8 models) ---
# for ax in axes[len(subplot_data):]:
#     ax.axis("off")

# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "ALL_MODELS_test_r2_subplots.png"), dpi=300)
# plt.close()

# print("\n✅ Finished! Created individual plots and combined subplot figure.")

# --- Convert subplot_data → single long DataFrame ---
grouped_df_list = []
for model_id, df in subplot_data.items():
    temp = df.copy()
    temp["model_id"] = model_id
    grouped_df_list.append(temp)

grouped_df = pd.concat(grouped_df_list, ignore_index=True)

result_df = (
    grouped_df
        .groupby(["model_id", "key"], as_index=False)
        .agg(
            median_rmspe=("rmspe", "median"),
            median_r2=("r2", "median"),
        )
        .sort_values("median_r2", ascending=False)
        .reset_index(drop=True)
)

print(result_df)

# ===============================
# HELPER: produce tight grouped boxplot with no gaps
# ===============================

def plot_grouped_boxplot(df, output_path, keys, model_color_map):
    """
    df must have columns: ['key', 'model_id', 'r2']
    """
    import itertools
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Key mapping
    key_mapping = {
        "a": "NTL, LC, and GEO",
        "b": "NTL",
        "c": "LC",
        "d": "GEO",
        "e": "NTL and LC",
        "f": "NTL and GEO",
        "g": "LC and GEO",
    }
    key_labels = [key_mapping[k] for k in keys]

    # Model legend names
    model_name_map = {
        # "lgbm_raw_gid0_agg_lc1_riv1_ntlL": "gbdt_ntlL_lc1",
        # "lgbm_raw_gid0_agg_lc1_riv1_ntlZ2": "gbdt_ntlZ_lc1",
        # "lgbm_raw_gid0_agg_lc2_riv1_ntlL": "gbdt_ntlL_lc2",
        # "lgbm_raw_gid0_agg_lc3_riv1_ntlL": "gbdt_ntlL_lc3",
        # "mlp_raw2_gid0_agg_lc1_riv1_ntlL": "ann_ntlL_lc1",
        # "mlp_raw2_gid0_agg_lc1_riv1_ntlZ2": "ann_ntlZ_lc1",
        # "mlp_raw2_gid0_agg_lc2_riv1_ntlL": "ann_ntlL_lc2",
        # "mlp_raw2_gid0_agg_lc3_riv1_ntlL": "ann_ntlL_lc3",
        "lgbm_rmspe_ntlL_lc1": "gbdt_rmspe_ntlL_lc1",
        "lgbm_rmspe_ntlZ_lc1": "gbdt_rmspe_ntlZ_lc1",
        "lgbm_rmspe_ntlL_lc2": "gbdt_rmspe_ntlL_lc2",
        "lgbm_rmspe_ntlL_lc3": "gbdt_rmspe_ntlL_lc3"
    }

    # Filter to models actually present
    available_models = sorted(df["model_id"].unique())
    local_color_map = {m: model_color_map[m] for m in available_models}

    # Build full key × model grid
    full_grid = pd.DataFrame(itertools.product(keys, available_models),
                             columns=["key", "model_id"])
    df_full = full_grid.merge(df, on=["key", "model_id"], how="left")
    df_full["model_id"] = pd.Categorical(df_full["model_id"], categories=available_models, ordered=True)
    df_full["key_label"] = df_full["key"].map(key_mapping)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df_full,
        x="key_label",
        y="r2",
        hue="model_id",            # <-- keep original model_id here
        order=key_labels,
        palette=local_color_map,   # works because keys match
        dodge=True,
        ax=ax
    )

    # Formatting
    ax.set_xlabel("Model features", fontsize=12)
    ax.set_ylabel("Test R²", fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("")
    ax.set_ylim(0, 0.9)

    # Relabel legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [model_name_map[l] for l in labels]
    ax.legend(
        handles,
        new_labels,
        title=None,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# def plot_lgbm_mlp_subplots(df, keys, model_color_map, output_path):
#     """
#     df must have columns: ['key', 'model_id', 'r2']
#     Produces a single figure with two subplots:
#         a) LGBM models (top)
#         b) MLP models (bottom)
#     """
#     import itertools
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     import pandas as pd

#     # Key mapping
#     key_mapping = {
#         "a": "NTL, LC, and GEO",
#         "b": "NTL",
#         "c": "LC",
#         "d": "GEO",
#         "e": "NTL and LC",
#         "f": "NTL and GEO",
#         "g": "LC and GEO",
#     }
#     key_labels = [key_mapping[k] for k in keys]

#     # Model legend names
#     model_name_map = {
#         "lgbm_raw_gid0_agg_lc1_riv1_ntlL": "gbdt_ntlL_lc1",
#         "lgbm_raw_gid0_agg_lc1_riv1_ntlZ2": "gbdt_ntlZ_lc1",
#         "lgbm_raw_gid0_agg_lc2_riv1_ntlL": "gbdt_ntlL_lc2",
#         "lgbm_raw_gid0_agg_lc3_riv1_ntlL": "gbdt_ntlL_lc3",
#         "mlp_raw2_gid0_agg_lc1_riv1_ntlL": "ann_ntlL_lc1",
#         "mlp_raw2_gid0_agg_lc1_riv1_ntlZ2": "ann_ntlZ_lc1",
#         "mlp_raw2_gid0_agg_lc2_riv1_ntlL": "ann_ntlL_lc2",
#         "mlp_raw2_gid0_agg_lc3_riv1_ntlL": "ann_ntlL_lc3",
#     }

#     # Panel mapping
#     panel_map = {"lgbm": "a", "mlp": "b"}

#     fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharey=True)

#     for i, model_type in enumerate(["lgbm", "mlp"]):
#         ax = axes[i]
#         df_sub = df[df.model_id.str.startswith(model_type)].copy()
#         available_models = sorted(df_sub["model_id"].unique())
#         local_color_map = {m: model_color_map[m] for m in available_models}

#         # Full grid for consistent grouping
#         full_grid = pd.DataFrame(itertools.product(keys, available_models),
#                                  columns=["key", "model_id"])
#         df_full = full_grid.merge(df_sub, on=["key", "model_id"], how="left")
#         df_full["model_id"] = pd.Categorical(df_full["model_id"],
#                                              categories=available_models,
#                                              ordered=True)
#         df_full["key_label"] = df_full["key"].map(key_mapping)

#         # Boxplot
#         sns.boxplot(
#             data=df_full,
#             x="key_label",
#             y="r2",
#             hue="model_id",
#             order=key_labels,
#             palette=local_color_map,
#             dodge=True,
#             ax=ax
#         )

#         # Axis labels
#         ax.set_xlabel("Model features" if i == 1 else "", fontsize=16)
#         ax.set_ylabel("Test R²", fontsize=16)
#         ax.tick_params(axis="x", labelsize=16)
#         ax.tick_params(axis="y", labelsize=12)
#         ax.set_ylim(0, 0.9)
#         ax.set_title("")

#         # Panel label
#         ax.text(-0.05, 1.05, panel_map[model_type], transform=ax.transAxes,
#                 fontsize=16, fontweight="bold", va="top")

#        # Legends
#         handles, labels = ax.get_legend_handles_labels()
#         new_labels = [model_name_map[l] for l in labels]

#         if model_type == "lgbm":
#             # Legend just above y=0 inside axes
#             ax.legend(
#                 handles,
#                 new_labels,
#                 title=None,
#                 fontsize=16,
#                 loc="lower center",
#                 bbox_to_anchor=(0.5, 0.02),  # slightly above y=0
#                 ncol=4,
#                 frameon=True
#             )
#         else:
#             # Legend below the axes
#             ax.legend(
#                 handles,
#                 new_labels,
#                 title=None,
#                 fontsize=16,
#                 loc="upper center",
#                 bbox_to_anchor=(0.5, 0.1),
#                 ncol=4,
#                 frameon=True
#             )
            
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.close()

def build_model_color_map(all_model_ids):
    """
    Returns a dictionary: {model_id: color}
    """

    # Sort for consistency
    all_model_ids = sorted(all_model_ids)

    # Split models by type
    lgbm_models = [m for m in all_model_ids if m.startswith("lgbm_raw")]
    mlp_models  = [m for m in all_model_ids if m.startswith("mlp")]
    lgbm_rmspe_models = [m for m in all_model_ids if m.startswith("lgbm_rmspe")]

    # Choose palettes
    cool_palette  = sns.color_palette("Blues", len(lgbm_models))
    warm_palette  = sns.color_palette("Reds",  len(mlp_models))
    purple_palette = sns.color_palette("Purples", len(lgbm_rmspe_models))

    # Assign colors
    color_map = {}

    for model, color in zip(lgbm_models, cool_palette):
        color_map[model] = color

    for model, color in zip(mlp_models, warm_palette):
        color_map[model] = color

    for model, color in zip(lgbm_rmspe_models, purple_palette):
        color_map[model] = color

    return color_map

# def plot_boxplots_key_model_pairs(df, title, output_path, model_color_map, figsize=(22, 6)):
#     """
#     df must have: ['key', 'model_id', 'r2']
#     Produces:
#       - 1 box per (key, model_id) pair
#       - Color = model_id (blues for lgbm, reds for mlp)
#       - Boxes ordered by median R²
#       - Legend manually drawn
#     """
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     import pandas as pd

#     df = df.copy()

#     # Unique ID for each pair
#     df["pair"] = df["key"] + " | " + df["model_id"]

#     # --- Compute medians for ordering ---
#     pair_medians = df.groupby("pair")["r2"].median()
#     sorted_pairs = pair_medians.sort_values().index.tolist()

#     # Sort df accordingly
#     df["pair"] = pd.Categorical(df["pair"], categories=sorted_pairs, ordered=True)
#     df_sorted = df.sort_values("pair")

#     # --- Build palette dict: pair → color (color based on model_id) ---
#     pair_palette = {
#         p: model_color_map[df_sorted[df_sorted["pair"] == p]["model_id"].iloc[0]]
#         for p in sorted_pairs
#     }

#     # --- Plot ---
#     fig, ax = plt.subplots(figsize=figsize)

#     sns.boxplot(
#         data=df_sorted,
#         x="pair",
#         y="r2",
#         hue="pair",              # required so seaborn accepts per-box palette
#         palette=pair_palette,
#         legend=False,            # we will add our own legend
#         ax=ax
#     )

#     # --- Titles & formatting ---
#     ax.set_title(title)
#     ax.set_ylabel("Test R²")
#     ax.set_xlabel("key | model_id")
#     ax.set_ylim(0, 0.9)

#     ax.set_xticklabels(ax.get_xticklabels(), rotation=65, ha="right", fontsize=9)

#     # --- Manual legend for model_ids ---
#     handles = []
#     labels = []
#     for m, c in model_color_map.items():
#         handles.append(plt.Line2D([0], [0], color=c, lw=10))
#         labels.append(m)

#     ax.legend(
#         handles,
#         labels,
#         title="Model ID",
#         loc="upper center",
#         bbox_to_anchor=(0.5, -0.25),
#         ncol=4,
#         frameon=True
#     )

#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.close()

def plot_boxplots_by_model_type(df, model_type, title, output_path, model_color_map, figsize=(14, 6)):
    """
    df must contain: ['key', 'model_id', 'r2']
    
    model_type: "lgbm" or "mlp"
    Produces one plot:
        - only models matching that prefix
        - one box per (key, model_id) pair
        - boxes sorted by median r2
        - x tick labels = key only
    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # --- Filter for model type ---
    df = df[df.model_id.str.startswith(model_type)].copy()

    # --- Create unique pair id ---
    df["pair"] = df["key"] + " | " + df["model_id"]

    # --- Compute ordering by median ---
    medians = df.groupby("pair")["r2"].median()
    sorted_pairs = medians.sort_values().index.tolist()

    df["pair"] = pd.Categorical(df["pair"], categories=sorted_pairs, ordered=True)
    df_sorted = df.sort_values("pair")

    # --- Per-pair colors (based on model_id only) ---
    pair_palette = {}
    for p in sorted_pairs:
        model_id = df_sorted[df_sorted["pair"] == p]["model_id"].iloc[0]
        pair_palette[p] = model_color_map[model_id]

    # --- Begin plot ---
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df_sorted,
        x="pair",
        y="r2",
        hue="pair",
        palette=pair_palette,
        legend=False,
        ax=ax
    )

    # --- Replace x-tick labels with key only ---
    new_xticks = [p.split(" | ")[0] for p in sorted_pairs]
    ax.set_xticks(range(len(sorted_pairs)))
    ax.set_xticklabels(new_xticks, rotation=0, ha="right", fontsize=14)

    # --- Axis labels, title, limits ---
    ax.set_title(title)
    ax.set_ylabel("Test R²")
    ax.set_xlabel("Key")
    ax.set_ylim(0, 0.9)

    # --- Manual legend (one entry per model_id) ---
    unique_models = sorted(df_sorted["model_id"].unique())
    handles = [plt.Line2D([0], [0], color=model_color_map[m], lw=10) for m in unique_models]

    ax.legend(
        handles,
        unique_models,
        title="Model ID",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=4,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# ===============================
# PRODUCE 4 PLOTS
# ===============================

# Build a global model_color_map
all_model_ids = grouped_df["model_id"].unique()
model_color_map = build_model_color_map(all_model_ids)

# ---- Plot LGBM RMSPE models ----
plot_grouped_boxplot(
    df=grouped_df,
    output_path=os.path.join(output_dir, "LGBM_RMSPE_grouped_boxplot.png"),
    keys=keys,
    model_color_map=model_color_map
)

# ---- Plot ALL models ----
plot_grouped_boxplot(
    df=grouped_df,
    # title="All Models",
    output_path=os.path.join(output_dir, "ALL_MODELS_grouped_boxplot.png"),
    keys=keys,
    model_color_map=model_color_map
)

# ---- LGBM-only ----
plot_grouped_boxplot(
    df=grouped_df[grouped_df.model_id.str.startswith("lgbm")],
    # title="LGBM Models",
    output_path=os.path.join(output_dir, "LGBM_grouped_boxplot.png"),
    keys=keys,
    model_color_map=model_color_map
)

# ---- MLP-only ----
plot_grouped_boxplot(
    df=grouped_df[grouped_df.model_id.str.startswith("mlp")],
    # title="MLP Models",
    output_path=os.path.join(output_dir, "MLP_grouped_boxplot.png"),
    keys=keys,
    model_color_map=model_color_map
)

# ---- LGBM + MLP subplots ----
plot_lgbm_mlp_subplots(
    df=grouped_df,
    keys=keys,
    model_color_map=model_color_map,
    output_path=os.path.join(output_dir, "LGBM_MLP_combined_boxplots.png")
)


# --------------------------------------------
# ---- Boxplots, sorted --------
# --------------------------------------------

# # LGBM only
# plot_boxplots_by_model_type(
#     df=grouped_df,
#     model_type="lgbm",
#     title="LGBM Models — Boxplots by Key",
#     output_path=os.path.join(output_dir, "LGBM_key_model_boxplots.png"),
#     model_color_map=model_color_map
# )

# # MLP only
# plot_boxplots_by_model_type(
#     df=grouped_df,
#     model_type="mlp",
#     title="MLP Models — Boxplots by Key",
#     output_path=os.path.join(output_dir, "MLP_key_model_boxplots.png"),
#     model_color_map=model_color_map
# )


# print("\n✓ Created all the boxplots.")

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import ast
# from collections import defaultdict
# import numpy as np

# # === Directories ===
# metrics_dir = "/p/projects/impactee/Josh/thesis_analysis/model_metrics"
# output_dir = "/p/projects/impactee/Josh/thesis_analysis/plots_per_country"
# os.makedirs(output_dir, exist_ok=True)

# keys = ['a','b','c','d','e','f','g']

# # === List of model_ids ===
# model_ids = [
#     "lgbm_raw_gid0_agg_lc1_riv1_ntlL",
#     "lgbm_raw_gid0_agg_lc1_riv1_ntlZ2",
#     "lgbm_raw_gid0_agg_lc2_riv1_ntlL",
#     "lgbm_raw_gid0_agg_lc3_riv1_ntlL",

#     "mlp_raw2_gid0_agg_lc1_riv1_ntlL",
#     "mlp_raw2_gid0_agg_lc1_riv1_ntlZ2",
#     "mlp_raw2_gid0_agg_lc2_riv1_ntlL",
#     "mlp_raw2_gid0_agg_lc3_riv1_ntlL",
# ]

# # Explicit list of (model_id, key) combos you want to include
# model_key_pairs = [
#     ("lgbm_raw_gid0_agg_lc1_riv1_ntlZ2", "a"),
#     ("lgbm_raw_gid0_agg_lc1_riv1_ntlL", "g"),
#     ("mlp_raw2_gid0_agg_lc1_riv1_ntlL", "b"),
#     ("mlp_raw2_gid0_agg_lc3_riv1_ntlL", "e"),
#     ("mlp_raw2_gid0_agg_lc1_riv1_ntlL", "f"),
# ]

# # Legend label mapping
# legend_map = {
#     ("lgbm_raw_gid0_agg_lc1_riv1_ntlZ2", "a"): "gbdt_4",
#     ("lgbm_raw_gid0_agg_lc1_riv1_ntlL", "g"): "gbdt_5",
#     ("mlp_raw2_gid0_agg_lc1_riv1_ntlL", "b"): "ann_1",
#     ("mlp_raw2_gid0_agg_lc3_riv1_ntlL", "e"): "ann_2",
#     ("mlp_raw2_gid0_agg_lc1_riv1_ntlL", "f"): "ann_3",
# }

# # Model → Family (GBDT or ANN)
# family_map = {
#     "gbdt_4": "GBDT",
#     "gbdt_5": "GBDT",
#     "ann_1": "ANN",
#     "ann_2": "ANN",
#     "ann_3": "ANN",
# }

# # ============================================================
# # === FUNCTION: EXTRACT COUNTRY-LEVEL R² FOR ONE MODEL-KEY ===
# # ============================================================

# def extract_country_r2(model_id, key):
#     fold_path = os.path.join(metrics_dir, f"{model_id}_{key}_fold_countries.csv")
#     test_path = os.path.join(metrics_dir, f"{model_id}_{key}_test_metrics.csv")

#     if not (os.path.exists(fold_path) and os.path.exists(test_path)):
#         print(f"⚠️ Missing files for {model_id}, key={key}. Skipping.")
#         return None

#     # Load data
#     fold_df = pd.read_csv(fold_path)
#     test_df = pd.read_csv(test_path)
#     fold_df["train_GID_0"] = fold_df["train_GID_0"].apply(ast.literal_eval)

#     # Collect test R² per country
#     country_r2 = defaultdict(list)

#     for _, row in test_df.iterrows():
#         rep, fold, r2 = row["repeat"], row["fold"], row["r2"]

#         frow = fold_df[(fold_df["repeat"] == rep) & (fold_df["fold"] == fold)]
#         if frow.empty:
#             continue

#         train_countries = frow["train_GID_0"].values[0]

#         for gid in train_countries:
#             country_r2[gid].append(r2)

#     if len(country_r2) == 0:
#         print(f"⚠️ No country results for {model_id}, key={key}.")
#         return None

#     df = pd.DataFrame({
#         "GID_0": list(country_r2.keys()),
#         "avg_test_r2": [np.mean(v) for v in country_r2.values()],
#         "std_test_r2": [np.std(v) for v in country_r2.values()],
#         "n_folds": [len(v) for v in country_r2.values()],
#     })

#     df["model_id"] = model_id
#     df["key"] = key

#     return df

# # ============================================================
# # === FUNCTION: RUN FULL PIPELINE FOR A SUBSET OF MODELS ===
# # ============================================================

# def process_models(model_key_pairs, order_reference, output_name):

#     # ----------------------
#     # Load data
#     # ----------------------
#     all_dfs = []

#     for model_id, key in model_key_pairs:
#         df = extract_country_r2(model_id, key)
#         if df is not None:
#             all_dfs.append(df)

#     if len(all_dfs) == 0:
#         print("⚠️ No valid model-key data found.")
#         return

#     combined = pd.concat(all_dfs, ignore_index=True)

#     # ----------------------
#     # Determine GID_0 ordering
#     # ----------------------
#     ref_model, ref_key = order_reference

#     ref_df = combined[
#         (combined["model_id"] == ref_model) &
#         (combined["key"] == ref_key)
#     ]

#     if ref_df.empty:
#         raise RuntimeError("Reference model/key not found among model_key_pairs")

#     gid_order = ref_df.sort_values("avg_test_r2")["GID_0"].tolist()

#     combined["GID_0"] = pd.Categorical(
#         combined["GID_0"],
#         categories=gid_order,
#         ordered=True
#     )

#     # ----------------------
#     # Label + family
#     # ----------------------
#     def get_label(row):
#         return legend_map.get(
#             (row["model_id"], row["key"]),
#             f"{row['model_id']}__{row['key']}"
#         )

#     combined["model_key"] = combined.apply(get_label, axis=1)
#     combined["family"] = combined["model_key"].map(family_map)

#     # ----------------------
#     # Color palettes
#     # ----------------------
#     gbdt_palette = sns.color_palette("Blues", n_colors=2)
#     ann_palette  = sns.color_palette("Reds",  n_colors=3)

#     unique_keys = sorted(combined["model_key"].unique())

#     palette = {}
#     gbdt_i = ann_i = 0

#     for mk in unique_keys:
#         if family_map[mk] == "GBDT":
#             palette[mk] = gbdt_palette[gbdt_i]
#             gbdt_i += 1
#         else:
#             palette[mk] = ann_palette[ann_i]
#             ann_i += 1

#     # ----------------------
#     # Plot
#     # ----------------------
#     plt.figure(figsize=(16, 7))

#     ax = sns.barplot(
#         data=combined,
#         x="GID_0",
#         y="avg_test_r2",
#         hue="model_key",
#         palette=palette,
#         errorbar=None
#     )

#     # Add matplotlib error bars manually
#     # ------------------------------------------------------------
#     # Loop over the bars in the seaborn plot
#     for bar, (_, row) in zip(ax.patches, combined.iterrows()):
#         x = bar.get_x() + bar.get_width() / 2
#         y = bar.get_height()
#         err = row["std_test_r2"]

#         plt.errorbar(
#             x, y,
#             yerr=err,
#             fmt='none',
#             capsize=3,
#             linewidth=1,
#             color='black',
#             alpha=0.3
#         )
    
#     # --- Make bars slightly darker ---
#     for bar in ax.patches:
#         facecolor = bar.get_facecolor()
#         # Darken by reducing lightness
#         bar.set_facecolor(facecolor[:3] + (0.85,))  # keep RGB, slightly darker alpha

#     # plt.xticks(rotation=90, fontsize=12)
#     # Remove the GID_0 tick labels but keep the axis label
#     ax.set_xticklabels([])       # remove tick text
#     ax.tick_params(axis='x', which='both', length=0)  # remove tick marks

#     # Remove extra left/right padding
#     ax.set_xlim(-0.5, len(combined['GID_0'].unique()) - 0.5)

#     plt.yticks(fontsize=20)
#     plt.ylabel("Test R²", fontsize=20)
#     #plt.xlabel("Country in training set (GID_0)", fontsize=16)
#     plt.xlabel("Country included in training set (one bar group per country)", fontsize=20)
#     plt.ylim(0.65, 0.90)
#     plt.legend(fontsize=20)
#     plt.tight_layout()

#     out_path = os.path.join(output_dir, output_name)
#     plt.savefig(out_path, dpi=300)
#     plt.close()

#     print(f"🎉 Saved plot: {out_path}")

#     return combined


# # ============================================================
# # === RUN THE TWO PIPELINES
# # ============================================================

# # GBDT-only subset
# gbdt_pairs = [(m, k) for (m, k) in model_key_pairs if m.startswith("lgbm")]
# # ANN-only subset
# ann_pairs  = [(m, k) for (m, k) in model_key_pairs if m.startswith("mlp")]

# # Choose appropriate reference for ordering in each family
# gbdt_reference = ("lgbm_raw_gid0_agg_lc1_riv1_ntlZ2", "a")   # gbdt_4
# ann_reference  = ("mlp_raw2_gid0_agg_lc1_riv1_ntlL", "b")    # ann_1

# # Run two passes
# gbdt_combined = process_models(gbdt_pairs, gbdt_reference, "gbdt_grouped_bar_r2.png")
# ann_combined  = process_models(ann_pairs,  ann_reference,  "ann_grouped_bar_r2.png")

# # ============================================================
# # === FUNCTION: BUILD TOP/BOTTOM TABLES WITH "mean ± std"
# # ============================================================

# def build_mean_std_tables(combined_subset, family_label, output_dir):
#     """
#     combined_subset: combined DataFrame for a family (ANN or GBDT)
#     family_label: 'ANN' or 'GBDT', used for output filenames
#     """
#     summary_top = {}
#     summary_bottom = {}

#     groups = combined_subset["model_key"].unique()

#     # --- Compute top/bottom 25% per model_key ---
#     for mk in groups:
#         df_mk = combined_subset[combined_subset["model_key"] == mk].copy()
#         df_mk = df_mk.sort_values("avg_test_r2")

#         n = len(df_mk)
#         k = max(1, int(0.25 * n))  # 25%

#         summary_top[mk] = df_mk.tail(k)
#         summary_bottom[mk] = df_mk.head(k)

#     # --- Find common GID_0 across all models in the group ---
#     top_sets = [set(df["GID_0"]) for df in summary_top.values()]
#     bottom_sets = [set(df["GID_0"]) for df in summary_bottom.values()]

#     common_top = set.intersection(*top_sets)
#     common_bottom = set.intersection(*bottom_sets)

#     # --- Function to build table ---
#     def build_table(common_set, label):
#         if len(common_set) == 0:
#             print(f"\n⚠️ No GID_0 appear in the {label} list for all models in {family_label}.")
#             return

#         rows = []
#         for gid in common_set:
#             row = {"GID_0": gid}
#             for mk in groups:
#                 df = combined_subset[(combined_subset["GID_0"] == gid) &
#                                      (combined_subset["model_key"] == mk)]
#                 if not df.empty:
#                     mean_val = float(df["avg_test_r2"].values[0])
#                     std_val  = float(df["std_test_r2"].values[0])
#                     row[mk] = f"{mean_val:.3f} ± {std_val:.3f}"
#                 else:
#                     row[mk] = np.nan
#             rows.append(row)

#         table = pd.DataFrame(rows)
#         table = table.set_index("GID_0")
#         table = table[sorted(table.columns)]
#         print(f"\n==============================")
#         print(f" TABLE: {label} 25% FOR ALL MODELS ({family_label})")
#         print("==============================")
#         print(table.to_string())
#         return table

#     table_top = build_table(common_top, "TOP")
#     table_bottom = build_table(common_bottom, "BOTTOM")

#     # --- Save CSVs ---
#     if table_top is not None:
#         out_top_csv = os.path.join(output_dir, f"{family_label}_common_top_25_percent_countries.csv")
#         table_top.to_csv(out_top_csv)
#         print(f"\n✅ Saved TOP 25% table for {family_label} to:\n{out_top_csv}")

#     if table_bottom is not None:
#         out_bottom_csv = os.path.join(output_dir, f"{family_label}_common_bottom_25_percent_countries.csv")
#         table_bottom.to_csv(out_bottom_csv)
#         print(f"\n✅ Saved BOTTOM 25% table for {family_label} to:\n{out_bottom_csv}")

#     return table_top, table_bottom

# # Build tables for ANN
# ann_subset = ann_combined.copy()  # contains only ANN models
# table_top_ann, table_bottom_ann = build_mean_std_tables(ann_subset, "ANN", output_dir)

# # Build tables for GBDT
# gbdt_subset = gbdt_combined.copy()  # contains only GBDT models
# table_top_gbdt, table_bottom_gbdt = build_mean_std_tables(gbdt_subset, "GBDT", output_dir)


# ============================================================
# === MASTER LOOP OVER ALL MODELS ===
# ============================================================
# for model_id in model_ids:
#     print(f"\nProcessing model: {model_id}")

#     all_country_r2 = []   # for storing results from all keys (submodels)

#     model_plot_dir = os.path.join(output_dir, model_id)
#     os.makedirs(model_plot_dir, exist_ok=True)


#     # ============================================================
#     # === LOOP OVER KEYS (submodels) ===
#     # ============================================================
#     for key in keys:

#         # --- File paths ---
#         fold_path = os.path.join(metrics_dir, f"{model_id}_{key}_fold_countries.csv")
#         test_path = os.path.join(metrics_dir, f"{model_id}_{key}_test_metrics.csv")

#         if not (os.path.exists(fold_path) and os.path.exists(test_path)):
#             print(f"  ⚠️ Missing data for {model_id}, key={key} — skipping.")
#             continue

#         # Load data
#         fold_countries_df = pd.read_csv(fold_path)
#         test_metrics_df = pd.read_csv(test_path)

#         # Convert stored stringified lists
#         fold_countries_df['train_GID_0'] = fold_countries_df['train_GID_0'].apply(ast.literal_eval)

#         # ============================================================
#         # === MAP TEST R² VALUES TO COUNTRIES ===
#         # ============================================================
#         country_r2 = defaultdict(list)

#         for _, row in test_metrics_df.iterrows():
#             rep, fold, test_r2 = row['repeat'], row['fold'], row['r2']

#             fold_row = fold_countries_df[
#                 (fold_countries_df['repeat'] == rep) &
#                 (fold_countries_df['fold'] == fold)
#             ]

#             if fold_row.empty:
#                 continue

#             train_countries = fold_row['train_GID_0'].values[0]

#             for gid in train_countries:
#                 country_r2[gid].append(test_r2)

#         # ============================================================
#         # === PER-KEY PER-COUNTRY R² PLOT ===
#         # ============================================================
#         country_r2_df = pd.DataFrame({
#             'GID_0': list(country_r2.keys()),
#             'avg_test_r2': [np.mean(vals) for vals in country_r2.values()],
#             'std_test_r2': [np.std(vals) for vals in country_r2.values()],
#             'n_folds': [len(vals) for vals in country_r2.values()]
#         }).sort_values('avg_test_r2')

#         # Plot per-key-per-country result
#         plt.figure(figsize=(12, 6))
#         sns.barplot(data=country_r2_df, x='GID_0', y='avg_test_r2')

#         ymin, ymax = country_r2_df['avg_test_r2'].min(), country_r2_df['avg_test_r2'].max()
#         buffer = (ymax - ymin) * 0.05 if ymax > ymin else 0.01
#         plt.ylim(ymin - buffer, ymax + buffer)

#         plt.xticks(rotation=90)
#         plt.ylabel("Average Test R²")
#         plt.title(f"Average Test R² per Country ({model_id}, key={key})")
#         plt.tight_layout()

#         plt.savefig(os.path.join(output_dir, f"{model_id}_per_country_avg_test_r2_{key}.png"), dpi=300)
#         plt.close()

#         # Store for combined plot
#         country_r2_df['key'] = key
#         all_country_r2.append(country_r2_df[['GID_0','avg_test_r2','key']])

#     # end key loop

#     # ============================================================
#     # === COMBINED ACROSS ALL SUBMODELS ===
#     # ============================================================
#     if not all_country_r2:
#         print(f"  ⚠️ No valid key-level data for {model_id}. Skipping.")
#         continue

#     combined_country_r2 = pd.concat(all_country_r2, ignore_index=True)

#     # Average R² across all submodels
#     country_avg_r2 = (
#         combined_country_r2.groupby('GID_0')['avg_test_r2']
#         .mean()
#         .reset_index()
#         .sort_values('avg_test_r2')
#     )

#     # Save CSV
#     combined_csv = os.path.join(
#         metrics_dir, f"{model_id}_per_country_avg_test_r2_all_submodels.csv"
#     )
#     country_avg_r2.to_csv(combined_csv, index=False)

#     # ============================================================
#     # === FINAL BARPLOT ACROSS ALL SUBMODELS ===
#     # ============================================================
#     plt.figure(figsize=(14, 7))
#     sns.barplot(
#         data=country_avg_r2,
#         x='GID_0',
#         y='avg_test_r2',
#         order=country_avg_r2['GID_0']
#     )

#     ymin, ymax = country_avg_r2['avg_test_r2'].min(), country_avg_r2['avg_test_r2'].max()
#     buffer = (ymax - ymin) * 0.05 if ymax > ymin else 0.01
#     plt.ylim(ymin - buffer, ymax + buffer)

#     plt.xticks(rotation=90)
#     plt.ylabel("Average Test R² (across all submodels)")
#     plt.title(f"Average Test R² per Country — {model_id} (mean across keys)")
#     plt.tight_layout()

#     plt.savefig(
#         os.path.join(output_dir, f"{model_id}_per_country_avg_test_r2_all_submodels.png"),
#         dpi=300
#     )
#     plt.close()

#     print(f"  ✔ Finished {model_id}")
