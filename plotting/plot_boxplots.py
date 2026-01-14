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
    "lgbm_rmspe_ntlL_lc3",

    "ann_rmspe_ntlL_lc1",
    "ann_rmspe_ntlZ_lc1",
    "ann_rmspe_ntlL_lc2",
    "ann_rmspe_ntlL_lc3"
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
        "lgbm_rmspe_ntlL_lc3": "gbdt_rmspe_ntlL_lc3",

        "ann_rmspe_ntlL_lc1": "ann_rmspe_ntlL_lc1",
        "ann_rmspe_ntlZ_lc1": "ann_rmspe_ntlZ_lc1",
        "ann_rmspe_ntlL_lc2": "ann_rmspe_ntlL_lc2",
        "ann_rmspe_ntlL_lc3": "ann_rmspe_ntlL_lc3"
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
    ann_rmspe_models = [m for m in all_model_ids if m.startswith("ann_rmspe")]

    # Choose palettes
    cool_palette  = sns.color_palette("Blues", len(lgbm_models))
    warm_palette  = sns.color_palette("Reds",  len(mlp_models))
    purple_palette = sns.color_palette("Purples", len(lgbm_rmspe_models))
    orange_palette = sns.color_palette("Oranges", len(ann_rmspe_models))

    # Assign colors
    color_map = {}

    for model, color in zip(lgbm_models, cool_palette):
        color_map[model] = color

    for model, color in zip(mlp_models, warm_palette):
        color_map[model] = color

    for model, color in zip(lgbm_rmspe_models, purple_palette):
        color_map[model] = color
    
    for model, color in zip(ann_rmspe_models, orange_palette):
        color_map[model] = color

    return color_map

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
    df=grouped_df[grouped_df.model_id.str.startswith("lgbm_rmspe")],
    output_path=os.path.join(output_dir, "LGBM_RMSPE_grouped_boxplot.png"),
    keys=keys,
    model_color_map=model_color_map
)

# ---- Plot ANN RMSPE models ----
plot_grouped_boxplot(
    df=grouped_df[grouped_df.model_id.str.startswith("ann_rmspe")],
    output_path=os.path.join(output_dir, "ANN_RMSPE_grouped_boxplot.png"),
    keys=keys,
    model_color_map=model_color_map
)

# # ---- Plot ALL models ----
# plot_grouped_boxplot(
#     df=grouped_df,
#     # title="All Models",
#     output_path=os.path.join(output_dir, "ALL_MODELS_grouped_boxplot.png"),
#     keys=keys,
#     model_color_map=model_color_map
# )

# # ---- LGBM-only ----
# plot_grouped_boxplot(
#     df=grouped_df[grouped_df.model_id.str.startswith("lgbm")],
#     # title="LGBM Models",
#     output_path=os.path.join(output_dir, "LGBM_grouped_boxplot.png"),
#     keys=keys,
#     model_color_map=model_color_map
# )

# # ---- MLP-only ----
# plot_grouped_boxplot(
#     df=grouped_df[grouped_df.model_id.str.startswith("mlp")],
#     # title="MLP Models",
#     output_path=os.path.join(output_dir, "MLP_grouped_boxplot.png"),
#     keys=keys,
#     model_color_map=model_color_map
# )

# # ---- LGBM + MLP subplots ----
# plot_lgbm_mlp_subplots(
#     df=grouped_df,
#     keys=keys,
#     model_color_map=model_color_map,
#     output_path=os.path.join(output_dir, "LGBM_MLP_combined_boxplots.png")
# )
