import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import optuna
import os
import logging
from datetime import datetime

# === Logging Setup ===
model_id = "lgbm_raw_gid0_agg_lc2_riv1_ntlL"
def setup_logger(key):
    os.makedirs(f"logs/{model_id}", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/{model_id}/{model_id}_{key}_{timestamp}.log"

    logger = logging.getLogger(key)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_filename, mode="w")
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# === Load Data ===
df = pd.read_csv("/p/projects/impactee/Josh/thesis_analysis/merged_data_final.csv")
csv_filename = os.path.basename("/p/projects/impactee/Josh/thesis_analysis/merged_data_final.csv")

# non-log variables for all sets...
# === Configuration ===

# --- Define feature groups ---
ntl_features = [
    "NTL_mean", "NTL_std"
]

lc_features = [
    'lccs_ag_101112_share', 'lccs_ag_20_share', 'lccs_ag_30_share', 'lccs_ag_40_share',
    'lccs_forest_50_share', 'lccs_forest_606162_share', 'lccs_forest_707172_share',
    'lccs_forest_808182_share', 'lccs_forest_90_share', 'lccs_forest_100_share',
    'lccs_forest_160_share', 'lccs_forest_170_share', 'lccs_grass_110_share',
    'lccs_grass_130_share', 'lccs_wet_180_share', 'lccs_urban_190_share',
    'lccs_shrub_120121122_share', 'lccs_sparse_140_share', 'lccs_sparse_150151152153_share',
    'lccs_bare_200201202_share', 'lccs_water_210_share', 'lccs_snow_220_share',
]

geo_features = [
    "avg_coast_dist", "avg_tri", "avg_lake_dist", "major_river_dist_mean",
    "std_coast_dist", "std_tri", "std_lake_dist", "major_river_dist_std"
]

gdp_features = [
    # "gdp_pc_lcu",  # optional alternative
    "gdp_pc_2015_usd"
]

# --- Combine groups into predictor sets ---
predictors_a = lc_features + geo_features + ntl_features + gdp_features
predictors_b = ntl_features + gdp_features
predictors_c = lc_features + gdp_features
predictors_d = geo_features + gdp_features
predictors_e = ntl_features + lc_features + gdp_features
predictors_f = ntl_features + geo_features + gdp_features
predictors_g = lc_features + geo_features + gdp_features

predictor_sets = {
    "a": predictors_a,
    # "b": predictors_b,
    "c": predictors_c,
    # "d": predictors_d,
    "e": predictors_e,
    # "f": predictors_f,
    "g": predictors_g,
}

os.makedirs(f"/p/projects/impactee/Josh/thesis_analysis/optuna_{model_id}/", exist_ok=True)
for key, predictors in predictor_sets.items():
    logger = setup_logger(key)
    logger.info(f"Predictors for set {key}: {predictors}")

    n_trials = 1
    optuna_storage = f"sqlite:////p/projects/impactee/Josh/thesis_analysis/optuna_{model_id}/optuna_lgbm_{key}.db"

    target = 'grp_pc_lcu2015_usd'
    df_model = df.dropna(subset=[target] + predictors)
    logger.info(f"Number of rows after filtering: {len(df_model)}")

    X = df_model[predictors + ["GID_1"]]
    y = df_model[target]

    # === Optuna Objective ===
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",

            # Complexity / tree structure
            "num_leaves": trial.suggest_int("num_leaves", 31, 512),     # allow deeper splits
            "max_depth": trial.suggest_int("max_depth", -1, 20),        # -1 = unlimited, tune up to 20

            # Learning dynamics
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 3000),  # let Optuna decide how long to train

            # Regularization
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 10.0, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 100.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 100.0, log=True),

            # Sampling / feature subsampling
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        }

        gkf = GroupKFold(n_splits=5)
        rmse_scores = []

        for train_idx, val_idx in gkf.split(df_model, groups=df_model['GID_0']):
            train_df, val_df = df_model.iloc[train_idx], df_model.iloc[val_idx]

            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), predictors),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['GID_1'])
            ])

            X_train = preprocessor.fit_transform(train_df[predictors + ['GID_1']])
            X_val   = preprocessor.transform(val_df[predictors + ['GID_1']])

            y_train, y_val = train_df[target], val_df[target]

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)

            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_val],
                valid_names=["train", "valid"],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            preds = model.predict(X_val, num_iteration=model.best_iteration)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    # === Run Optuna ===
    study_name = f"lgbm_study_{key}"
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=optuna_storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"Best params: {best_params}")

    ################################################
    ### === Train and Evaluate Multiple Runs === ###
    ################################################

    n_splits = 5 # for GroupKFold CV only
    n_repeats = 20  # adjust as needed
    unique_countries = df_model['GID_0'].unique() # for LOCO CV only

    # Lists to store results
    all_val_metrics = []
    all_test_metrics = []
    all_fold_country_mapping = []

    #####################################
    #### ====== GroupKFold CV ====== ####
    #####################################

    # for rep in range(n_repeats):
    #     logger.info(f"=== CV Repeat {rep + 1}/{n_repeats} ===")
        
    #     # Shuffle groups differently for each repeat
    #     unique_groups = df_model['GID_0'].unique()
    #     rng = np.random.default_rng(seed=rep)
    #     rng.shuffle(unique_groups)

    #     gkf = GroupKFold(n_splits=n_splits)
        
    #     for fold_idx, (_, test_group_idx) in enumerate(gkf.split(unique_groups, groups=unique_groups)):
    #         test_groups = unique_groups[test_group_idx]
    #         test_df = df_model[df_model['GID_0'].isin(test_groups)]
    #         train_val_df = df_model[~df_model['GID_0'].isin(test_groups)]

    #         # Split train into train/val (by groups)
    #         gids_train, gids_val = train_test_split(
    #             train_val_df['GID_0'].unique(),
    #             test_size=0.2,
    #             random_state=rep  # ensures reproducible train/val split per repeat
    #         )
    #         val_df = train_val_df[train_val_df['GID_0'].isin(gids_val)]
    #         train_df = train_val_df[train_val_df['GID_0'].isin(gids_train)]

    #         # Preprocessing
    #         preprocessor = ColumnTransformer([
    #             ('num', StandardScaler(), predictors),
    #             ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['GID_1'])
    #         ])

    #         X_train = preprocessor.fit_transform(train_df[predictors + ['GID_1']])
    #         X_val   = preprocessor.transform(val_df[predictors + ['GID_1']])
    #         X_test  = preprocessor.transform(test_df[predictors + ['GID_1']])

    #         y_train, y_val, y_test = train_df[target], val_df[target], test_df[target]

    #         # Train LightGBM
    #         lgb_train = lgb.Dataset(X_train, y_train)
    #         lgb_val   = lgb.Dataset(X_val, y_val)
    #         model = lgb.train(
    #             {**best_params, "objective": "regression", "metric": "rmse", "verbosity": -1},
    #             lgb_train,
    #             valid_sets=[lgb_train, lgb_val],
    #             num_boost_round=1000,
    #             callbacks=[lgb.early_stopping(50, verbose=False)]
    #         )

    #         # === Validation metrics ===
    #         y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    #         val_r2 = r2_score(y_val, y_val_pred)
    #         val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    #         all_val_metrics.append({
    #             "repeat": rep,
    #             "fold": fold_idx,
    #             "r2": val_r2,
    #             "rmse": val_rmse
    #         })

    #         # === Test metrics ===
    #         y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    #         test_r2 = r2_score(y_test, y_test_pred)
    #         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    #         all_test_metrics.append({
    #             "repeat": rep,
    #             "fold": fold_idx,
    #             "r2": test_r2,
    #             "rmse": test_rmse
    #         })

    #         # === Track countries in this fold ===
    #         all_fold_country_mapping.append({
    #             "repeat": rep,
    #             "fold": fold_idx,
    #             "train_GID_0": gids_train.tolist(),
    #             "val_GID_0": gids_val.tolist(),
    #             "test_GID_0": test_groups.tolist()
    #         })

    # # Convert to DataFrames for analysis
    # val_metrics_df = pd.DataFrame(all_val_metrics)
    # test_metrics_df = pd.DataFrame(all_test_metrics)
    # fold_countries_df = pd.DataFrame(all_fold_country_mapping)

    # # Create directory if it doesn't exist
    # metrics_dir = f"/p/projects/impactee/Josh/thesis_analysis/model_metrics"
    # os.makedirs(metrics_dir, exist_ok=True)

    # # Save validation metrics
    # val_metrics_df.to_csv(
    #     f"{metrics_dir}/{model_id}_{key}_val_metrics.csv",
    #     index=False
    # )

    # # Save test metrics
    # test_metrics_df.to_csv(
    #     f"{metrics_dir}/{model_id}_{key}_test_metrics.csv",
    #     index=False
    # )

    # # Save fold-country mapping
    # fold_countries_df.to_csv(
    #     f"{metrics_dir}/{model_id}_{key}_fold_countries.csv",
    #     index=False
    # )

    #############################
    ### ====== LOCO CV ====== ###
    #############################
    
    all_subnational_results = []

    for rep in range(n_repeats):
        logger.info(f"=== LOCO CV Repeat {rep + 1}/{n_repeats} ===")
        
        # shuffle country order each repeat
        rng = np.random.default_rng(seed=rep)
        shuffled_countries = unique_countries.copy()
        rng.shuffle(shuffled_countries)

        for fold_idx, test_country in enumerate(shuffled_countries):
            logger.info(f"--- Test country: {test_country} ---")

            # === 1. Split LOCO test set ===
            test_df = df_model[df_model['GID_0'] == test_country]
            train_val_df = df_model[df_model['GID_0'] != test_country]

            # === 2. Train/validation split (by GID_0 — group-wise) ===
            train_countries, val_countries = train_test_split(
                train_val_df['GID_0'].unique(),
                test_size=0.2,
                random_state=rep
            )
            
            train_df = train_val_df[train_val_df['GID_0'].isin(train_countries)]
            val_df   = train_val_df[train_val_df['GID_0'].isin(val_countries)]

            # === 3. Preprocessing ===
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), predictors),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['GID_1'])
            ])

            X_train = preprocessor.fit_transform(train_df[predictors + ['GID_1']])
            X_val   = preprocessor.transform(val_df[predictors + ['GID_1']])
            X_test  = preprocessor.transform(test_df[predictors + ['GID_1']])

            y_train, y_val, y_test = train_df[target], val_df[target], test_df[target]

            # === 4. Train LightGBM ===
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val   = lgb.Dataset(X_val, y_val)

            model = lgb.train(
                {**best_params, "objective": "regression", "metric": "rmse", "verbosity": -1},
                lgb_train,
                valid_sets=[lgb_train, lgb_val],
                num_boost_round=2000,
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            # === 5. Validation metrics ===
            y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            val_r2 = r2_score(y_val, y_val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

            all_val_metrics.append({
                "repeat": rep,
                "fold": fold_idx,
                "test_country": test_country,
                "r2": val_r2,
                "rmse": val_rmse
            })

            # === 6. Test metrics (LOCO held-out country) ===
            y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            all_test_metrics.append({
                "repeat": rep,
                "fold": fold_idx,
                "test_country": test_country,
                "r2": test_r2,
                "rmse": test_rmse
            })

            # --- Append subnational-level predictions ---
            subnational_df = pd.DataFrame({
                "repeat": rep,
                "fold": fold_idx,
                "GID_0": test_df["GID_0"].values,
                "GID_1": test_df["GID_1"].values,
                "y_true": y_test.values,
                "y_pred": y_test_pred
            })
            all_subnational_results.append(subnational_df)

            # === 7. Save mapping of which countries were train/val/test ===
            all_fold_country_mapping.append({
                "repeat": rep,
                "fold": fold_idx,
                "test_country": test_country,
                "train_GID_0": train_countries.tolist(),
                "val_GID_0": val_countries.tolist()
            })


    # === Save outputs ===
    val_metrics_df = pd.DataFrame(all_val_metrics)
    test_metrics_df = pd.DataFrame(all_test_metrics)
    fold_countries_df = pd.DataFrame(all_fold_country_mapping)

    metrics_dir = f"/p/projects/impactee/Josh/thesis_analysis/model_metrics_loco/{model_id}"
    os.makedirs(metrics_dir, exist_ok=True)

    # Combine all subnational predictions and save
    all_subnational_df = pd.concat(all_subnational_results, ignore_index=True)
    subnational_csv_path = f"{metrics_dir}/{model_id}_{key}_loco_subnational_predictions.csv"
    all_subnational_df.to_csv(subnational_csv_path, index=False)
    logger.info(f"Saved subnational predictions to: {subnational_csv_path}")

    val_metrics_df.to_csv(f"{metrics_dir}/{model_id}_{key}_loco_val_metrics.csv", index=False)
    test_metrics_df.to_csv(f"{metrics_dir}/{model_id}_{key}_loco_test_metrics.csv", index=False)
    fold_countries_df.to_csv(f"{metrics_dir}/{model_id}_{key}_loco_fold_countries.csv", index=False)

'''
    # ======================================================
    # SHAP INTERACTION + PERMUTATION IMPORTANCE ANALYSIS
    # ======================================================

    import shap
    import itertools
    import matplotlib.pyplot as plt

    def plot_interaction(fname, interact_with, save_path):
        try:
            shap.dependence_plot(
                fname,
                shap_values.values,
                X_test,
                interaction_index=interact_with,
                show=False
            )
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            return f"Saved {fname} vs {interact_with}"
        except Exception as e:
            return f"Failed {fname} vs {interact_with}: {e}"

    # NOTE: Update these as needed
    base_path = "/p/projects/impactee/Josh/thesis_analysis/shap_feat_importance"
    output_dir = os.path.join(base_path, f"{model_id}")
    os.makedirs(output_dir, exist_ok=True)

    # === Recreate preprocessor (must match training) ===
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), predictors),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['GID_1'])
    ])

    # Fit only on training data (if not already in memory)
    preprocessor.fit(train_df[predictors + ['GID_1']])

    # Transform test data (same as training input to LightGBM)
    X_test_encoded = preprocessor.transform(test_df[predictors + ['GID_1']])

    # Convert to dense array for SHAP (LightGBM expects numpy, not sparse)
    X_test_dense = X_test_encoded.toarray()

    # Keep only the first N columns (your numeric predictors)
    X_test_num = X_test_dense[:, :len(predictors)]
    X_test_num_df = pd.DataFrame(X_test_num, columns=predictors)

    # --- SHAP ANALYSIS ---
    logger.info(f"Starting SHAP interaction analysis for predictor set '{key}'")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_dense)
    interaction_values = explainer.shap_interaction_values(X_test_dense)

    # === NEW SECTION: SHAP SUMMARY PLOT ===
    logger.info("Generating SHAP summary plot...")

    summary_folder = os.path.join(output_dir, "shap_summary_plots")
    os.makedirs(summary_folder, exist_ok=True)
    summary_path = os.path.join(summary_folder, f"shap_summary_{key}.png")
    plt.figure()
    shap.summary_plot(
        shap_values[:, :len(predictors)],
        X_test_num_df,
        show=False,
        plot_size=(10, 8),
    )
    plt.tight_layout()
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved to {summary_path}")

    # === Focus on numeric predictors only ===
    n_features = len(predictors)
    mean_abs_interactions = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                mean_abs_interactions[i, j] = np.abs(interaction_values[:, i, j]).mean()

    interaction_df = pd.DataFrame(mean_abs_interactions, index=predictors, columns=predictors)
    interaction_folder = os.path.join(output_dir, "shap_interaction_matrices")
    os.makedirs(interaction_folder, exist_ok=True)
    interaction_df.to_csv(os.path.join(interaction_folder, f"shap_interaction_matrix_{key}.csv"))
    logger.info(f"SHAP interaction matrix saved")

    # Rank and plot top interactions
    feature_pairs = list(itertools.combinations(range(n_features), 2))
    ranked_pairs = sorted(feature_pairs, key=lambda ij: mean_abs_interactions[ij[0], ij[1]], reverse=True)

    top_k = 10
    topk_dir = os.path.join(output_dir, f"shap_top_interactions_{key}")
    os.makedirs(topk_dir, exist_ok=True)

    topk_tasks = []
    for idx, (i, j) in enumerate(ranked_pairs[:top_k]):
        fname_i, fname_j = predictors[i], predictors[j]
        shap.dependence_plot(
            fname_i,                            # main feature
            shap_values[:, :n_features],
            X_test_num_df.values,
            interaction_index=fname_j,          # secondary (color) feature
            display_features=X_test_num_df,
            show=False
        )
        plt.savefig(os.path.join(topk_dir, f"top{idx+1}_{fname_i}_vs_{fname_j}.png"), bbox_inches="tight")
        plt.close()

    logger.info(f"Top {top_k} SHAP interaction plots saved to {topk_dir}")

    # ======================================================
    # PERMUTATION FEATURE IMPORTANCE
    # ======================================================
    logger.info(f"Starting permutation importance analysis for predictor set '{key}'")

    # Compute baseline performance
    y_pred_base = model.predict(X_test_dense)
    r2_base = r2_score(y_test, y_pred_base)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))

    results = []
    n_repeats = 10

    # Compute baseline metrics
    y_pred = model.predict(X_test_dense)
    baseline_r2 = r2_score(y_test, y_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    for feature in predictors:
        r2_scores, rmse_scores = [], []

        for _ in range(n_repeats):
            X_permuted = X_test_dense.copy()
            feat_idx = predictors.index(feature)
            X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])

            y_pred_perm = model.predict(X_permuted)
            r2_scores.append(r2_score(y_test, y_pred_perm))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred_perm)))

        results.append({
            "feature": feature,
            "baseline_r2": baseline_r2,
            "mean_r2_after_permutation": np.mean(r2_scores),
            "mean_r2_drop": baseline_r2 - np.mean(r2_scores),
            "baseline_rmse": baseline_rmse,
            "mean_rmse_after_permutation": np.mean(rmse_scores),
            "mean_rmse_increase": np.mean(rmse_scores) - baseline_rmse,
        })

    # --- Group feature permutation ---
    # Define groups (example)
    feature_groups = {
        "ntl_features": ntl_features,
        "lc_features": lc_features,
        "geo_features": geo_features
    }

    for group_name, group_features in feature_groups.items():
        # Get indices of all features in this group
        group_indices = [predictors.index(f) for f in group_features if f in predictors]
        if not group_indices:
            logger.warning(f"Skipping group '{group_name}' — no matching features found.")
            continue

        r2_scores, rmse_scores = [], []

        for _ in range(n_repeats):
            X_permuted = X_test_dense.copy()

            # Permute all features in this group independently
            for idx in group_indices:
                X_permuted[:, idx] = np.random.permutation(X_permuted[:, idx])

            y_pred_perm = model.predict(X_permuted)
            r2_scores.append(r2_score(y_test, y_pred_perm))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred_perm)))

        results.append({
            "feature": group_name,
            # "type": "group",
            "baseline_r2": r2_base,
            "mean_r2_after_permutation": np.mean(r2_scores),
            "mean_r2_drop": r2_base - np.mean(r2_scores),
            "baseline_rmse": rmse_base,
            "mean_rmse_after_permutation": np.mean(rmse_scores),
            "mean_rmse_increase": np.mean(rmse_scores) - rmse_base,
        })


    # 2. Save results
    perm_results_df = pd.DataFrame(results)
    perm_folder = os.path.join(output_dir, "permutation_importance")
    os.makedirs(perm_folder, exist_ok=True)
    perm_path = os.path.join(perm_folder, f"permutation_importance_{key}.csv")
    perm_results_df.to_csv(perm_path, index=False)

    logger.info(f"Permutation importance results saved to {perm_path}")
    logger.info(perm_results_df)

    del X_test_dense, X_test_num_df, shap_values, interaction_values
    import gc; gc.collect()
'''