import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
import os
import copy
import logging
from datetime import datetime
from shapely.geometry import shape
import matplotlib.pyplot as plt
import shap
import json

# === Logging Setup ===
# model_id = "mlp_raw_gid0_agg_lc3_riv1_ntlL"
model_id = "mlp_raw2_gid0_agg_lc3_riv1_ntlL"
# model_id = "ann_adamw_raw2_gid0_agg_lc3_riv1_ntlL"

def setup_logger(key):
    os.makedirs(f"logs/{model_id}", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/{model_id}/{model_id}_{key}_{timestamp}.log"

    logger = logging.getLogger(key)
    logger.setLevel(logging.INFO)

    # Remove previous handlers if they exist
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

# === Configuration ===

# --- Define feature groups ---
ntl_features = [
    "log_NTL_mean", "NTL_std"
]

lc_features = [
    'lccs_ag_ipcc_share', 'lccs_forest_ipcc_share', 'log_lccs_grass_ipcc_share',
    'log_lccs_wet_ipcc_share', 'log_lccs_urban_ipcc_share', 'log_lccs_shrub_ipcc_share',
    'log_lccs_sparse_ipcc_share', 'log_lccs_bare_ipcc_share', 'log_lccs_water_ipcc_share',
]

geo_features = [
    "log_avg_coast_dist", "log_avg_tri", "log_avg_lake_dist", "log_major_river_dist_mean",
    "log_std_coast_dist", "std_tri", "log_std_lake_dist", "log_major_river_dist_std"
]

gdp_features = [
    # "gdp_pc_lcu",  # optional alternative
    "log_gdp_pc_2015_usd"
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
    # "a": predictors_a,
    # "b": predictors_b,
    # "c": predictors_c,
    # "d": predictors_d,
    "e": predictors_e,
    # "f": predictors_f,
    # "g": predictors_g,
}

os.makedirs(f"/p/projects/impactee/Josh/thesis_analysis/optuna/optuna_{model_id}/", exist_ok=True)
for key, predictors in predictor_sets.items():
    logger = setup_logger(key)
    logger.info(f"Predictors for set {key}: {predictors}")

    n_trials = 1 # usually set to 100 or 200, but for testing purposes we set it to 1
    optuna_storage = f"sqlite:////p/projects/impactee/Josh/thesis_analysis/optuna/optuna_{model_id}/optuna_mlp_{key}.db"

    # create target variable column
    # df['log_ratio_gdppc_grppc'] = df['log_grp_pc_lcu'] - df['log_gdp_pc_lcu']
    # target = 'log_ratio_gdppc_grppc'
    target = 'log_grp_pc_lcu2015_usd'

    logger.info(f"Target: {target}")

    # Drop missing
    df_model = df.dropna(subset=[target] + predictors)
    logger.info(f"Number of rows after filtering: {len(df_model)}")

    X = df_model[predictors]
    y = df_model[target]

    gid = df_model['GID_1']
    year = df_model['year']

    # === Preprocessing ===
    # Define a flexible MLP model
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout, use_batchnorm,
        l2_lambda
        ):
            super().__init__()
            layers = []
            in_dim = input_dim

            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = h_dim

            layers.append(nn.Linear(in_dim, 1))  # Output layer
            self.net = nn.Sequential(*layers)
            self.l2_lambda = l2_lambda

        def forward(self, x):
            return self.net(x)  # Return shape (batch, 1)

        # commenting out L2 penalty, as it is a double penalty when using weight_decay
        def l2_penalty(self):
            return sum((param**2).sum() for name, param in self.named_parameters() if 'weight' in name)

    # Optuna objective function
    def objective(trial):
        # Hyperparameter search space
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_units = trial.suggest_int("n_units", 32, 256, step=32)
        use_batchnorm = trial.suggest_categorical('batchnorm', [True, False])
        l2_lambda = trial.suggest_float('l2_lambda', 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

        hidden_dims = [hidden_units] * n_layers

        # Prepare GroupKFold on unique GID_1 groups
        gkf = GroupKFold(n_splits=5)

        val_rmse_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df_model, groups=df_model['GID_0'])):
            train_df = df_model.iloc[train_idx]
            val_df   = df_model.iloc[val_idx]

            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), predictors),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['GID_1'])
            ])

            X_train_proc = preprocessor.fit_transform(train_df[predictors + ['GID_1']])
            X_val_proc   = preprocessor.transform(val_df[predictors + ['GID_1']])

            y_train = train_df[target].values.reshape(-1, 1)
            y_val   = val_df[target].values.reshape(-1, 1)

            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_train_proc.toarray()), torch.FloatTensor(y_train)),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )

            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val_proc.toarray()), torch.FloatTensor(y_val)),
                batch_size=batch_size
            )

            # Create a fresh model for each fold
            model = MLP(
                input_dim=X_train_proc.shape[1],
                hidden_dims=hidden_dims,
                dropout=dropout_rate,
                use_batchnorm=use_batchnorm,
                l2_lambda=l2_lambda
            )
            criterion = nn.MSELoss()
            # old Adam optimizer:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            # new AdamW optimizer with weight decay:
            # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Basic early stopping
            best_val_loss = float('inf')
            best_state = None
            patience = 10
            min_delta = 1e-4
            epochs_no_improve = 0

            for epoch in range(200):
                model.train()
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    preds = model(xb)
                    loss = criterion(preds, yb) + model.l2_penalty() * l2_lambda
                    # loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                val_preds = []
                val_targets = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        preds = model(xb)
                        val_preds.append(preds.cpu())
                        val_targets.append(yb.cpu())
                val_preds = torch.cat(val_preds)
                val_targets = torch.cat(val_targets)
                val_loss = criterion(val_preds, val_targets).item()

                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break

            if best_state:
                model.load_state_dict(best_state)

            # Compute RMSE for this fold
            model.eval()
            final_preds = []
            final_targets = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    final_preds.append(model(xb))
                    final_targets.append(yb)
            final_preds = torch.cat(final_preds).numpy()
            final_targets = torch.cat(final_targets).numpy()

            fold_rmse = np.sqrt(mean_squared_error(final_targets, final_preds))
            val_rmse_scores.append(fold_rmse)

        # Return average RMSE over all folds
        logger.info(f'Average RMSE across folds: {np.mean(val_rmse_scores):.4f}')
        return np.mean(val_rmse_scores)
    
    # === Run or Resume Optuna Study ===
    logger.info("Starting Optuna study for hyperparameter optimization")
    study_name = f"mlp_study_{key}"

    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=optuna_storage,
        load_if_exists=True
    )

    study.optimize(objective, n_trials=n_trials)

    ################################################
    ### === Train and Evaluate Multiple Runs === ###
    ################################################

    # logger.info("Training and evaluating final model with best hyperparameters")
    best_params = study.best_params
    logger.info(f"Best parameters: {best_params}")

    # n_splits = 5
    # n_repeats = 20  # default is 20; reduce to 1 for other testing

    # # Lists to store results
    # all_val_metrics = []
    # all_test_metrics = []
    # all_fold_country_mapping = []
    # all_region_metrics = []

    #####################################
    #### ====== GroupKFold CV ====== ####
    #####################################

    # # Loop over repeats
    # for rep in range(n_repeats):
    #     logger.info(f"=== CV Repeat {rep + 1}/{n_repeats} ===")
        
    #     # Shuffle groups differently for each repeat
    #     unique_groups = df_model['GID_0'].unique()
    #     rng = np.random.default_rng(seed=rep)
    #     rng.shuffle(unique_groups)

    #     gkf = GroupKFold(n_splits=5)  # 5-fold CV on GID_0 groups

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

    #         # Preprocess
    #         preprocessor = ColumnTransformer([
    #             ('num', StandardScaler(), predictors),
    #             ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['GID_1'])
    #         ])
    #         X_train_proc = preprocessor.fit_transform(train_df[predictors + ['GID_1']])
    #         X_val_proc   = preprocessor.transform(val_df[predictors + ['GID_1']])
    #         X_test_proc  = preprocessor.transform(test_df[predictors + ['GID_1']])

    #         y_train = train_df[target].values.reshape(-1, 1)
    #         y_val   = val_df[target].values.reshape(-1, 1)
    #         y_test  = test_df[target].values.reshape(-1, 1)

    #         train_loader = DataLoader(
    #             TensorDataset(torch.FloatTensor(X_train_proc.toarray()), torch.FloatTensor(y_train)),
    #             batch_size=best_params['batch_size'],
    #             shuffle=True,
    #             drop_last=True
    #         )
    #         val_loader = DataLoader(
    #             TensorDataset(torch.FloatTensor(X_val_proc.toarray()), torch.FloatTensor(y_val)),
    #             batch_size=best_params['batch_size']
    #         )
    #         test_loader = DataLoader(
    #             TensorDataset(torch.FloatTensor(X_test_proc.toarray()), torch.FloatTensor(y_test)),
    #             batch_size=best_params['batch_size']
    #         )

    #         # Build model
    #         hidden_dims = [best_params['n_units']] * best_params['n_layers']
    #         model = MLP(
    #             input_dim=X_train_proc.shape[1],
    #             hidden_dims=hidden_dims,
    #             dropout=best_params['dropout_rate'],
    #             use_batchnorm=best_params['batchnorm'],
    #             l2_lambda=best_params['l2_lambda']
    #         )
    #         criterion = nn.MSELoss()
    #         optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    #         # optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'])


    #         # Train with simple early stopping
    #         best_val_loss = float('inf')
    #         best_state = None
    #         patience = 10
    #         min_delta = 1e-4
    #         epochs_no_improve = 0

    #         for epoch in range(200):
    #             model.train()
    #             for xb, yb in train_loader:
    #                 optimizer.zero_grad()
    #                 loss = criterion(model(xb), yb) + model.l2_penalty() * best_params['l2_lambda']
    #                 # loss = criterion(model(xb), yb)
    #                 loss.backward()
    #                 optimizer.step()

    #             # Validation
    #             model.eval()
    #             val_preds, val_targets = [], []
    #             with torch.no_grad():
    #                 for xb, yb in val_loader:
    #                     preds = model(xb)
    #                     val_preds.append(preds.cpu())
    #                     val_targets.append(yb.cpu())
    #             val_preds = torch.cat(val_preds)
    #             val_targets = torch.cat(val_targets)
    #             val_loss = criterion(val_preds, val_targets).item()

    #             if best_val_loss - val_loss > min_delta:
    #                 best_val_loss = val_loss
    #                 best_state = copy.deepcopy(model.state_dict())
    #                 epochs_no_improve = 0
    #             else:
    #                 epochs_no_improve += 1

    #             if epochs_no_improve >= patience:
    #                 break

    #         if best_state:
    #             model.load_state_dict(best_state)

    #         # Validation metrics
    #         model.eval()
    #         y_val_pred = []
    #         with torch.no_grad():
    #             for xb, _ in val_loader:
    #                 y_val_pred.append(model(xb).cpu().numpy())
    #         y_val_pred = np.vstack(y_val_pred)

    #         # --- Back-transform predictions and true targets ---
    #         y_val_exp = np.exp(y_val)
    #         y_val_pred_exp = np.exp(y_val_pred)

    #         # --- Compute metrics on raw scale ---
    #         val_r2 = r2_score(y_val_exp, y_val_pred_exp)
    #         val_rmse = np.sqrt(mean_squared_error(y_val_exp, y_val_pred_exp))
    #         # val_metrics.append((val_r2, val_rmse))
    #         all_val_metrics.append({
    #             "repeat": rep,
    #             "fold": fold_idx,
    #             "r2": val_r2,
    #             "rmse": val_rmse
    #         })

    #         # update Oct 27: these are the old test metrics; we now compute test metrics on the exp-transformed values above
    #         # val_r2 = r2_score(y_val, y_val_pred)
    #         # val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    #         # val_metrics.append((val_r2, val_rmse))

    #         # Test metrics
    #         y_test_pred = []
    #         with torch.no_grad():
    #             for xb, _ in test_loader:
    #                 y_test_pred.append(model(xb).cpu().numpy())
    #         y_test_pred = np.vstack(y_test_pred)

    #         # --- Back-transform predictions and true targets ---
    #         y_test_exp = np.exp(y_test)
    #         y_test_pred_exp = np.exp(y_test_pred)

    #         # --- Compute metrics on raw scale ---
    #         test_r2 = r2_score(y_test_exp, y_test_pred_exp)
    #         test_rmse = np.sqrt(mean_squared_error(y_test_exp, y_test_pred_exp))
    #         # metrics.append((test_r2, test_rmse))
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

    #         # === Per-region metrics (ALSO on raw scale) ===
    #         test_df = test_df.copy()
    #         test_df["y_pred_raw"] = np.exp(y_test_pred)       # store raw preds
    #         test_df["y_true_raw"] = np.exp(test_df[target])    # store raw truths

    #         region_metrics = []

    #         for region, group_df in test_df.groupby("GID_1"):

    #             y_true_region = group_df["y_true_raw"].values
    #             y_pred_region = group_df["y_pred_raw"].values

    #             # Compute region-level metrics on raw (exp) scale
    #             r2 = r2_score(y_true_region, y_pred_region) if len(y_true_region) > 1 else np.nan
    #             rmse = np.sqrt(mean_squared_error(y_true_region, y_pred_region))
    #             n_obs = len(group_df)

    #             region_metrics.append({
    #                 "GID_1": region,
    #                 "repeat": rep,
    #                 "fold": fold_idx,
    #                 "n_obs": n_obs,
    #                 "r2": r2,
    #                 "rmse": rmse,

    #                 # Store region-level RAW values as JSON
    #                 "y_true_values": json.dumps(y_true_region.tolist()),
    #                 "y_pred_values": json.dumps(y_pred_region.tolist())
    #             })

    #         region_metrics_df = pd.DataFrame(region_metrics)
    #         all_region_metrics.append(region_metrics_df)
            
    #         # update Oct 27: these are the old test metrics; we now compute test metrics on the exp-transformed values above
    #         # test_r2 = r2_score(y_test, y_test_pred)
    #         # test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    #         # metrics.append((test_r2, test_rmse))
    
    # # Convert to DataFrames for analysis
    # val_metrics_df = pd.DataFrame(all_val_metrics)
    # test_metrics_df = pd.DataFrame(all_test_metrics)
    # fold_countries_df = pd.DataFrame(all_fold_country_mapping)
    # all_region_metrics_df = pd.concat(all_region_metrics, ignore_index=True)

    # ### Comment out csv saving while running SHAP analysis, since I'm only doing n_repeats = 1 ###
    # # Create directory if it doesn't exist
    # metrics_dir = f"/p/projects/impactee/Josh/thesis_analysis/model_metrics"
    # os.makedirs(metrics_dir, exist_ok=True)

    # # Save region metrics
    # all_region_metrics_df.to_csv(
    #     f"{metrics_dir}/{model_id}_{key}_region_metrics.csv",
    #     index=False
    # )
    
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

    # # Report CV results
    # r2_mean, r2_std = test_metrics_df['r2'].mean(), test_metrics_df['r2'].std()
    # rmse_mean, rmse_std = test_metrics_df['rmse'].mean(), test_metrics_df['rmse'].std()
    # logger.info(f"Test RMSE (CV): Mean = {rmse_mean:.4f}, Std = {rmse_std:.4f}")
    # logger.info(f"Test R2 (CV): Mean = {r2_mean:.4f}, Std = {r2_std:.4f}")

    # val_r2_mean, val_r2_std = val_metrics_df['r2'].mean(), val_metrics_df['r2'].std()
    # val_rmse_mean, val_rmse_std = val_metrics_df['rmse'].mean(), val_metrics_df['rmse'].std()
    # logger.info(f"Validation RMSE (CV): Mean = {val_rmse_mean:.4f}, Std = {val_rmse_std:.4f}")
    # logger.info(f"Validation R2 (CV): Mean = {val_r2_mean:.4f}, Std = {val_r2_std:.4f}")

    #############################
    ### ====== LOCO CV ====== ###
    #############################

    # # === Leave-One-Country-Out CV ===
    # unique_countries = df_model['GID_0'].unique()

    # # === Initialize list to store subnational predictions ===
    # all_subnational_results = []

    # for rep in range(n_repeats):
    #     logger.info(f"=== LOCO Repeat {rep + 1}/{n_repeats} ===")

    #     # shuffle countries each repeat
    #     rng = np.random.default_rng(seed=rep)
    #     countries_shuffled = unique_countries.copy()
    #     rng.shuffle(countries_shuffled)

    #     for fold_idx, test_country in enumerate(countries_shuffled):
            
    #         logger.info(f"=== LOCO Fold {fold_idx + 1}/{len(countries_shuffled)}: "
    #                     f"Test country = {test_country} ===")

    #         # Test set = one country
    #         test_df = df_model[df_model['GID_0'] == test_country]

    #         # Remaining countries for train + val
    #         remaining = df_model[df_model['GID_0'] != test_country]
    #         remaining_countries = remaining['GID_0'].unique()

    #         # Split remaining into train/val groups
    #         gids_train, gids_val = train_test_split(
    #             remaining_countries,
    #             test_size=0.2,
    #             random_state=rep  # stable per repeat
    #         )
    #         train_df = remaining[remaining['GID_0'].isin(gids_train)]
    #         val_df   = remaining[remaining['GID_0'].isin(gids_val)]

    #         # === Preprocessing ===
    #         preprocessor = ColumnTransformer([
    #             ('num', StandardScaler(), predictors),
    #             ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['GID_1'])
    #         ])
    #         X_train_proc = preprocessor.fit_transform(train_df[predictors + ['GID_1']])
    #         X_val_proc   = preprocessor.transform(val_df[predictors + ['GID_1']])
    #         X_test_proc  = preprocessor.transform(test_df[predictors + ['GID_1']])

    #         y_train = train_df[target].values.reshape(-1, 1)
    #         y_val   = val_df[target].values.reshape(-1, 1)
    #         y_test  = test_df[target].values.reshape(-1, 1)

    #         train_loader = DataLoader(
    #             TensorDataset(torch.FloatTensor(X_train_proc.toarray()), torch.FloatTensor(y_train)),
    #             batch_size=best_params['batch_size'], shuffle=True, drop_last=True
    #         )
    #         val_loader = DataLoader(
    #             TensorDataset(torch.FloatTensor(X_val_proc.toarray()), torch.FloatTensor(y_val)),
    #             batch_size=best_params['batch_size']
    #         )
    #         test_loader = DataLoader(
    #             TensorDataset(torch.FloatTensor(X_test_proc.toarray()), torch.FloatTensor(y_test)),
    #             batch_size=best_params['batch_size']
    #         )

    #         # === Build + train MLP ===
    #         hidden_dims = [best_params['n_units']] * best_params['n_layers']
    #         model = MLP(
    #             input_dim=X_train_proc.shape[1],
    #             hidden_dims=hidden_dims,
    #             dropout=best_params['dropout_rate'],
    #             use_batchnorm=best_params['batchnorm'],
    #             l2_lambda=best_params['l2_lambda']
    #         )
    #         criterion = nn.MSELoss()
    #         optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

    #         # Early stopping
    #         best_val_loss = float('inf')
    #         best_state = None
    #         patience = 10
    #         min_delta = 1e-4
    #         epochs_no_improve = 0

    #         for epoch in range(200):
    #             model.train()
    #             for xb, yb in train_loader:
    #                 optimizer.zero_grad()
    #                 loss = criterion(model(xb), yb) + model.l2_penalty() * best_params['l2_lambda']
    #                 loss.backward()
    #                 optimizer.step()

    #             model.eval()
    #             val_preds, val_targets = [], []
    #             with torch.no_grad():
    #                 for xb, yb in val_loader:
    #                     val_preds.append(model(xb))
    #                     val_targets.append(yb)

    #             val_preds = torch.cat(val_preds)
    #             val_targets = torch.cat(val_targets)
    #             val_loss = criterion(val_preds, val_targets).item()

    #             if best_val_loss - val_loss > min_delta:
    #                 best_val_loss = val_loss
    #                 best_state = copy.deepcopy(model.state_dict())
    #                 epochs_no_improve = 0
    #             else:
    #                 epochs_no_improve += 1
                
    #             if epochs_no_improve >= patience:
    #                 break

    #         if best_state:
    #             model.load_state_dict(best_state)

    #         # === Validation metrics ===
    #         y_val_pred = []
    #         with torch.no_grad():
    #             for xb, _ in val_loader:
    #                 y_val_pred.append(model(xb).cpu().numpy())
    #         y_val_pred = np.vstack(y_val_pred)

    #         y_val_exp = np.exp(y_val)
    #         y_val_pred_exp = np.exp(y_val_pred)

    #         all_val_metrics.append({
    #             "repeat": rep,
    #             "fold": fold_idx,
    #             "test_country": test_country,
    #             "r2": r2_score(y_val_exp, y_val_pred_exp),
    #             "rmse": np.sqrt(mean_squared_error(y_val_exp, y_val_pred_exp))
    #         })

    #         # === Test metrics ===
    #         y_test_pred = []
    #         with torch.no_grad():
    #             for xb, _ in test_loader:
    #                 y_test_pred.append(model(xb).cpu().numpy())
    #         y_test_pred = np.vstack(y_test_pred)

    #         y_test_exp = np.exp(y_test)
    #         y_test_pred_exp = np.exp(y_test_pred)

    #         # --- Append subnational-level predictions ---
    #         subnational_df = pd.DataFrame({
    #             "repeat": rep,
    #             "fold": fold_idx,
    #             "GID_0": test_df["GID_0"].values,
    #             "GID_1": test_df["GID_1"].values,
    #             "y_true": y_test_exp.flatten(),
    #             "y_pred": y_test_pred_exp.flatten()
    #         })
    #         all_subnational_results.append(subnational_df)

    #         all_test_metrics.append({
    #             "repeat": rep,
    #             "fold": fold_idx,
    #             "test_country": test_country,
    #             "r2": r2_score(y_test_exp, y_test_pred_exp),
    #             "rmse": np.sqrt(mean_squared_error(y_test_exp, y_test_pred_exp))
    #         })

    #         # === Save the actual countries in each role ===
    #         all_fold_country_mapping.append({
    #             "repeat": rep,
    #             "fold": fold_idx,
    #             "test_GID_0": [test_country],
    #             "train_GID_0": gids_train.tolist(),
    #             "val_GID_0": gids_val.tolist(),
    #         })
    
    # # Convert to DataFrames for analysis
    # val_metrics_df = pd.DataFrame(all_val_metrics)
    # test_metrics_df = pd.DataFrame(all_test_metrics)
    # fold_countries_df = pd.DataFrame(all_fold_country_mapping)

    # ### Comment out csv saving while running SHAP analysis ###
    # # Create directory if it doesn't exist
    # metrics_dir = f"/p/projects/impactee/Josh/thesis_analysis/model_metrics_loco/{model_id}"
    # os.makedirs(metrics_dir, exist_ok=True)

    # # Combine all subnational predictions and save
    # all_subnational_df = pd.concat(all_subnational_results, ignore_index=True)
    # subnational_csv_path = f"{metrics_dir}/{model_id}_{key}_loco_subnational_predictions.csv"
    # all_subnational_df.to_csv(subnational_csv_path, index=False)
    # logger.info(f"Saved subnational predictions to: {subnational_csv_path}")

    # # Save validation metrics
    # val_metrics_df.to_csv(
    #     f"{metrics_dir}/{model_id}_{key}_loco_val_metrics.csv",
    #     index=False
    # )

    # # Save test metrics
    # test_metrics_df.to_csv(
    #     f"{metrics_dir}/{model_id}_{key}_loco_test_metrics.csv",
    #     index=False
    # )

    # # Save fold-country mapping
    # fold_countries_df.to_csv(
    #     f"{metrics_dir}/{model_id}_{key}_loco_fold_countries.csv",
    #     index=False
    # )

    #############################
    ### === SHAP ANALYSIS === ###
    #############################

    # === SHAP ANALYSIS USING DeepExplainer ===
    logger.info(f"Starting SHAP analysis for predictor set '{key}'")

    # === TRAIN FINAL MODEL ON FULL DATA ===
    final_preprocessor = ColumnTransformer([
        ('num', StandardScaler(), predictors),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['GID_1'])
    ])
    X_all_proc = final_preprocessor.fit_transform(df_model[predictors + ['GID_1']])
    y_all = df_model[target].values.reshape(-1,1)

    final_model = MLP(
        input_dim=X_all_proc.shape[1],
        hidden_dims=[best_params['n_units']] * best_params['n_layers'],
        dropout=best_params['dropout_rate'],
        use_batchnorm=best_params['batchnorm'],
        l2_lambda=best_params['l2_lambda']
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])

    # Simple training loop (could include early stopping if desired)
    X_tensor = torch.tensor(X_all_proc.toarray(), dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.float32)

    for epoch in range(200):
        final_model.train()
        optimizer.zero_grad()
        loss = criterion(final_model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    # Now pass this fully trained model to SHAP
    model = final_model  # overwrite old 'model' so your existing SHAP code works

    model.eval()

    # Save output location
    shap_dir = f"/p/projects/impactee/Josh/thesis_analysis/shap_feat_importance/{model_id}/"
    os.makedirs(shap_dir, exist_ok=True)

    # --- Prepare data for SHAP ---
    # Note: X_all_proc comes from final_preprocessor.fit_transform(df_model)
    X_all_dense = X_all_proc.toarray() if hasattr(X_all_proc, "toarray") else X_all_proc

    # Subsample for speed (DeepExplainer is sensitive to memory)
    n_background = min(200, X_all_dense.shape[0])
    n_test       = min(800, X_all_dense.shape[0])

    rng = np.random.default_rng(42)
    bg_idx   = rng.choice(X_all_dense.shape[0], n_background, replace=False)
    test_idx = rng.choice(X_all_dense.shape[0], n_test, replace=False)

    background_data = torch.tensor(X_all_dense[bg_idx], dtype=torch.float32)
    X_test_tensor   = torch.tensor(X_all_dense[test_idx], dtype=torch.float32)

    # --- Build DeepExplainer ---
    explainer = shap.DeepExplainer(model, background_data)

    logger.info("Computing SHAP values with DeepExplainer...")
    shap_values = explainer.shap_values(X_test_tensor)

    # --- Convert shape (1, N, F) → (N, F) if needed ---
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)

    logger.info(f"SHAP values shape after squeeze: {shap_values.shape}")

    # --- Feature names from final preprocessing ---
    feature_names_proc = (
        final_preprocessor.named_transformers_["num"].get_feature_names_out(predictors).tolist()
        + final_preprocessor.named_transformers_["cat"].get_feature_names_out(["GID_1"]).tolist()
    )

    # log a sample of the first 10 feature names
    logger.info(f"Processed feature names (sample): {feature_names_proc[:10]}")

    # Convert to DataFrame
    shap_df = pd.DataFrame(shap_values, columns=feature_names_proc)

    # Optionally drop OneHot columns for cleaner plots
    mask = ~pd.Series(feature_names_proc).str.startswith("GID_1")

    shap_values_filtered = shap_values[:, mask.values]
    feature_names_filtered = np.array(feature_names_proc)[mask.values]

    shap_df_filtered = pd.DataFrame(shap_values_filtered, columns=feature_names_filtered)

    logger.info(f"Filtered SHAP shape = {shap_df_filtered.shape}")

    # === Plot SHAP summary ===
    summary_dir = os.path.join(shap_dir, "shap_summary_plots")
    os.makedirs(summary_dir, exist_ok=True)

    summary_path = os.path.join(summary_dir, f"shap_summary_{key}.png")

    # --- Convert X_test_tensor to NumPy first ---
    X_test_np = X_test_tensor.numpy()  # shape (n_test, n_features)

    # --- Apply mask to remove categorical features ---
    X_test_filtered = X_test_np[:, mask.values]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_filtered,
        X_test_filtered,
        feature_names=feature_names_filtered,
        show=False,
        color_bar=True,
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"SHAP summary plot saved to {summary_path}")

    # ---------------------------------------------------
    # === SHAP INTERACTION (DEPENDENCE) PLOTS FOR ANN ===
    # ---------------------------------------------------

    interaction_dir = os.path.join(shap_dir, f"shap_interaction_plots_{key}")
    os.makedirs(interaction_dir, exist_ok=True)

    top_k = 6  # keep small for interpretability

    # Compute mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values_filtered).mean(axis=0)

    # Get indices of top_k features
    top_idx = np.argsort(mean_abs_shap)[::-1]  # descending order

    top_features = feature_names_filtered[top_idx[:top_k]]

    logger.info(f"Computing SHAP interaction plots for top features: {top_features}")

    for i, fname in enumerate(top_features):
        for j, interact_with in enumerate(top_features):
            if i >= j:
                continue  # avoid duplicates and self-interactions

            save_path = os.path.join(
                interaction_dir,
                f"interaction_{fname}_x_{interact_with}.png"
            )

            try:
                plt.figure(figsize=(6, 5))
                shap.dependence_plot(
                    fname,
                    shap_values_filtered,
                    X_test_filtered,
                    interaction_index=interact_with,
                    feature_names=feature_names_filtered,
                    show=False
                )
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()

                logger.info(f"Saved interaction plot: {fname} × {interact_with}")

            except Exception as e:
                logger.warning(f"Failed interaction {fname} × {interact_with}: {e}")
    
    def interaction_strength(shap_vals, x_vals):
        return np.std(shap_vals) * np.std(x_vals)

    interaction_scores = []

    for i, fi in enumerate(top_features):
        for j, fj in enumerate(top_features):
            if i >= j:
                continue
            score = interaction_strength(
                shap_values_filtered[:, top_idx[i]],
                X_test_filtered[:, top_idx[j]]
            )
            interaction_scores.append((fi, fj, score))

    interaction_df = (
        pd.DataFrame(interaction_scores, columns=["feature_1", "feature_2", "score"])
        .sort_values("score", ascending=False)
    )

    logger.info("Top ANN interaction pairs:")
    logger.info(interaction_df.head(10))
