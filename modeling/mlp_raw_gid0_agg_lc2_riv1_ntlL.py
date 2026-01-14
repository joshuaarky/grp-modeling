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

# === Logging Setup ===
# model_id = "mlp_raw_gid0_agg_lc2_riv1_ntlL"
# model_id = "mlp_raw2_gid0_agg_lc2_riv1_ntlL"
model_id = "ann_adamw_raw2_gid0_agg_lc2_riv1_ntlL"

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
    'lccs_ag_101112_share', 'log_lccs_ag_20_share', 'log_lccs_ag_30_share', 'log_lccs_ag_40_share',
    'lccs_forest_50_share', 'log_lccs_forest_606162_share', 'log_lccs_forest_707172_share',
    'log_lccs_forest_808182_share', 'log_lccs_forest_90_share', 'log_lccs_forest_100_share',
    'log_lccs_forest_160_share', 'log_lccs_forest_170_share', 'log_lccs_grass_110_share',
    'log_lccs_grass_130_share', 'log_lccs_wet_180_share', 'log_lccs_urban_190_share',
    'log_lccs_shrub_120121122_share', 'log_lccs_sparse_140_share', 'log_lccs_sparse_150151152153_share',
    'log_lccs_bare_200201202_share', 'log_lccs_water_210_share', 'log_lccs_snow_220_share',
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
    "a": predictors_a,
    # "b": predictors_b,
    "c": predictors_c,
    # "d": predictors_d,
    "e": predictors_e,
    # "f": predictors_f,
    "g": predictors_g,
}

os.makedirs(f"/p/projects/impactee/Josh/thesis_analysis/optuna/optuna_{model_id}/", exist_ok=True)
for key, predictors in predictor_sets.items():
    logger = setup_logger(key)
    logger.info(f"Predictors for set {key}: {predictors}")

    n_trials = 100 # usually set to 100 or 200, but for testing purposes we set it to 1
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
        # l2_lambda
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
            # self.l2_lambda = l2_lambda

        def forward(self, x):
            return self.net(x)  # Return shape (batch, 1)

        # commenting out L2 penalty, as it is a double penalty when using weight_decay
        # def l2_penalty(self):
        #     return sum((param**2).sum() for name, param in self.named_parameters() if 'weight' in name)

    # Optuna objective function
    def objective(trial):
        # Hyperparameter search space
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_units = trial.suggest_int("n_units", 32, 256, step=32)
        use_batchnorm = trial.suggest_categorical('batchnorm', [True, False])
        # l2_lambda = trial.suggest_float('l2_lambda', 1e-6, 1e-2, log=True)
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
                # l2_lambda=l2_lambda
            )
            criterion = nn.MSELoss()
            # old Adam optimizer:
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            # new AdamW optimizer with weight decay:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
                    # loss = criterion(preds, yb) + model.l2_penalty() * l2_lambda
                    loss = criterion(preds, yb)
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

    logger.info("Training and evaluating final model with best hyperparameters")
    best_params = study.best_params
    logger.info(f"Best parameters: {best_params}")

    n_splits = 5
    n_repeats = 20  # default is 20; reduce to 1 for other testing

    # Lists to store results
    all_val_metrics = []
    all_test_metrics = []
    all_fold_country_mapping = []

    #####################################
    #### ====== GroupKFold CV ====== ####
    #####################################

    # Loop over repeats
    for rep in range(n_repeats):
        logger.info(f"=== CV Repeat {rep + 1}/{n_repeats} ===")
        
        # Shuffle groups differently for each repeat
        unique_groups = df_model['GID_0'].unique()
        rng = np.random.default_rng(seed=rep)
        rng.shuffle(unique_groups)

        gkf = GroupKFold(n_splits=5)  # 5-fold CV on GID_0 groups

        for fold_idx, (_, test_group_idx) in enumerate(gkf.split(unique_groups, groups=unique_groups)):
            test_groups = unique_groups[test_group_idx]
            test_df = df_model[df_model['GID_0'].isin(test_groups)]
            train_val_df = df_model[~df_model['GID_0'].isin(test_groups)]

            # Split train into train/val (by groups)
            gids_train, gids_val = train_test_split(
                train_val_df['GID_0'].unique(),
                test_size=0.2,
                random_state=rep  # ensures reproducible train/val split per repeat
            )
            val_df = train_val_df[train_val_df['GID_0'].isin(gids_val)]
            train_df = train_val_df[train_val_df['GID_0'].isin(gids_train)]

            # Preprocess
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), predictors),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['GID_1'])
            ])
            X_train_proc = preprocessor.fit_transform(train_df[predictors + ['GID_1']])
            X_val_proc   = preprocessor.transform(val_df[predictors + ['GID_1']])
            X_test_proc  = preprocessor.transform(test_df[predictors + ['GID_1']])

            y_train = train_df[target].values.reshape(-1, 1)
            y_val   = val_df[target].values.reshape(-1, 1)
            y_test  = test_df[target].values.reshape(-1, 1)

            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_train_proc.toarray()), torch.FloatTensor(y_train)),
                batch_size=best_params['batch_size'],
                shuffle=True,
                drop_last=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val_proc.toarray()), torch.FloatTensor(y_val)),
                batch_size=best_params['batch_size']
            )
            test_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_test_proc.toarray()), torch.FloatTensor(y_test)),
                batch_size=best_params['batch_size']
            )

            # Build model
            hidden_dims = [best_params['n_units']] * best_params['n_layers']
            model = MLP(
                input_dim=X_train_proc.shape[1],
                hidden_dims=hidden_dims,
                dropout=best_params['dropout_rate'],
                use_batchnorm=best_params['batchnorm'],
                # l2_lambda=best_params['l2_lambda']
            )
            criterion = nn.MSELoss()
            # optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'])


            # Train with simple early stopping
            best_val_loss = float('inf')
            best_state = None
            patience = 10
            min_delta = 1e-4
            epochs_no_improve = 0

            for epoch in range(200):
                model.train()
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    #loss = criterion(model(xb), yb) + model.l2_penalty() * best_params['l2_lambda']
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                val_preds, val_targets = [], []
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

            # Validation metrics
            model.eval()
            y_val_pred = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    y_val_pred.append(model(xb).cpu().numpy())
            y_val_pred = np.vstack(y_val_pred)

            # --- Back-transform predictions and true targets ---
            y_val_exp = np.exp(y_val)
            y_val_pred_exp = np.exp(y_val_pred)

            # --- Compute metrics on raw scale ---
            val_r2 = r2_score(y_val_exp, y_val_pred_exp)
            val_rmse = np.sqrt(mean_squared_error(y_val_exp, y_val_pred_exp))
            # val_metrics.append((val_r2, val_rmse))
            all_val_metrics.append({
                "repeat": rep,
                "fold": fold_idx,
                "r2": val_r2,
                "rmse": val_rmse
            })

            # update Oct 27: these are the old test metrics; we now compute test metrics on the exp-transformed values above
            # val_r2 = r2_score(y_val, y_val_pred)
            # val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            # val_metrics.append((val_r2, val_rmse))

            # Test metrics
            y_test_pred = []
            with torch.no_grad():
                for xb, _ in test_loader:
                    y_test_pred.append(model(xb).cpu().numpy())
            y_test_pred = np.vstack(y_test_pred)

            # --- Back-transform predictions and true targets ---
            y_test_exp = np.exp(y_test)
            y_test_pred_exp = np.exp(y_test_pred)

            # --- Compute metrics on raw scale ---
            test_r2 = r2_score(y_test_exp, y_test_pred_exp)
            test_rmse = np.sqrt(mean_squared_error(y_test_exp, y_test_pred_exp))
            # metrics.append((test_r2, test_rmse))
            all_test_metrics.append({
                "repeat": rep,
                "fold": fold_idx,
                "r2": test_r2,
                "rmse": test_rmse
            })

            # === Track countries in this fold ===
            all_fold_country_mapping.append({
                "repeat": rep,
                "fold": fold_idx,
                "train_GID_0": gids_train.tolist(),
                "val_GID_0": gids_val.tolist(),
                "test_GID_0": test_groups.tolist()
            })
            
            # update Oct 27: these are the old test metrics; we now compute test metrics on the exp-transformed values above
            # test_r2 = r2_score(y_test, y_test_pred)
            # test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            # metrics.append((test_r2, test_rmse))
    
    # Convert to DataFrames for analysis
    val_metrics_df = pd.DataFrame(all_val_metrics)
    test_metrics_df = pd.DataFrame(all_test_metrics)
    fold_countries_df = pd.DataFrame(all_fold_country_mapping)

    ### Comment out csv saving while running SHAP analysis, since I'm only doing n_repeats = 1 ###
    # Create directory if it doesn't exist
    metrics_dir = f"/p/projects/impactee/Josh/thesis_analysis/model_metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    # Save validation metrics
    val_metrics_df.to_csv(
        f"{metrics_dir}/{model_id}_{key}_val_metrics.csv",
        index=False
    )

    # Save test metrics
    test_metrics_df.to_csv(
        f"{metrics_dir}/{model_id}_{key}_test_metrics.csv",
        index=False
    )

    # Save fold-country mapping
    fold_countries_df.to_csv(
        f"{metrics_dir}/{model_id}_{key}_fold_countries.csv",
        index=False
    )

    # Report CV results
    r2_mean, r2_std = test_metrics_df['r2'].mean(), test_metrics_df['r2'].std()
    rmse_mean, rmse_std = test_metrics_df['rmse'].mean(), test_metrics_df['rmse'].std()
    logger.info(f"Test RMSE (CV): Mean = {rmse_mean:.4f}, Std = {rmse_std:.4f}")
    logger.info(f"Test R2 (CV): Mean = {r2_mean:.4f}, Std = {r2_std:.4f}")

    val_r2_mean, val_r2_std = val_metrics_df['r2'].mean(), val_metrics_df['r2'].std()
    val_rmse_mean, val_rmse_std = val_metrics_df['rmse'].mean(), val_metrics_df['rmse'].std()
    logger.info(f"Validation RMSE (CV): Mean = {val_rmse_mean:.4f}, Std = {val_rmse_std:.4f}")
    logger.info(f"Validation R2 (CV): Mean = {val_r2_mean:.4f}, Std = {val_r2_std:.4f}")


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

    # # === SHAP ANALYSIS ===
    # logger.info(f"Starting SHAP analysis for predictor set '{key}'")

    # import shap

    # # Put model in eval mode
    # model.eval()

    # # Update these as needed
    # base_path = "/p/projects/impactee/Josh/thesis_analysis/shap_feat_importance"
    # output_dir = os.path.join(base_path, f"{model_id}")
    # os.makedirs(output_dir, exist_ok=True)

    # # Subsample for SHAP speed
    # n_background = min(100, X_train_proc.shape[0])
    # n_test = min(500, X_test_proc.shape[0])

    # # Randomly choose subsets
    # background_idx = np.random.choice(X_train_proc.shape[0], n_background, replace=False)
    # test_idx = np.random.choice(X_test_proc.shape[0], n_test, replace=False)

    # background_data = torch.tensor(X_train_proc[background_idx].toarray(), dtype=torch.float32)
    # X_test_tensor = torch.tensor(X_test_proc[test_idx].toarray(), dtype=torch.float32)

    # # Wrap model for SHAP
    # explainer = shap.DeepExplainer(model, background_data)

    # # Compute SHAP values on test data
    # X_test_tensor = torch.tensor(X_test_proc.toarray(), dtype=torch.float32)
    # shap_values = explainer.shap_values(X_test_tensor)

    # # Convert to numpy array if needed
    # if isinstance(shap_values, list):
    #     shap_values = shap_values[0]  # Take the single output case

    # # Get all feature names after preprocessing
    # feature_names = preprocessor.get_feature_names_out()

    # # Convert to DataFrame
    # # Remove trailing output dimension if present
    # if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
    #     shap_values = shap_values.squeeze(-1)
    # shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # # Drop categorical (GID_1) columns consistently
    # mask = ~pd.Series(feature_names).str.startswith('cat__')
    # shap_values = shap_values[:, mask.values]
    # feature_names = np.array(feature_names)[mask.values]
    # shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # logger.info(f"After filtering: shap_values shape = {shap_values.shape}, shap_df shape = {shap_df.shape}")
    # logger.info(f"Feature names: {list(feature_names)[:10]} ...")

    # # Plot SHAP summary
    # summary_folder = os.path.join(output_dir, "shap_summary_plots")
    # os.makedirs(summary_folder, exist_ok=True)
    # summary_path = os.path.join(summary_folder, f"shap_summary_{key}.png")

    # plt.figure(figsize=(10, 8))
    # shap.summary_plot(
    #     shap_values,
    #     shap_df,
    #     feature_names=feature_names,
    #     show=False,
    #     color_bar=True,
    # )
    # plt.tight_layout()
    # plt.savefig(summary_path, bbox_inches="tight", dpi=300)
    # plt.close()
    # logger.info(f"SHAP summary plot saved to {summary_path}")
