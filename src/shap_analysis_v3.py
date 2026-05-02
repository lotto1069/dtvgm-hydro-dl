# Hydro-DL: Publication-Quality Driver Analysis (Fixed)
import os
import re
import pandas as pd
import numpy as np
import rasterio
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.metrics import r2_score

import warnings

warnings.filterwarnings('ignore')

# --- 1. Style Configuration ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

VAR_NAMES = {
    'prec': 'Precipitation',
    'prec_lag_1': 'Precipitation (Lag-1)',
    'prec_lag_3': 'Precipitation (Lag-3)',
    'temp': 'Temperature',
    'temp_lag_1': 'Temperature (Lag-1)',
    'rhu': 'Relative Humidity',
    'petpm': 'Potential Evap. (PET)',
    'pres': 'Atmospheric Pressure',
    'wind': 'Wind Speed',
    'snow': 'Snow Cover Fraction',
    'ndvi': 'NDVI',
    'ndvi_lag_1': 'NDVI (Lag-1)'
}

train_paths = {
    'ndvi': r"G:\train\MODIS_NDVI_daily_interpolated-1984-2013",
    'temp': r"G:\train\meteorology_1984_2013\xiinghe_001deg_tmpmean_flattened_tif",
    'rhu': r"G:\train\meteorology_1984_2013\xiinghe_001deg_rhu_flattened_tif",
    'petpm': r"G:\train\meteorology_1984_2013\xiinghe_001deg_petPM_flattened_tif",
    'prec': r"G:\train\meteorology_1984_2013\xiinghe_001deg_prec_flattened_tif",
    'pres': r"G:\train\meteorology_1984_2013\xiinghe_001deg_pres_flattened_tif",
    'wind': r"G:\train\meteorology_1984_2013\xiinghe_001deg_windflat_tif",
    'snow': r"G:\SNOW"
}
flow_train_path = r"G:\train\runoff1984-2013.xlsx"

test_paths = {
    'ndvi': r"G:\test\MODIS_NDVI_daily_interpolated",
    'temp': r"G:\test\xiinghe_001deg_tmpmean_flattened_tif",
    'rhu': r"G:\test\xiinghe_001deg_rhu_flattened_tif",
    'petpm': r"G:\test\xiinghe_001deg_petPM_flattened_tif",
    'prec': r"G:\test\xiinghe_001deg_prec_flattened_tif",
    'pres': r"G:\test\xiinghe_001deg_pres_flattened_tif",
    'wind': r"G:\test\xiinghe_001deg_windflat_tif",
    'snow': r"G:\SNOW"
}
# --- 3. Data Loading ---
def parse_date_from_filename(filename):
    match = re.search(r'(\d{4})(\d{2})(\d{2})|(\d{4})_(\d{2})_(\d{2})', os.path.basename(filename))
    if match:
        parts = [p for p in match.groups() if p is not None]
        year, month, day = map(int, parts[-3:])
        try:
            return datetime(year, month, day)
        except:
            return None
    return None
def load_dataset(data_folders, flow_path, start_date, end_date, dataset_name):
    print(f"\n--- Loading {dataset_name} ---")
    data = {}
    for var, folder_path in data_folders.items():
        if not os.path.isdir(folder_path): continue
        var_data = []
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
        # Simply limit files for speed during debug if needed, remove slice for full run
        for file in tqdm(files, desc=f"Reading {var}", leave=False):
            date = parse_date_from_filename(file)
            if date and (pd.Timestamp(start_date) <= date <= pd.Timestamp(end_date)):
                try:
                    with rasterio.open(file) as src:
                        array = src.read(1).astype(float)
                        if src.nodata is not None: array[array == src.nodata] = np.nan

                        if np.all(np.isnan(array)):
                            val = np.nan
                        elif var == 'snow' and np.nanmax(array) > 1:
                            val = np.sum((array == 100) | (array == 200)) / np.sum(np.isfinite(array)) if np.any(
                                np.isfinite(array)) else 0
                        else:
                            val = np.nanmean(array)
                        var_data.append({'date': date, var: val})
                except:
                    pass

        if var_data:
            df = pd.DataFrame(var_data).set_index('date')
            if df.index.has_duplicates: df = df.groupby(df.index).mean()
            data[var] = df

    if not data: return pd.DataFrame()

    df_merged = pd.concat(data.values(), axis=1).sort_index()
    flow_df = pd.read_excel(flow_path)
    flow_df.columns = ['date', 'flow']
    flow_df['date'] = pd.to_datetime(flow_df['date'], errors='coerce')
    flow_df = flow_df.dropna(subset=['date']).set_index('date')
    if flow_df.index.has_duplicates: flow_df = flow_df.groupby(flow_df.index).mean()

    df_final = df_merged.join(flow_df, how='inner').loc[start_date:end_date]
    df_filled = df_final.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')

    # Lags
    df_filled['prec_lag_1'] = df_filled['prec'].shift(1)
    df_filled['prec_lag_3'] = df_filled['prec'].shift(3)
    df_filled['temp_lag_1'] = df_filled['temp'].shift(1)
    if 'ndvi' in df_filled.columns:
        df_filled['ndvi_lag_1'] = df_filled['ndvi'].shift(1)

    return df_filled.dropna()
# --- 4. Anti-Collinearity ---
def remove_collinear_features(X, y, threshold=0.90):
    print(f"\n--- Removing Collinear Features (Threshold: {threshold}) ---")
    corr_matrix = X.corr(method='spearman').abs()
    dist_linkage = hierarchy.ward(corr_matrix)
    cluster_ids = hierarchy.fcluster(dist_linkage, t=threshold, criterion='distance')

    cluster_id_to_feature_ids = {}
    for i, cluster_id in enumerate(cluster_ids):
        if cluster_id not in cluster_id_to_feature_ids: cluster_id_to_feature_ids[cluster_id] = []
        cluster_id_to_feature_ids[cluster_id].append(X.columns[i])

    selected_features = []
    correlations_with_target = X.apply(lambda x: spearmanr(x, y)[0]).abs()

    for cluster_id, features in cluster_id_to_feature_ids.items():
        if len(features) == 1:
            selected_features.append(features[0])
        else:
            best_feature = max(features, key=lambda f: correlations_with_target[f])
            selected_features.append(best_feature)

    print(f"Features kept: {len(selected_features)} / {X.shape[1]}")
    return X[selected_features]
def get_label(col_name):
    return VAR_NAMES.get(col_name, col_name)
# --- 5. Main ---
if __name__ == "__main__":
    out_dir = "analysis_figures_english6"
    os.makedirs(out_dir, exist_ok=True)

    # Load
    df_train = load_dataset(train_paths, flow_train_path, "1984-01-01", "2013-12-31", "Train Set")
    df_test = load_dataset(test_paths, flow_test_path, "2014-01-01", "2022-12-31", "Test Set")
    full_df = pd.concat([df_train, df_test]).sort_index()

    target_col = 'flow'
    feature_cols = [c for c in full_df.columns if c != target_col and not c.startswith('flow')]
    X = full_df[feature_cols]
    y = full_df[target_col]

    # Filter
    X_refined = remove_collinear_features(X, y, threshold=0.90)

    # 1. Correlation Plot
    print("Generating Plot 1: Correlation Matrix...")
    plt.figure(figsize=(12, 10))
    corr = X_refined.join(y).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    labels = [get_label(c) for c in corr.columns]
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r',
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8},
                xticklabels=labels, yticklabels=labels)
    plt.title('Pearson Correlation Matrix', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/Fig1_Correlation_Matrix.png', dpi=600)

    # 2. Model & Validation
    print("Training XGBoost...")
    model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                             reg_alpha=0.5, reg_lambda=1.0, n_jobs=-1, random_state=42)
    model.fit(X_refined, y)

    print("Generating Plot 2: Validation...")
    y_pred = model.predict(X_refined)
    r2 = r2_score(y, y_pred)
    plt.figure(figsize=(8, 8))
    plt.scatter(y, y_pred, alpha=0.3, color='#1f77b4', edgecolor='k', s=10)
    max_val = max(y.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2)
    plt.text(0.05 * max_val, 0.9 * max_val, f'$R^2 = {r2:.2f}$', fontsize=14)
    plt.title('Model Validation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/Fig2_Model_Validation.png', dpi=600)

    # 3. SHAP
    explainer = shap.Explainer(model, X_refined)
    shap_values = explainer(X_refined)

    shap_values.feature_names = [get_label(c) for c in X_refined.columns]

    print("Generating Plot 3: Summary...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_refined, show=False, cmap='coolwarm', alpha=0.7)
    plt.title('Global Importance & Impact Direction', fontsize=16, fontweight='bold')
    plt.xlabel('SHAP value (Impact on Streamflow)')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/Fig3_SHAP_Summary.png', dpi=600)

    print("Generating Plot 4: Importance Bar...")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title('Feature Importance Ranking', fontsize=16, fontweight='bold')
    plt.xlabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/Fig4_SHAP_Importance_Bar.png', dpi=600)

    # 4. Dependence Plots (FIXED LOOP)
    print("Generating Plots 5+: Dependence Plots...")
    feature_order = np.argsort(np.abs(shap_values.values).mean(0))[::-1]

    for i, idx in enumerate(feature_order):

        plt.figure(figsize=(8, 6))
        shap.plots.scatter(shap_values[:, idx], color=shap_values, show=False, alpha=0.5, dot_size=10)

        clean_name = shap_values.feature_names[idx].replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")

        plt.title(f'Response Curve: {shap_values.feature_names[idx]}', fontsize=14, fontweight='bold')
        plt.ylabel('SHAP Value (Impact)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/Fig5_{i + 1}_Dependence_{clean_name}.png', dpi=600)
        plt.close()

    # 5. Interaction Plot (FIXED)
    print("Generating Plot 11: Interaction Effect...")
    try:
        prec_idx = next((i for i, c in enumerate(X_refined.columns) if 'prec' in c), None)
        rhu_idx = next((i for i, c in enumerate(X_refined.columns) if 'rhu' in c), None)

        if prec_idx is not None and rhu_idx is not None:
            plt.figure(figsize=(10, 8))
            shap.plots.scatter(shap_values[:, prec_idx], color=shap_values[:, rhu_idx], show=False)

            p_name = shap_values.feature_names[prec_idx]
            r_name = shap_values.feature_names[rhu_idx]

            plt.title(f'Interaction: {p_name} vs {r_name}', fontsize=16, fontweight='bold')
            plt.ylabel(f'Impact of {p_name}', fontsize=12)
            plt.xlabel(p_name, fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{out_dir}/Fig6_Interaction_Prec_RHU.png', dpi=600)
    except Exception as e:
        print(f"Skipping interaction plot due to missing features: {e}")

