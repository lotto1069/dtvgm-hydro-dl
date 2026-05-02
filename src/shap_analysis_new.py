# Hydro-DL: Publication-Quality Driver Analysis (Final Optimized)
import os
import re
import warnings
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from datetime import datetime
from tqdm import tqdm
from scipy.stats import spearmanr, zscore
from scipy.cluster import hierarchy
from sklearn.metrics import r2_score
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['ytick.labelsize'] = 14

VAR_NAMES = {
    'flow': 'Q',
    'prec': 'P',
    'prec_lag_1': 'P_lag1',
    'prec_lag_3': 'P_lag3',
    'temp': 'T',
    'temp_lag_1': 'T_lag1',
    'rhu': 'RH',
    'petpm': 'PET',
    'pres': 'AP',
    'wind': 'WS',
    'snow': 'SC',
    'ndvi': 'NDVI',
    'ndvi_lag_1': 'NDVI_lag1'
}

train_paths = {
    'ndvi': r"C:\train\MODIS_NDVI_daily_interpolated-1984-2013",
    'temp': r"C:\train\meteorology_1984_2013\xiinghe_001deg_tmpmean_flattened_tif",
    'rhu': r"C:\train\meteorology_1984_2013\xiinghe_001deg_rhu_flattened_tif",
    'petpm': r"C:\train\meteorology_1984_2013\xiinghe_001deg_petPM_flattened_tif",
    'prec': r"C:\train\meteorology_1984_2013\xiinghe_001deg_prec_flattened_tif",
    'pres': r"C:\train\meteorology_1984_2013\xiinghe_001deg_pres_flattened_tif",
    'wind': r"C:\train\meteorology_1984_2013\xiinghe_001deg_windflat_tif",
    'snow': r"C:\SNOW"
}
flow_train_path = r"C:\train\runoff1984-2013.xlsx"

test_paths = {
    'ndvi': r"C:\test\MODIS_NDVI_daily_interpolated",
    'temp': r"C:\test\xiinghe_001deg_tmpmean_flattened_tif",
    'rhu': r"C:\test\xiinghe_001deg_rhu_flattened_tif",
    'petpm': r"C:\test\xiinghe_001deg_petPM_flattened_tif",
    'prec': r"C:\test\xiinghe_001deg_prec_flattened_tif",
    'pres': r"C:\test\xiinghe_001deg_pres_flattened_tif",
    'wind': r"C:\test\xiinghe_001deg_windflat_tif",
    'snow': r"C:\SNOW"
}
def parse_date_from_filename(filename):
    match = re.search(r'(\d{4})(\d{2})(\d{2})|(\d{4})_(\d{2})_(\d{2})', os.path.basename(filename))
    if match:
        parts = [p for p in match.groups() if p is not None]
        try:
            return datetime(*map(int, parts[-3:]))
        except:
            return None
    return None
def get_label(col_name):
    return VAR_NAMES.get(col_name, col_name)
def process_file_task(file_path, var_name, start_date, end_date):
    try:
        dt = parse_date_from_filename(file_path)
        if dt is None or not (start_date <= dt <= end_date): return None
        with rasterio.open(file_path) as src:
            array = src.read(1).astype(float)
            if src.nodata is not None: array[array == src.nodata] = np.nan

            if np.all(np.isnan(array)):
                val = np.nan
            elif var_name == 'snow':
                valid = np.sum(np.isfinite(array))
                val = np.sum((array == 100) | (array == 200)) / valid if valid > 0 else 0
            else:
                val = np.nanmean(array)
            return {'date': dt, var_name: val}
    except:
        return None
def load_dataset(data_folders, flow_path, start_date_str, end_date_str, dataset_name):
    print(f"\n--- Loading {dataset_name} (Fast Mode) ---")
    start_date, end_date = pd.to_datetime(start_date_str), pd.to_datetime(end_date_str)
    data = {}

    for var, folder in data_folders.items():
        if not os.path.isdir(folder): continue
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.tif')]
        print(f"Reading {var}...")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_file_task, f, var, start_date, end_date) for f in files]
            results = [f.result() for f in tqdm(as_completed(futures), total=len(files), leave=False) if f.result()]

        if results:
            df = pd.DataFrame(results).set_index('date')
            data[var] = df.groupby(df.index).mean()

    if not data: return pd.DataFrame()

    print("Merging data...")
    df_merged = pd.concat(data.values(), axis=1).sort_index()
    flow_df = pd.read_excel(flow_path)
    flow_df.columns = ['date', 'flow']
    flow_df['date'] = pd.to_datetime(flow_df['date'], errors='coerce')
    flow_df = flow_df.dropna(subset=['date']).set_index('date').groupby('date').mean()

    df_final = df_merged.join(flow_df, how='inner').loc[start_date:end_date]
    df_filled = df_final.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')

    # Lags
    df_filled['prec_lag_1'] = df_filled['prec'].shift(1)
    df_filled['prec_lag_3'] = df_filled['prec'].shift(3)
    df_filled['temp_lag_1'] = df_filled['temp'].shift(1)
    if 'ndvi' in df_filled.columns: df_filled['ndvi_lag_1'] = df_filled['ndvi'].shift(1)

    return df_filled.dropna()
def remove_collinear_features(X, y, threshold=0.90):
    print(f"\n--- Removing Collinear Features ---")
    X = X.dropna(axis=1, how='all')

    non_const = [c for c in X.columns if X[c].nunique() > 1]
    X = X[non_const]
    if X.empty: raise ValueError("No valid features left!")

    corr = X.corr(method='spearman').abs().fillna(0)

    try:
        linkage = hierarchy.ward(corr)
    except:
        return X  # Fallback

    cluster_ids = hierarchy.fcluster(linkage, t=threshold, criterion='distance')

    selected = []
    corrs_y = X.apply(lambda x: spearmanr(x, y)[0]).abs().fillna(0)

    cluster_map = {}
    for i, cid in enumerate(cluster_ids):
        cluster_map.setdefault(cid, []).append(X.columns[i])

    for cid, feats in cluster_map.items():
        best = max(feats, key=lambda f: corrs_y[f])
        selected.append(best)

    print(f"Features kept: {len(selected)} / {X.shape[1]}")
    return X[selected]
if __name__ == "__main__":
    out_dir = "analysis_figures_final_fixed"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Data
    df_train = load_dataset(train_paths, flow_train_path, "1984-01-01", "2013-12-31", "Train")
    df_test = load_dataset(test_paths, flow_test_path, "2014-01-01", "2022-12-31", "Test")
    full_df = pd.concat([df_train, df_test]).sort_index()

    target_col = 'flow'
    feature_cols = [c for c in full_df.columns if c != target_col and not c.startswith('flow')]
    X = full_df[feature_cols]
    y = full_df[target_col]

    # 2. Filter Features
    X_refined = remove_collinear_features(X, y, threshold=0.90)

    # --- Plot 1: Correlation Matrix ---
    print("Plotting Correlation Matrix...")
    plt.figure(figsize=(8, 7))
    corr = X_refined.join(y).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    labels = [get_label(c) for c in corr.columns]

    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r',
                center=0, square=True, linewidths=2, linecolor='white',
                cbar_kws={"shrink": .7},
                annot_kws={"size": 14, "weight": "bold"},
                xticklabels=labels, yticklabels=labels)

    plt.title('Pearson Correlation Matrix', fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=1.5)
    plt.savefig(f'{out_dir}/Fig1_Correlation_Matrix.png', dpi=600)
    plt.close()

    # --- Model Training ---
    print("Training XGBoost...")
    model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)
    model.fit(X_refined, y)

    # --- Validation Plot ---
    print("Plotting Validation...")
    y_pred = model.predict(X_refined)
    residuals = np.abs(y - y_pred)
    mask_val = zscore(residuals) < 3
    y_c, y_p = y.values[mask_val], y_pred[mask_val]

    plt.figure(figsize=(7, 7))
    plt.scatter(y_c, y_p, alpha=0.3, color='#444444', s=20, edgecolors='none')
    mx = max(y_c.max(), y_p.max())
    plt.plot([0, mx], [0, mx], 'r--', lw=2.5)
    plt.text(0.05 * mx, 0.9 * mx, f'$R^2 = {r2_score(y_c, y_p):.2f}$', fontsize=16)
    plt.xlabel('Observed Q ($m^3/s$)', fontsize=14)
    plt.ylabel('Simulated Q ($m^3/s$)', fontsize=14)
    plt.title('Model Validation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/Fig2_Model_Validation.png', dpi=600)
    plt.close()

    # --- SHAP Analysis ---
    explainer = shap.Explainer(model, X_refined)
    shap_values = explainer(X_refined)
    shap_values.feature_names = [get_label(c) for c in X_refined.columns]

    # --- SHAP Summary Plots ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_refined, show=False, cmap='coolwarm', alpha=0.8)
    plt.title('Global Feature Importance', fontsize=16, fontweight='bold')
    plt.savefig(f'{out_dir}/Fig3_SHAP_Summary.png', dpi=600)
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title('Feature Importance Ranking', fontsize=16, fontweight='bold')
    plt.savefig(f'{out_dir}/Fig4_SHAP_Importance_Bar.png', dpi=600)
    plt.close()

    print("Plotting Clean Dependence Plots with Trends...")
    order = np.argsort(np.abs(shap_values.values).mean(0))[::-1]

    Q_LOW, Q_HIGH = 0.01, 0.99

    for idx in order:
        feat_name = shap_values.feature_names[idx]
        clean_name = feat_name.replace(" ", "_")

        x_raw = shap_values[:, idx].data
        y_raw = shap_values[:, idx].values

        x_lim_min, x_lim_max = np.nanquantile(x_raw, Q_LOW), np.nanquantile(x_raw, Q_HIGH)
        y_lim_min, y_lim_max = np.nanquantile(y_raw, Q_LOW), np.nanquantile(y_raw, Q_HIGH)

        mask_plot = (x_raw >= x_lim_min) & (x_raw <= x_lim_max) & \
                    (y_raw >= y_lim_min) & (y_raw <= y_lim_max)

        if mask_plot.sum() < 20: continue

        x_clean = x_raw[mask_plot]
        y_clean = y_raw[mask_plot]

        plt.figure(figsize=(7, 6))

        sc = plt.scatter(x_clean, y_clean,
                         c=c_clean,
                         cmap='coolwarm',
                         s=20, alpha=0.4, edgecolors='none')

        sort_idx = np.argsort(x_clean)
        x_sorted = x_clean[sort_idx]
        y_sorted = y_clean[sort_idx]

        y_trend = pd.Series(y_sorted).rolling(window=win_size, center=True, min_periods=1).mean()

        plt.plot(x_sorted, y_trend, color='red', linewidth=3, label='Trend')

        plt.xlim(x_lim_min, x_lim_max)
        plt.ylim(y_lim_min, y_lim_max)

        cbar = plt.colorbar(sc)
        cbar.set_label(f'{feat_name} Value')
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel(feat_name, fontsize=14, fontweight='bold')
        plt.ylabel('SHAP Value (Impact on Q)', fontsize=14, fontweight='bold')
        plt.title(f'Response Curve: {feat_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/Fig5_Dependence_{clean_name}.png', dpi=600)
        plt.close()

    print(f"\nAll Fixed Figures generated in: {os.path.abspath(out_dir)}")