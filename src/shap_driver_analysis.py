# Hydro-DL: Scientifically Robust Driver Analysis (Anti-Collinearity)
import os
import re
import pandas as pd
import numpy as np
import rasterio
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import shap
import xgboost as xgb
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

import warnings

warnings.filterwarnings('ignore')

try:
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    if os.path.exists(font_path):
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

train_paths = {
    'ndvi': r"D:\Desktop\train\MODIS_NDVI_daily_interpolated-1984-2013",
    'temp': r"D:\Desktop\train\meteorology_1984_2013\xiinghe_001deg_tmpmean_flattened_tif",
    'rhu': r"D:\Desktop\train\meteorology_1984_2013\xiinghe_001deg_rhu_flattened_tif",
    'petpm': r"D:\Desktop\train\meteorology_1984_2013\xiinghe_001deg_petPM_flattened_tif",
    'prec': r"D:\Desktop\train\meteorology_1984_2013\xiinghe_001deg_prec_flattened_tif",
    'pres': r"D:\Desktop\train\meteorology_1984_2013\xiinghe_001deg_pres_flattened_tif",
    'wind': r"D:\Desktop\train\meteorology_1984_2013\xiinghe_001deg_windflat_tif",
    'snow': r"D:\Desktop\SNOW"
}
flow_train_path = r"D:\Desktop\train\runoff1984-2013.xlsx"

test_paths = {
    'ndvi': r"D:\Desktop\test\MODIS_NDVI_daily_interpolated",
    'temp': r"D:\Desktop\test\xiinghe_001deg_tmpmean_flattened_tif",
    'rhu': r"D:\Desktop\test\xiinghe_001deg_rhu_flattened_tif",
    'petpm': r"D:\Desktop\test\xiinghe_001deg_petPM_flattened_tif",
    'prec': r"D:\Desktop\test\xiinghe_001deg_prec_flattened_tif",
    'pres': r"D:\Desktop\test\xiinghe_001deg_pres_flattened_tif",
    'wind': r"D:\Desktop\test\xiinghe_001deg_windflat_tif",
    'snow': r"D:\Desktop\SNOW"
}
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
    data = {}
    for var, folder_path in data_folders.items():
        if not os.path.isdir(folder_path): continue
        var_data = []
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
            date = parse_date_from_filename(file)
            if date and (pd.Timestamp(start_date) <= date <= pd.Timestamp(end_date)):
                try:
                    with rasterio.open(file) as src:
                        array = src.read(1).astype(float)
                        nodata = src.nodata
                        if nodata is not None: array[array == nodata] = np.nan
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

    df_merged = pd.concat(data.values(), axis=1).sort_index()
    flow_df = pd.read_excel(flow_path)
    flow_df.columns = ['date', 'flow']
    flow_df['date'] = pd.to_datetime(flow_df['date'], errors='coerce')
    flow_df = flow_df.dropna(subset=['date']).set_index('date')
    if flow_df.index.has_duplicates: flow_df = flow_df.groupby(flow_df.index).mean()
    df_final = df_merged.join(flow_df, how='inner').loc[start_date:end_date]
    df_filled = df_final.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    df_filled['prec_lag_1'] = df_filled['prec'].shift(1)
    df_filled['prec_lag_3'] = df_filled['prec'].shift(3)

    df_filled['temp_lag_1'] = df_filled['temp'].shift(1)

    if 'ndvi' in df_filled.columns:

    df_filled = df_filled.dropna()
    return df_filled
def remove_collinear_features(X, y, threshold=0.90):
    """
    """
    corr_matrix = X.corr(method='spearman').abs()

    dist_linkage = hierarchy.ward(corr_matrix)
    cluster_ids = hierarchy.fcluster(dist_linkage, t=threshold, criterion='distance')

    cluster_id_to_feature_ids = {}
    for i, cluster_id in enumerate(cluster_ids):
        if cluster_id not in cluster_id_to_feature_ids:
            cluster_id_to_feature_ids[cluster_id] = []
        cluster_id_to_feature_ids[cluster_id].append(X.columns[i])

    selected_features = []
    dropped_log = []

    correlations_with_target = X.apply(lambda x: spearmanr(x, y)[0]).abs()

    for cluster_id, features in cluster_id_to_feature_ids.items():
        if len(features) == 1:
            selected_features.append(features[0])
        else:
            best_feature = max(features, key=lambda f: correlations_with_target[f])
            selected_features.append(best_feature)
            for f in features:
                if f != best_feature:

    if dropped_log:
        for log in dropped_log:
            print(f"  - {log}")

    return X[selected_features]
if __name__ == "__main__":
    os.makedirs("analysis_results_refined", exist_ok=True)

    df_test = load_dataset(test_paths, flow_test_path, "2014-01-01", "2022-12-31", "test")
    full_df = pd.concat([df_train, df_test]).sort_index()

    target_col = 'flow'
    feature_cols = [c for c in full_df.columns if c != target_col and not c.startswith('flow')]
    X = full_df[feature_cols]
    y = full_df[target_col]

    X_refined = remove_collinear_features(X, y, threshold=0.90)

    plt.figure(figsize=(14, 12))
    sns.heatmap(X_refined.join(y).corr(), annot=True, cmap='RdBu_r', fmt='.2f', center=0)
    plt.tight_layout()
    plt.savefig('analysis_results_refined/correlation_heatmap_refined.png', dpi=300)

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_refined, y)

    explainer = shap.Explainer(model, X_refined)
    shap_values = explainer(X_refined)

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_refined, show=False)
    plt.tight_layout()
    plt.savefig('analysis_results_refined/shap_summary_refined.png', dpi=300)

    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig('analysis_results_refined/shap_importance_refined.png', dpi=300)

