import os
import re
import pandas as pd
import numpy as np
import rasterio
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import warnings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    plt.rcParams['font.family'] = 'Arial'
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False

# Paths (modify as needed)
train_paths = {
    'ndvi': r"D:\Desktop\\MODIS_NDVI_daily_interpolated-1984-2013",
    'temp': r"D:\Desktop\\_1984_2013\xiinghe_001deg_tmpmean_flattened_tif",
    'rhu': r"D:\Desktop\\_1984_2013\xiinghe_001deg_rhu_flattened_tif",
    'petpm': r"D:\Desktop\\_1984_2013\xiinghe_001deg_petPM_flattened_tif",
    'prec': r"D:\Desktop\\_1984_2013\xiinghe_001deg_prec_flattened_tif",
    'pres': r"D:\Desktop\\_1984_2013\xiinghe_001deg_pres_flattened_tif",
    'wind': r"D:\Desktop\\_1984_2013\xiinghe_001deg_windflat_tif",
    'snow': r"D:\Desktop\SNOW"
}
flow_train_path = r"D:\Desktop\\1984-2013.xlsx"
test_paths = {
    'ndvi': r"D:\Desktop\\MODIS_NDVI_daily_interpolated",
    'temp': r"D:\Desktop\\xiinghe_001deg_tmpmean_flattened_tif",
    'rhu': r"D:\Desktop\\xiinghe_001deg_rhu_flattened_tif",
    'petpm': r"D:\Desktop\\xiinghe_001deg_petPM_flattened_tif",
    'prec': r"D:\Desktop\\xiinghe_001deg_prec_flattened_tif",
    'pres': r"D:\Desktop\\xiinghe_001deg_pres_flattened_tif",
    'wind': r"D:\Desktop\\xiinghe_001deg_windflat_tif",
    'snow': r"D:\Desktop\SNOW"
}
flow_test_path = r"D:\Desktop\\2013-2022.xlsx"
def parse_date_from_filename(filename):
    match = re.search(
        r'(\d{4})(\d{2})(\d{2})|(\d{4})_(\d{2})_(\d{2})',
        os.path.basename(filename)
    )
    if not match:
        return None
    parts = [p for p in match.groups() if p is not None]
    year, month, day = map(int, parts[-3:])
    if month == 2 and day == 29:
        if not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            day = 28
    try:
        return datetime(year, month, day)
    except ValueError:
        return None
def load_features_from_tifs(data_folders, flow_path, start_date, end_date, set_name):
    print(f"\nLoading {set_name} data: {start_date} to {end_date}")
    data = {}
    for var, folder_path in data_folders.items():
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Directory not found: {folder_path}")

        var_data = []
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith('.tif')]
        for file in tqdm(files, desc=f"  {var}", leave=False):
            date = parse_date_from_filename(file)
            if date is None:
                continue
            try:
                with rasterio.open(file) as src:
                    array = src.read(1).astype(float)
                    nodata_val = src.nodata
                    if nodata_val is not None:
                        array[array == nodata_val] = np.nan
                    if np.all(np.isnan(array)):
                        value = np.nan
                    elif var == 'snow' and np.nanmax(array) > 1:
                        snow_mask = (array == 100) | (array == 200)
                        value = (np.sum(snow_mask) / np.sum(np.isfinite(array))
                                 if np.any(np.isfinite(array)) else 0)
                    else:
                        value = np.nanmean(array)
                    var_data.append({'date': date, var: value})
            except Exception as e:
                print(f"Warning: {file}: {e}")

        if not var_data:
            continue
        df_var = pd.DataFrame(var_data).set_index('date')
        if df_var.index.has_duplicates:
            df_var = df_var.groupby(df_var.index).mean()
        data[var] = df_var

    df_list = [df for df in data.values() if not df.empty]
    if not df_list:
        raise ValueError("No TIF data loaded.")
    df_merged = pd.concat(df_list, axis=1).sort_index()

    flow_df = pd.read_excel(flow_path)
    flow_df.columns = ['date', 'flow']
    flow_df['date'] = pd.to_datetime(flow_df['date'], errors='coerce')
    flow_df = flow_df.dropna(subset=['date']).set_index('date')
    if flow_df.index.has_duplicates:
        flow_df = flow_df.groupby(flow_df.index).mean()

    df_final = df_merged.join(flow_df, how='inner').loc[start_date:end_date]
    if df_final.empty:
        raise ValueError("Empty dataframe after join.")

    df_filled = (df_final.interpolate(method='time', limit_direction='both')
                 .fillna(method='bfill').fillna(method='ffill'))
    if df_filled.isnull().values.any():
        df_filled = df_filled.fillna(0)

    for lag in [1, 3, 7, 14, 30]:
        df_filled[f'prec_lag_{lag}'] = df_filled['prec'].shift(lag)
        df_filled[f'temp_lag_{lag}'] = df_filled['temp'].shift(lag)
        df_filled[f'flow_lag_{lag}'] = df_filled['flow'].shift(lag)
    df_filled = df_filled.dropna()
    return df_filled
def analyze_feature_importance(model, X, y, feature_names, top_n=20):
    print("  Computing permutation feature importance...")
    result = permutation_importance(model, X, y, n_repeats=10,
                                    random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean
    })
    filtered = importance_df[~importance_df['Feature'].str.startswith('flow')].copy()
    sorted_df = filtered.sort_values('Importance', ascending=False).reset_index(drop=True)
    print(sorted_df.to_string())

    os.makedirs("output_results", exist_ok=True)
    sorted_df.to_csv("output_results/driving_factors_importance.csv",
                     index=False, encoding='utf-8-sig')

    top = sorted_df.head(top_n)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(top['Feature'][::-1], top['Importance'][::-1], color='dodgerblue')
    ax.set_xlabel('Permutation Importance (Mean Decrease in Score)', fontsize=12)
    ax.set_title(f'Top {top_n} Driving Factors (Runoff Excluded)', fontsize=16)
    plt.tight_layout()
    plt.savefig('feature_importance_no_runoff.png', dpi=300)
    plt.close()
def train_dtvgm_baseline(train_df, test_df, feature_cols, target_col):
    print("\nTraining enhanced DTVGM baseline...")
    full_df = pd.concat([train_df, test_df])

    full_df['api'] = full_df['prec'].ewm(alpha=0.1, adjust=False).mean()
    full_df['snow_melt'] = (full_df['temp'] > 0) * full_df['temp'] * full_df['snow']
    physical_features = ['api', 'snow_melt']

    phys_model = LinearRegression()
    phys_model.fit(full_df.loc[train_df.index, physical_features],
                   train_df[target_col])
    q_base_train = np.maximum(0, phys_model.predict(
        full_df.loc[train_df.index, physical_features]))
    q_base_test = np.maximum(0, phys_model.predict(
        full_df.loc[test_df.index, physical_features]))

    ml_target = train_df[target_col] - q_base_train
    ml_booster = HistGradientBoostingRegressor(
        random_state=42, max_iter=300, learning_rate=0.05,
        max_leaf_nodes=31, l2_regularization=0.5)
    ml_booster.fit(train_df[feature_cols], ml_target)

    analyze_feature_importance(ml_booster, train_df[feature_cols],
                               ml_target, feature_cols)

    q_ml_train = ml_booster.predict(train_df[feature_cols])
    q_ml_test = ml_booster.predict(test_df[feature_cols])

    q_tvgm_train = np.maximum(0, q_base_train + q_ml_train)
    q_tvgm_test = np.maximum(0, q_base_test + q_ml_test)
    return q_tvgm_train, q_tvgm_test
# Deep learning models (matching paper Section 3.2.3)
class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride,
                 dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding,
                               dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding,
                               dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
class ResidualTCN(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()
        num_channels = [32, 32, 32, 32]
        kernel_size = 3
        layers = []
        for i, n_out in enumerate(num_channels):
            dilation = 2 ** i
            n_in = input_size if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(
                n_in, n_out, kernel_size, stride=1, dilation=dilation,
                padding=(kernel_size - 1) * dilation, dropout=0.2))
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        return self.linear(self.network(x.permute(0, 2, 1))[:, :, -1])
class ResidualTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2,
                 output_size=1, dropout=0.2):
        super().__init__()
        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 2, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)
        self.output_fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_fc(x) + self.pos_encoder[:, :x.size(1), :]
        return self.output_fc(self.transformer_encoder(x)[:, -1, :])
def create_sequences(features, targets, seq_length):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i:(i + seq_length)])
        ys.append(targets[i + seq_length])
    return np.array(xs), np.array(ys).reshape(-1, 1)
def train_dl_model(model, train_loader, val_loader, epochs, lr, patience,
                   model_name):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    pbar = tqdm(range(epochs), desc=f"Training {model_name}", leave=False)
    for epoch in pbar:
        model.train()
        for seq, labels in train_loader:
            seq, labels = seq.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(seq), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, labels in val_loader:
                val_loss += criterion(model(seq.to(DEVICE)),
                                      labels.to(DEVICE)).item()
        val_loss /= len(val_loader)
        scheduler.step()
        pbar.set_postfix({'Val Loss': f'{val_loss:.6f}'})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'best_{model_name}.pth')
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break
    model.load_state_dict(torch.load(f'best_{model_name}.pth'))
    return model
def nse_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum(
        (y_true - np.mean(y_true)) ** 2)
def pearson_r2_score(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    return np.corrcoef(y_true, y_pred)[0, 1] ** 2
def evaluate_model(observed, predicted):
    mask = np.isfinite(observed) & np.isfinite(predicted)
    obs, pred = observed[mask], predicted[mask]
    return {
        'NSE': nse_score(obs, pred),
        'R2': pearson_r2_score(obs, pred),
        'RMSE': np.sqrt(mean_squared_error(obs, pred))
    }
if __name__ == "__main__":
    print("=" * 60)
    print("DTVGM-DL Hybrid Streamflow Simulation Framework")
    print("=" * 60)

    # ---- Step 1: Load and engineer features ----
    train_df = load_features_from_tifs(
        train_paths, flow_train_path, "1984-01-01", "2013-12-31", "TRAINING")
    test_df = load_features_from_tifs(
        test_paths, flow_test_path, "2014-01-01", "2022-12-31", "TESTING")

    common_features = list(set(train_df.columns) & set(test_df.columns) - {'flow'})
    train_df = train_df[common_features + ['flow']]
    test_df = test_df[common_features + ['flow']]
    print(f"\n  {len(common_features)} common features loaded.")

    # ---- Step 2: Min-Max normalization (per Section 2.3) ----
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(train_df[common_features])
    X_test_norm = scaler.transform(test_df[common_features])

    # ---- Step 3: Train DTVGM baseline ----
    tvgm_train, tvgm_test = train_dtvgm_baseline(
        train_df, test_df, common_features, 'flow')
    train_df['dtvgm_sim'] = tvgm_train
    test_df['dtvgm_sim'] = tvgm_test

    # ---- Step 4: Add baseline output to DL input features (per Section 3.2.2) ----
    train_df['dtvgm_sim_norm'] = (tvgm_train - scaler.data_min_[0]) / (
        scaler.data_max_[0] - scaler.data_min_[0])
    test_df['dtvgm_sim_norm'] = (tvgm_test - scaler.data_min_[0]) / (
        scaler.data_max_[0] - scaler.data_min_[0])

    dl_feature_cols = common_features + ['dtvgm_sim_norm']
    X_train_dl = np.column_stack([X_train_norm, train_df['dtvgm_sim_norm'].values])
    X_test_dl = np.column_stack([X_test_norm, test_df['dtvgm_sim_norm'].values])

    # ---- Step 5: Compute residuals ----
    residual_train = train_df['flow'].values - tvgm_train
    residual_test = test_df['flow'].values - tvgm_test

    # ---- Step 6: Create sequences (seq_length=14 per Section 3.2.2) ----
    SEQ_LENGTH = 14
    BATCH_SIZE = 64
    split_idx = int(len(X_train_dl) * 0.8)
    X_train_seq, y_train_seq = create_sequences(
        X_train_dl[:split_idx], residual_train[:split_idx], SEQ_LENGTH)
    X_val_seq, y_val_seq = create_sequences(
        X_train_dl[split_idx:], residual_train[split_idx:], SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(
        X_test_dl, residual_test, SEQ_LENGTH)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_seq),
                      torch.FloatTensor(y_train_seq)),
        BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_seq),
                      torch.FloatTensor(y_val_seq)), BATCH_SIZE)
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_seq),
                      torch.FloatTensor(y_test_seq)), BATCH_SIZE)

    # ---- Step 7: Train DL residual correction models ----
    n_features = len(dl_feature_cols)
    dl_configs = {
        'LSTM': ResidualLSTM(n_features),
        'TCN': ResidualTCN(n_features),
        'Transformer': ResidualTransformer(n_features, nhead=4),
    }
    train_params = {'epochs': 100, 'lr': 0.001, 'patience': 10}

    dl_predictions = {}
    for name, model in dl_configs.items():
        print(f"\nTraining DTVGM+{name}...")
        model = train_dl_model(model.to(DEVICE), train_loader, val_loader,
                               model_name=name, **train_params)
        model.eval()
        preds = []
        with torch.no_grad():
            for seq, _ in test_loader:
                preds.extend(model(seq.to(DEVICE)).cpu().numpy().flatten())

        final_preds = np.maximum(
            0, tvgm_test[SEQ_LENGTH:SEQ_LENGTH + len(preds)] + np.array(preds))
        col_name = f'DTVGM+{name}'
        test_df[col_name] = np.nan
        test_df.loc[test_df.index[SEQ_LENGTH:SEQ_LENGTH + len(final_preds)],
                    col_name] = final_preds
        dl_predictions[name] = final_preds

    # ---- Step 8: Evaluate all models ----
    print("\n" + "=" * 60)
    print("Model Performance Comparison")
    print("=" * 60)

    model_keys = {
        'DTVGM': 'dtvgm_sim',
        'DTVGM+LSTM': 'DTVGM+LSTM',
        'DTVGM+TCN': 'DTVGM+TCN',
        'DTVGM+Transformer': 'DTVGM+Transformer',
    }

    results = {}
    for display_name, col in model_keys.items():
        pred = test_df[col].dropna()
        obs = test_df['flow'].loc[pred.index]
        results[display_name] = evaluate_model(obs.values, pred.values)
        print(f"  {display_name:25s}  "
              f"NSE={results[display_name]['NSE']:.3f}  "
              f"R2={results[display_name]['R2']:.3f}  "
              f"RMSE={results[display_name]['RMSE']:.3f}")

    results_df = pd.DataFrame(results).T
    results_df.to_excel("output_results/metrics_results.xlsx")
    print("\nMetrics saved to output_results/metrics_results.xlsx")

    # ---- Save predictions ----
    os.makedirs("output_results", exist_ok=True)
    train_out = train_df[['flow', 'dtvgm_sim']].rename(
        columns={'flow': 'Observed', 'dtvgm_sim': 'Predicted_DTVGM'})
    train_out.to_csv("output_results/train_set_results.csv")
    test_out = test_df.rename(columns={
        'flow': 'Observed', 'dtvgm_sim': 'Predicted_DTVGM',
        'DTVGM+LSTM': 'Predicted_DTVGM+LSTM',
        'DTVGM+TCN': 'Predicted_DTVGM+TCN',
        'DTVGM+Transformer': 'Predicted_DTVGM+Transformer'
    })
    pred_cols = ['Observed'] + [c for c in test_out.columns if 'Predicted' in c]
    test_out[pred_cols].to_csv("output_results/test_set_results.csv")
    print("Predictions saved to output_results/")

    # ---- Plot hydrograph comparison ----
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_df = test_df.dropna(subset=['DTVGM+LSTM']).copy()
    ax.plot(plot_df.index, plot_df['flow'], label='Observed',
            color='black', linewidth=1.5)
    ax.plot(plot_df.index, plot_df['dtvgm_sim'],
            label=f'DTVGM (NSE={results["DTVGM"]["NSE"]:.3f})',
            color='red', linestyle='--')
    colors = {'LSTM': 'blue', 'TCN': 'green', 'Transformer': 'purple'}
    for name in dl_configs:
        col = f'DTVGM+{name}'
        if col in plot_df.columns:
            ax.plot(plot_df.index, plot_df[col],
                    label=f'DTVGM+{name} (NSE={results[f"DTVGM+{name}"]["NSE"]:.3f})',
                    color=colors[name])
    ax.set_ylabel('Streamflow (m$^3$/s)', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("output_results/prediction_comparison.png", dpi=300)
    plt.close()
    print("Figure saved to output_results/prediction_comparison.png")
    print("\nDone.")
