# Hydro-DL Final Model - TVGM_v15.6 (Fixed Feature Mismatch)

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
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import warnings

VERSION = "TVGM_v15.6_Fixed_Feature_Mismatch"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch is configured to use device: {DEVICE}")
if torch.cuda.is_available(): print(f"  - Device Name: {torch.cuda.get_device_name(0)}")
print("=" * 40 + "\n")

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    if os.path.exists(font_path):
        my_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = my_font.get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:

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
def parse_date_from_filename(filename):
    match = re.search(r'(\d{4})(\d{2})(\d{2})|(\d{4})_(\d{2})_(\d{2})', os.path.basename(filename))
    if match:
        parts = [p for p in match.groups() if p is not None]
        year, month, day = map(int, parts[-3:])
        if month == 2 and day == 29:
            if not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)): day = 28
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
    return None
def prepare_and_engineer_features(data_folders, flow_path, start_date, end_date, set_name):
    print(f"\n--- Loading data for period: {start_date} to {end_date} for {set_name} set ---")
    data = {}
    for var, folder_path in data_folders.items():
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Directory not found for {var}: {folder_path}")

        var_data = []
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.tif')]

        for file in tqdm(files, desc=f"Processing {var}", leave=False):
            date = parse_date_from_filename(file)
            if date:
                try:
                    with rasterio.open(file) as src:
                        array = src.read(1).astype(float)
                        nodata_val = src.nodata
                        if nodata_val is not None: array[array == nodata_val] = np.nan

                        if np.all(np.isnan(array)):
                            value = np.nan
                        elif var == 'snow' and np.nanmax(array) > 1:
                            snow_mask = (array == 100) | (array == 200)
                            value = np.sum(snow_mask) / np.sum(np.isfinite(array)) if np.any(np.isfinite(array)) else 0
                        else:
                            value = np.nanmean(array)

                        var_data.append({'date': date, var: value})
                except Exception as e:
                    print(f"Warning: Could not process file {file}. Error: {e}")

        if not var_data: continue
        df_single_var = pd.DataFrame(var_data).set_index('date')
        if df_single_var.index.has_duplicates:
            df_single_var = df_single_var.groupby(df_single_var.index).mean()
        data[var] = df_single_var

    df_list = [df for df in data.values() if not df.empty]
    if not df_list: raise ValueError("No data loaded.")

    df_merged = pd.concat(df_list, axis=1).sort_index()

    flow_df = pd.read_excel(flow_path)
    flow_df.columns = ['date', 'flow']
    flow_df['date'] = pd.to_datetime(flow_df['date'], errors='coerce')
    flow_df = flow_df.dropna(subset=['date']).set_index('date')
    if flow_df.index.has_duplicates: flow_df = flow_df.groupby(flow_df.index).mean()

    df_final = df_merged.join(flow_df, how='inner').loc[start_date:end_date]
    if df_final.empty: raise ValueError("Dataframe empty after join/slice.")

    df_filled = df_final.interpolate(method='time', limit_direction='both').fillna(method='bfill').fillna(
        method='ffill')
    if df_filled.isnull().values.any(): df_filled = df_filled.fillna(0)

    for lag in [1, 3, 7, 14, 30]:
        df_filled[f'prec_lag_{lag}'] = df_filled['prec'].shift(lag)
        df_filled[f'temp_lag_{lag}'] = df_filled['temp'].shift(lag)
    df_filled = df_filled.dropna()
    return df_filled
def analyze_and_plot_feature_importance(model, X_full, y, feature_names_full, top_n=20):
    """
    """
    print("  - Analyzing feature importance (excluding runoff & past runoff)...")

    non_runoff_indices = [
        i for i, col in enumerate(feature_names_full)
        if not (col.lower().startswith("flow") or "flow" in col.lower())
    ]
    non_runoff_features = [feature_names_full[i] for i in non_runoff_indices]

    if len(non_runoff_features) == 0:
        print("Warning: No non-runoff features found. Skipping importance analysis.")
        return

    result = permutation_importance(
        model, X_full, y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importances = result.importances_mean

    plt.figure(figsize=(12, 10))
    plt.barh(
        range(top_n),
        color='dodgerblue',
        align='center'
    )
    plt.yticks(
        range(top_n),
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig('feature_importance_no_runoff.png', dpi=300)
    plt.close()
def train_and_predict_tvgm_v15(train_df, test_df, feature_cols, target_col):
    print("\n--- Training TVGM (Based on successful v15.1 logic) ---")
    full_df = pd.concat([train_df, test_df])
    print("  - Step 1: Engineering physics-driven features (API & Snow Melt)...")
    full_df['api'] = full_df['prec'].ewm(alpha=0.1, adjust=False).mean()
    full_df['snow_melt'] = (full_df['temp'] > 0) * full_df['temp'] * full_df['snow']
    physical_features = ['api', 'snow_melt']

    print("  - Step 2: Training a simple linear model as the physical base (Q_base)...")
    phys_model = LinearRegression()
    phys_model.fit(full_df.loc[train_df.index, physical_features], train_df[target_col])
    q_base_train = phys_model.predict(full_df.loc[train_df.index, physical_features])
    q_base_test = phys_model.predict(full_df.loc[test_df.index, physical_features])

    print("  - Step 3: Training a Gradient Boosting model to learn residuals (Q_ml_boost)...")
    ml_target = train_df[target_col] - q_base_train
    ml_booster = HistGradientBoostingRegressor(random_state=42, max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=0.5)

    analyze_and_plot_feature_importance(
        model=ml_booster,
        y=ml_target,
    )

    q_ml_boost_train = ml_booster.predict(train_df[feature_cols])
    q_ml_boost_test = ml_booster.predict(test_df[feature_cols])

    print("  - Step 4: Combining Q_base and Q_ml_boost for final TVGM prediction...")
    q_tvgm_train = np.maximum(0, q_base_train + q_ml_boost_train)
    q_tvgm_test = np.maximum(0, q_base_test + q_ml_boost_test)

    print("TVGM v15 training and prediction finished.")
    return q_tvgm_train, q_tvgm_test
class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
class ResidualTCN(nn.Module):
    def __init__(self, input_size, output_size=1, num_channels=[32] * 3, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, n_out in enumerate(num_channels):
            dilation = 2 ** i
            n_in = input_size if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(n_in, n_out, kernel_size, stride=1, dilation=dilation,
                                        padding=(kernel_size - 1) * dilation, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        return self.linear(self.network(x.permute(0, 2, 1))[:, :, -1])
class ResidualTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=8, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_fc(x) + self.pos_encoder[:, :x.size(1), :]
        return self.output_fc(self.transformer_encoder(x)[:, -1, :])
def create_sequences(input_data, target_data, seq_length):
    xs, ys = [], []
    for i in range(len(input_data) - seq_length):
        xs.append(input_data[i:(i + seq_length)])
        ys.append(target_data[i + seq_length])
    return np.array(xs), np.array(ys).reshape(-1, 1)
def train_dl_model(model, train_loader, val_loader, epochs, lr, patience):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience // 2, factor=0.5)
    best_val_loss, epochs_no_improve = float('inf'), 0
    pbar = tqdm(range(epochs), desc=f"Training {model.__class__.__name__}", leave=False)
    for epoch in pbar:
        model.train()
        for seq, labels in train_loader:
            seq, labels = seq.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(seq), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, labels in val_loader:
                val_loss += criterion(model(seq.to(DEVICE)), labels.to(DEVICE)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        pbar.set_postfix({'Val Loss': f'{val_loss:.6f}'})
        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = val_loss, 0
            torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pth')
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break
    model.load_state_dict(torch.load(f'best_{model.__class__.__name__}.pth'))
    return model
if __name__ == "__main__":
    print("Step 1/7: Loading and engineering features...")
    train_df = prepare_and_engineer_features(train_paths, flow_train_path, "1984-01-01", "2013-12-31", "TRAINING")
    test_df = prepare_and_engineer_features(test_paths, flow_test_path, "2014-01-01", "2022-12-31", "TESTING")

    common_features = list(set(train_df.columns) & set(test_df.columns) - {'flow'})
    train_df = train_df[common_features + ['flow']]
    test_df = test_df[common_features + ['flow']]
    print(f"\n- Using {len(common_features)} common features for all models.")

    print("\nStep 2/7: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[common_features])
    X_test_scaled = scaler.transform(test_df[common_features])

    print("\nStep 3/7: Training TVGM...")
    tvgm_pred_train, tvgm_pred_test = train_and_predict_tvgm_v15(train_df, test_df, common_features, 'flow')
    train_df['TVGM_pred'] = tvgm_pred_train
    test_df['TVGM_pred'] = tvgm_pred_test

    print("\nStep 4/7: Calculating TVGM residuals...")
    residual_train = train_df['flow'].values - tvgm_pred_train
    residual_test = test_df['flow'].values - tvgm_pred_test

    print("\nStep 5/7: Creating sequence samples...")
    SEQ_LENGTH, BATCH_SIZE = 30, 64
    split_idx = int(len(X_train_scaled) * 0.8)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled[:split_idx], residual_train[:split_idx], SEQ_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_train_scaled[split_idx:], residual_train[split_idx:], SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, residual_test, SEQ_LENGTH)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq)), BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq)), BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(y_test_seq)), BATCH_SIZE)

    print("\nStep 6/7: Training DL models...")
    dl_models_config = {
        'LSTM': {'model': ResidualLSTM(len(common_features)), 'params': {'epochs': 100, 'lr': 0.001, 'patience': 10}},
        'TCN': {'model': ResidualTCN(len(common_features)), 'params': {'epochs': 100, 'lr': 0.001, 'patience': 10}},
        'Transformer': {'model': ResidualTransformer(len(common_features)),
                        'params': {'epochs': 100, 'lr': 0.001, 'patience': 10}}
    }

    num_features = len(common_features)
    nhead = 1
    for i in [8, 4, 2, 1]:
        if (num_features * 2) % i == 0 and 64 % i == 0:
            nhead = i
            break
    dl_models_config['Transformer'] = {'model': ResidualTransformer(num_features, nhead=nhead),
                                       'params': {'epochs': 100, 'lr': 0.001, 'patience': 10}}
    print(f"Transformer nhead automatically set to {nhead}.")

    for name, config in dl_models_config.items():
        model = train_dl_model(config['model'].to(DEVICE), train_loader, val_loader, **config['params'])
        model.eval()
        preds = []
        with torch.no_grad():
            for seq, _ in test_loader:
                preds.extend(model(seq.to(DEVICE)).cpu().numpy().flatten())

        final_preds = np.maximum(0, tvgm_pred_test[SEQ_LENGTH:SEQ_LENGTH + len(preds)] + np.array(preds))
        test_df[f'TVGM+{name}_pred'] = pd.Series(final_preds,
                                                 index=test_df.index[SEQ_LENGTH:SEQ_LENGTH + len(final_preds)])

    print("\nStep 7/7: Evaluating models, saving results, and plotting...")
    def nse_score(y_true, y_pred):
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    metrics = {}
    model_keys = ['TVGM_pred', 'TVGM+LSTM_pred', 'TVGM+TCN_pred', 'TVGM+Transformer_pred']
    model_names = [VERSION, f'{VERSION}+LSTM', f'{VERSION}+TCN', f'{VERSION}+Transformer']

    for key, name in zip(model_keys, model_names):
        pred_train = train_df['TVGM_pred'] if key == 'TVGM_pred' else pd.Series(np.nan, index=train_df.index)
        pred_test = test_df[key]

        true_train, pred_train = train_df['flow'].align(pred_train.dropna(), join='inner')
        true_test, pred_test = test_df['flow'].align(pred_test.dropna(), join='inner')

        metrics[name] = {
            'NSE_Train': nse_score(true_train, pred_train) if not pred_train.empty else np.nan,
            'R2_Train': r2_score(true_train, pred_train) if not pred_train.empty else np.nan,
            'RMSE_Train': np.sqrt(mean_squared_error(true_train, pred_train)) if not pred_train.empty else np.nan,
            'NSE_Test': nse_score(true_test, pred_test),
            'R2_Test': r2_score(true_test, pred_test),
            'RMSE_Test': np.sqrt(mean_squared_error(true_test, pred_test))
        }

    results_df = pd.DataFrame(metrics).T
    results_df.columns = pd.MultiIndex.from_tuples([(c.split('_')[0], c.split('_')[1]) for c in results_df.columns])
    results_df = results_df.reindex(columns=['NSE', 'R2', 'RMSE'], level=0).reindex(columns=['Train', 'Test'], level=1)

    print("\n=== Model Performance Comparison (Train & Test) ===")
    print(results_df.to_string(float_format="%.4f"))
    results_df.to_excel(f"metrics_results_{VERSION}_final.xlsx")

    output_folder = "output_results"
    os.makedirs(output_folder, exist_ok=True)
    train_results_df = train_df[['flow', 'TVGM_pred']].rename(
        columns={'flow': 'Observed', 'TVGM_pred': 'Predicted_TVGM'})
    train_results_df.to_csv(os.path.join(output_folder, 'train_set_results.csv'))
    test_results_df = test_df.rename(
        columns={'flow': 'Observed', 'TVGM_pred': 'Predicted_TVGM', 'TVGM+LSTM_pred': 'Predicted_TVGM+LSTM',
                 'TVGM+TCN_pred': 'Predicted_TVGM+TCN', 'TVGM+Transformer_pred': 'Predicted_TVGM+Transformer'})
    test_results_df[['Observed'] + [col for col in test_results_df.columns if 'Predicted' in col]].to_csv(
        os.path.join(output_folder, 'test_set_results.csv'))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_df = test_df.dropna(subset=['TVGM+LSTM_pred']).copy()
            color='red', linestyle='--')
    colors = {'LSTM': 'blue', 'TCN': 'green', 'Transformer': 'purple'}
    for name in dl_models_config.keys():
        full_name = f'{VERSION}+{name}'
        ax.plot(plot_df.index, plot_df[f'TVGM+{name}_pred'],

    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"prediction_comparison_{VERSION}_final.png", dpi=300)
    plt.show()

    print("\nProcess finished successfully.")