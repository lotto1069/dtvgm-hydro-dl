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
import warnings
import copy
import math

VERSION = "DTVGM_DL_v1.0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch is configured to use device: {DEVICE}")
if torch.cuda.is_available(): print(f"  - Device Name: {torch.cuda.get_device_name(0)}")
print("=" * 60 + "\n")

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
pd.options.mode.chained_assignment = None

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
    'ndvi': r"D:\Desktop\train\MODIS_NDVI_daily_interpolated-1984-2013",
    'temp': r"D:\Desktop\train\meteorology_1984_2013\xiinghe_001deg_tmpmean_flattened_tif",
    'rhu': r"D:\Desktop\train\meteorology_1984_2013\xiinghe_001deg_rhu_flattened_tif",
    'petpm': r"D:\Desktop\train\meteorology_1984_2013\xiinghe_001deg_petPM_flattened_tif",
    'prec': r"D:\Desktop\train\meteorology_1984_2013\xiinghe_001deg_prec_flattened_tif",
    'pres': r"D:\Desktop\train\meteorology_1984_2013\xiinghe_001deg_pres_flattened_tif",
    'wind': r"D:\Desktop\train\meteorology_1984_2013\xiinghe_001deg_windflat_tif", 'snow': r"D:\Desktop\SNOW"
}
flow_train_path = r"D:\Desktop\train\runoff1984-2013.xlsx"
validation_feature_paths = {
    'ndvi': r"D:\Desktop\test\MODIS_NDVI_daily_interpolated",
    'temp': r"D:\Desktop\test\xiinghe_001deg_tmpmean_flattened_tif",
    'rhu': r"D:\Desktop\test\xiinghe_001deg_rhu_flattened_tif",
    'petpm': r"D:\Desktop\test\xiinghe_001deg_petPM_flattened_tif",
    'prec': r"D:\Desktop\test\xiinghe_001deg_prec_flattened_tif",
    'pres': r"D:\Desktop\test\xiinghe_001deg_pres_flattened_tif",
    'wind': r"D:\Desktop\test\xiinghe_001deg_windflat_tif", 'snow': r"D:\Desktop\SNOW"
}
validation_stations = {
}
OUTPUT_DIR = "transformer_revolution_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINETUNE_TRANSFORMER_CONFIG = {
                     "split_ratio": None},
}
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=256, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_embed = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_size]
        src = self.input_embed(src) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        # PyTorch Transformer expects [seq_len, batch_size, d_model] if batch_first=False
        # Since we use batch_first=True, we keep it as [batch_size, seq_len, d_model]
        # But positional encoding is often added on seq_len first dim, so let's check
        # Our PositionalEncoding is designed for [seq_len, batch, d_model], let's adapt
        src = src.transpose(0, 1)  # [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # [batch_size, seq_len, d_model]

        memory = self.transformer_encoder(src)  # [batch_size, seq_len, d_model]

        # We use the output of the last time step for prediction
        output = self.output_head(memory[:, -1, :])  # [batch_size, 1]
        return output
def nse_score(y_true, y_pred): return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
def parse_date_from_filename(filename):
    match = re.search(r'(\d{4})(\d{2})(\d{2})|(\d{4})_(\d{2})_(\d{2})', os.path.basename(filename))
    if match:
        parts = [p for p in match.groups() if p is not None];
        year, month, day = map(int, parts[-3:])
        if month == 2 and day == 29 and not (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)): day = 28
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
    return None
def prepare_and_engineer_features(data_folders, flow_path, start_date, end_date, set_name):
    print(f"\n--- Loading data for period: {start_date} to {end_date} for {set_name} set ---")
    data = {}
    for var, folder_path in data_folders.items():
        var_data = []
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
        for file in tqdm(files, desc=f"Processing {var}", leave=False):
            date = parse_date_from_filename(file)
            if date and start_date <= date <= end_date:
                try:
                    with rasterio.open(file) as src:
                        array = src.read(1).astype(float);
                        nodata_val = src.nodata
                        if nodata_val is not None: array[array == nodata_val] = np.nan
                        if np.all(np.isnan(array)):
                            value = np.nan
                        elif var == 'snow' and np.nanmax(array) > 1:
                            value = np.sum((array == 100) | (array == 200)) / np.sum(np.isfinite(array)) if np.any(
                                np.isfinite(array)) else 0
                        else:
                            value = np.nanmean(array)
                        var_data.append({'date': date, var: value})
                except Exception as e:
                    print(f"Warning: Could not process file {file}. Error: {e}")
        if not var_data: continue
        df_single_var = pd.DataFrame(var_data).set_index('date');
        data[var] = df_single_var.groupby(df_single_var.index).mean()
    df_list = [df for df in data.values() if not df.empty]
    if not df_list: raise ValueError("No feature data loaded.")
    df_merged = pd.concat(df_list, axis=1).sort_index()
    flow_df = pd.read_excel(flow_path, header=0);
    flow_df.columns = ['date', 'flow']
    try:
        flow_df['date'] = pd.to_datetime(flow_df['date'], format='%Y%m%d')
    except (ValueError, TypeError):
        flow_df['date'] = pd.to_datetime(flow_df['date'])
    flow_df = flow_df.dropna(subset=['date']).set_index('date')
    if flow_df.index.has_duplicates: flow_df = flow_df.groupby(flow_df.index).mean()
    df_final = df_merged.join(flow_df, how='inner')
    if df_final.empty: raise ValueError(f"Dataframe empty after join for {set_name}.")
    df_filled = df_final.interpolate(method='time', limit_direction='both').fillna(method='bfill').fillna(
        method='ffill')
    if df_filled.isnull().values.any(): df_filled = df_filled.fillna(0)
    for lag in [1, 3, 7, 14, 30]:
        df_filled[f'prec_lag_{lag}'] = df_filled['prec'].shift(lag)
        df_filled[f'temp_lag_{lag}'] = df_filled['temp'].shift(lag)
        if 'flow' in df_filled.columns: df_filled[f'flow_lag_{lag}'] = df_filled['flow'].shift(lag)
    print("  - Applying advanced feature engineering (seasonality & rolling averages)...")
    df_filled['day_of_year'] = df_filled.index.dayofyear
    df_filled['sin_day'] = np.sin(2 * np.pi * df_filled['day_of_year'] / 365.25)
    df_filled['cos_day'] = np.cos(2 * np.pi * df_filled['day_of_year'] / 365.25)
    df_filled = df_filled.drop(columns=['day_of_year'])
    df_filled['prec_MA_7'] = df_filled['prec'].rolling(window=7, min_periods=1).mean()
    df_filled['temp_MA_7'] = df_filled['temp'].rolling(window=7, min_periods=1).mean()
    df_filled = df_filled.dropna()
    return df_filled
def create_sequences(input_data, target_data, seq_length):
    xs, ys = [], []
    for i in range(len(input_data) - seq_length):
        xs.append(input_data[i:(i + seq_length)])
        ys.append(target_data[i + seq_length])
    return np.array(xs), np.array(ys).reshape(-1, 1)
def train_or_finetune_dl_model(model, train_loader, val_loader, epochs, lr, patience, model_save_path):
    criterion = nn.MSELoss().to(DEVICE)
    print(f"  - Training {model.__class__.__name__} with stable MSE Loss and Gradient Clipping.")
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params_to_update, lr=lr, weight_decay=1e-5)  # Added weight decay for regularization
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience // 2, factor=0.5)
    best_val_loss, epochs_no_improve = float('inf'), 0
    pbar_desc = f"Fine-tuning {model.__class__.__name__}" if "finetune" in model_save_path else f"Pre-training {model.__class__.__name__}"
    pbar = tqdm(range(epochs), desc=pbar_desc, leave=False)
    for epoch in pbar:
        model.train()
        for seq, labels in train_loader:
            seq, labels = seq.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(seq)  # Model now returns only predictions
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval();
        val_loss = 0
        with torch.no_grad():
            for seq, labels in val_loader:
                outputs = model(seq.to(DEVICE))
                val_loss += criterion(outputs, labels.to(DEVICE)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss);
        pbar.set_postfix({'Val MSE Loss': f'{val_loss:.6f}'})
        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = val_loss, 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience: print(f"Early stopping at epoch {epoch + 1}."); break
    model.load_state_dict(torch.load(model_save_path));
    return model
def plot_hydrograph_simple(dates, observed, predicted, station_name, nse, save_path):
    plt.figure(figsize=(20, 8))
    plt.legend(loc='upper right');
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout();
    plt.savefig(save_path, dpi=300);
    plt.close()
def main_transformer_run():
    print("=" * 60);
    print("PART 1: Pre-training the Master Transformer Model");
    print("=" * 60)
    print("Step 1.1: Loading and engineering data...")
    train_df = prepare_and_engineer_features(train_paths, flow_train_path, datetime(1984, 1, 1), datetime(2013, 12, 31),
                                             "REFERENCE TRAIN")
    feature_cols = [col for col in train_df.columns if col != 'flow'];
    target_col = 'flow'
    print(f"\n- Using {len(feature_cols)} features for the model.")
    print("Step 1.2: Fitting the StandardScaler...");
    scaler = StandardScaler();
    X_train_scaled = scaler.fit_transform(train_df[feature_cols])
    print("Step 1.3: Training TVGM components...");
    train_df['api'] = train_df['prec'].ewm(alpha=0.1, adjust=False).mean();
    train_df['snow_melt'] = (train_df['temp'] > 0) * train_df['temp'] * train_df['snow']
    physical_features = ['api', 'snow_melt'];
    phys_model = LinearRegression();
    phys_model.fit(train_df[physical_features], train_df[target_col])
    q_base_train = phys_model.predict(train_df[physical_features]);
    ml_target = train_df[target_col] - q_base_train
    ml_booster = HistGradientBoostingRegressor(random_state=42, max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=0.5)
    ml_booster.fit(train_df[feature_cols], ml_target);
    q_ml_boost_train = ml_booster.predict(train_df[feature_cols]);
    q_tvgm_train = np.maximum(0, q_base_train + q_ml_boost_train);
    print("  - TVGM components pre-trained.")

    print("Step 1.4: Pre-training TransformerModel on TVGM residuals...")
    SEQ_LENGTH, BATCH_SIZE = 30, 64;
    residual_train = train_df[target_col].values - q_tvgm_train;
    split_idx = int(len(X_train_scaled) * 0.8)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled[:split_idx], residual_train[:split_idx], SEQ_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_train_scaled[split_idx:], residual_train[split_idx:], SEQ_LENGTH)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq)), BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq)), BATCH_SIZE)

    master_transformer_path = "master_transformer_final.pth"
    pre_trained_transformer = TransformerModel(input_size=len(feature_cols)).to(DEVICE)
    pre_trained_transformer = train_or_finetune_dl_model(pre_trained_transformer, train_loader, val_loader, epochs=100,
                                                         lr=1e-4, patience=10, model_save_path=master_transformer_path)
    print("  - Master Transformer model pre-trained and saved.")
    print("\nMaster Transformer Model is Ready for Customized Application.\n")

    print("=" * 60);
    print("PART 2: Applying Transformer and Validating");
    print("=" * 60)
    validation_results = {}
    for station_name, flow_path in validation_stations.items():
        try:
            print(f"\n--- Processing Station: {station_name} ---")
            config = FINETUNE_TRANSFORMER_CONFIG.get(station_name);
            print(f"  - Applying TRANSFORMER strategy: {config}")
            temp_flow_df = pd.read_excel(flow_path);
            temp_flow_df.columns = ['date', 'flow']
            try:
                temp_flow_df['date'] = pd.to_datetime(temp_flow_df['date'], format='%Y%m%d')
            except (ValueError, TypeError):
                temp_flow_df['date'] = pd.to_datetime(temp_flow_df['date'])
            start_date, end_date = temp_flow_df['date'].min(), temp_flow_df['date'].max()
            val_df = prepare_and_engineer_features(validation_feature_paths, flow_path, start_date, end_date,
                                                   f"DATASET-{station_name}")
            val_df = val_df[feature_cols + ['flow']]

            if config['finetune']:
                split_point = int(len(val_df) * config['split_ratio']); finetune_df = val_df.iloc[
                                                                                      :split_point]; test_df = val_df.iloc[
                                                                                                               split_point:]
            else:
                test_df = val_df

            if config['finetune']:
                print("  - Fine-tuning is ENABLED. Preparing data and model...")
                X_finetune_scaled = scaler.transform(finetune_df[feature_cols])
                finetune_df['api'] = finetune_df['prec'].ewm(alpha=0.1, adjust=False).mean();
                finetune_df['snow_melt'] = (finetune_df['temp'] > 0) * finetune_df['temp'] * finetune_df['snow']
                q_base_finetune = phys_model.predict(finetune_df[physical_features]);
                finetune_booster = copy.deepcopy(ml_booster)
                ml_target_finetune = finetune_df[target_col] - q_base_finetune;
                finetune_booster.fit(finetune_df[feature_cols], ml_target_finetune)
                q_ml_boost_finetune = finetune_booster.predict(finetune_df[feature_cols]);
                q_tvgm_finetune = np.maximum(0, q_base_finetune + q_ml_boost_finetune)
                residual_finetune = finetune_df[target_col].values - q_tvgm_finetune
                X_ft_seq, y_ft_seq = create_sequences(X_finetune_scaled, residual_finetune, SEQ_LENGTH)

                finetune_transformer = TransformerModel(len(feature_cols)).to(DEVICE)
                finetune_transformer.load_state_dict(torch.load(master_transformer_path))
                if config['freeze_layers']:
                    print("    - Freezing the first Transformer encoder layer...")
                    for name, param in finetune_transformer.named_parameters():
                        if 'transformer_encoder.layers.0' in name: param.requires_grad = False

                ft_train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_ft_seq), torch.FloatTensor(y_ft_seq)),
                                             batch_size=16, shuffle=True)
                finetuned_transformer_model = train_or_finetune_dl_model(
                    finetune_transformer, ft_train_loader, ft_train_loader, epochs=config['epochs'],
                    lr=config['learning_rate'], patience=5,
                    model_save_path=f"finetuned_transformer_{station_name}.pth"
                )
                final_booster = finetune_booster;
                final_dl_model = finetuned_transformer_model
            else:
                print("  - Fine-tuning is DISABLED. Using pre-trained master models directly.")
                final_booster = ml_booster;
                final_dl_model = pre_trained_transformer

            X_test_scaled = scaler.transform(test_df[feature_cols])
            test_df['api'] = test_df['prec'].ewm(alpha=0.1, adjust=False).mean();
            test_df['snow_melt'] = (test_df['temp'] > 0) * test_df['temp'] * test_df['snow']
            q_base_test = phys_model.predict(test_df[physical_features]);
            q_ml_boost_test = final_booster.predict(test_df[feature_cols])
            q_tvgm_test = np.maximum(0, q_base_test + q_ml_boost_test)
            residual_test = test_df[target_col].values - q_tvgm_test
            X_test_seq, _ = create_sequences(X_test_scaled, residual_test, SEQ_LENGTH)

            final_dl_model.eval()
            with torch.no_grad():
                res_dl_pred = final_dl_model(torch.FloatTensor(X_test_seq).to(DEVICE))
                res_dl_pred = res_dl_pred.cpu().numpy().flatten()

            final_pred_aligned = np.maximum(0, q_tvgm_test[SEQ_LENGTH:] + res_dl_pred);
            observed_aligned = test_df[target_col].values[SEQ_LENGTH:];
            dates_aligned = test_df.index[SEQ_LENGTH:]
            nse = nse_score(observed_aligned, final_pred_aligned);
            r2 = r2_score(observed_aligned, final_pred_aligned);
            rmse = np.sqrt(mean_squared_error(observed_aligned, final_pred_aligned))
            validation_results[station_name] = {'NSE': nse, 'R2': r2, 'RMSE': rmse}
            print(f"  - Final Metrics for {station_name} (Transformer): NSE={nse:.4f}, R2={r2:.4f}, RMSE={rmse:.4f}")

            results_df = pd.DataFrame({'Observed_Flow': observed_aligned, 'Predicted_Flow': final_pred_aligned},
                                      index=dates_aligned)
            results_df.to_csv(os.path.join(OUTPUT_DIR, f"predictions_{station_name}.csv"), encoding='utf-8-sig');
            plot_path = os.path.join(OUTPUT_DIR, f"plot_{station_name}.png")
            plot_hydrograph_simple(dates_aligned, observed_aligned, final_pred_aligned, station_name, nse, plot_path)
            print(f"  - Final results and plots saved to '{OUTPUT_DIR}'.")

        except Exception as e:
            print(f"!!!!!!!! FAILED to process station {station_name}. Error: {e} !!!!!!!!")
            import traceback;
            traceback.print_exc();
            validation_results[station_name] = {'NSE': np.nan, 'R2': np.nan, 'RMSE': np.nan}

    print("\n" + "=" * 60);
    print("PART 3: Transformer Revolution Summary");
    print("=" * 60)
    summary_df = pd.DataFrame(validation_results).T;
    print(summary_df.to_string(float_format="%.4f"))
    summary_path = os.path.join(OUTPUT_DIR, "transformer_summary.xlsx");
    summary_df.to_excel(summary_path)
    print("\nTransformer Revolution process finished. This tests the architectural limits.")
if __name__ == "__main__":
    main_transformer_run()