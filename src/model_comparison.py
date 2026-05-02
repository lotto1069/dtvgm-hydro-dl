import os
import re
import pandas as pd
import numpy as np
import rasterio
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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
    plt.rcParams['font.family'] = 'Arial'
except Exception:
    print("Arial font not found, using Matplotlib default.")
plt.rcParams['axes.unicode_minus'] = False

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
OUTPUT_DIR = "model_comparison_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        return self.fc(last_hidden_state)
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
    def __init__(self, input_size, d_model=128, nhead=8, num_encoder_layers=2, dim_feedforward=256, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.input_embed = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_embed(src)
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        output = self.transformer_encoder(src)
        return self.output_head(output[:, -1, :])
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
class TCNModel(nn.Module):
    def __init__(self, input_size, output_size=1, num_channels=[64, 128], kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        self.input_embed = nn.Linear(input_size, num_channels[0])
        self.tcn = TemporalConvNet(num_channels[0], num_channels, kernel_size=kernel_size, dropout=dropout)
        self.output_head = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):  # inputs: (batch, seq_len, input_size)
        embedded = self.input_embed(inputs)  # (batch, seq_len, num_channels[0])
        y = embedded.permute(0, 2, 1)  # (batch, num_channels[0], seq_len) for TCN
        y = self.tcn(y)  # (batch, num_channels[-1], seq_len)
        return self.output_head(y[:, :, -1])  # (batch, output_size)
def nse_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
def pearson_r2_score(y_true, y_pred):
    """
    """
    if len(y_true) < 2: return np.nan
    return np.corrcoef(y_true, y_pred)[0, 1] ** 2
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience // 2, factor=0.5)
    best_val_loss, epochs_no_improve = float('inf'), 0
    pbar_desc = f"Fine-tuning {model.__class__.__name__}" if "finetune" in model_save_path else f"Pre-training {model.__class__.__name__}"
    pbar = tqdm(range(epochs), desc=pbar_desc, leave=False)
    for epoch in pbar:
        model.train()
        for seq, labels in train_loader:
            seq, labels = seq.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(seq);
            loss = criterion(outputs, labels)
            loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval();
        val_loss = 0
        with torch.no_grad():
            for seq, labels in val_loader: val_loss += criterion(model(seq.to(DEVICE)), labels.to(DEVICE)).item()
        val_loss /= len(val_loader);
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
def plot_model_comparison(dates, observed, predictions_dict, metrics_dict, station_name, save_path):
    """
    Generates and saves an academic-style plot comparing model predictions against observed data.
    """
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
    })

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(dates, observed, label='Observed', color='black', linewidth=2.5, zorder=10, linestyle='-', alpha=0.9)

    colors = {
        'LSTM': '#0072B2',
        'Transformer': '#D55E00',
        'TCN': '#009E73'
    }
    for model_name, preds in predictions_dict.items():
        metrics = metrics_dict[model_name]
        nse = metrics['NSE']
        r2 = metrics['R2']
        ax.plot(dates, preds, label=label, color=colors[model_name], linestyle='-', alpha=0.8, linewidth=1.5)

    ax.set_title(f'Daily Streamflow Prediction at {station_name} Station', fontweight='bold', pad=20, y=0.98)
    ax.set_xlabel('Date', labelpad=10)
    ax.set_ylabel('Streamflow ($m^3/s$)', labelpad=10)

    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray', facecolor='white', labelspacing=0.8)
    ax.grid(True, linestyle='--', color='gray', alpha=0.3, linewidth=0.8)

    ax.tick_params(axis='x', rotation=30, ha='right', pad=5)
    ax.tick_params(axis='y', pad=5)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close(fig)
def main_model_comparison():
    print("=" * 60);
    print("PART 1: Pre-training Master Models");
    print("=" * 60)
    print("Step 1.1: Loading and engineering reference data...")
    train_df = prepare_and_engineer_features(train_paths, flow_train_path, datetime(1984, 1, 1), datetime(2013, 12, 31),
                                             "REFERENCE TRAIN")
    feature_cols = [col for col in train_df.columns if col != 'flow'];
    target_col = 'flow'
    print(f"\n- Using {len(feature_cols)} features for all models.")
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

    print("Step 1.4: Pre-training all DL models on TVGM residuals...")
    SEQ_LENGTH, BATCH_SIZE = 30, 64;
    residual_train = train_df[target_col].values - q_tvgm_train;
    split_idx = int(len(X_train_scaled) * 0.8)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled[:split_idx], residual_train[:split_idx], SEQ_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_train_scaled[split_idx:], residual_train[split_idx:], SEQ_LENGTH)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq)), BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq)), BATCH_SIZE)

    models_to_train = {
        "LSTM": LSTMModel(input_size=len(feature_cols)),
        "Transformer": TransformerModel(input_size=len(feature_cols)),
        "TCN": TCNModel(input_size=len(feature_cols))
    }
    for model_name, model_instance in models_to_train.items():
        print(f"\n--- Pre-training Master {model_name} Model ---")
        model_path = f"master_{model_name.lower()}.pth"
        model_instance.to(DEVICE)
        train_or_finetune_dl_model(model_instance, train_loader, val_loader, epochs=100, lr=1e-4, patience=10,
                                   model_save_path=model_path)
        print(f"  - Master {model_name} model saved to {model_path}.")
    print("\nAll Master Models are Ready for Evaluation.\n")

    print("=" * 60);
    print(f"PART 2: Model Evaluation on '{validation_station_name}' Station");
    print("=" * 60)
    print(f"Step 2.1: Loading validation data for {validation_station_name}...")
    temp_flow_df = pd.read_excel(validation_station_path);
    temp_flow_df.columns = ['date', 'flow']
    try:
        temp_flow_df['date'] = pd.to_datetime(temp_flow_df['date'], format='%Y%m%d')
    except (ValueError, TypeError):
        temp_flow_df['date'] = pd.to_datetime(temp_flow_df['date'])
    start_date, end_date = temp_flow_df['date'].min(), temp_flow_df['date'].max()
    val_df = prepare_and_engineer_features(validation_feature_paths, validation_station_path, start_date, end_date,
                                           f"DATASET-{validation_station_name}")
    val_df = val_df[feature_cols + ['flow']]

    split_ratio = 0.7;
    split_point = int(len(val_df) * split_ratio)
    finetune_df, test_df = val_df.iloc[:split_point], val_df.iloc[split_point:]

    print("\nStep 2.2: Preparing fine-tuning data...")
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
    ft_train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_ft_seq), torch.FloatTensor(y_ft_seq)), batch_size=16,
                                 shuffle=True)

    print("\nStep 2.3: Preparing test data...")
    X_test_scaled = scaler.transform(test_df[feature_cols])
    test_df['api'] = test_df['prec'].ewm(alpha=0.1, adjust=False).mean();
    test_df['snow_melt'] = (test_df['temp'] > 0) * test_df['temp'] * test_df['snow']
    q_base_test = phys_model.predict(test_df[physical_features]);
    q_ml_boost_test = finetune_booster.predict(test_df[feature_cols])
    q_tvgm_test = np.maximum(0, q_base_test + q_ml_boost_test)
    residual_test = test_df[target_col].values - q_tvgm_test
    X_test_seq, _ = create_sequences(X_test_scaled, residual_test, SEQ_LENGTH)
    observed_aligned = test_df[target_col].values[SEQ_LENGTH:];
    dates_aligned = test_df.index[SEQ_LENGTH:]

    all_predictions, all_metrics = {}, {}

    print("\nStep 2.4: Running each model through fine-tuning and prediction...")
    for model_name, model_class in [("LSTM", LSTMModel), ("Transformer", TransformerModel), ("TCN", TCNModel)]:
        print(f"\n--- Processing Model: {model_name} ---")
        model = model_class(input_size=len(feature_cols)).to(DEVICE)
        model.load_state_dict(torch.load(f"master_{model_name.lower()}.pth"))

        finetuned_model = train_or_finetune_dl_model(
            model, ft_train_loader, ft_train_loader, epochs=40, lr=5e-5, patience=5,
            model_save_path=f"finetuned_{model_name.lower()}_{validation_station_name}.pth"
        )

        finetuned_model.eval()
        with torch.no_grad():
            res_dl_pred = finetuned_model(torch.FloatTensor(X_test_seq).to(DEVICE)).cpu().numpy().flatten()

        final_pred = np.maximum(0, q_tvgm_test[SEQ_LENGTH:] + res_dl_pred)
        all_predictions[model_name] = final_pred

        nse = nse_score(observed_aligned, final_pred)
        rmse = np.sqrt(mean_squared_error(observed_aligned, final_pred))

        all_metrics[model_name] = {'NSE': nse, 'R2': r2, 'RMSE': rmse}
        print(f"  - Final Metrics for {model_name}: NSE={nse:.4f}, R2={r2:.4f}, RMSE={rmse:.4f}")

    print("\n" + "=" * 60);
    print("PART 3: Results Summary and Visualization");
    print("=" * 60)
    summary_df = pd.DataFrame(all_metrics).T
    print(summary_df.to_string(float_format="%.4f"))
    summary_path = os.path.join(OUTPUT_DIR, "model_comparison_summary.xlsx")
    summary_df.to_excel(summary_path)

    flow_data_df = pd.DataFrame({
        'Date': dates_aligned,
        'Observed_Flow': observed_aligned,
        'LSTM_Predicted': all_predictions['LSTM'],
        'Transformer_Predicted': all_predictions['Transformer'],
        'TCN_Predicted': all_predictions['TCN']
    })
    flow_data_path = os.path.join(OUTPUT_DIR, "flow_observation_prediction_data.xlsx")
    flow_data_df.to_excel(flow_data_path, index=False)

    plot_path = os.path.join(OUTPUT_DIR, f"model_comparison_plot_{validation_station_name}.png")
    plot_model_comparison(dates_aligned, observed_aligned, all_predictions, all_metrics, validation_station_name,
                          plot_path)

    print("\nModel comparison process finished successfully.")
if __name__ == "__main__":
    main_model_comparison()