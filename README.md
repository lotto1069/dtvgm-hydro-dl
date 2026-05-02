# DTVGM-DL: A Hybrid Physics-Deep Learning Framework for Runoff Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Code repository for the paper submitted to **Environmental Modelling & Software**.

## Overview

This repository provides a hybrid hydrological modeling framework that couples the **Distributed Time-Variant Gain Model (DTVGM)** with deep learning residual correction. The framework simulates daily runoff in semi-arid watersheds by combining:

- **Physical base model**: DTVGM with antecedent precipitation index (API) and snowmelt modules
- **ML booster**: Histogram-based Gradient Boosting for learning complex physical residuals
- **Deep learning correctors**: LSTM, TCN, and Transformer models for temporal residual correction

The framework was developed and tested on the **Xilin River Basin** in Inner Mongolia, China, using daily meteorological forcing data and MODIS NDVI from 1984-2022.

## Key Features

- Multi-source data fusion (meteorology, NDVI, snow cover, runoff)
- Physics-guided feature engineering (lag effects, seasonal encoding, rolling statistics)
- Three deep learning architectures for residual correction: LSTM, TCN, Transformer
- Transfer learning across gauging stations
- SHAP-based driver importance analysis
- Publication-quality visualization suite

## Repository Structure

```
├── config.py                          # Centralized path and parameter configuration
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── README.md                          # This file
├── src/
│   ├── main_train_evaluate.py         # Main pipeline: train DTVGM+DL, evaluate, output results
│   ├── model_comparison.py            # Fair comparison of LSTM vs TCN vs Transformer
│   ├── transfer_learning_3station.py  # Transfer learning across 3 stations (Transformer)
│   ├── finetune_3station.py           # Fine-tuning experiments (LSTM-based)
│   ├── shap_driver_analysis.py        # SHAP-based scientific driver analysis
│   ├── driving_factor_analysis.py     # Permutation importance analysis
│   ├── feature_importance_analysis.py # Feature importance with XGBoost
│   └── *_archive.py                   # Archived intermediate versions
├── scripts/
│   ├── plot_figure7.py                # Generate Figure 7: 4-panel model evaluation
│   ├── plot_four_panel_figure.py      # 4-panel figure (Chinese labels)
│   ├── plot_three_station.py          # Three-station comparison plots
│   ├── plot_animation.py              # Animated runoff hydrograph
│   ├── plot_stmx_eoc.py              # Scatter plots
│   ├── kg_analysis.py                 # Kling-Gupta efficiency analysis
│   └── kg_runoff.py                   # Additional KG metrics
├── data/
│   ├── README.md                      # Data availability and format description
│   ├── test_set_results.csv           # Test set predictions (2014-2022)
│   └── train_set_results.csv          # Training set predictions (1984-2013)
└── outputs/                           # Generated figures and results
```

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/dtvgm-hydro-dl.git
cd dtvgm-hydro-dl

# Create and activate conda environment (recommended)
conda create -n hydro_dl python=3.9
conda activate hydro_dl

# Install dependencies
pip install -r requirements.txt
```

**Note**: PyTorch installation may vary by system. See [pytorch.org](https://pytorch.org) for platform-specific instructions (CPU/CUDA).

## Usage

### 1. Configure Data Paths

Edit `config.py` to set the paths to your local data directories (see [data/README.md](data/README.md) for data format requirements).

### 2. Run the Main Training & Evaluation Pipeline

```bash
python src/main_train_evaluate.py
```

This script:
1. Loads and engineers features from TIF raster files and Excel runoff data
2. Trains the DTVGM physical + ML booster model
3. Trains LSTM, TCN, and Transformer residual correctors
4. Evaluates all models (NSE, R², RMSE)
5. Saves results to `outputs/`

### 3. Model Comparison (Single Station)

```bash
python src/model_comparison.py
```

Fairly compares LSTM, TCN, and Transformer at the Langrengu station with correct R²/NSE metrics.

### 4. Transfer Learning Across Stations

```bash
python src/transfer_learning_3station.py
```

Pre-trains a master Transformer model and applies/fine-tunes it across three stations.

### 5. Driver Analysis

```bash
python src/shap_driver_analysis.py
```

Generates SHAP summary plots, dependence plots, and interaction plots for identifying key runoff drivers.

### 6. Generate Figures

```bash
python scripts/plot_figure7.py      # 4-panel figure (radar, bar, KDE, radial)
python scripts/plot_three_station.py # Three-station hydrograph comparison
python scripts/plot_animation.py     # Animated runoff GIF
```

## Data Requirements

The model requires the following input data (see `data/README.md` for detailed format):

| Variable | Format | Resolution | Source |
|----------|--------|------------|--------|
| Temperature | GeoTIFF | 0.01° daily | ERA5-Land |
| Precipitation | GeoTIFF | 0.01° daily | ERA5-Land |
| Relative Humidity | GeoTIFF | 0.01° daily | ERA5-Land |
| PET (PM) | GeoTIFF | 0.01° daily | Calculated |
| Wind Speed | GeoTIFF | 0.01° daily | ERA5-Land |
| Surface Pressure | GeoTIFF | 0.01° daily | ERA5-Land |
| NDVI | GeoTIFF | 0.01° daily | MODIS MOD13A2 |
| Snow Cover | GeoTIFF | 0.01° daily | MODIS MOD10A1 |
| Observed Runoff | Excel (.xlsx) | Daily | Gauging station |

## Model Architecture

### DTVGM Physical Module
The Distributed Time-Variant Gain Model provides the physical backbone:
- **API (Antecedent Precipitation Index)**: Exponential weighted mean of precipitation
- **Snowmelt**: Temperature-threshold snowmelt module
- **Base flow**: Linear regression on physical features

### ML Booster (Residual Learning)
- **Histogram Gradient Boosting** (HistGradientBoostingRegressor)
- Learns the residual between observed runoff and DTVGM physical predictions
- Feature importance via permutation importance

### Deep Learning Correctors
Three architectures correct the remaining temporal residuals:
- **ResidualLSTM**: 2-layer LSTM with residual connection
- **ResidualTCN**: Temporal Convolutional Network with dilation
- **ResidualTransformer**: Transformer Encoder with positional encoding

## Results

Performance metrics on the test set (2014-2022):

| Model | NSE | R² | RMSE (m³/s) |
|-------|-----|-----|-------------|
| DTVGM | 0.855 | 0.856 | 0.236 |
| DTVGM+LSTM | 0.886 | 0.887 | 0.212 |
| DTVGM+TCN | 0.871 | 0.877 | 0.225 |
| DTVGM+Transformer | 0.860 | 0.865 | 0.219 |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your2025paper,
  title={Your Paper Title},
  author={Your Name et al.},
  journal={Environmental Modelling & Software},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration, please contact: [your-email]
