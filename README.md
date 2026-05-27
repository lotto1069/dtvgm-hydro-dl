# DTVGM-DL: A Hybrid Physics-Deep Learning Framework for Runoff Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Code repository for the paper.

**Data DOI**: [10.17632/wfgsnpdpm3.3](https://doi.org/10.17632/wfgsnpdpm3.3) (Mendeley Data)

## Overview

This repository provides a hybrid hydrological modeling framework that couples the **Distributed Time-Variant Gain Model (DTVGM)** with deep learning residual correction. The framework simulates daily runoff in semi-arid watersheds by combining:

- **Physical base model**: DTVGM with antecedent precipitation index (API) and snowmelt modules
- **ML booster**: Histogram-based Gradient Boosting for learning complex physical residuals
- **Deep learning correctors**: BiLSTM-Attention, TCN, and Transformer models for temporal residual correction

The framework was developed and tested on the **Xilin River Basin** in Inner Mongolia, China, using daily meteorological forcing data and MODIS NDVI from 1984-2022.

## Key Features

- Multi-source data fusion (meteorology, NDVI, snow cover, runoff)
- Physics-guided feature engineering (lag effects, seasonal encoding, rolling statistics)
- Three deep learning architectures for residual correction: BiLSTM-Attention, TCN, Transformer
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
│   ├── model_comparison.py            # Fair comparison of BiLSTM-Attention vs TCN vs Transformer
│   ├── transfer_learning_3station.py  # Transfer learning across 3 stations (exploratory)
│   ├── finetune_3station.py           # Fine-tuning experiments (BiLSTM-based)
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
git clone https://github.com/lotto1069/dtvgm-hydro-dl.git
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
3. Trains BiLSTM-Attention, TCN, and Transformer residual correctors
4. Evaluates all models (NSE, R², RMSE)
5. Saves results to `outputs/`

### 3. Model Comparison (Single Station)

```bash
python src/model_comparison.py
```

Fairly compares BiLSTM-Attention, TCN, and Transformer at the Langrengu station with correct R²/NSE metrics.

### 4. Transfer Learning Across Stations

```bash
python src/transfer_learning_3station.py
```

Exploratory transfer learning experiments across three stations.

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

## Data Availability

The raw meteorological, NDVI, snow cover, and observed runoff datasets are archived at Mendeley Data:

[![DOI](https://img.shields.io/badge/DOI-10.17632%2Fwfgsnpdpm3.3-blue)](https://doi.org/10.17632/wfgsnpdpm3.3)

See [data/README.md](data/README.md) for detailed data format and source descriptions.

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
- **ResidualBiLSTM**: 2-layer BiLSTM with attention mechanism and residual connection
- **ResidualTCN**: Temporal Convolutional Network with dilated convolutions
- **ResidualTransformer**: Transformer Encoder with positional encoding

## Results

Performance metrics on the test set (2014-2022):

| Model | NSE |
|-------|-----|
| DTVGM | 0.855 |
| DTVGM+BiLSTM-Attention | 0.940 |
| DTVGM+TCN | 0.947 |
| DTVGM+Transformer | 0.760 |

DTVGM+TCN achieves the best performance, reducing high-flow RMSE by 25.4% relative to the physical baseline. DTVGM+BiLSTM-Attention follows closely (NSE = 0.940). The Transformer architecture underperforms, failing to improve upon the DTVGM baseline in this residual-correction context.

## Citation

If you use this code in your research, please cite:

Zhao, S., Duan, L., Wang, Y., Luo, Y., Wang, X., Singh, V. P., & Liu, T. (2026). Hybrid Streamflow Simulation in a Semi-Arid Basin Using an Enhanced Distributed Time-Variant Gain Model and Deep Learning Residual Correction. *Submitted*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration, please contact: 18018351069@qq.com
