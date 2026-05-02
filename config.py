"""
Configuration file for the DTVGM-DL Hydrological Modeling project.
Modify the paths below to match your local data and output directories.
"""

import os

# Data Paths - MODIFY THESE to match your local setup

# Training data (1984-2013)
TRAIN_PATHS = {
    'ndvi': r"path/to/train/MODIS_NDVI_daily_interpolated-1984-2013",
    'temp': r"path/to/train/xiinghe_001deg_tmpmean_flattened_tif",
    'rhu': r"path/to/train/xiinghe_001deg_rhu_flattened_tif",
    'petpm': r"path/to/train/xiinghe_001deg_petPM_flattened_tif",
    'prec': r"path/to/train/xiinghe_001deg_prec_flattened_tif",
    'pres': r"path/to/train/xiinghe_001deg_pres_flattened_tif",
    'wind': r"path/to/train/xiinghe_001deg_windflat_tif",
    'snow': r"path/to/train/SNOW"
}
FLOW_TRAIN_PATH = r"path/to/train/runoff_1984_2013.xlsx"

# Test/validation data (2014-2022)
TEST_PATHS = {
    'ndvi': r"path/to/test/MODIS_NDVI_daily_interpolated",
    'temp': r"path/to/test/xiinghe_001deg_tmpmean_flattened_tif",
    'rhu': r"path/to/test/xiinghe_001deg_rhu_flattened_tif",
    'petpm': r"path/to/test/xiinghe_001deg_petPM_flattened_tif",
    'prec': r"path/to/test/xiinghe_001deg_prec_flattened_tif",
    'pres': r"path/to/test/xiinghe_001deg_pres_flattened_tif",
    'wind': r"path/to/test/xiinghe_001deg_windflat_tif",
    'snow': r"path/to/test/SNOW"
}
FLOW_TEST_PATH = r"path/to/test/runoff_2013_2022.xlsx"

# Station-specific validation data
VALIDATION_STATIONS = {
    "Langrengu": r"path/to/langrengu.xlsx",
    "Jiuquwan": r"path/to/jiuquwan.xlsx",
    "Hadeng": r"path/to/hadeng.xlsx"
}

# Model Configuration
VERSION = "TVGM_v15.7"
DEVICE = "cuda"  # or "cpu"
SEQ_LENGTH = 30
BATCH_SIZE = 64

# Output Paths
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
