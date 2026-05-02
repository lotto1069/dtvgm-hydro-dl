# Data Availability and Format Description

## Data Sources

The input data for this study are derived from the following sources:

### Meteorological Forcing Data
- **Source**: ERA5-Land reanalysis (Copernicus Climate Data Store)
- **Variables**: Temperature, Precipitation, Relative Humidity, Wind Speed, Surface Pressure, Potential Evapotranspiration (Penman-Monteith)
- **Spatial Resolution**: 0.01° (~1 km), resampled to the study area
- **Temporal Resolution**: Daily
- **Format**: GeoTIFF (.tif) files, one per day, with date in filename (YYYYMMDD)

### Vegetation Index
- **Source**: MODIS MOD13A2 (NASA LP DAAC)
- **Variable**: Normalized Difference Vegetation Index (NDVI)
- **Spatial Resolution**: 0.01° (~1 km)
- **Temporal Resolution**: 16-day, interpolated to daily
- **Format**: GeoTIFF (.tif)

### Snow Cover
- **Source**: MODIS MOD10A1 (NASA NSIDC)
- **Variable**: Snow cover fraction
- **Spatial Resolution**: 0.01° (~1 km)
- **Temporal Resolution**: Daily
- **Format**: GeoTIFF (.tif)
- **Processing**: Snow pixel ratio calculated per basin (pixels = 100 or 200 / total valid pixels)

### Observed Runoff
- **Source**: Xilin River Basin gauging stations
- **Stations**: Langrengu (狼人谷), Jiuquwan (九曲湾), Hadeng (哈登)
- **Temporal Resolution**: Daily
- **Period**: 1984-2022
- **Format**: Excel (.xlsx) with columns: date (YYYYMMDD), flow (m³/s)

## Data Directory Structure

Users should organize their data as follows:

```
data/
├── train/
│   ├── MODIS_NDVI_daily_interpolated-1984-2013/
│   │   ├── 19840101.tif
│   │   ├── 19840102.tif
│   │   └── ...
│   ├── xiinghe_001deg_tmpmean_flattened_tif/
│   ├── xiinghe_001deg_rhu_flattened_tif/
│   ├── xiinghe_001deg_petPM_flattened_tif/
│   ├── xiinghe_001deg_prec_flattened_tif/
│   ├── xiinghe_001deg_pres_flattened_tif/
│   ├── xiinghe_001deg_windflat_tif/
│   ├── SNOW/
│   └── runoff_1984_2013.xlsx
├── test/
│   ├── MODIS_NDVI_daily_interpolated/
│   ├── xiinghe_001deg_tmpmean_flattened_tif/
│   ├── xiinghe_001deg_rhu_flattened_tif/
│   ├── xiinghe_001deg_petPM_flattened_tif/
│   ├── xiinghe_001deg_prec_flattened_tif/
│   ├── xiinghe_001deg_pres_flattened_tif/
│   ├── xiinghe_001deg_windflat_tif/
│   ├── SNOW/
│   └── runoff_2013_2022.xlsx
└── stations/
    ├── langrengu.xlsx
    ├── jiuquwan.xlsx
    └── hadeng.xlsx
```

## Data Availability Statement

Due to the size of the meteorological and remote sensing datasets (>20 GB), the raw TIF files are not included in this repository. Researchers can obtain the data from:

1. **ERA5-Land**: https://cds.climate.copernicus.eu/
2. **MODIS NDVI (MOD13A2)**: https://lpdaac.usgs.gov/products/mod13a2v061/
3. **MODIS Snow Cover (MOD10A1)**: https://nsidc.org/data/mod10a1

The observed runoff data from the gauging stations may be available upon reasonable request to the corresponding author, subject to data-sharing agreements with the monitoring agencies.

A minimal example dataset with synthetic data is provided for testing the code pipeline:
- `data/train_set_results.csv`: Training period model predictions
- `data/test_set_results.csv`: Test period model predictions
- `data/driving_factors_importance_*.csv`: Feature importance results

## Feature Engineering

The following features are automatically engineered from the raw data:

| Feature | Description |
|---------|-------------|
| `prec_lag_{1,3,7,14,30}` | Precipitation lagged by N days |
| `temp_lag_{1,3,7,14,30}` | Temperature lagged by N days |
| `flow_lag_{1,3,7,14,30}` | Runoff lagged by N days (autoregressive) |
| `sin_day`, `cos_day` | Seasonal encoding (day of year) |
| `prec_MA_7` | 7-day moving average of precipitation |
| `temp_MA_7` | 7-day moving average of temperature |
| `api` | Antecedent Precipitation Index |
| `snow_melt` | Snowmelt proxy (T>0 * T * snow_fraction) |
