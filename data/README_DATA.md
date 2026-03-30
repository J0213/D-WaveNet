# Data Acquisition and Preprocessing Guide

## Data Source

The datasets used in this study were obtained from the **National Marine Data Center National Science and Technology Resource Sharing Platform (NMDCNS-TRSSPC)**:

- **URL**: http://mds.nmdis.org.cn/
- **Data Format**: Fixed-width ASCII text (see format specification below)
- **Temporal Resolution**: Hourly observations
- **Variable**: Significant Wave Height (SWH, columns 59-61 of Data Record, in units of 0.1 m)

## Monitoring Stations

| Station Code | Station Name | Latitude | Longitude | Mean Depth |
|-------------|-------------|----------|-----------|------------|
| 001 | ShiDao | 36.89N | 122.43E | ~5m |
| 002 | XiaoMaiDao | 35.95N | 120.41E | <10m |
| 003 | LianYunGang | 34.79N | 119.39E | <10m |

## Step 1: Obtain Raw Data

Download the wave and wind data (delayed mode) from NMDCNS-TRSSPC and place all `.txt` files into `data/raw/`:

Raw files follow the naming pattern `YYYYMMNNN.txt` (e.g., `202107001.txt` = July 2021, ShiDao).

## Step 2: Parse Raw Data to CSV

```bash
python scripts/parse_raw_data.py --input_dir ./data/raw --output_dir ./data
```

This produces: `data/ShiDao.csv`, `data/XiaoMaiDao.csv`, `data/LianYunGang.csv`

Each CSV contains columns: `datetime` (YYYY-MM-DD HH:MM:SS) and `SWH` (meters).

## Step 3: Run D-WaveNet

```bash
python run.py --data_path ./data/LianYunGang.csv --dataset_name LianYunGang --pred_len 168
```

## Key Columns in Data Record (1-indexed)

| Column | Parameter | Format |
|--------|-----------|--------|
| 59-61 | **Significant wave height (SWH)** | **0.1 m** |
| 63-65 | SWH period | 0.1 s |

Missing/invalid codes: `997`, `999`, blank

## Important Note on Data Reproducibility

The experimental results reported in Table 2 of the manuscript were generated using observation data spanning **January 2014 to December 2022** (approximately 7-8 years per station, yielding 63,900-77,800 effective data points as detailed in Table 1).

Due to the data platform's rolling access policies, the specific historical data partitions available for download may vary depending on the time of access. Users who obtain data covering a different temporal range should expect **numerically different (though qualitatively consistent)** performance metrics.

The code and model architecture are fully reproducible. Minor variations in absolute metric values arise solely from differences in the underlying observation time series, not from the implementation. The key qualitative findings -- such as the sub-linear RMSE growth, the superiority of cross-scale attention over decoupled encoders, and the physical consistency of predictions -- are robust to the choice of temporal window.

## Quick Verification with Synthetic Data

To verify that the code pipeline runs correctly without real buoy data:

```bash
python scripts/generate_synthetic_data.py
python run.py --synthetic --pred_len 24 --train_epochs 3
```

**Note**: Synthetic data results will NOT match the paper's reported metrics.
