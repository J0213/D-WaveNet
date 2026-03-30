"""
Generate Synthetic SWH Data for Code Verification
===================================================
Creates physically plausible synthetic SWH time series for testing
the D-WaveNet pipeline when real buoy data is not available.

WARNING: Results obtained with synthetic data will NOT reproduce
the paper's reported metrics. This script is provided solely for
code verification purposes.

Usage:
    python scripts/generate_synthetic_data.py
"""

import os
import numpy as np
import pandas as pd


def generate_synthetic_swh(n_hours=70000, seed=42):
    """
    Generate a synthetic SWH time series that mimics realistic
    wave dynamics: seasonal variation, swell cycles, wind-sea
    fluctuations, and occasional extreme events.

    Parameters
    ----------
    n_hours : int
        Number of hourly data points (~8 years)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame with 'datetime' and 'SWH' columns
    """
    np.random.seed(seed)
    t = np.arange(n_hours)

    # Base swell: low-frequency seasonal component
    annual_cycle = 0.4 * np.sin(2 * np.pi * t / (24 * 365) - np.pi / 2)
    monthly_cycle = 0.15 * np.sin(2 * np.pi * t / (24 * 30))

    # Mean SWH baseline (typical for China's eastern coast)
    baseline = 0.8

    # Wind-sea: high-frequency stochastic component
    wind_sea = 0.2 * np.sin(2 * np.pi * t / 12) + 0.15 * np.random.randn(n_hours)

    # Diurnal variation (sea-land breeze effect)
    diurnal = 0.08 * np.sin(2 * np.pi * t / 24)

    # Occasional extreme events (typhoons) — ~3 events per year
    extreme_events = np.zeros(n_hours)
    n_events = int(n_hours / (24 * 365) * 3)
    event_centers = np.random.choice(range(24 * 30, n_hours - 24 * 30), n_events, replace=False)
    for center in event_centers:
        duration = np.random.randint(48, 120)  # 2-5 days
        peak = np.random.uniform(2.0, 5.0)
        event_shape = peak * np.exp(-0.5 * ((np.arange(n_hours) - center) / (duration / 4)) ** 2)
        extreme_events += event_shape

    # Combine all components
    swh = baseline + annual_cycle + monthly_cycle + wind_sea + diurnal + extreme_events

    # Ensure physical validity: SWH must be positive
    swh = np.maximum(swh, 0.05)

    # Apply depth-induced ceiling (~0.78 * 10m = 7.8m)
    swh = np.minimum(swh, 7.8)

    # Create DataFrame
    start_date = pd.Timestamp('2014-01-01')
    dates = pd.date_range(start=start_date, periods=n_hours, freq='h')
    df = pd.DataFrame({
        'datetime': dates,
        'SWH': np.round(swh, 3)
    })

    return df


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(output_dir, exist_ok=True)

    stations = {
        'ShiDao': {'seed': 42, 'n_hours': 63900},
        'XiaoMaiDao': {'seed': 123, 'n_hours': 77800},
        'LianYunGang': {'seed': 456, 'n_hours': 66000},
    }

    for name, params in stations.items():
        print(f"Generating synthetic data for {name}...")
        df = generate_synthetic_swh(n_hours=params['n_hours'], seed=params['seed'])
        path = os.path.join(output_dir, f'{name}.csv')
        df.to_csv(path, index=False)
        print(f"  Saved: {path} ({len(df)} records)")

    print("\n[WARNING] Synthetic data generated for CODE VERIFICATION ONLY.")
    print("Replace with real buoy data from http://mds.nmdis.org.cn/ to reproduce paper results.")


if __name__ == '__main__':
    main()
