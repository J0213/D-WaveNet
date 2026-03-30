"""
NMDCNS Raw Data Parser
=======================
Parses the fixed-width format wave and wind data files from the
National Marine Data Center (NMDCNS-TRSSPC, http://mds.nmdis.org.cn/)
and exports them as CSV files compatible with the D-WaveNet data loader.

Input format: YYYYMMNNN.txt (fixed-width ASCII, see data/README_DATA.md)
Output format: CSV with columns [datetime, SWH]

Station codes:
  001 = ShiDao
  002 = XiaoMaiDao
  003 = LianYunGang

Usage:
    python scripts/parse_raw_data.py --input_dir ./data/raw --output_dir ./data

Notes:
    - Records with SWH code 997 or 999 are treated as missing values
    - SWH values are stored as integers in units of 0.1 m (e.g., 35 = 3.5 m)
    - Duplicate timestamps are removed (keeping the first occurrence)
    - The SWH column (columns 59-61 in the data record) corresponds to
      "Significant wave height" as defined in the format specification
"""

import os
import sys
import glob
import csv
import argparse
from datetime import datetime


STATION_NAMES = {
    1: 'ShiDao',
    2: 'XiaoMaiDao',
    3: 'LianYunGang',
}

MISSING_CODES = {'997', '999', ''}


def parse_monthly_file(filepath):
    """
    Parse a single monthly fixed-width data file.

    Parameters
    ----------
    filepath : str
        Path to the YYYYMMNNN.txt file.

    Returns
    -------
    list of (datetime, float) tuples
        Parsed (timestamp, SWH_in_meters) records.
    """
    fname = os.path.basename(filepath)
    year = int(fname[:4])
    month = int(fname[4:6])

    records = []
    with open(filepath, 'r', encoding='ascii', errors='ignore') as f:
        for line in f:
            line = line.rstrip()

            # Data records start with '2'
            if len(line) < 65 or line[0] != '2':
                continue

            try:
                day = int(line[2:4].strip())
                hour = int(line[4:6].strip())

                # Columns 59-61 (0-indexed: 58:61): Significant wave height
                # Format: **.* in units of 0.1 m
                swh_str = line[58:61].strip()

                if swh_str in MISSING_CODES:
                    continue

                swh = float(swh_str) / 10.0  # convert to meters

                if swh <= 0 or swh > 20.0:
                    continue

                dt = datetime(year, month, day, hour)
                records.append((dt, swh))

            except (ValueError, IndexError):
                continue

    return records


def parse_station(station_code, input_dir):
    """
    Parse all monthly files for a given station.

    Parameters
    ----------
    station_code : int
        Station code (1, 2, or 3).
    input_dir : str
        Directory containing the raw .txt files.

    Returns
    -------
    list of (datetime, float) tuples
        All valid records, sorted by time, duplicates removed.
    """
    pattern = os.path.join(input_dir, f'*{station_code:03d}.txt')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"  WARNING: No files found for station {station_code:03d} "
              f"in {input_dir}")
        return []

    all_records = []
    for fpath in files:
        recs = parse_monthly_file(fpath)
        all_records.extend(recs)

    # Sort by datetime
    all_records.sort(key=lambda x: x[0])

    # Remove duplicate timestamps (keep first occurrence)
    seen = set()
    unique = []
    for dt, swh in all_records:
        if dt not in seen:
            seen.add(dt)
            unique.append((dt, swh))

    return unique


def export_csv(records, output_path):
    """Export parsed records to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['datetime', 'SWH'])
        for dt, swh in records:
            writer.writerow([dt.strftime('%Y-%m-%d %H:%M:%S'), f'{swh:.2f}'])


def main():
    parser = argparse.ArgumentParser(
        description='Parse NMDCNS raw wave data to CSV for D-WaveNet'
    )
    parser.add_argument('--input_dir', type=str, default='./data/raw',
                        help='Directory containing raw YYYYMMNNN.txt files')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for CSV files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for code, name in STATION_NAMES.items():
        print(f'Parsing station {code:03d} ({name})...')
        records = parse_station(code, args.input_dir)

        if not records:
            print(f'  No valid records found. Skipping.')
            continue

        output_path = os.path.join(args.output_dir, f'{name}.csv')
        export_csv(records, output_path)

        dates = [r[0] for r in records]
        swh = [r[1] for r in records]
        import numpy as np
        swh_arr = np.array(swh)

        print(f'  Records: {len(records)}')
        print(f'  Date range: {dates[0]} to {dates[-1]}')
        print(f'  SWH: min={swh_arr.min():.2f}m, max={swh_arr.max():.2f}m, '
              f'mean={swh_arr.mean():.2f}m, std={swh_arr.std():.2f}m')
        print(f'  Exported to: {output_path}')
        print()

    print('Done. CSV files are ready for D-WaveNet data loader.')


if __name__ == '__main__':
    main()
