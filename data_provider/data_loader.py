"""
D-WaveNet Data Provider
=======================
Key features:
  1. Real CSV loading (no synthetic default)
  2. Gap-aware sliding window: never straddles gaps > max_gap_hours
  3. Strictly causal moving average and DWT (mode='zero')
  4. Z-score fitted on training set only
  5. Returns D1,D2,D3,A3 wavelet sub-bands + 4-ch kinematic swell state
"""
import os, numpy as np, pandas as pd, pywt, torch
from torch.utils.data import Dataset, DataLoader


class WaveDataset(Dataset):
    def __init__(self, data_path, seq_len=96, pred_len=168, flag='train',
                 ma_window=12, wavelet='db4', decomp_level=3,
                 mean_depth=8.0, max_gap_hours=6, synthetic=False):
        self.seq_len, self.pred_len = seq_len, pred_len
        self.ma_window, self.wavelet, self.decomp_level = ma_window, wavelet, decomp_level
        self.mean_depth = mean_depth

        # --- Load ---
        if synthetic:
            print("[WARNING] Synthetic data — will NOT reproduce paper results.")
            np.random.seed(42); n = 70000; t = np.arange(n)
            raw = np.abs(0.8 + 0.3*np.sin(2*np.pi*t/(24*30)) + 0.2*np.sin(2*np.pi*t/12)
                         + 0.1*np.random.randn(n) + 0.4*np.sin(2*np.pi*t/(24*365)))
            timestamps = pd.date_range('2014-01-01', periods=n, freq='h')
        else:
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Data not found: {data_path}\nSee data/README_DATA.md or use --synthetic")
            df = pd.read_csv(data_path)
            assert 'SWH' in df.columns, f"CSV must have 'SWH' column. Found: {list(df.columns)}"
            timestamps = pd.to_datetime(df['datetime']) if 'datetime' in df.columns else None
            raw = df['SWH'].values.astype(np.float64)

        # --- Quality control ---
        raw = np.where((raw < 0) | (raw > 20.0), np.nan, raw)
        mu, sigma = np.nanmean(raw), np.nanstd(raw)
        raw[np.abs(raw - mu) > 3 * sigma] = np.nan
        s = pd.Series(raw).interpolate(method='linear', limit=max_gap_hours)
        raw = s.values

        # --- Build valid index mask (no NaN, no long gaps) ---
        valid = ~np.isnan(raw)
        # Mark runs of consecutive valid hours
        if timestamps is not None:
            dt_hours = np.diff(timestamps).astype('timedelta64[h]').astype(float)
            gap_mask = np.ones(len(raw), dtype=bool)
            gap_mask[0] = valid[0]
            for i in range(1, len(raw)):
                if not valid[i] or dt_hours[i-1] > max_gap_hours:
                    gap_mask[i] = False  # break point
                else:
                    gap_mask[i] = True
        else:
            gap_mask = valid

        # Build segments of consecutive valid data
        segments = []
        start = None
        for i in range(len(raw)):
            if gap_mask[i] and valid[i]:
                if start is None: start = i
            else:
                if start is not None:
                    segments.append((start, i))
                    start = None
        if start is not None:
            segments.append((start, len(raw)))

        # Store raw for inverse transform
        self.raw_data = raw

        # --- 7:1:2 split on the full valid data ---
        total_valid = sum(e - s for s, e in segments)
        n_train = int(total_valid * 0.7)
        n_val = int(total_valid * 0.1)

        # Compute train stats from first 70% of valid points
        train_vals = []
        count = 0
        for s, e in segments:
            for i in range(s, e):
                if count < n_train:
                    train_vals.append(raw[i])
                count += 1
        self.train_mean = np.mean(train_vals)
        self.train_std = np.std(train_vals) if np.std(train_vals) > 1e-8 else 1.0

        # Normalize entire array
        data_norm = (raw - self.train_mean) / self.train_std

        # --- Build sample indices (within segments, respecting split) ---
        # CRITICAL: Split by the END of the prediction window (g + seq_len + pred_len),
        # not the start. This ensures no training target leaks into val/test regions.
        window = seq_len + pred_len
        all_indices = []
        cum = 0
        for seg_s, seg_e in segments:
            seg_len = seg_e - seg_s
            for i in range(seg_len - window + 1):
                global_idx = seg_s + i
                # sample_end: cumulative position of the LAST prediction target point
                sample_end = cum + i + seq_len + pred_len
                all_indices.append((global_idx, sample_end))
            cum += seg_len

        # Split by prediction window end position:
        #   train: entire sample (input + target) falls within [0, n_train]
        #   val:   input may start in train region, but target starts after n_train
        #          and entire target falls within [n_train, n_train + n_val]
        #   test:  target starts after n_train + n_val
        n_val_end = n_train + n_val
        train_idx = [(g, e) for g, e in all_indices if e <= n_train]
        val_idx   = [(g, e) for g, e in all_indices if e > n_train and e <= n_val_end]
        test_idx  = [(g, e) for g, e in all_indices if e > n_val_end]

        if flag == 'train': self.indices = [g for g, _ in train_idx]
        elif flag == 'val': self.indices = [g for g, _ in val_idx]
        else: self.indices = [g for g, _ in test_idx]

        self.data_norm = data_norm
        print(f"[{flag.upper()}] {len(self.indices)} samples | "
              f"mean={self.train_mean:.4f} std={self.train_std:.4f}")

    def __len__(self):
        return len(self.indices)

    def _causal_ma(self, x, w):
        result = np.zeros_like(x, dtype=np.float64)
        cs = np.cumsum(np.insert(x, 0, 0.0))
        for i in range(len(x)):
            s = max(0, i - w + 1)
            result[i] = (cs[i+1] - cs[s]) / (i - s + 1)
        return result

    def _causal_dwt(self, x):
        coeffs = pywt.wavedec(x, self.wavelet, level=self.decomp_level, mode='zero')
        n = len(x)
        A3 = pywt.upcoef('a', coeffs[0], self.wavelet, level=self.decomp_level, take=n)
        D3 = pywt.upcoef('d', coeffs[1], self.wavelet, level=self.decomp_level, take=n)
        D2 = pywt.upcoef('d', coeffs[2], self.wavelet, level=self.decomp_level-1, take=n)
        D1 = pywt.upcoef('d', coeffs[3], self.wavelet, level=1, take=n)
        return A3, D1, D2, D3

    def _kinematic(self, swell):
        """Construct kinematic state vector: [position, velocity, acceleration, breaking_proximity].

        The fourth channel is the depth-limited breaking proximity ratio ξ = H_s / H_max,
        where H_max = 0.78 * d. This measures how close the swell is to the depth-induced
        breaking threshold (ξ → 1 means imminent breaking). Note: this is distinct from the
        Stokes steepness H/L used in the physics-guided loss function.
        """
        vel = np.zeros_like(swell); vel[1:] = swell[1:] - swell[:-1]
        acc = np.zeros_like(swell); acc[1:] = vel[1:] - vel[:-1]
        breaking_proximity = swell / (0.78 * self.mean_depth)
        return np.stack([swell, vel, acc, breaking_proximity], axis=-1)

    def __getitem__(self, idx):
        g = self.indices[idx]
        seq = self.data_norm[g:g+self.seq_len].copy()
        tgt = self.data_norm[g+self.seq_len:g+self.seq_len+self.pred_len].copy()

        # Last observed value in input window (for persistence baseline)
        last_obs = seq[-1]

        swell = self._causal_ma(seq, self.ma_window)
        wind = seq - swell
        A3, D1, D2, D3 = self._causal_dwt(wind)
        state = self._kinematic(swell)

        return (torch.FloatTensor(D1).unsqueeze(-1),
                torch.FloatTensor(D2).unsqueeze(-1),
                torch.FloatTensor(D3).unsqueeze(-1),
                torch.FloatTensor(A3).unsqueeze(-1),
                torch.FloatTensor(state),
                torch.FloatTensor(tgt),
                torch.FloatTensor([last_obs]))


def data_provider(args, flag):
    ds = WaveDataset(args.data_path, args.seq_len, args.pred_len, flag,
                     args.ma_window, args.wavelet, args.decomp_level,
                     args.mean_depth, getattr(args, 'max_gap_hours', 6),
                     args.synthetic)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=(flag == 'train'),
                        num_workers=getattr(args, 'num_workers', 0),
                        drop_last=True)
    return ds, loader
