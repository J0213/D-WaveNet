"""
D-WaveNet Model (with Ablation Support)
========================================
Ablation modes (configs.ablation):
  None        -> Full D-WaveNet
  'no_lambda' -> Model A: freeze lambda=0
  'no_wcft'   -> Model B: independent encoders (no cross-scale attention)
  'no_kdcm'   -> Model C: scalar swell regression (no kinematic features)
  'no_phy'    -> Model D: model unchanged, loss weight set to 0 in run.py
"""
import torch, torch.nn as nn, math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1)])

class IntraScaleSelfAttn(nn.Module):
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(drop)
    def forward(self, x):
        o, _ = self.attn(x, x, x); return self.norm(x + self.drop(o))

class InterScaleCrossAttn(nn.Module):
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(drop)
    def forward(self, q, kv):
        o, _ = self.attn(query=q, key=kv, value=kv); return self.norm(q + self.drop(o))

class WCFTBlock(nn.Module):
    def __init__(self, d, h, drop=0.1, use_cross=True):
        super().__init__()
        self.use_cross = use_cross
        self.intra = nn.ModuleList([IntraScaleSelfAttn(d, h, drop) for _ in range(4)])
        if use_cross:
            self.cross = nn.ModuleList([InterScaleCrossAttn(d, h, drop) for _ in range(4)])
        ff = d * 4
        self.ffns = nn.ModuleList([nn.Sequential(
            nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop),
            nn.Linear(ff, d), nn.Dropout(drop)) for _ in range(4)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])

    def forward(self, streams):
        s = [self.intra[i](streams[i]) for i in range(4)]
        if self.use_cross:
            c = []
            for i in range(4):
                others = torch.cat([s[j] for j in range(4) if j != i], dim=1)
                c.append(self.cross[i](s[i], others))
        else:
            c = s
        return [self.norms[i](c[i] + self.ffns[i](c[i])) for i in range(4)]

class WCFT(nn.Module):
    def __init__(self, seq_len, pred_len, d=512, h=8, layers=3, drop=0.1, use_cross=True):
        super().__init__()
        self.embs = nn.ModuleList([nn.Linear(1, d) for _ in range(4)])
        self.pe = PositionalEncoding(d, seq_len, drop)
        self.blocks = nn.ModuleList([WCFTBlock(d, h, drop, use_cross) for _ in range(layers)])
        self.proj = nn.Linear(d * 4, pred_len)
    def forward(self, D1, D2, D3, A3):
        st = [self.pe(self.embs[i](x)) for i, x in enumerate([D1, D2, D3, A3])]
        for b in self.blocks: st = b(st)
        return self.proj(torch.cat([s.mean(1) for s in st], -1))

class KDCM(nn.Module):
    def __init__(self, seq_len, pred_len, d=128):
        super().__init__()
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(1, d, 3, padding=1), nn.ReLU(),
            nn.Conv1d(d, d, 3, padding=1), nn.ReLU()) for _ in range(4)])
        self.mlp = nn.Sequential(
            nn.Linear(d * 4 * seq_len, d * 4), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d * 4, pred_len))
    def forward(self, state):
        feats = [self.convs[c](state[:, :, c].unsqueeze(1)) for c in range(4)]
        return self.mlp(torch.cat(feats, 1).flatten(1))

class SimpleSwellEncoder(nn.Module):
    """Ablation C: no kinematic features, scalar swell only."""
    def __init__(self, seq_len, pred_len, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(seq_len, d*2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d*2, pred_len))
    def forward(self, state):
        return self.net(state[:, :, 0])

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.ablation = getattr(configs, 'ablation', None)
        if self.ablation == 'no_lambda':
            self.register_buffer('lambda_factor', torch.tensor(0.0))
        else:
            self.lambda_factor = nn.Parameter(torch.tensor(0.1))
        use_cross = (self.ablation != 'no_wcft')
        self.wcft = WCFT(configs.seq_len, configs.pred_len, configs.d_model,
                         configs.n_heads, configs.e_layers, configs.dropout, use_cross)
        kd = getattr(configs, 'kdcm_dim', 128)
        if self.ablation == 'no_kdcm':
            self.swell_enc = SimpleSwellEncoder(configs.seq_len, configs.pred_len, kd)
        else:
            self.swell_enc = KDCM(configs.seq_len, configs.pred_len, kd)

    def forward(self, D1, D2, D3, A3, swell_state):
        # Energy Dissipation-Guided Optimization (Eq. 4-5 in paper)
        # λ transfers residual high-frequency energy from the final-level
        # approximation A3 back to all detail components D1, D2, D3.
        # This corrects spectral leakage at the last decomposition level.
        lam = 0.0 if self.ablation == 'no_lambda' else torch.sigmoid(self.lambda_factor)
        D1o, D2o, D3o = D1 + lam * A3, D2 + lam * A3, D3 + lam * A3
        A3o = (1.0 - lam) * A3
        wind = self.wcft(D1o, D2o, D3o, A3o)
        swell = self.swell_enc(swell_state)
        return wind + swell, swell
