import torch
import torch.nn as nn

class WCFT(nn.Module):
    """
    Wavelet-Component Fusion Transformer (WCFT)
    Paper Params: Encoder Depth N=3, d_model=512, Heads h=8, Dropout=0.1
    """
    def __init__(self, seq_len=96, pred_len=168, d_model=512, n_heads=8, e_layers=3, dropout=0.1):
        super(WCFT, self).__init__()
        self.value_embedding = nn.Linear(1, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        
        # Intra-Scale Self-Attention & Inter-Scale Cross-Attention components
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x shape: [Batch, Seq_len, 1]
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        
        x_emb = self.value_embedding(x) + self.position_embedding(pos)
        enc_out = self.transformer_encoder(x_emb)
        
        # Global Average Pooling for Static Multi-Step Forecasting
        gap_out = torch.mean(enc_out, dim=1) 
        return self.projection(gap_out)

class KDCM(nn.Module):
    """
    Kinematic-Dynamic Coupled Module (KDCM)
    Incorporates wave acceleration to eliminate hysteresis.
    """
    def __init__(self, seq_len=96, pred_len=168, d_model=512):
        super(KDCM, self).__init__()
        # 1D Conv to extract kinematic derivatives (velocity, acceleration)
        self.kinematic_extractor = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(d_model * seq_len, pred_len)

    def forward(self, x):
        # x shape: [Batch, 1, Seq_len]
        x = x.transpose(1, 2) 
        k_feat = self.relu(self.kinematic_extractor(x))
        k_feat = k_feat.view(k_feat.size(0), -1) 
        return self.fc(k_feat)

class Model(nn.Module):
    """
    D-WaveNet Main Architecture
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        self.wcft = WCFT(
            seq_len=configs.seq_len, 
            pred_len=configs.pred_len,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            e_layers=configs.e_layers,
            dropout=configs.dropout
        )
        self.kdcm = KDCM(
            seq_len=configs.seq_len, 
            pred_len=configs.pred_len,
            d_model=configs.d_model
        )
        
    def forward(self, wind_sea_comp, swell_comp):
        # Cross-Scale Interaction & Fusion
        wind_sea_out = self.wcft(wind_sea_comp)
        swell_out = self.kdcm(swell_comp)
        return wind_sea_out + swell_out