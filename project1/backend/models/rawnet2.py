"""
Simplified RawNet2 for anti-spoofing (binary: real=1, fake=0).

Architecture (Tak et al., 2021 — condensed):
  Raw waveform → SincConv1d → ResBlocks × 6 (with max-pool) → GRU → FC → sigmoid

Key innovation: SincConv1d learns band-pass filters directly parameterised by
their lower and upper cutoff frequencies, constrained to stay physically valid.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

SAMPLE_RATE = 16000


# ------------------------------------------------------------------ #
#  SincConv1d — learnable sinc filterbank                            #
# ------------------------------------------------------------------ #
class SincConv1d(nn.Module):
    """
    Learns N band-pass filters: h[n] = sinc(2*f_hi*n) - sinc(2*f_lo*n),
    windowed with Hamming. Initialised with mel-spaced cutoffs.
    """
    def __init__(self, out_channels: int = 128, kernel_size: int = 251,
                 sample_rate: int = SAMPLE_RATE):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.sample_rate  = sample_rate

        # Mel-spaced initialisation
        low_hz  = 30.0
        high_hz = sample_rate / 2 - 100

        mel_lo = np.linspace(self._hz2mel(low_hz),
                             self._hz2mel(high_hz), out_channels + 1)
        hz_lo  = self._mel2hz(mel_lo)

        f_lo = torch.tensor(hz_lo[:-1], dtype=torch.float32)
        f_hi = torch.tensor(hz_lo[1:],  dtype=torch.float32)

        self.f_lo = nn.Parameter(f_lo)
        self.f_hi = nn.Parameter(f_hi)

        # Hamming window (fixed)
        n = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1,
                         dtype=torch.float32)
        self.register_buffer("window", 0.54 - 0.46 * torch.cos(
            2 * math.pi * n / (kernel_size - 1)))
        self.register_buffer("n", n)

    @staticmethod
    def _hz2mel(hz):  return 2595 * np.log10(1 + hz / 700)
    @staticmethod
    def _mel2hz(mel): return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) → (B, out_channels, T')"""
        # Clamp to valid range
        f_lo = torch.clamp(torch.abs(self.f_lo), 30.0, self.sample_rate / 2 - 100.0)
        f_hi = torch.clamp(torch.abs(self.f_hi), 80.0, float(self.sample_rate / 2))
        f_hi = torch.maximum(f_hi, f_lo + 50.0)

        # Normalise to [0, 1]
        f_lo_n = (2 * f_lo / self.sample_rate).unsqueeze(1)  # (C, 1)
        f_hi_n = (2 * f_hi / self.sample_rate).unsqueeze(1)

        n = self.n.unsqueeze(0)   # (1, K)
        pi_n = math.pi * n.clamp(min=1e-9)

        low_pass_hi = 2 * f_hi_n * torch.sinc(f_hi_n * n)
        low_pass_lo = 2 * f_lo_n * torch.sinc(f_lo_n * n)
        band_pass   = (low_pass_hi - low_pass_lo) * self.window  # (C, K)

        # Normalise each filter
        band_pass = band_pass / (band_pass.norm(dim=1, keepdim=True) + 1e-9)
        filters   = band_pass.unsqueeze(1)   # (C, 1, K)

        return F.conv1d(x, filters, padding=self.kernel_size // 2)


# ------------------------------------------------------------------ #
#  Residual block with Feature Map Scaling (FMS)                     #
# ------------------------------------------------------------------ #
class FMS(nn.Module):
    """Feature Map Scaling: learns per-channel scale + bias via sigmoid."""
    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = torch.sigmoid(self.fc(x.mean(dim=2)))   # (B, C)
        return x * s.unsqueeze(2) + s.unsqueeze(2)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.fms   = FMS(out_ch)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.pool  = nn.MaxPool1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.bn1(self.conv1(x)), 0.3)
        h = self.bn2(self.conv2(h))
        h = h + self.skip(x)
        h = F.leaky_relu(h, 0.3)
        h = self.fms(h)
        return self.pool(h)


# ------------------------------------------------------------------ #
#  RawNet2                                                            #
# ------------------------------------------------------------------ #
class RawNet2(nn.Module):
    """
    Simplified RawNet2. Input: (B, T) raw waveform at 16 kHz.
    Output: (B,) logit — positive → real, negative → fake.
    """
    def __init__(self, sinc_channels: int = 128,
                 gru_hidden: int = 1024, gru_layers: int = 3):
        super().__init__()

        self.sinc   = SincConv1d(sinc_channels, kernel_size=251)
        self.bn_sinc = nn.BatchNorm1d(sinc_channels)

        self.res_blocks = nn.Sequential(
            ResBlock(sinc_channels, 20),
            ResBlock(20, 64),
            ResBlock(64, 64),
            ResBlock(64, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
        )

        self.gru = nn.GRU(128, gru_hidden, num_layers=gru_layers,
                          batch_first=True, dropout=0.1)
        self.fc  = nn.Linear(gru_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) → logit (B,)"""
        x = x.unsqueeze(1)                          # (B, 1, T)
        x = torch.abs(self.sinc(x))                 # (B, C, T')
        x = F.leaky_relu(self.bn_sinc(x), 0.3)
        x = self.res_blocks(x)                      # (B, 128, T'')
        x = x.permute(0, 2, 1)                      # (B, T'', 128)
        _, h = self.gru(x)                          # h: (layers, B, H)
        h = h[-1]                                   # (B, H) — last layer
        return self.fc(h).squeeze(1)                # (B,)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probability of being real (0-1)."""
        return torch.sigmoid(self.forward(x))


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
def load_rawnet2(path: str, device: str = "cpu") -> RawNet2:
    model = RawNet2()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
