"""
Anti-spoofing detector.  Three modes:

  rule_based  — works immediately, no training needed. Uses acoustic
                heuristics (jitter, spectral flatness, modulation energy).
                Threshold-calibrated against a simple real/fake assumption.

  gmm         — Gaussian Mixture Model trained on LFCC features.
                Train: SpoofDetector('gmm').train(real_dir, fake_dir)

  rawnet2     — End-to-end deep model on raw audio.
                Train: SpoofDetector('rawnet2').train(real_dir, fake_dir)

Pipeline integration:
    All three expose the same interface:
        result = detector.detect(audio, sr)
        result.is_real   → bool
        result.score     → float in [0, 1], higher = more real
        result.attack_type → 'genuine' | 'synthetic_tts' | 'voice_conversion' | 'replay' | 'unknown_spoof'
"""
from __future__ import annotations
import os, sys, time, warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import joblib
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from backend.core.spoof_features import (
    extract_spoof_feature_vector,
    _sub_band_flatness, _pitch_jitter_shimmer,
    _modulation_energy, _hnr_proxy, _sub_band_energy_ratio,
)

SAMPLE_RATE   = 16000
MODELS_DIR    = "data/models"
DEVICE        = "mps" if torch.backends.mps.is_available() else "cpu"


# ------------------------------------------------------------------ #
#  Result dataclass                                                   #
# ------------------------------------------------------------------ #
@dataclass
class SpoofResult:
    is_real:     bool
    score:       float          # 0=definitely fake, 1=definitely real
    threshold:   float
    attack_type: str
    mode:        str
    latency_ms:  float

    def __str__(self):
        verdict = "GENUINE" if self.is_real else "SPOOF DETECTED"
        return (f"[{verdict}] score={self.score:.4f} "
                f"threshold={self.threshold:.4f} "
                f"attack={self.attack_type} "
                f"mode={self.mode} "
                f"{self.latency_ms:.0f}ms")


# ------------------------------------------------------------------ #
#  SpoofDetector                                                      #
# ------------------------------------------------------------------ #
class SpoofDetector:
    def __init__(self, mode: str = "rule_based",
                 model_path: Optional[str] = None,
                 threshold: Optional[float] = None):
        if mode not in ("rule_based", "gmm", "rawnet2"):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode      = mode
        self.threshold = threshold
        self._model    = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        elif mode == "rawnet2" and model_path is None:
            # Default save path
            default = os.path.join(MODELS_DIR, "rawnet2_spoof.pt")
            if os.path.exists(default):
                self.load(default)

    # ---------------------------------------------------------------- #
    #  Public API                                                       #
    # ---------------------------------------------------------------- #
    def detect(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> SpoofResult:
        t0 = time.time()
        if self.mode == "rule_based":
            score, attack = self._rule_based(audio, sr)
            thresh = self.threshold if self.threshold is not None else 0.50
        elif self.mode == "gmm":
            score, attack = self._gmm_score(audio, sr)
            thresh = self.threshold if self.threshold is not None else 0.50
        else:
            score, attack = self._rawnet2_score(audio, sr)
            thresh = self.threshold if self.threshold is not None else 0.50

        return SpoofResult(
            is_real    = score >= thresh,
            score      = score,
            threshold  = thresh,
            attack_type= attack,
            mode       = self.mode,
            latency_ms = (time.time() - t0) * 1000,
        )

    def train(self, real_dir: str, fake_dir: str,
              sr: int = SAMPLE_RATE, save_path: Optional[str] = None):
        """Train GMM or RawNet2 on directories of real/fake .wav files."""
        from backend.core.audio_utils import load_audio, preprocess

        real_files = _list_wavs(real_dir)
        fake_files = _list_wavs(fake_dir)
        print(f"  Training {self.mode}: {len(real_files)} real, {len(fake_files)} fake")

        if self.mode == "gmm":
            self._train_gmm(real_files, fake_files, sr)
            path = save_path or os.path.join(MODELS_DIR, "gmm_spoof.pkl")
            self.save(path)

        elif self.mode == "rawnet2":
            self._train_rawnet2(real_files, fake_files, sr)
            path = save_path or os.path.join(MODELS_DIR, "rawnet2_spoof.pt")
            self.save(path)

        else:
            print("  rule_based mode does not require training.")

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if self.mode == "gmm" and self._model is not None:
            joblib.dump(self._model, path)
        elif self.mode == "rawnet2" and self._model is not None:
            torch.save(self._model.state_dict(), path)
        print(f"  Model saved: {path}")

    def load(self, path: str):
        if self.mode == "gmm":
            self._model = joblib.load(path)
        elif self.mode == "rawnet2":
            from backend.models.rawnet2 import RawNet2
            self._model = RawNet2().to(DEVICE)
            state = torch.load(path, map_location=DEVICE, weights_only=True)
            self._model.load_state_dict(state)
            self._model.eval()

    # ---------------------------------------------------------------- #
    #  Rule-based scoring                                               #
    # ---------------------------------------------------------------- #
    def _rule_based(self, audio: np.ndarray, sr: int) -> tuple[float, str]:
        """
        Composite naturalness score in [0, 1].
        Each component is scored 0-1 (1 = natural), then weighted.
        Weights are tuned heuristically against typical TTS/VC artifacts.
        """
        jitter, shimmer = _pitch_jitter_shimmer(audio, sr)
        flatness        = _sub_band_flatness(audio, sr)
        mod_e           = _modulation_energy(audio, sr)
        hnr             = _hnr_proxy(audio, sr)
        energy          = _sub_band_energy_ratio(audio, sr)

        # --- Pitch jitter score (natural: 0.005-0.05; TTS: <0.001)
        jitter_score = float(np.clip(jitter / 0.02, 0, 1))

        # --- Modulation energy score (natural: >0.25; TTS: <0.10)
        mod_score = float(np.clip(mod_e / 0.25, 0, 1))

        # --- Spectral flatness score (natural: lower; TTS: higher in mid/hi)
        # Low flatness in high bands = more natural
        hi_flat = flatness[2:].mean()
        flat_score = float(np.clip(1 - hi_flat * 2, 0, 1))

        # --- HNR score (natural: 0.5-0.9; over-cleaned TTS: >0.95)
        hnr_score = float(np.clip(1 - abs(hnr - 0.70) / 0.30, 0, 1))

        # --- Sub-band energy: real mics capture natural low/high balance
        # Vocoders often reduce very high-freq content
        hi_ratio  = energy[2]
        hi_score  = float(np.clip(hi_ratio / 0.15, 0, 1))

        # Weighted composite
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        scores  = [jitter_score, mod_score, flat_score, hnr_score, hi_score]
        total   = sum(w * s for w, s in zip(weights, scores))

        # Classify attack type from dominant failure
        attack = _classify_attack(jitter, mod_e, hi_flat, hnr, hi_ratio)
        return float(total), attack

    # ---------------------------------------------------------------- #
    #  GMM scoring                                                      #
    # ---------------------------------------------------------------- #
    def _gmm_score(self, audio: np.ndarray, sr: int) -> tuple[float, str]:
        if self._model is None:
            raise RuntimeError("GMM not trained. Call .train() first.")
        vec = extract_spoof_feature_vector(audio, sr).reshape(1, -1).astype(np.float64)
        gmm_real, gmm_fake = self._model
        ll_real = gmm_real.score(vec)
        ll_fake = gmm_fake.score(vec)
        raw     = ll_real - ll_fake
        score   = float(1 / (1 + np.exp(-raw / 10)))   # sigmoid normalise
        attack  = "genuine" if score >= 0.5 else "unknown_spoof"
        return score, attack

    def _train_gmm(self, real_files, fake_files, sr):
        from sklearn.mixture import GaussianMixture
        from backend.core.audio_utils import load_audio, preprocess

        def feats(files):
            out = []
            for f in files:
                try:
                    a = preprocess(load_audio(f, sr), sr)
                    out.append(extract_spoof_feature_vector(a, sr))
                except Exception:
                    pass
            return np.vstack(out) if out else np.zeros((1, 143))

        X_real = feats(real_files)
        X_fake = feats(fake_files)

        n_comp   = max(1, min(16, len(X_real) // 2, len(X_fake) // 2))
        gmm_real = GaussianMixture(n_components=n_comp, covariance_type='diag',
                                    max_iter=200, reg_covar=1e-3, random_state=42)
        gmm_fake = GaussianMixture(n_components=n_comp, covariance_type='diag',
                                    max_iter=200, reg_covar=1e-3, random_state=42)
        gmm_real.fit(X_real.astype(np.float64))
        gmm_fake.fit(X_fake.astype(np.float64))
        self._model = (gmm_real, gmm_fake)
        print(f"  GMM trained on {len(X_real)} real, {len(X_fake)} fake samples")

    # ---------------------------------------------------------------- #
    #  RawNet2 scoring                                                  #
    # ---------------------------------------------------------------- #
    def _rawnet2_score(self, audio: np.ndarray, sr: int) -> tuple[float, str]:
        if self._model is None:
            raise RuntimeError("RawNet2 not trained. Call .train() first.")
        x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = self._model(x)
        score  = float(torch.sigmoid(logit).cpu().item())
        attack = "genuine" if score >= 0.5 else "unknown_spoof"
        return score, attack

    def _train_rawnet2(self, real_files, fake_files, sr,
                        epochs=20, lr=1e-4, batch_size=16):
        from backend.models.rawnet2 import RawNet2
        from backend.core.audio_utils import load_audio, preprocess
        from torch.utils.data import TensorDataset, DataLoader

        CLIP_LEN = sr * 4   # 4-second clips

        def load_clips(files, label):
            xs, ys = [], []
            for f in files:
                try:
                    a = preprocess(load_audio(f, sr), sr)
                    a = _pad_or_trim(a, CLIP_LEN)
                    xs.append(torch.tensor(a, dtype=torch.float32))
                    ys.append(label)
                except Exception:
                    pass
            return xs, ys

        x_r, y_r = load_clips(real_files, 1.0)
        x_f, y_f = load_clips(fake_files, 0.0)
        if not x_r or not x_f:
            raise RuntimeError("No audio files loaded for training.")

        X = torch.stack(x_r + x_f)
        Y = torch.tensor(y_r + y_f, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size,
                            shuffle=True, drop_last=False)

        self._model = RawNet2().to(DEVICE)
        opt  = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        self._model.train()
        for ep in range(1, epochs + 1):
            total_loss, correct, total = 0.0, 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits = self._model(xb)
                loss   = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                preds   = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == yb).sum().item()
                total   += len(yb)
            acc = correct / total
            print(f"  Epoch {ep:02d}/{epochs}  loss={total_loss/len(loader):.4f}  acc={acc:.4f}")
        self._model.eval()


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
def _classify_attack(jitter, mod_e, hi_flat, hnr, hi_ratio) -> str:
    if jitter < 0.001 and mod_e < 0.10:
        return "synthetic_tts"
    if hi_flat > 0.6 and hi_ratio < 0.05:
        return "voice_conversion"
    if hnr < 0.30:
        return "replay"
    if jitter > 0.001 and mod_e > 0.10:
        return "genuine"
    return "unknown_spoof"


def _list_wavs(directory: str) -> list:
    if not os.path.exists(directory):
        return []
    return [os.path.join(directory, f)
            for f in os.listdir(directory) if f.endswith(".wav")]


def _pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))
