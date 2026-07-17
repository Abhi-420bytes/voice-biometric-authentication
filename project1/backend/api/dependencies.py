"""
Singleton pipeline instances shared across all requests.
Models are loaded once at startup and reused — avoids reloading on every call.
"""
from __future__ import annotations
import os, sys
import numpy as np
import soundfile as sf
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from backend.core.auth_pipeline          import AuthPipeline
from backend.core.text_dependent_pipeline import TextDependentPipeline
from backend.core.spoof_detector          import SpoofDetector
from backend.core.audio_utils             import preprocess

SR                 = 16000
DEFAULT_PASSPHRASE = "my voice is my password"

# ── Singletons (initialised at startup) ─────────────────────────
_ti_pipeline: AuthPipeline           | None = None
_td_pipeline: TextDependentPipeline  | None = None
_spoof_only:  SpoofDetector          | None = None


def get_ti_pipeline() -> AuthPipeline:
    global _ti_pipeline
    if _ti_pipeline is None:
        _ti_pipeline = AuthPipeline(
            spoof_mode      = "rule_based",
            verify_backend  = "resemblyzer",
            spoof_threshold = 0.20,
            verify_threshold = 0.75,
        )
    return _ti_pipeline


def get_td_pipeline() -> TextDependentPipeline:
    global _td_pipeline
    if _td_pipeline is None:
        _td_pipeline = TextDependentPipeline(
            passphrase      = DEFAULT_PASSPHRASE,
            whisper_model   = "base",
            spoof_mode      = "rule_based",
            verify_backend  = "resemblyzer",
            spoof_threshold = 0.20,
            voice_threshold = 0.75,
            text_threshold  = 0.70,
        )
    return _td_pipeline


def get_spoof_detector() -> SpoofDetector:
    global _spoof_only
    if _spoof_only is None:
        _spoof_only = SpoofDetector(mode="rule_based", threshold=0.20)
    return _spoof_only


# ── Audio loading helper ─────────────────────────────────────────
def load_audio_bytes(data: bytes) -> np.ndarray:
    """Read raw WAV bytes → preprocessed float32 numpy array at 16 kHz."""
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return preprocess(audio, SR)
