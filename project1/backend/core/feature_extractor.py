"""
Feature extraction for voice biometric authentication.
Extracts MFCCs, Mel Spectrograms, Pitch (F0), and a combined feature vector.
"""
import numpy as np
import librosa

SAMPLE_RATE = 16000
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 160   # 10ms at 16kHz
WIN_LENGTH = 400   # 25ms at 16kHz
N_FFT = 512


def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Returns (N_MFCC, T) — raw MFCC frames."""
    return librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
    )


def extract_mfcc_delta(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Returns (N_MFCC*3, T) — MFCC + delta + delta-delta stacked."""
    mfcc = extract_mfcc(audio, sr)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2])


def extract_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Returns (N_MELS, T) log-mel spectrogram in dB."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
    )
    return librosa.power_to_db(mel, ref=np.max)


def extract_pitch(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Returns (T,) F0 pitch contour (voiced frames only, unvoiced = 0)."""
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=HOP_LENGTH
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0


def extract_spectral_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Returns (4, T) — centroid, bandwidth, rolloff, ZCR stacked."""
    centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=HOP_LENGTH)
    rolloff   = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=HOP_LENGTH)
    zcr       = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    return np.vstack([centroid, bandwidth, rolloff, zcr])


def extract_all(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Extract all features and return as a dict."""
    return {
        "mfcc":          extract_mfcc(audio, sr),
        "mfcc_delta":    extract_mfcc_delta(audio, sr),
        "mel_spec":      extract_mel_spectrogram(audio, sr),
        "pitch":         extract_pitch(audio, sr),
        "spectral":      extract_spectral_features(audio, sr),
    }


def get_mean_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Fixed-length feature vector: mean + std of MFCC-delta (120,) + spectral (8,) = 128-dim.
    Used as a compact embedding for text-dependent auth (Phase 3).
    """
    mfcc_delta = extract_mfcc_delta(audio, sr)   # (120, T)
    spectral   = extract_spectral_features(audio, sr)  # (4, T)

    mfcc_mean = np.mean(mfcc_delta, axis=1)   # (120,)
    mfcc_std  = np.std(mfcc_delta, axis=1)    # (120,)
    spec_mean = np.mean(spectral, axis=1)     # (4,)
    spec_std  = np.std(spectral, axis=1)      # (4,)

    return np.concatenate([mfcc_mean, mfcc_std, spec_mean, spec_std])  # (248,)
