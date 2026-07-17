"""
Anti-spoofing feature extraction.

Features used to distinguish real (genuine) from fake (TTS / voice-conversion / replay) audio:
  - LFCC  : Linear Frequency Cepstral Coefficients (linear filterbank, not mel)
  - Spectral flatness per sub-band (TTS tends to be too smooth)
  - Pitch jitter / shimmer  (natural voice has micro-variations; TTS is often too regular)
  - Sub-band energy ratios  (vocoders attenuate highs differently from real microphones)
  - HNR   : Harmonic-to-Noise Ratio proxy
  - Modulation energy (4-16 Hz syllabic rate band)
"""
import numpy as np
import librosa
from scipy.fft import dct as scipy_dct

SAMPLE_RATE = 16000
N_LFCC      = 60
N_FFT       = 512
HOP_LENGTH  = 160   # 10 ms
WIN_LENGTH  = 400   # 25 ms
N_LINEAR    = 70    # number of linear filter bands


# ------------------------------------------------------------------ #
#  LFCC                                                               #
# ------------------------------------------------------------------ #
def _linear_filterbank(n_filters: int, n_fft: int, sr: int) -> np.ndarray:
    """(n_filters, n_fft//2+1) filterbank with linearly spaced triangular filters."""
    freq_bins = np.linspace(0, sr / 2, n_fft // 2 + 1)
    centres   = np.linspace(0, sr / 2, n_filters + 2)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        lo, mid, hi = centres[m - 1], centres[m], centres[m + 1]
        fb[m - 1] = np.maximum(
            0, np.minimum(
                (freq_bins - lo) / (mid - lo + 1e-10),
                (hi - freq_bins) / (hi - mid + 1e-10)
            )
        )
    return fb


_FB_CACHE: dict = {}

def extract_lfcc(audio: np.ndarray, sr: int = SAMPLE_RATE,
                 n_lfcc: int = N_LFCC) -> np.ndarray:
    """Return (n_lfcc, T) LFCC matrix."""
    key = (sr, n_lfcc)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _linear_filterbank(N_LINEAR, N_FFT, sr)
    fb = _FB_CACHE[key]

    S  = np.abs(librosa.stft(audio, n_fft=N_FFT,
                              hop_length=HOP_LENGTH, win_length=WIN_LENGTH)) ** 2
    lf = np.dot(fb, S)                            # (N_LINEAR, T)
    log_lf = np.log(lf + 1e-9)
    lfcc   = scipy_dct(log_lf, axis=0, norm='ortho')[:n_lfcc]  # (n_lfcc, T)
    return lfcc


# ------------------------------------------------------------------ #
#  Spectral flatness per sub-band                                     #
# ------------------------------------------------------------------ #
def _sub_band_flatness(audio: np.ndarray, sr: int,
                       bands=((0, 500), (500, 2000), (2000, 4000), (4000, 8000))
                       ) -> np.ndarray:
    """Geometric/arithmetic mean ratio per band → (n_bands,)."""
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    flatness = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() == 0:
            flatness.append(0.0)
            continue
        sub = S[mask].mean(axis=1)          # mean over time
        geo = np.exp(np.mean(np.log(sub + 1e-9)))
        ari = np.mean(sub) + 1e-9
        flatness.append(float(geo / ari))
    return np.array(flatness)


# ------------------------------------------------------------------ #
#  Pitch jitter & shimmer                                             #
# ------------------------------------------------------------------ #
def _pitch_jitter_shimmer(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Jitter  = mean |F0[t] - F0[t-1]| / mean(F0)   (pitch regularity)
    Shimmer = mean |A[t]  - A[t-1]|  / mean(A)     (amplitude regularity)
    Natural speech: moderate jitter/shimmer. TTS: near-zero.
    """
    f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=500,
                                  sr=sr, hop_length=HOP_LENGTH)
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced_f0 = f0[f0 > 0]

    if len(voiced_f0) < 4:
        return 0.0, 0.0

    jitter = float(np.mean(np.abs(np.diff(voiced_f0))) / (np.mean(voiced_f0) + 1e-9))

    rms_frames = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
    shimmer    = float(np.mean(np.abs(np.diff(rms_frames))) / (np.mean(rms_frames) + 1e-9))

    return jitter, shimmer


# ------------------------------------------------------------------ #
#  Sub-band energy ratios                                             #
# ------------------------------------------------------------------ #
def _sub_band_energy_ratio(audio: np.ndarray, sr: int) -> np.ndarray:
    """High-freq / low-freq energy ratios in several bands → (3,)."""
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    total = S.sum() + 1e-9

    lo  = S[freqs <  1000].sum()
    mid = S[(freqs >= 1000) & (freqs < 4000)].sum()
    hi  = S[freqs >= 4000].sum()
    return np.array([lo / total, mid / total, hi / total])


# ------------------------------------------------------------------ #
#  Modulation energy (syllabic rate 4-16 Hz)                         #
# ------------------------------------------------------------------ #
def _modulation_energy(audio: np.ndarray, sr: int) -> float:
    """
    Envelope modulation energy in the 4-16 Hz band.
    Natural speech has strong syllabic modulation; TTS can lack it.
    """
    envelope = np.abs(librosa.effects.harmonic(audio))
    # Resample envelope to 100 Hz
    hop      = sr // 100
    env_ds   = np.array([envelope[i: i + hop].mean()
                          for i in range(0, len(envelope) - hop, hop)])
    if len(env_ds) < 32:
        return 0.0
    mod_spec = np.abs(np.fft.rfft(env_ds))
    mod_freq = np.fft.rfftfreq(len(env_ds), d=1 / 100)
    band     = (mod_freq >= 4) & (mod_freq <= 16)
    total    = mod_spec.sum() + 1e-9
    return float(mod_spec[band].sum() / total)


# ------------------------------------------------------------------ #
#  HNR proxy                                                          #
# ------------------------------------------------------------------ #
def _hnr_proxy(audio: np.ndarray, sr: int) -> float:
    """
    Approximate HNR: ratio of harmonic energy to total energy.
    Computed as: harmonic_component energy / original energy.
    """
    harmonic = librosa.effects.harmonic(audio, margin=3.0)
    h_energy = np.mean(harmonic ** 2) + 1e-9
    t_energy = np.mean(audio ** 2) + 1e-9
    return float(h_energy / t_energy)


# ------------------------------------------------------------------ #
#  Combined feature vector                                            #
# ------------------------------------------------------------------ #
def extract_spoof_feature_vector(audio: np.ndarray,
                                  sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Fixed-length 143-dim feature vector for GMM / rule-based detection.
    Breakdown:
      LFCC stats    : 120  (mean 60 + std 60)
      Sub-band flat :   4
      Energy ratio  :   3
      Jitter/shimmer:   2
      Modulation    :   1
      HNR proxy     :   1
      Spectral centroid stats: 2 (mean, std)
      ZCR stats     : 2 (mean, std)
      MFCC delta std: 8  (first 8 MFCC delta stds — capture temporal regularity)
    Total           : 143
    """
    lfcc      = extract_lfcc(audio, sr)                         # (60, T)
    lfcc_mean = np.mean(lfcc, axis=1)                           # (60,)
    lfcc_std  = np.std(lfcc,  axis=1)                           # (60,)

    flatness  = _sub_band_flatness(audio, sr)                   # (4,)
    energy    = _sub_band_energy_ratio(audio, sr)               # (3,)
    jitter, shimmer = _pitch_jitter_shimmer(audio, sr)          # scalars
    mod_e     = _modulation_energy(audio, sr)                   # scalar
    hnr       = _hnr_proxy(audio, sr)                           # scalar

    centroid  = librosa.feature.spectral_centroid(
                    y=audio, sr=sr, hop_length=HOP_LENGTH)[0]
    cen_feats = np.array([centroid.mean(), centroid.std()])      # (2,)

    zcr       = librosa.feature.zero_crossing_rate(
                    audio, hop_length=HOP_LENGTH)[0]
    zcr_feats = np.array([zcr.mean(), zcr.std()])               # (2,)

    mfcc      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=8,
                                      hop_length=HOP_LENGTH)
    delta     = librosa.feature.delta(mfcc)
    mfcc_d_std = np.std(delta, axis=1)                          # (8,)

    return np.concatenate([
        lfcc_mean, lfcc_std,
        flatness, energy,
        [jitter, shimmer, mod_e, hnr],
        cen_feats, zcr_feats,
        mfcc_d_std,
    ]).astype(np.float32)                                        # (143,)
