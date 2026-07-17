"""
Data augmentation for voice samples.
Generates realistic variations to improve model robustness.
"""
import numpy as np
import librosa

SAMPLE_RATE = 16000


def add_white_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    signal_power = np.mean(audio ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)


def add_background_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
    """Mix in a separate noise clip (e.g. room noise) at a given SNR."""
    if len(noise) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(audio)]

    signal_power = np.mean(audio ** 2) + 1e-9
    noise_power  = np.mean(noise ** 2) + 1e-9
    scale        = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return (audio + scale * noise).astype(np.float32)


def pitch_shift(audio: np.ndarray, sr: int = SAMPLE_RATE, n_steps: float = 2.0) -> np.ndarray:
    """Shift pitch by n_steps semitones (positive = higher, negative = lower)."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps).astype(np.float32)


def time_stretch(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """Stretch/compress time. rate > 1 = faster, rate < 1 = slower."""
    return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)


def add_reverb(audio: np.ndarray, sr: int = SAMPLE_RATE, room_scale: float = 0.3) -> np.ndarray:
    """Simulate room reverb by convolving with a simple exponential decay IR."""
    decay_samples = int(sr * room_scale)
    ir = np.exp(-np.linspace(0, 6, decay_samples)).astype(np.float32)
    ir /= ir.sum()
    reverbed = np.convolve(audio, ir, mode='full')[:len(audio)]
    return reverbed.astype(np.float32)


def random_crop(audio: np.ndarray, sr: int = SAMPLE_RATE, min_duration: float = 2.0) -> np.ndarray:
    """Randomly crop a segment of at least min_duration seconds."""
    min_samples = int(min_duration * sr)
    if len(audio) <= min_samples:
        return audio
    start = np.random.randint(0, len(audio) - min_samples)
    return audio[start: start + min_samples]


def volume_perturbation(audio: np.ndarray, low: float = 0.7, high: float = 1.3) -> np.ndarray:
    factor = np.random.uniform(low, high)
    return np.clip(audio * factor, -1.0, 1.0).astype(np.float32)


def augment_sample(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Apply all augmentations and return a dict of variants.
    Each variant is a separate augmented copy for training data expansion.
    """
    return {
        "original":        audio,
        "white_noise_20":  add_white_noise(audio, snr_db=20),
        "white_noise_10":  add_white_noise(audio, snr_db=10),
        "pitch_up":        pitch_shift(audio, sr, n_steps=1.5),
        "pitch_down":      pitch_shift(audio, sr, n_steps=-1.5),
        "time_stretch_fast": time_stretch(audio, rate=1.1),
        "time_stretch_slow": time_stretch(audio, rate=0.9),
        "reverb":          add_reverb(audio, sr, room_scale=0.3),
        "volume_low":      volume_perturbation(audio, low=0.5, high=0.7),
        "volume_high":     volume_perturbation(audio, low=1.2, high=1.4),
    }
