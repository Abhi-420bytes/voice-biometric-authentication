"""
Audio recording and preprocessing utilities.
Used across all phases of the pipeline.
"""
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import noisereduce as nr
import os

SAMPLE_RATE = 16000

def record_audio(duration: int = 5, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def load_audio(file_path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.load(file_path, sr=sample_rate)
    return y

def save_audio(audio: np.ndarray, file_path: str, sample_rate: int = SAMPLE_RATE):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    sf.write(file_path, audio, sample_rate)

def reduce_noise(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return nr.reduce_noise(y=audio, sr=sample_rate)

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return audio / max_val

def trim_silence(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, top_db: int = 20) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed

def preprocess(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio = reduce_noise(audio, sample_rate)
    audio = trim_silence(audio, sample_rate)
    audio = normalize_audio(audio)
    return audio
