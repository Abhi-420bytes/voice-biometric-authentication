"""
Test script: verifies microphone input, saves a 5-second recording,
and checks librosa can load and process it.
"""
import sounddevice as sd
import numpy as np
import soundfile as sf
import librosa
import os

SAMPLE_RATE = 16000
DURATION = 5
OUTPUT_PATH = "data/raw/mic_test.wav"

def list_devices():
    print("\nAvailable audio devices:")
    print(sd.query_devices())

def record_test():
    os.makedirs("data/raw", exist_ok=True)
    print(f"\nRecording {DURATION} seconds at {SAMPLE_RATE}Hz... Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("Recording done.")

    audio = audio.flatten()
    sf.write(OUTPUT_PATH, audio, SAMPLE_RATE)
    print(f"Saved to {OUTPUT_PATH}")
    return audio

def verify_librosa(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(f"\nLibrosa load OK — shape: {y.shape}, sr: {sr}")
    print(f"MFCC shape: {mfccs.shape}")
    print(f"Audio duration: {len(y)/sr:.2f}s")
    print(f"Max amplitude: {np.max(np.abs(y)):.4f}")

if __name__ == "__main__":
    list_devices()
    audio = record_test()
    verify_librosa(OUTPUT_PATH)
    print("\nPhase 1 microphone test PASSED")
