"""
Phase 2 unit test — verifies feature extraction and augmentation
work correctly on a generated sine wave (no microphone needed).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from backend.core.audio_utils import preprocess, save_audio
from backend.core.feature_extractor import extract_all, get_mean_features
from backend.core.augmentation import augment_sample

SR = 16000

def make_test_audio(duration=3.0, freq=200.0):
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return audio

def test_preprocessing():
    audio = make_test_audio()
    clean = preprocess(audio, SR)
    assert clean.ndim == 1
    assert np.max(np.abs(clean)) <= 1.0 + 1e-6
    print(f"  [OK] preprocess — shape={clean.shape}, max_amp={np.max(np.abs(clean)):.4f}")

def test_feature_extraction():
    audio = make_test_audio()
    feats = extract_all(audio, SR)

    assert feats["mfcc"].shape[0] == 40
    assert feats["mfcc_delta"].shape[0] == 120
    assert feats["mel_spec"].shape[0] == 128
    assert feats["pitch"].ndim == 1
    assert feats["spectral"].shape[0] == 4

    vec = get_mean_features(audio, SR)
    assert vec.shape == (248,), f"Expected (248,), got {vec.shape}"

    print(f"  [OK] MFCC shape:       {feats['mfcc'].shape}")
    print(f"  [OK] MFCC+delta shape: {feats['mfcc_delta'].shape}")
    print(f"  [OK] Mel spec shape:   {feats['mel_spec'].shape}")
    print(f"  [OK] Pitch shape:      {feats['pitch'].shape}")
    print(f"  [OK] Spectral shape:   {feats['spectral'].shape}")
    print(f"  [OK] Mean vector dim:  {vec.shape[0]}")

def test_augmentation():
    audio = make_test_audio()
    variants = augment_sample(audio, SR)
    expected_keys = [
        "original", "white_noise_20", "white_noise_10",
        "pitch_up", "pitch_down", "time_stretch_fast",
        "time_stretch_slow", "reverb", "volume_low", "volume_high"
    ]
    for key in expected_keys:
        assert key in variants, f"Missing augmentation: {key}"
        assert variants[key].ndim == 1
        print(f"  [OK] augmentation '{key}' — shape={variants[key].shape}")

def test_save_load():
    audio = make_test_audio()
    path = "data/processed/_test/test_sine.wav"
    save_audio(audio, path, SR)
    from backend.core.audio_utils import load_audio
    loaded = load_audio(path, SR)
    assert loaded.ndim == 1
    print(f"  [OK] save/load — saved {len(audio)} samples, loaded {len(loaded)} samples")
    os.remove(path)

if __name__ == "__main__":
    print("\n=== Phase 2 Tests ===\n")

    print("1. Preprocessing")
    test_preprocessing()

    print("\n2. Feature Extraction")
    test_feature_extraction()

    print("\n3. Augmentation")
    test_augmentation()

    print("\n4. Save / Load")
    test_save_load()

    print("\nAll Phase 2 tests PASSED.\n")
