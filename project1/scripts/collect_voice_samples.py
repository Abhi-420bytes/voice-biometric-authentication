"""
Interactive voice sample collection script.
Records N samples for a user (text-dependent or text-independent),
preprocesses them, extracts features, and saves everything to disk.

Usage:
    python3 scripts/collect_voice_samples.py --user abhiram --mode text_dependent --samples 5
    python3 scripts/collect_voice_samples.py --user abhiram --mode text_independent --samples 5
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np

from backend.core.audio_utils import record_audio, save_audio, preprocess
from backend.core.feature_extractor import extract_all, get_mean_features
from backend.core.augmentation import augment_sample

PASSPHRASE        = "my voice is my password"
RECORD_DURATION   = 4       # seconds per sample
SAMPLE_RATE       = 16000
BASE_RAW_DIR      = "data/raw"
BASE_PROC_DIR     = "data/processed"
BASE_FEAT_DIR     = "data/features"
BASE_AUG_DIR      = "data/augmented"


def collect_samples(username: str, mode: str, n_samples: int, augment: bool):
    raw_dir  = os.path.join(BASE_RAW_DIR,  username, mode)
    proc_dir = os.path.join(BASE_PROC_DIR, username, mode)
    feat_dir = os.path.join(BASE_FEAT_DIR, username, mode)
    aug_dir  = os.path.join(BASE_AUG_DIR,  username, mode)

    for d in [raw_dir, proc_dir, feat_dir]:
        os.makedirs(d, exist_ok=True)
    if augment:
        os.makedirs(aug_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Voice Collection — User: {username} | Mode: {mode}")
    print(f"{'='*55}")

    if mode == "text_dependent":
        print(f"\n  Passphrase to say each time:")
        print(f"  >> \"{PASSPHRASE}\" <<\n")
    else:
        print("\n  Speak freely — any sentence works (text-independent).\n")

    features_list = []

    for i in range(1, n_samples + 1):
        input(f"  [Sample {i}/{n_samples}] Press ENTER when ready, then speak...")
        print(f"  Recording {RECORD_DURATION}s...")

        audio_raw = record_audio(duration=RECORD_DURATION, sample_rate=SAMPLE_RATE)
        audio_clean = preprocess(audio_raw, sample_rate=SAMPLE_RATE)

        # Save raw and processed
        raw_path  = os.path.join(raw_dir,  f"sample_{i:02d}.wav")
        proc_path = os.path.join(proc_dir, f"sample_{i:02d}.wav")
        save_audio(audio_raw,   raw_path,  SAMPLE_RATE)
        save_audio(audio_clean, proc_path, SAMPLE_RATE)

        # Extract features
        feats = extract_all(audio_clean)
        mean_vec = get_mean_features(audio_clean)
        feat_path = os.path.join(feat_dir, f"sample_{i:02d}.npz")
        np.savez(feat_path,
                 mfcc=feats["mfcc"],
                 mfcc_delta=feats["mfcc_delta"],
                 mel_spec=feats["mel_spec"],
                 pitch=feats["pitch"],
                 spectral=feats["spectral"],
                 mean_vector=mean_vec)
        features_list.append(mean_vec)

        # Augmentation
        if augment:
            aug_variants = augment_sample(audio_clean, SAMPLE_RATE)
            aug_sample_dir = os.path.join(aug_dir, f"sample_{i:02d}")
            os.makedirs(aug_sample_dir, exist_ok=True)
            for aug_name, aug_audio in aug_variants.items():
                save_audio(aug_audio, os.path.join(aug_sample_dir, f"{aug_name}.wav"), SAMPLE_RATE)

        print(f"  Saved. Feature vector dim: {mean_vec.shape[0]}")

    # Save stacked mean feature matrix for quick loading
    feature_matrix = np.vstack(features_list)
    np.save(os.path.join(feat_dir, "feature_matrix.npy"), feature_matrix)

    print(f"\n  Collection complete.")
    print(f"  Feature matrix shape: {feature_matrix.shape}  ({n_samples} samples x {feature_matrix.shape[1]} features)")
    print(f"  Raw:       {raw_dir}")
    print(f"  Processed: {proc_dir}")
    print(f"  Features:  {feat_dir}")
    if augment:
        print(f"  Augmented: {aug_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect voice samples for enrollment")
    parser.add_argument("--user",    required=True,  help="Username (e.g. abhiram)")
    parser.add_argument("--mode",    required=True,  choices=["text_dependent", "text_independent"])
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to record")
    parser.add_argument("--augment", action="store_true", default=True, help="Generate augmented variants")
    args = parser.parse_args()

    collect_samples(args.user, args.mode, args.samples, args.augment)
