"""
Train the GMM or RawNet2 spoof detector on real/fake audio directories.

Without ASVspoof dataset — generate fake data from real recordings using TTS
or by providing a directory of synthetic samples.

Usage:
    # With ASVspoof or any real/fake wav directories
    python3 scripts/train_spoof_detector.py \\
        --mode gmm \\
        --real  data/spoof_data/real \\
        --fake  data/spoof_data/fake

    # RawNet2 (deep model, slower)
    python3 scripts/train_spoof_detector.py \\
        --mode rawnet2 \\
        --real  data/spoof_data/real \\
        --fake  data/spoof_data/fake \\
        --epochs 30

    # Generate synthetic fake data from real recordings (demo / no external data needed)
    python3 scripts/train_spoof_detector.py --generate-fake --real data/raw/abhiram
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import soundfile as sf

from backend.core.audio_utils   import load_audio, preprocess, save_audio
from backend.core.spoof_detector import SpoofDetector

SAMPLE_RATE = 16000


def generate_fake_from_real(real_dir: str, out_dir: str):
    """
    Create 'fake' training samples from real recordings by applying
    vocoder-like transformations (pitch-sync resynthesis, spectral smoothing).
    These mimic TTS artifacts without needing actual TTS software.
    """
    import librosa
    os.makedirs(out_dir, exist_ok=True)
    wav_files = [f for f in _list_wavs(real_dir)]
    print(f"  Generating {len(wav_files)} fake samples from {real_dir}")

    for fpath in wav_files:
        audio = preprocess(load_audio(fpath, SAMPLE_RATE), SAMPLE_RATE)

        # Resynthesise via Griffin-Lim with reduced iterations (introduces
        # phase artifacts similar to vocoders)
        S    = librosa.stft(audio, n_fft=512, hop_length=160)
        mag  = np.abs(S)
        # Smooth the magnitude spectrum (vocoder-like)
        from scipy.ndimage import uniform_filter1d
        mag_smooth = uniform_filter1d(mag, size=5, axis=0)
        fake = librosa.griffinlim(mag_smooth, n_iter=8,
                                   hop_length=160, win_length=400)
        fake = fake[:len(audio)]
        fake = fake / (np.max(np.abs(fake)) + 1e-9)

        out_path = os.path.join(out_dir, os.path.basename(fpath))
        save_audio(fake.astype(np.float32), out_path, SAMPLE_RATE)

    print(f"  Saved fake samples to {out_dir}")
    return out_dir


def _list_wavs(directory: str) -> list:
    result = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".wav"):
                result.append(os.path.join(root, f))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spoof detector")
    parser.add_argument("--mode",  default="gmm",
                        choices=["gmm", "rawnet2"])
    parser.add_argument("--real",  default="data/spoof_data/real",
                        help="Directory of real (genuine) wav files")
    parser.add_argument("--fake",  default="data/spoof_data/fake",
                        help="Directory of fake (spoof) wav files")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs (RawNet2 only)")
    parser.add_argument("--generate-fake", action="store_true",
                        help="Auto-generate fake samples from --real directory")
    parser.add_argument("--save",  default=None, help="Output model path")
    args = parser.parse_args()

    fake_dir = args.fake
    if args.generate_fake:
        fake_dir = "data/spoof_data/fake_generated"
        generate_fake_from_real(args.real, fake_dir)

    if not os.path.exists(args.real) or not _list_wavs(args.real):
        print(f"ERROR: No wav files found in --real={args.real}")
        sys.exit(1)
    if not os.path.exists(fake_dir) or not _list_wavs(fake_dir):
        print(f"ERROR: No wav files found in --fake={fake_dir}")
        sys.exit(1)

    detector = SpoofDetector(mode=args.mode)

    if args.mode == "rawnet2":
        # Patch epoch count
        def _train(real_files, fake_files, sr):
            detector._train_rawnet2(real_files, fake_files, sr, epochs=args.epochs)
        real_files = _list_wavs(args.real)
        fake_files = _list_wavs(fake_dir)
        _train(real_files, fake_files, SAMPLE_RATE)
        path = args.save or "data/models/rawnet2_spoof.pt"
        detector.save(path)
    else:
        detector.train(args.real, fake_dir, save_path=args.save)

    print(f"\n  Training complete. Mode: {args.mode}")
