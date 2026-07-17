#!/usr/bin/env python3
"""
Enroll a user for text-dependent authentication.

Usage:
    python scripts/enroll_passphrase.py --user abhiram
    python scripts/enroll_passphrase.py --user abhiram --passphrase "open sesame" --n 5
    python scripts/enroll_passphrase.py --user abhiram --wav-dir data/raw/abhiram/text_dependent
"""
import argparse, os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf

from backend.core.audio_utils              import preprocess
from backend.core.text_dependent_pipeline  import TextDependentPipeline

SR                 = 16000
DEFAULT_PASSPHRASE = "my voice is my password"


def record_clip(duration: int, sr: int, label: str) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:
        print("  sounddevice not installed. Use --wav-dir instead.")
        sys.exit(1)
    print(f"\n  {label}")
    input("  Press ENTER then start speaking...")
    print(f"  Recording {duration}s ... ", end="", flush=True)
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("done.")
    return audio.flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user",       required=True,                    help="Username to enroll")
    ap.add_argument("--passphrase", default=DEFAULT_PASSPHRASE,       help="Passphrase to enroll")
    ap.add_argument("--n",          type=int, default=5,              help="Number of recordings (mic mode)")
    ap.add_argument("--duration",   type=int, default=5,              help="Seconds per recording (mic mode)")
    ap.add_argument("--wav-dir",    default=None,                     help="Load existing wav files from this folder")
    args = ap.parse_args()

    pipeline = TextDependentPipeline(passphrase=args.passphrase)

    print(f"\n{'='*55}")
    print(f"  Text-Dependent Enrollment")
    print(f"  User       : {args.user}")
    print(f"  Passphrase : \"{args.passphrase}\"")
    print(f"{'='*55}")

    audios = []

    if args.wav_dir:
        files = sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")))
        if not files:
            print(f"  No .wav files found in {args.wav_dir}")
            sys.exit(1)
        print(f"  Loading {len(files)} wav files from {args.wav_dir} ...")
        for f in files:
            y, _ = sf.read(f, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            audios.append(preprocess(y, SR))
            print(f"    ✓  {os.path.basename(f)}")
    else:
        print(f"\n  You will record the passphrase {args.n} times.")
        print(f"  Say exactly: \"{args.passphrase}\" each time.\n")
        for i in range(args.n):
            audio = record_clip(args.duration, SR,
                                f"Recording {i+1}/{args.n} — say the passphrase now")
            audios.append(preprocess(audio, SR))

    print(f"\n  Enrolling {len(audios)} recordings ...")
    centroid = pipeline.enroll(args.user, audios, SR)
    print(f"  Enrolled!  Centroid shape: {centroid.shape}")
    print(f"\n  '{args.user}' is ready for text-dependent authentication.")
    print(f"  Run:  python scripts/verify_passphrase.py --user {args.user}")


if __name__ == "__main__":
    main()
