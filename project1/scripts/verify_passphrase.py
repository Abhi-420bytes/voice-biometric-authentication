#!/usr/bin/env python3
"""
Verify a user with text-dependent authentication (passphrase + speaker).

Usage:
    python scripts/verify_passphrase.py --user abhiram
    python scripts/verify_passphrase.py --user abhiram --wav path/to/audio.wav
    python scripts/verify_passphrase.py --user abhiram --passphrase "open sesame"
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf

from backend.core.audio_utils              import preprocess
from backend.core.text_dependent_pipeline  import TextDependentPipeline

SR                 = 16000
DEFAULT_PASSPHRASE = "my voice is my password"


def record_clip(duration: int, sr: int) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:
        print("  sounddevice not installed. Use --wav instead.")
        sys.exit(1)
    print(f"\n  Say your passphrase now.")
    input("  Press ENTER then start speaking...")
    print(f"  Recording {duration}s ... ", end="", flush=True)
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("done.")
    return audio.flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user",       required=True)
    ap.add_argument("--passphrase", default=DEFAULT_PASSPHRASE)
    ap.add_argument("--wav",        default=None, help="Path to wav file instead of mic")
    ap.add_argument("--duration",   type=int, default=5)
    args = ap.parse_args()

    pipeline = TextDependentPipeline(passphrase=args.passphrase)

    if not pipeline.is_enrolled(args.user):
        print(f"\n  User '{args.user}' is not enrolled.")
        print(f"  Run:  python scripts/enroll_passphrase.py --user {args.user}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Text-Dependent Verification")
    print(f"  User       : {args.user}")
    print(f"  Passphrase : \"{args.passphrase}\"")
    print(f"{'='*55}")

    if args.wav:
        y, _ = sf.read(args.wav, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        audio = preprocess(y, SR)
        print(f"  Loaded: {args.wav}")
    else:
        audio = record_clip(args.duration, SR)
        audio = preprocess(audio, SR)

    print("  Authenticating ...")
    result = pipeline.authenticate(args.user, audio, SR)
    print(result)

    if result.accepted:
        print(f"\n  ✓  ACCESS GRANTED — welcome, {args.user}!")
    else:
        reason = {
            "REJECTED_SPOOF": "audio flagged as synthetic/replayed",
            "REJECTED_TEXT":  f"passphrase mismatch (got: '{result.text_result.transcript}')",
            "REJECTED_VOICE": "voice does not match enrolled speaker",
        }.get(result.final_decision, result.final_decision)
        print(f"\n  ✗  ACCESS DENIED  — {reason}")


if __name__ == "__main__":
    main()
