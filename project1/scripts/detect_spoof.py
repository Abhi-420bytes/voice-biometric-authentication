"""
CLI: check if an audio recording is real or spoofed.

Usage:
    # Live microphone (4 seconds)
    python3 scripts/detect_spoof.py

    # Pre-recorded file
    python3 scripts/detect_spoof.py --wav path/to/audio.wav

    # With trained GMM or RawNet2
    python3 scripts/detect_spoof.py --mode gmm --model data/models/gmm_spoof.pkl
    python3 scripts/detect_spoof.py --mode rawnet2 --model data/models/rawnet2_spoof.pt
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np

from backend.core.audio_utils  import record_audio, load_audio, preprocess
from backend.core.spoof_detector import SpoofDetector

SAMPLE_RATE    = 16000
RECORD_SECONDS = 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anti-spoofing detector")
    parser.add_argument("--wav",   default=None, help="Path to wav file")
    parser.add_argument("--mode",  default="rule_based",
                        choices=["rule_based", "gmm", "rawnet2"])
    parser.add_argument("--model", default=None, help="Path to trained model file")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    detector = SpoofDetector(mode=args.mode,
                              model_path=args.model,
                              threshold=args.threshold)

    if args.wav:
        print(f"\n  Loading: {args.wav}")
        audio = preprocess(load_audio(args.wav, SAMPLE_RATE), SAMPLE_RATE)
    else:
        input(f"\n  Press ENTER and speak for {RECORD_SECONDS}s...")
        print(f"  Recording...")
        audio = preprocess(record_audio(RECORD_SECONDS, SAMPLE_RATE), SAMPLE_RATE)

    print(f"\n  Running {args.mode} spoof detector...")
    result = detector.detect(audio, SAMPLE_RATE)

    print(f"\n  {result}")

    # Visual score bar
    bar_len  = 40
    filled   = int(result.score * bar_len)
    bar      = "█" * filled + "░" * (bar_len - filled)
    t_pos    = int(result.threshold * bar_len)
    print(f"\n  Real  [{bar}] {result.score:.4f}")
    print(f"  Thresh[{'─'*t_pos}▲{'─'*(bar_len-t_pos)}] {result.threshold:.4f}")
