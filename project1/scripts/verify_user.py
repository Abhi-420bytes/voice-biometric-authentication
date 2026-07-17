"""
Verify a user's identity or identify an unknown speaker (text-independent).

Usage:
    # 1-to-1 verification (claim identity)
    python3 scripts/verify_user.py --user abhiram

    # 1-to-N identification (who is this?)
    python3 scripts/verify_user.py --identify

    # Use a pre-recorded wav instead of mic
    python3 scripts/verify_user.py --user abhiram --wav path/to/audio.wav

    # Use ECAPA-TDNN backend
    python3 scripts/verify_user.py --user abhiram --backend ecapa_tdnn
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np

from backend.core.audio_utils import record_audio, load_audio, preprocess
from backend.core.verification import VoiceVerifier

SAMPLE_RATE    = 16000
RECORD_SECONDS = 4


def get_audio(wav_path: str = None) -> np.ndarray:
    if wav_path:
        print(f"  Loading: {wav_path}")
        audio = load_audio(wav_path, SAMPLE_RATE)
    else:
        input("  Press ENTER and start speaking (4s)...")
        print(f"  Recording {RECORD_SECONDS}s...")
        audio = record_audio(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE)
    return preprocess(audio, SAMPLE_RATE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice verification / identification")
    parser.add_argument("--user",    default=None,  help="Claimed username for 1-to-1 verify")
    parser.add_argument("--identify",action="store_true", help="1-to-N identification mode")
    parser.add_argument("--wav",     default=None,  help="Path to a wav file (skip mic)")
    parser.add_argument("--backend", default="resemblyzer",
                        choices=["resemblyzer", "ecapa_tdnn"])
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override cosine similarity threshold")
    args = parser.parse_args()

    verifier = VoiceVerifier(backend=args.backend, threshold=args.threshold)

    print(f"\n  Voice Authentication  |  backend: {args.backend}")
    print(f"  Threshold: {verifier.threshold:.3f}\n")

    audio = get_audio(args.wav)

    if args.identify:
        print("\n  Running 1-to-N identification...")
        scores = verifier.identify(audio, SAMPLE_RATE)
        if not scores:
            print("  No users enrolled yet.")
        else:
            print(f"\n  {'Rank':<5} {'Username':<20} {'Score':<10} {'Decision'}")
            print("  " + "-" * 50)
            for rank, (user, score) in enumerate(scores.items(), 1):
                decision = "MATCH" if (rank == 1 and score >= verifier.threshold) else ""
                print(f"  {rank:<5} {user:<20} {score:<10.4f} {decision}")

    elif args.user:
        if not verifier.store.is_enrolled(args.user, args.backend):
            print(f"  ERROR: User '{args.user}' is not enrolled. Run enroll_user.py first.")
            sys.exit(1)

        result = verifier.verify(args.user, audio, SAMPLE_RATE)
        print(f"\n  {result}")

        bar_len = 40
        filled  = int(result.score * bar_len)
        bar     = "█" * filled + "░" * (bar_len - filled)
        thresh_pos = int(result.threshold * bar_len)
        print(f"\n  Score  [{bar}] {result.score:.4f}")
        print(f"  Thresh [{'─' * thresh_pos}▲{'─' * (bar_len - thresh_pos)}] {result.threshold:.4f}")

    else:
        print("  Provide --user <name> or --identify")
        sys.exit(1)
