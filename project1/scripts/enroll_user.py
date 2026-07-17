"""
Enroll a user into the voice authentication system (text-independent).
Records N samples of free speech, extracts d-vector embeddings,
stores the centroid in both disk and SQLite.

Usage:
    python3 scripts/enroll_user.py --user abhiram
    python3 scripts/enroll_user.py --user abhiram --samples 7 --backend ecapa_tdnn
    python3 scripts/enroll_user.py --user abhiram --wav-dir data/raw/abhiram/text_independent
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np

from backend.core.audio_utils import record_audio, load_audio, preprocess
from backend.core.verification import VoiceVerifier

SAMPLE_RATE    = 16000
RECORD_SECONDS = 5


def enroll_from_mic(username: str, n_samples: int, verifier: VoiceVerifier):
    print(f"\n  Enrolling: {username}  |  backend: {verifier.backend}")
    print(f"  Speak freely — say any sentence for {RECORD_SECONDS}s each time.\n")

    audios = []
    for i in range(1, n_samples + 1):
        input(f"  [Sample {i}/{n_samples}] Press ENTER and start speaking...")
        print(f"  Recording {RECORD_SECONDS}s...")
        raw   = record_audio(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE)
        clean = preprocess(raw, SAMPLE_RATE)
        audios.append(clean)
        print(f"  Done.\n")
    return audios


def enroll_from_wav_dir(wav_dir: str, verifier: VoiceVerifier):
    files = sorted(f for f in os.listdir(wav_dir) if f.endswith(".wav"))
    if not files:
        print(f"No .wav files found in {wav_dir}")
        sys.exit(1)
    print(f"\n  Loading {len(files)} wav files from {wav_dir}")
    audios = []
    for f in files:
        audio = load_audio(os.path.join(wav_dir, f), SAMPLE_RATE)
        audio = preprocess(audio, SAMPLE_RATE)
        audios.append(audio)
    return audios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a user via voice (text-independent)")
    parser.add_argument("--user",    required=True,  help="Username")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples (mic mode)")
    parser.add_argument("--backend", default="resemblyzer",
                        choices=["resemblyzer", "ecapa_tdnn"])
    parser.add_argument("--wav-dir", default=None,
                        help="Use pre-recorded wavs instead of mic")
    args = parser.parse_args()

    verifier = VoiceVerifier(backend=args.backend)

    if args.wav_dir:
        audios = enroll_from_wav_dir(args.wav_dir, verifier)
    else:
        audios = enroll_from_mic(args.user, args.samples, verifier)

    print(f"\n  Extracting embeddings ({verifier.backend})...")
    centroid = verifier.enroll(args.user, audios, SAMPLE_RATE)

    print(f"\n  Enrollment complete.")
    print(f"  User:     {args.user}")
    print(f"  Samples:  {len(audios)}")
    print(f"  Centroid: shape={centroid.shape}, norm={np.linalg.norm(centroid):.4f}")
    print(f"  Stored at: data/enrolled/{args.backend}/{args.user}/")
