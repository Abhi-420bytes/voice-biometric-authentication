"""
Visualize extracted features for a given user sample.
Plots waveform, MFCC, Mel Spectrogram, and Pitch contour side by side.

Usage:
    python3 scripts/visualize_features.py --user abhiram --mode text_dependent --sample 1
    python3 scripts/visualize_features.py --file data/raw/abhiram/text_dependent/sample_01.wav
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa.display

from backend.core.audio_utils import load_audio, preprocess
from backend.core.feature_extractor import (
    extract_mfcc, extract_mel_spectrogram,
    extract_pitch, extract_spectral_features
)

SAMPLE_RATE = 16000


def visualize(audio: np.ndarray, sr: int, title: str, save_path: str = None):
    mfcc    = extract_mfcc(audio, sr)
    mel     = extract_mel_spectrogram(audio, sr)
    pitch   = extract_pitch(audio, sr)
    times   = np.arange(len(audio)) / sr
    t_frames = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=160)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(4, 1, hspace=0.5)

    # 1. Waveform
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, audio, color='steelblue', linewidth=0.6)
    ax1.set_title("Waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim([0, times[-1]])

    # 2. Mel Spectrogram
    ax2 = fig.add_subplot(gs[1])
    img = librosa.display.specshow(mel, sr=sr, hop_length=160, x_axis='time',
                                   y_axis='mel', ax=ax2, cmap='magma')
    ax2.set_title("Mel Spectrogram (dB)")
    fig.colorbar(img, ax=ax2, format='%+2.0f dB')

    # 3. MFCC
    ax3 = fig.add_subplot(gs[2])
    img2 = librosa.display.specshow(mfcc, sr=sr, hop_length=160, x_axis='time',
                                    ax=ax3, cmap='coolwarm')
    ax3.set_title(f"MFCCs ({mfcc.shape[0]} coefficients)")
    ax3.set_ylabel("MFCC #")
    fig.colorbar(img2, ax=ax3)

    # 4. Pitch (F0)
    ax4 = fig.add_subplot(gs[3])
    voiced = pitch > 0
    ax4.plot(t_frames[voiced], pitch[voiced], 'o', color='crimson',
             markersize=1.5, label='Voiced F0')
    ax4.set_title("Pitch Contour (F0)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Frequency (Hz)")
    ax4.set_xlim([0, t_frames[-1]])
    ax4.legend(fontsize=8)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user",   default=None)
    parser.add_argument("--mode",   default="text_dependent",
                        choices=["text_dependent", "text_independent"])
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--file",   default=None, help="Direct path to a .wav file")
    parser.add_argument("--save",   default=None, help="Save plot to this path instead of displaying")
    args = parser.parse_args()

    if args.file:
        path = args.file
    elif args.user:
        path = f"data/processed/{args.user}/{args.mode}/sample_{args.sample:02d}.wav"
    else:
        print("Provide --user or --file")
        sys.exit(1)

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    audio = load_audio(path, SAMPLE_RATE)
    title = f"Feature Visualization — {os.path.basename(path)}"
    visualize(audio, SAMPLE_RATE, title, save_path=args.save)
