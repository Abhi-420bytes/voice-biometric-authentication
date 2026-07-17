"""
Text-dependent verification: ensures the user says the expected passphrase.
Uses Whisper (offline ASR) for transcription + fuzzy string matching.
"""
from __future__ import annotations
import re, time
from difflib import SequenceMatcher
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_PASSPHRASE = "my voice is my password"
TEXT_THRESHOLD = 0.70   # fuzzy score >= this → passphrase accepted


@dataclass
class TextVerifyResult:
    transcript:    str
    expected:      str
    text_score:    float      # 0–1 combined fuzzy match
    text_accepted: bool
    latency_ms:    float

    def __str__(self):
        verdict = "ACCEPTED" if self.text_accepted else "REJECTED"
        return (f"[{verdict}] transcript='{self.transcript}' "
                f"expected='{self.expected}' score={self.text_score:.3f}")


# ── Text normalisation ──────────────────────────────────────────────
def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


def _word_overlap(a: str, b: str) -> float:
    """Fraction of expected words present in transcript (recall-style)."""
    wa = set(a.split())
    wb = set(b.split())
    if not wb:
        return 0.0
    return len(wa & wb) / len(wb)


def text_similarity(transcript: str, expected: str) -> float:
    """
    0.6 × SequenceMatcher character ratio  +  0.4 × word recall
    Combining both gives robustness to word order and small ASR errors.
    """
    t = _normalize(transcript)
    e = _normalize(expected)
    seq = SequenceMatcher(None, t, e).ratio()
    wov = _word_overlap(t, e)
    return 0.6 * seq + 0.4 * wov


# ── Main class ──────────────────────────────────────────────────────
class TextVerifier:
    """
    Transcribes audio with Whisper and checks against the enrolled passphrase.

    model_size:  "tiny" (fastest, ~39 MB) | "base" | "small" | "medium"
    """
    def __init__(self,
                 passphrase:  str = DEFAULT_PASSPHRASE,
                 model_size:  str = "base",
                 threshold:   float = TEXT_THRESHOLD):
        self.passphrase = _normalize(passphrase)
        self.threshold  = threshold
        self._size      = model_size
        self._model     = None          # lazy-loaded on first use

    def _load(self):
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self._size)
        return self._model

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
        """Return raw Whisper transcript (not normalised)."""
        model = self._load()
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio.astype(np.float32)
        result = model.transcribe(audio, language="en", fp16=False, verbose=False)
        return result["text"].strip()

    def verify(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> TextVerifyResult:
        """Transcribe audio and compare to the enrolled passphrase."""
        t0         = time.time()
        transcript = self.transcribe(audio, sr)
        score      = text_similarity(transcript, self.passphrase)
        accepted   = score >= self.threshold
        return TextVerifyResult(
            transcript    = transcript,
            expected      = self.passphrase,
            text_score    = score,
            text_accepted = accepted,
            latency_ms    = (time.time() - t0) * 1000,
        )
