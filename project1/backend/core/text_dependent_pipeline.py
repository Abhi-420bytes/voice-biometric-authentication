"""
Text-Dependent Authentication Pipeline.

Three-stage check (all must pass):
  1. Anti-spoofing  — is the audio genuine?
  2. Passphrase     — did the user say the correct phrase? (Whisper ASR)
  3. Speaker        — does the voice match the enrolled speaker?

Combined score = 0.4 × text_score + 0.6 × voice_score
"""
from __future__ import annotations
import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from backend.core.spoof_detector         import SpoofDetector, SpoofResult
from backend.core.verification           import VoiceVerifier, VerificationResult
from backend.core.text_verifier          import TextVerifier, TextVerifyResult
from backend.db.database                 import get_connection

SAMPLE_RATE        = 16000
DEFAULT_PASSPHRASE = "my voice is my password"


@dataclass
class TextAuthResult:
    username:       str
    final_decision: str           # ACCEPTED | REJECTED_SPOOF | REJECTED_TEXT | REJECTED_VOICE
    spoof_result:   SpoofResult
    text_result:    TextVerifyResult  | None
    voice_result:   VerificationResult | None
    combined_score: float
    total_ms:       float

    @property
    def accepted(self) -> bool:
        return self.final_decision == "ACCEPTED"

    def __str__(self):
        lines = [
            "  ╔══ Text-Dependent Auth ══════════════════════╗",
            f"  ║  User        : {self.username}",
            f"  ║  Decision    : {self.final_decision}",
            f"  ║  Spoof score : {self.spoof_result.score:.4f}  "
            f"({'REAL' if self.spoof_result.is_real else 'FAKE'})",
        ]
        if self.text_result:
            lines.append(
                f"  ║  Text score  : {self.text_result.text_score:.4f}  "
                f"transcript='{self.text_result.transcript}'"
            )
        if self.voice_result:
            lines.append(
                f"  ║  Voice score : {self.voice_result.score:.4f}  "
                f"({'OK' if self.voice_result.accepted else 'FAIL'})"
            )
        lines += [
            f"  ║  Combined    : {self.combined_score:.4f}",
            f"  ║  Total time  : {self.total_ms:.0f} ms",
            "  ╚════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


class TextDependentPipeline:
    """
    Text-dependent voice authentication system.

    Usage:
        pipe = TextDependentPipeline(passphrase="my voice is my password")
        pipe.enroll("abhiram", [audio1, audio2, ...])
        result = pipe.authenticate("abhiram", audio)
        if result.accepted:
            grant_access()
    """
    def __init__(self,
                 passphrase:       str   = DEFAULT_PASSPHRASE,
                 whisper_model:    str   = "base",
                 spoof_mode:       str   = "rule_based",
                 verify_backend:   str   = "resemblyzer",
                 spoof_threshold:  float | None = None,
                 text_threshold:   float | None = None,
                 voice_threshold:  float | None = None,
                 spoof_model_path: str   | None = None):

        self.passphrase = passphrase

        self.spoof_detector = SpoofDetector(
            mode=spoof_mode,
            threshold=spoof_threshold,
            model_path=spoof_model_path,
        )
        self.text_verifier = TextVerifier(
            passphrase=passphrase,
            model_size=whisper_model,
        )
        if text_threshold is not None:
            self.text_verifier.threshold = text_threshold

        self.voice_verifier = VoiceVerifier(
            backend=verify_backend,
            threshold=voice_threshold,
        )

    # ---------------------------------------------------------------- #
    #  Enrollment                                                        #
    # ---------------------------------------------------------------- #
    def enroll(self, username: str, audios: list,
               sr: int = SAMPLE_RATE) -> "np.ndarray":
        """
        Enroll from passphrase recordings.
        Builds speaker embedding centroid (same as text-independent).
        """
        centroid = self.voice_verifier.enroll(username, audios, sr)
        self._log(username, "enroll", None, None)
        return centroid

    # ---------------------------------------------------------------- #
    #  Authentication                                                    #
    # ---------------------------------------------------------------- #
    def authenticate(self, username: str, audio,
                     sr: int = SAMPLE_RATE) -> TextAuthResult:
        import numpy as np
        t0 = time.time()

        # ── Step 1: Anti-spoofing ────────────────────────────────────
        spoof_result = self.spoof_detector.detect(audio, sr)
        if not spoof_result.is_real:
            return TextAuthResult(
                username       = username,
                final_decision = "REJECTED_SPOOF",
                spoof_result   = spoof_result,
                text_result    = None,
                voice_result   = None,
                combined_score = 0.0,
                total_ms       = (time.time() - t0) * 1000,
            )

        # ── Step 2: Passphrase text check ────────────────────────────
        text_result = self.text_verifier.verify(audio, sr)
        if not text_result.text_accepted:
            self._log(username, "verify", text_result.text_score, "rejected_text")
            return TextAuthResult(
                username       = username,
                final_decision = "REJECTED_TEXT",
                spoof_result   = spoof_result,
                text_result    = text_result,
                voice_result   = None,
                combined_score = text_result.text_score,
                total_ms       = (time.time() - t0) * 1000,
            )

        # ── Step 3: Speaker verification ─────────────────────────────
        voice_result = self.voice_verifier.verify(username, audio, sr)
        combined     = 0.4 * text_result.text_score + 0.6 * voice_result.score
        decision     = "ACCEPTED" if voice_result.accepted else "REJECTED_VOICE"

        self._log(username, "verify", combined,
                  "accepted" if decision == "ACCEPTED" else "rejected")
        return TextAuthResult(
            username       = username,
            final_decision = decision,
            spoof_result   = spoof_result,
            text_result    = text_result,
            voice_result   = voice_result,
            combined_score = combined,
            total_ms       = (time.time() - t0) * 1000,
        )

    def is_enrolled(self, username: str) -> bool:
        return self.voice_verifier.store.is_enrolled(
            username, self.voice_verifier.backend)

    # ---------------------------------------------------------------- #
    #  Internal                                                          #
    # ---------------------------------------------------------------- #
    def _log(self, username: str, action: str, score, result):
        try:
            conn = get_connection()
            cur  = conn.cursor()
            if action == "verify":
                cur.execute(
                    "INSERT INTO auth_logs "
                    "(username, auth_type, result, confidence) VALUES (?,?,?,?)",
                    (username, "text_dependent", result, score)
                )
                conn.commit()
            conn.close()
        except Exception:
            pass
