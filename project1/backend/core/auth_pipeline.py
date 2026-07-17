"""
Full authentication pipeline:
  Step 1 — Anti-spoofing check  (is this audio real or synthesised?)
  Step 2 — Voice verification   (is this person who they claim to be?)

Both must pass for authentication to succeed.

For text-dependent mode, use TextDependentPipeline directly.
"""
from __future__ import annotations
import os, sys, time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from backend.core.spoof_detector          import SpoofDetector, SpoofResult
from backend.core.verification            import VoiceVerifier, VerificationResult
from backend.core.text_dependent_pipeline import TextDependentPipeline, TextAuthResult

SAMPLE_RATE = 16000


@dataclass
class AuthResult:
    username:      str
    final_decision: str           # "ACCEPTED" | "REJECTED_SPOOF" | "REJECTED_VOICE"
    spoof_result:  SpoofResult
    voice_result:  VerificationResult | None
    total_ms:      float

    @property
    def accepted(self) -> bool:
        return self.final_decision == "ACCEPTED"

    def __str__(self):
        lines = [
            f"  ╔══ Auth Result ══════════════════════════╗",
            f"  ║  User       : {self.username}",
            f"  ║  Decision   : {self.final_decision}",
            f"  ║  Spoof score: {self.spoof_result.score:.4f}  "
            f"({'REAL' if self.spoof_result.is_real else 'FAKE'})"
            f"  [{self.spoof_result.attack_type}]",
        ]
        if self.voice_result:
            lines.append(
                f"  ║  Voice score: {self.voice_result.score:.4f}  "
                f"({'OK' if self.voice_result.accepted else 'FAIL'})"
            )
        lines += [
            f"  ║  Total time : {self.total_ms:.0f} ms",
            f"  ╚════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


class AuthPipeline:
    """
    Ties anti-spoofing and speaker verification into one call.

    Usage:
        pipeline = AuthPipeline()
        result   = pipeline.authenticate("abhiram", audio)
        if result.accepted:
            grant_access()
    """
    def __init__(self,
                 spoof_mode:   str   = "rule_based",
                 verify_backend: str = "resemblyzer",
                 spoof_threshold: float | None = None,
                 verify_threshold: float | None = None,
                 spoof_model_path: str | None = None):

        self.spoof_detector = SpoofDetector(
            mode=spoof_mode,
            threshold=spoof_threshold,
            model_path=spoof_model_path,
        )
        self.verifier = VoiceVerifier(
            backend=verify_backend,
            threshold=verify_threshold,
        )

    # ---------------------------------------------------------------- #
    #  Main entry point                                                 #
    # ---------------------------------------------------------------- #
    def authenticate(self, username: str, audio,
                     sr: int = SAMPLE_RATE) -> AuthResult:
        import numpy as np
        t0 = time.time()

        # Step 1 — Spoof detection
        spoof_result = self.spoof_detector.detect(audio, sr)

        if not spoof_result.is_real:
            return AuthResult(
                username       = username,
                final_decision = "REJECTED_SPOOF",
                spoof_result   = spoof_result,
                voice_result   = None,
                total_ms       = (time.time() - t0) * 1000,
            )

        # Step 2 — Voice verification
        voice_result = self.verifier.verify(username, audio, sr)

        decision = "ACCEPTED" if voice_result.accepted else "REJECTED_VOICE"

        return AuthResult(
            username       = username,
            final_decision = decision,
            spoof_result   = spoof_result,
            voice_result   = voice_result,
            total_ms       = (time.time() - t0) * 1000,
        )

    def enroll(self, username: str, audios: list, sr: int = SAMPLE_RATE):
        """Enroll a user (delegates to VoiceVerifier)."""
        return self.verifier.enroll(username, audios, sr)

    def is_enrolled(self, username: str) -> bool:
        return self.verifier.store.is_enrolled(
            username, self.verifier.backend)

    # ---------------------------------------------------------------- #
    #  Text-dependent mode (convenience wrapper)                        #
    # ---------------------------------------------------------------- #
    def authenticate_text_dependent(self,
                                    username:    str,
                                    audio,
                                    passphrase:  str   = "my voice is my password",
                                    sr:          int   = SAMPLE_RATE,
                                    whisper_model: str = "base") -> TextAuthResult:
        """
        Run the full text-dependent pipeline (spoof + passphrase + speaker).
        Builds a fresh TextDependentPipeline with the same spoof/voice settings.
        """
        td = TextDependentPipeline(
            passphrase      = passphrase,
            whisper_model   = whisper_model,
            spoof_mode      = self.spoof_detector.mode,
            verify_backend  = self.verifier.backend,
            spoof_threshold = self.spoof_detector.threshold,
            voice_threshold = self.verifier.threshold,
        )
        return td.authenticate(username, audio, sr)
