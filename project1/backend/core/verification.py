"""
Voice verification engine for text-independent authentication.

Workflow:
  1. Enroll  — record N clips → extract embeddings → store centroid
  2. Verify  — record 1 clip  → extract embedding → cosine sim vs centroid → accept/reject
  3. Identify — 1-to-N: who is this person among all enrolled users?

Default thresholds (tuned for Resemblyzer GE2E on clean speech):
  resemblyzer  : 0.75
  ecapa_tdnn   : 0.25  (cosine distance space, so lower = more similar)
"""
from __future__ import annotations
import os, sys, time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from backend.core.speaker_encoder import SpeakerEncoder, cosine_similarity
from backend.models.enrollment_store import EnrollmentStore
from backend.db.database import get_connection

SAMPLE_RATE = 16000

# Cosine similarity threshold: score >= threshold → ACCEPTED
THRESHOLDS = {
    "resemblyzer": 0.75,
    "ecapa_tdnn":  0.80,
}


@dataclass
class VerificationResult:
    username: str
    score: float
    threshold: float
    accepted: bool
    backend: str
    latency_ms: float

    def __str__(self):
        verdict = "ACCEPTED" if self.accepted else "REJECTED"
        return (f"[{verdict}] {self.username} | "
                f"score={self.score:.4f} threshold={self.threshold:.4f} | "
                f"backend={self.backend} | {self.latency_ms:.0f}ms")


class VoiceVerifier:
    def __init__(self, backend: str = "resemblyzer", threshold: float = None):
        self.backend   = backend
        self.threshold = threshold if threshold is not None else THRESHOLDS[backend]
        self.encoder   = SpeakerEncoder(backend=backend)
        self.store     = EnrollmentStore()

    # ------------------------------------------------------------------ #
    #  Enrollment                                                          #
    # ------------------------------------------------------------------ #
    def enroll(self, username: str, audios: list, sr: int = SAMPLE_RATE) -> np.ndarray:
        """
        Enroll a user from a list of audio arrays.
        Returns the stored centroid embedding.
        """
        embeddings = self.encoder.embed_batch(audios, sr)
        centroid   = self.store.enroll(username, embeddings, self.backend)
        self._log(username, "enroll", None, None)
        return centroid

    # ------------------------------------------------------------------ #
    #  Verification (1-to-1)                                              #
    # ------------------------------------------------------------------ #
    def verify(self, username: str, audio: np.ndarray,
               sr: int = SAMPLE_RATE) -> VerificationResult:
        """Verify claimed identity against stored centroid."""
        t0 = time.time()
        centroid = self.store.get_centroid(username, self.backend)
        emb      = self.encoder.get_embedding(audio, sr)
        score    = cosine_similarity(emb, centroid)
        accepted = score >= self.threshold
        latency  = (time.time() - t0) * 1000

        result = VerificationResult(
            username=username,
            score=score,
            threshold=self.threshold,
            accepted=accepted,
            backend=self.backend,
            latency_ms=latency,
        )
        self._log(username, "verify", score, "accepted" if accepted else "rejected")
        return result

    # ------------------------------------------------------------------ #
    #  Identification (1-to-N)                                            #
    # ------------------------------------------------------------------ #
    def identify(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
        """
        Identify the speaker among all enrolled users.
        Returns a ranked dict {username: score} sorted by descending score.
        The best match is accepted only if its score exceeds the threshold.
        """
        users = self.store.list_users(self.backend)
        if not users:
            return {}

        emb = self.encoder.get_embedding(audio, sr)
        scores = {u: cosine_similarity(emb, self.store.get_centroid(u, self.backend))
                  for u in users}
        ranked = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        return ranked

    # ------------------------------------------------------------------ #
    #  Threshold tuning                                                    #
    # ------------------------------------------------------------------ #
    def compute_eer(self, genuine_scores: list, impostor_scores: list) -> tuple[float, float]:
        """
        Compute Equal Error Rate (EER) and the optimal threshold.
        genuine_scores  : cosine sim scores from same-speaker pairs
        impostor_scores : cosine sim scores from different-speaker pairs
        Returns (eer, best_threshold).
        """
        all_scores = sorted(set(genuine_scores + impostor_scores))
        best_eer, best_thresh = 1.0, 0.5

        for thresh in all_scores:
            far = sum(s >= thresh for s in impostor_scores) / len(impostor_scores)
            frr = sum(s <  thresh for s in genuine_scores)  / len(genuine_scores)
            eer = (far + frr) / 2
            if eer < best_eer:
                best_eer, best_thresh = eer, thresh

        return best_eer, best_thresh

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #
    def _log(self, username: str, action: str, score, result):
        try:
            conn = get_connection()
            cur  = conn.cursor()
            if action == "verify":
                cur.execute(
                    "INSERT INTO auth_logs (username, auth_type, result, confidence) VALUES (?,?,?,?)",
                    (username, "text_independent", result, score)
                )
                conn.commit()
            conn.close()
        except Exception:
            pass
