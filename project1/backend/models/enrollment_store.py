"""
Enrollment store — persists speaker embeddings for enrolled users.

Layout on disk:
    data/enrolled/<backend>/<username>/
        sample_00.npy … sample_N.npy   ← individual embeddings
        centroid.npy                   ← mean L2-normalised embedding (used for verification)

SQLite mirrors: users + voice_embeddings tables (see backend/db/database.py).
"""
import os
import numpy as np
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from backend.db.database import get_connection, init_db
from backend.core.speaker_encoder import _l2_norm

BASE_DIR = "data/enrolled"


class EnrollmentStore:
    def __init__(self, base_dir: str = BASE_DIR):
        self.base_dir = base_dir
        init_db()

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #
    def enroll(self, username: str, embeddings: np.ndarray, backend: str) -> np.ndarray:
        """
        Save individual embeddings and compute/store the centroid.
        embeddings: (N, D) array — one row per enrollment sample.
        Returns the centroid embedding (D,).
        """
        user_dir = self._user_dir(username, backend)
        os.makedirs(user_dir, exist_ok=True)

        for i, emb in enumerate(embeddings):
            np.save(os.path.join(user_dir, f"sample_{i:02d}.npy"), emb)

        centroid = _l2_norm(embeddings.mean(axis=0))
        np.save(os.path.join(user_dir, "centroid.npy"), centroid)

        self._persist_to_db(username, embeddings, centroid, backend)
        return centroid

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #
    def get_centroid(self, username: str, backend: str) -> np.ndarray:
        path = os.path.join(self._user_dir(username, backend), "centroid.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"User '{username}' not enrolled for backend '{backend}'")
        return np.load(path)

    def get_all_samples(self, username: str, backend: str) -> np.ndarray:
        """Return (N, D) array of all stored sample embeddings."""
        user_dir = self._user_dir(username, backend)
        files = sorted(f for f in os.listdir(user_dir) if f.startswith("sample_"))
        return np.vstack([np.load(os.path.join(user_dir, f)) for f in files])

    def list_users(self, backend: str) -> list:
        backend_dir = os.path.join(self.base_dir, backend)
        if not os.path.exists(backend_dir):
            return []
        return [d for d in os.listdir(backend_dir)
                if os.path.isdir(os.path.join(backend_dir, d))]

    def is_enrolled(self, username: str, backend: str) -> bool:
        path = os.path.join(self._user_dir(username, backend), "centroid.npy")
        return os.path.exists(path)

    def delete_user(self, username: str, backend: str):
        import shutil
        user_dir = self._user_dir(username, backend)
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row:
            cur.execute(
            "DELETE FROM voice_embeddings WHERE user_id = ? AND auth_type = ?",
            (row["id"], "text_independent")
        )
            conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #
    def _user_dir(self, username: str, backend: str) -> str:
        return os.path.join(self.base_dir, backend, username)

    def _persist_to_db(self, username: str, embeddings: np.ndarray,
                       centroid: np.ndarray, backend: str):
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO users (username) VALUES (?)", (username,)
        )
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_id = cur.fetchone()["id"]

        # Remove old embeddings for this user + backend
        cur.execute(
            "DELETE FROM voice_embeddings WHERE user_id = ? AND auth_type = ?",
            (user_id, "text_independent")
        )

        # Store centroid as a single row
        cur.execute(
            "INSERT INTO voice_embeddings (user_id, embedding, auth_type, sample_index) VALUES (?,?,?,?)",
            (user_id, centroid.tobytes(), "text_independent", 0)
        )
        conn.commit()
        conn.close()
