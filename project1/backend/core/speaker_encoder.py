"""
Speaker embedding extractor.

Two backends:
  - "resemblyzer" : 256-dim GE2E d-vector (fast, no download required)
  - "ecapa_tdnn"  : 192-dim ECAPA-TDNN x-vector (more accurate, ~80 MB download on first use)

All embeddings are L2-normalised before returning.
"""
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

SAMPLE_RATE = 16000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class SpeakerEncoder:
    def __init__(self, backend: str = "resemblyzer"):
        if backend not in ("resemblyzer", "ecapa_tdnn"):
            raise ValueError(f"Unknown backend: {backend}")
        self.backend = backend
        self._model = None

    # ------------------------------------------------------------------ #
    #  Lazy model loading                                                  #
    # ------------------------------------------------------------------ #
    def _load(self):
        if self._model is not None:
            return

        if self.backend == "resemblyzer":
            from resemblyzer import VoiceEncoder
            self._model = VoiceEncoder(device=DEVICE)

        elif self.backend == "ecapa_tdnn":
            from speechbrain.inference.speaker import EncoderClassifier
            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="data/models/ecapa_tdnn",
                run_opts={"device": DEVICE},
            )

    # ------------------------------------------------------------------ #
    #  Core extraction                                                     #
    # ------------------------------------------------------------------ #
    def get_embedding(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Return a single L2-normalised embedding vector for one audio clip."""
        self._load()

        if self.backend == "resemblyzer":
            from resemblyzer import preprocess_wav
            wav = preprocess_wav(audio, source_sr=sr)
            emb = self._model.embed_utterance(wav)          # (256,)

        elif self.backend == "ecapa_tdnn":
            signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # (1, T)
            with torch.no_grad():
                emb = self._model.encode_batch(signal)       # (1, 1, 192)
            emb = emb.squeeze().cpu().numpy()                # (192,)

        return _l2_norm(emb)

    def embed_batch(self, audios: list, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Return (N, D) matrix of L2-normalised embeddings."""
        return np.vstack([self.get_embedding(a, sr) for a in audios])

    @property
    def embedding_dim(self) -> int:
        return 256 if self.backend == "resemblyzer" else 192


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #
def _l2_norm(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-10)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors (range -1 … 1)."""
    return float(np.dot(_l2_norm(a), _l2_norm(b)))
