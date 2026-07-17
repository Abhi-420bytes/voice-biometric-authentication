"""
Unit tests for core pipeline modules:
  audio_utils · feature_extractor · speaker_encoder · spoof_detector · verification
"""
import os, sys, glob
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TI_WAVS  = sorted(glob.glob(f"{DATA_DIR}/raw/abhiram/text_independent/sample_*.wav"))
SR       = 16000


def load(path):
    y, _ = sf.read(path, dtype="float32")
    return y.mean(axis=1) if y.ndim > 1 else y


# ── audio_utils ──────────────────────────────────────────────────
class TestAudioUtils:
    def test_preprocess_returns_float32(self):
        from backend.core.audio_utils import preprocess
        y = load(TI_WAVS[0])
        out = preprocess(y, SR)
        assert out.dtype == np.float32

    def test_preprocess_normalises_amplitude(self):
        from backend.core.audio_utils import preprocess
        y = load(TI_WAVS[0])
        out = preprocess(y, SR)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_preprocess_output_nonempty(self):
        from backend.core.audio_utils import preprocess
        y = load(TI_WAVS[0])
        out = preprocess(y, SR)
        assert len(out) > 0


# ── feature_extractor ────────────────────────────────────────────
class TestFeatureExtractor:
    @pytest.fixture(autouse=True)
    def audio(self):
        from backend.core.audio_utils import preprocess
        y = load(TI_WAVS[0])
        self.y = preprocess(y, SR)

    def test_mfcc_shape(self):
        from backend.core.feature_extractor import extract_mfcc
        m = extract_mfcc(self.y)
        assert m.shape[0] == 40

    def test_mel_shape(self):
        from backend.core.feature_extractor import extract_mel_spectrogram
        m = extract_mel_spectrogram(self.y)
        assert m.shape[0] == 128

    def test_mean_features_dim(self):
        from backend.core.feature_extractor import get_mean_features
        v = get_mean_features(self.y)
        assert v.ndim == 1
        assert len(v) > 0

    def test_pitch_length_matches_frames(self):
        from backend.core.feature_extractor import extract_pitch, extract_mfcc
        p = extract_pitch(self.y)
        m = extract_mfcc(self.y)
        assert len(p) == m.shape[1]


# ── speaker_encoder ──────────────────────────────────────────────
class TestSpeakerEncoder:
    @pytest.fixture(autouse=True, scope="class")
    def encoder(self):
        from backend.core.speaker_encoder import SpeakerEncoder
        self.__class__.enc = SpeakerEncoder(backend="resemblyzer")

    def test_embedding_shape(self):
        from backend.core.audio_utils import preprocess
        y   = preprocess(load(TI_WAVS[0]), SR)
        emb = self.enc.get_embedding(y, SR)
        assert emb.ndim == 1
        assert emb.shape[0] == 256

    def test_embedding_is_unit_norm(self):
        from backend.core.audio_utils import preprocess
        y   = preprocess(load(TI_WAVS[0]), SR)
        emb = self.enc.get_embedding(y, SR)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_same_speaker_high_similarity(self):
        from backend.core.audio_utils    import preprocess
        from backend.core.speaker_encoder import cosine_similarity
        e1 = self.enc.get_embedding(preprocess(load(TI_WAVS[0]), SR), SR)
        e2 = self.enc.get_embedding(preprocess(load(TI_WAVS[1]), SR), SR)
        sim = cosine_similarity(e1, e2)
        assert sim > 0.70, f"Same-speaker sim too low: {sim:.4f}"

    def test_batch_embed_count(self):
        from backend.core.audio_utils import preprocess
        audios = [preprocess(load(p), SR) for p in TI_WAVS[:3]]
        embs   = self.enc.embed_batch(audios, SR)
        assert len(embs) == 3


# ── spoof_detector ───────────────────────────────────────────────
class TestSpoofDetector:
    @pytest.fixture(autouse=True, scope="class")
    def detector(self):
        from backend.core.spoof_detector import SpoofDetector
        self.__class__.det = SpoofDetector(mode="rule_based", threshold=0.20)

    def test_real_audio_score_above_threshold(self):
        from backend.core.audio_utils import preprocess
        y      = preprocess(load(TI_WAVS[0]), SR)
        result = self.det.detect(y)
        assert result.score >= 0.20, f"Real audio score too low: {result.score:.4f}"

    def test_real_audio_is_real(self):
        from backend.core.audio_utils import preprocess
        y = preprocess(load(TI_WAVS[0]), SR)
        assert self.det.detect(y).is_real

    def test_result_has_all_fields(self):
        from backend.core.audio_utils import preprocess
        y = preprocess(load(TI_WAVS[0]), SR)
        r = self.det.detect(y)
        assert hasattr(r, "score")
        assert hasattr(r, "is_real")
        assert hasattr(r, "attack_type")
        assert hasattr(r, "latency_ms")
        assert r.latency_ms >= 0


# ── verification ─────────────────────────────────────────────────
class TestVerification:
    TEST_USER = "_unit_test_verifier_"

    @pytest.fixture(autouse=True, scope="class")
    def setup(self):
        from backend.core.verification import VoiceVerifier
        from backend.core.audio_utils  import preprocess
        self.__class__.verifier = VoiceVerifier(backend="resemblyzer", threshold=0.75)
        audios = [preprocess(load(p), SR) for p in TI_WAVS[:5]]
        self.__class__.verifier.enroll(self.TEST_USER, audios, SR)
        yield
        self.__class__.verifier.store.delete_user(self.TEST_USER, "resemblyzer")

    def test_verify_same_speaker_accepted(self):
        from backend.core.audio_utils import preprocess
        y      = preprocess(load(TI_WAVS[5]), SR)
        result = self.verifier.verify(self.TEST_USER, y, SR)
        assert result.accepted, f"Same speaker rejected, score={result.score:.4f}"

    def test_verify_returns_score_in_range(self):
        from backend.core.audio_utils import preprocess
        y = preprocess(load(TI_WAVS[6]), SR)
        r = self.verifier.verify(self.TEST_USER, y, SR)
        assert 0.0 <= r.score <= 1.0

    def test_identify_returns_ranked_dict(self):
        from backend.core.audio_utils import preprocess
        y = preprocess(load(TI_WAVS[7]), SR)
        ranked = self.verifier.identify(y, SR)
        assert isinstance(ranked, dict)
        assert self.TEST_USER in ranked
