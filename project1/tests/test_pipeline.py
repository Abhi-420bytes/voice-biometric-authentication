"""
End-to-end pipeline tests.

Simulates a full user journey:
  Enroll → Authenticate (same speaker) → Authenticate (wrong speaker) → Delete
"""
import os, sys, glob
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.core.audio_utils  import preprocess
from backend.core.auth_pipeline import AuthPipeline

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
TI_WAVS      = sorted(glob.glob(f"{DATA_DIR}/raw/abhiram/text_independent/sample_*.wav"))
IMPOSTOR_WAVS= sorted(glob.glob(f"{DATA_DIR}/raw/friend/*.wav"))
SR           = 16000
E2E_USER     = "_e2e_test_user_"


def load_clean(path):
    y, _ = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return preprocess(y, SR)


@pytest.fixture(scope="module")
def pipeline():
    return AuthPipeline(
        spoof_mode       = "rule_based",
        verify_backend   = "resemblyzer",
        spoof_threshold  = 0.20,
        verify_threshold = 0.75,
    )


@pytest.fixture(scope="module", autouse=True)
def enroll_e2e(pipeline):
    audios = [load_clean(p) for p in TI_WAVS[:5]]
    pipeline.enroll(E2E_USER, audios, SR)
    yield
    pipeline.verifier.store.delete_user(E2E_USER, "resemblyzer")


# ── 1. Same-speaker authentication ───────────────────────────────
class TestSameSpeaker:
    def test_genuine_samples_accepted(self, pipeline):
        accepted = 0
        for path in TI_WAVS[5:15]:
            audio  = load_clean(path)
            result = pipeline.authenticate(E2E_USER, audio, SR)
            if result.accepted:
                accepted += 1
        rate = accepted / 10
        assert rate >= 0.80, f"Acceptance rate too low: {rate:.0%} (expected ≥80%)"

    def test_voice_score_above_threshold(self, pipeline):
        audio  = load_clean(TI_WAVS[5])
        result = pipeline.authenticate(E2E_USER, audio, SR)
        assert result.voice_result is not None
        assert result.voice_result.score >= 0.75, (
            f"Voice score {result.voice_result.score:.4f} below threshold")

    def test_spoof_score_passes(self, pipeline):
        audio  = load_clean(TI_WAVS[6])
        result = pipeline.authenticate(E2E_USER, audio, SR)
        assert result.spoof_result.score >= 0.20, (
            f"Spoof score {result.spoof_result.score:.4f} too low for real audio")


# ── 2. Result structure ───────────────────────────────────────────
class TestResultStructure:
    def test_result_has_all_fields(self, pipeline):
        audio  = load_clean(TI_WAVS[7])
        result = pipeline.authenticate(E2E_USER, audio, SR)
        assert result.username        == E2E_USER
        assert result.final_decision  in ("ACCEPTED", "REJECTED_SPOOF", "REJECTED_VOICE")
        assert result.spoof_result    is not None
        assert result.total_ms        >= 0

    def test_accepted_has_voice_result(self, pipeline):
        audio  = load_clean(TI_WAVS[8])
        result = pipeline.authenticate(E2E_USER, audio, SR)
        if result.accepted:
            assert result.voice_result is not None

    def test_str_output_not_empty(self, pipeline):
        audio  = load_clean(TI_WAVS[9])
        result = pipeline.authenticate(E2E_USER, audio, SR)
        assert len(str(result)) > 0


# ── 3. Impostor rejection ─────────────────────────────────────────
class TestImpostor:
    @pytest.mark.skipif(not IMPOSTOR_WAVS, reason="No impostor recordings found")
    def test_impostor_mean_score_lower_than_genuine(self, pipeline):
        genuine_scores = []
        for p in TI_WAVS[5:10]:
            r = pipeline.authenticate(E2E_USER, load_clean(p), SR)
            if r.voice_result:
                genuine_scores.append(r.voice_result.score)

        impostor_scores = []
        for path in IMPOSTOR_WAVS:
            try:
                audio = load_clean(path)
            except Exception:
                continue       # skip corrupted / wrong-format files
            r = pipeline.authenticate(E2E_USER, audio, SR)
            if r.voice_result:
                impostor_scores.append(r.voice_result.score)

        if not impostor_scores:
            pytest.skip("All impostor files unreadable — re-record them louder")

        assert np.mean(genuine_scores) > np.mean(impostor_scores), (
            f"Genuine mean {np.mean(genuine_scores):.4f} should be > "
            f"impostor mean {np.mean(impostor_scores):.4f}")


# ── 4. Full metrics ───────────────────────────────────────────────
class TestMetrics:
    def test_far_is_zero_on_genuine(self, pipeline):
        """All genuine samples at threshold 0.75 should be accepted."""
        false_rejects = 0
        total = 10
        for path in TI_WAVS[10:20]:
            result = pipeline.authenticate(E2E_USER, load_clean(path), SR)
            if result.final_decision == "REJECTED_VOICE":
                false_rejects += 1
        frr = false_rejects / total
        assert frr == 0.0, f"FRR={frr:.0%} — {false_rejects}/{total} genuine samples rejected"
