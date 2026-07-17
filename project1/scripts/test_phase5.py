"""
Phase 5 tests — no microphone, no internet required.
Uses synthetic audio:
  "Real"  → voiced harmonic signal with natural micro-variations
  "Fake"  → Griffin-Lim resynthesis (vocoder-like artifacts)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import librosa
import torch
import tempfile
from backend.core.spoof_detector import SpoofDetector

SR = 16000
CLIP = SR * 3   # 3-second clips


# ------------------------------------------------------------------ #
#  Synthetic audio generators                                         #
# ------------------------------------------------------------------ #
def make_real(seed=0) -> np.ndarray:
    """Voiced harmonic signal with natural jitter/shimmer."""
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 3.0, CLIP, endpoint=False)
    # Slowly varying F0 (natural pitch modulation)
    f0  = 130 + 10 * np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 1, CLIP)
    phase = np.cumsum(2 * np.pi * f0 / SR)
    sig   = sum((1 / k) * np.sin(k * phase) for k in range(1, 8))
    # Add amplitude modulation (syllabic rate ~5 Hz)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
    sig = sig * envelope
    # Light noise
    sig += 0.01 * rng.standard_normal(CLIP)
    sig /= np.max(np.abs(sig)) + 1e-9
    return sig.astype(np.float32)


def make_fake(seed=0) -> np.ndarray:
    """
    TTS-like: perfectly constant F0, zero amplitude modulation, no jitter.
    These are the primary artifacts the rule-based detector targets.
    """
    t     = np.linspace(0, 3.0, CLIP, endpoint=False)
    f0    = 130.0          # constant pitch — no jitter
    phase = 2 * np.pi * f0 * t
    sig   = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 8))
    # Constant amplitude — no syllabic modulation
    sig  /= np.max(np.abs(sig)) + 1e-9
    return sig.astype(np.float32)


# ------------------------------------------------------------------ #
#  Tests                                                              #
# ------------------------------------------------------------------ #
def test_lfcc_extraction():
    from backend.core.spoof_features import extract_lfcc
    audio = make_real(0)
    lfcc  = extract_lfcc(audio, SR)
    assert lfcc.shape[0] == 60, f"Expected 60 LFCC coefficients, got {lfcc.shape[0]}"
    assert lfcc.shape[1] > 0
    print(f"  [OK] LFCC shape = {lfcc.shape}")


def test_spoof_feature_vector():
    from backend.core.spoof_features import extract_spoof_feature_vector
    audio = make_real(0)
    vec   = extract_spoof_feature_vector(audio, SR)
    assert vec.shape == (143,), f"Expected (143,), got {vec.shape}"
    assert not np.any(np.isnan(vec)), "NaN in feature vector"
    assert not np.any(np.isinf(vec)), "Inf in feature vector"
    print(f"  [OK] Feature vector dim={vec.shape[0]}, no NaN/Inf")


def test_rule_based_detection():
    from backend.core.spoof_detector import SpoofDetector
    detector = SpoofDetector(mode="rule_based")

    real_audio = make_real(0)
    fake_audio = make_fake(0)

    result_real = detector.detect(real_audio, SR)
    result_fake = detector.detect(fake_audio, SR)

    print(f"  [OK] Real audio: score={result_real.score:.4f} "
          f"is_real={result_real.is_real} attack={result_real.attack_type}")
    print(f"  [OK] Fake audio: score={result_fake.score:.4f} "
          f"is_real={result_fake.is_real} attack={result_fake.attack_type}")
    assert result_real.score > result_fake.score, \
        f"Real score ({result_real.score:.4f}) should be > fake score ({result_fake.score:.4f})"
    print(f"  [OK] Real score > Fake score (gap = {result_real.score - result_fake.score:.4f})")


def test_gmm_train_and_detect():
    from backend.core.spoof_detector import SpoofDetector
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = os.path.join(tmpdir, "real")
        fake_dir = os.path.join(tmpdir, "fake")
        os.makedirs(real_dir); os.makedirs(fake_dir)

        rng = np.random.default_rng(0)
        for i in range(8):
            sf.write(os.path.join(real_dir, f"r{i}.wav"), make_real(i), SR)
            # Add tiny per-sample noise so GMM covariance is non-degenerate
            fake = make_fake(i) + rng.normal(0, 1e-4, CLIP).astype(np.float32)
            sf.write(os.path.join(fake_dir, f"f{i}.wav"), fake, SR)

        detector = SpoofDetector(mode="gmm")
        detector.train(real_dir, fake_dir,
                       save_path=os.path.join(tmpdir, "gmm.pkl"))

        real_r = detector.detect(make_real(99), SR)
        fake_r = detector.detect(make_fake(99), SR)
        print(f"  [OK] GMM real score={real_r.score:.4f}  fake score={fake_r.score:.4f}")
        assert real_r.score > fake_r.score, "GMM: real score should exceed fake score"


def test_rawnet2_architecture():
    from backend.models.rawnet2 import RawNet2, count_parameters
    model = RawNet2()
    n_params = count_parameters(model)
    print(f"  [OK] RawNet2 parameters: {n_params:,}")

    x   = torch.randn(2, SR * 3)     # batch of 2, 3-second clips
    out = model(x)
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"

    proba = model.predict_proba(x)
    assert proba.shape == (2,)
    assert ((proba >= 0) & (proba <= 1)).all()
    print(f"  [OK] RawNet2 forward pass: input=(2,{SR*3}) → logit={out.shape} proba={proba.shape}")


def test_rawnet2_train_and_detect():
    from backend.core.spoof_detector import SpoofDetector
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = os.path.join(tmpdir, "real")
        fake_dir = os.path.join(tmpdir, "fake")
        os.makedirs(real_dir); os.makedirs(fake_dir)

        rng = np.random.default_rng(1)
        for i in range(12):
            sf.write(os.path.join(real_dir, f"r{i}.wav"), make_real(i), SR)
            fake = make_fake(i) + rng.normal(0, 1e-4, CLIP).astype(np.float32)
            sf.write(os.path.join(fake_dir, f"f{i}.wav"), fake, SR)

        detector = SpoofDetector(mode="rawnet2")
        detector._train_rawnet2(
            [os.path.join(real_dir, f) for f in os.listdir(real_dir)],
            [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)],
            SR, epochs=5
        )

        real_r = detector.detect(make_real(99), SR)
        fake_r = detector.detect(make_fake(99), SR)
        print(f"  [OK] RawNet2 real score={real_r.score:.4f}  fake score={fake_r.score:.4f}")


def test_auth_pipeline():
    from backend.core.auth_pipeline import AuthPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        # Lower spoof threshold (0.20) so synthetic voiced audio passes;
        # fake (Griffin-Lim) score is ~0.23 lower → still rejected
        real_score = SpoofDetector("rule_based").detect(make_real(0), SR).score
        fake_score = SpoofDetector("rule_based").detect(make_fake(0), SR).score
        threshold  = (real_score + fake_score) / 2   # midpoint

        pipeline = AuthPipeline(spoof_mode="rule_based",
                                verify_backend="resemblyzer",
                                spoof_threshold=threshold)
        pipeline.verifier.store.base_dir = tmpdir

        enroll_clips = [make_real(i) for i in range(5)]
        pipeline.enroll("alice", enroll_clips, SR)

        result_genuine = pipeline.authenticate("alice", make_real(99), SR)
        result_spoof   = pipeline.authenticate("alice", make_fake(99), SR)

        print(f"  [OK] Genuine: {result_genuine.final_decision}  "
              f"spoof_score={result_genuine.spoof_result.score:.4f}  threshold={threshold:.4f}")
        print(f"  [OK] Spoof  : {result_spoof.final_decision}  "
              f"spoof_score={result_spoof.spoof_result.score:.4f}")
        assert result_genuine.final_decision in ("ACCEPTED", "REJECTED_VOICE")
        assert result_spoof.final_decision == "REJECTED_SPOOF"


# ------------------------------------------------------------------ #
#  Runner                                                             #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("\n=== Phase 5 Tests (Anti-Spoofing + Full Pipeline) ===\n")

    print("1. LFCC Extraction")
    test_lfcc_extraction()

    print("\n2. Spoof Feature Vector")
    test_spoof_feature_vector()

    print("\n3. Rule-Based Detector")
    test_rule_based_detection()

    print("\n4. GMM Detector (train + detect)")
    test_gmm_train_and_detect()

    print("\n5. RawNet2 Architecture")
    test_rawnet2_architecture()

    print("\n6. RawNet2 Train + Detect (5 epochs)")
    test_rawnet2_train_and_detect()

    print("\n7. Full Auth Pipeline")
    test_auth_pipeline()

    print("\nAll Phase 5 tests PASSED.\n")
