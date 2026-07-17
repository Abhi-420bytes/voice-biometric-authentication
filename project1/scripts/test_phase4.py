"""
Phase 4 tests — no microphone required.
Uses synthesised speech-like noise bursts to exercise the full pipeline:
  encoder → enroll → verify → identify → EER
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

SR = 16000


# ------------------------------------------------------------------ #
#  Synthetic audio helpers                                            #
# ------------------------------------------------------------------ #
def make_voiced_audio(duration=3.0, fundamental=150.0, formants=(700, 1200, 2600),
                      seed=None) -> np.ndarray:
    """
    Generate a harmonic signal that resembles a vowel —
    enough voiced energy for Resemblyzer's VAD to keep it.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, duration, int(SR * duration), endpoint=False)
    sig = np.zeros_like(t)
    for k in range(1, 10):          # harmonics
        sig += (1.0 / k) * np.sin(2 * np.pi * fundamental * k * t)
    for f in formants:              # bandpass resonance bumps
        sig += 0.3 * np.sin(2 * np.pi * f * t)
    sig += 0.01 * rng.standard_normal(len(t))
    sig /= np.max(np.abs(sig)) + 1e-9
    return sig.astype(np.float32)


# Speaker A uses fundamental=130 Hz, Speaker B uses 220 Hz
def speaker_a(seed=0): return make_voiced_audio(fundamental=130, seed=seed)
def speaker_b(seed=0): return make_voiced_audio(fundamental=220,
                                                 formants=(900, 1500, 2800), seed=seed)


# ------------------------------------------------------------------ #
#  Tests                                                              #
# ------------------------------------------------------------------ #
def test_encoder():
    from backend.core.speaker_encoder import SpeakerEncoder, cosine_similarity
    enc = SpeakerEncoder(backend="resemblyzer")

    emb_a = enc.get_embedding(speaker_a(0), SR)
    emb_b = enc.get_embedding(speaker_b(0), SR)

    assert emb_a.shape == (256,), f"Expected (256,), got {emb_a.shape}"
    assert abs(np.linalg.norm(emb_a) - 1.0) < 1e-5, "Embedding not L2-normalised"

    sim_same  = cosine_similarity(emb_a, enc.get_embedding(speaker_a(1), SR))
    sim_cross = cosine_similarity(emb_a, emb_b)
    print(f"  [OK] embedding dim=256, L2-normalised")
    print(f"  [OK] same-speaker sim={sim_same:.4f}  cross-speaker sim={sim_cross:.4f}")


def test_enroll_and_verify():
    from backend.core.verification import VoiceVerifier

    # use a temp enrollment dir so tests don't pollute real data
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        verifier = VoiceVerifier(backend="resemblyzer")
        verifier.store.base_dir = tmpdir

        enroll_clips = [speaker_a(seed=i) for i in range(5)]
        centroid     = verifier.enroll("alice", enroll_clips, SR)
        assert centroid.shape == (256,)
        assert abs(np.linalg.norm(centroid) - 1.0) < 1e-5

        # Same speaker → should be high score
        result_genuine = verifier.verify("alice", speaker_a(seed=99), SR)
        # Different speaker → should be low score
        result_impostor = verifier.verify("alice", speaker_b(seed=99), SR)

        print(f"  [OK] enroll: centroid shape={centroid.shape}")
        print(f"  [OK] genuine  score={result_genuine.score:.4f}  accepted={result_genuine.accepted}")
        print(f"  [OK] impostor score={result_impostor.score:.4f}  accepted={result_impostor.accepted}")
        assert result_genuine.score > result_impostor.score, \
            "Genuine score should be higher than impostor score"


def test_identify():
    from backend.core.verification import VoiceVerifier
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        verifier = VoiceVerifier(backend="resemblyzer")
        verifier.store.base_dir = tmpdir

        verifier.enroll("alice", [speaker_a(seed=i) for i in range(5)], SR)
        verifier.enroll("bob",   [speaker_b(seed=i) for i in range(5)], SR)

        scores_a = verifier.identify(speaker_a(seed=42), SR)
        scores_b = verifier.identify(speaker_b(seed=42), SR)

        top_for_a = list(scores_a.keys())[0]
        top_for_b = list(scores_b.keys())[0]

        print(f"  [OK] identify alice→ top={top_for_a}  scores={dict(list(scores_a.items()))}")
        print(f"  [OK] identify bob  → top={top_for_b}  scores={dict(list(scores_b.items()))}")
        assert top_for_a == "alice", f"Expected alice, got {top_for_a}"
        assert top_for_b == "bob",   f"Expected bob,   got {top_for_b}"


def test_eer():
    from backend.core.verification import VoiceVerifier
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        verifier = VoiceVerifier(backend="resemblyzer")
        verifier.store.base_dir = tmpdir

        verifier.enroll("alice", [speaker_a(seed=i) for i in range(5)], SR)

        from backend.core.speaker_encoder import cosine_similarity
        centroid = verifier.store.get_centroid("alice", "resemblyzer")
        enc      = verifier.encoder

        genuine  = [cosine_similarity(enc.get_embedding(speaker_a(s), SR), centroid)
                    for s in range(10, 20)]
        impostor = [cosine_similarity(enc.get_embedding(speaker_b(s), SR), centroid)
                    for s in range(10, 20)]

        eer, thresh = verifier.compute_eer(genuine, impostor)
        print(f"  [OK] EER={eer:.4f}  best_threshold={thresh:.4f}")
        print(f"       genuine  mean={np.mean(genuine):.4f}  min={np.min(genuine):.4f}")
        print(f"       impostor mean={np.mean(impostor):.4f}  max={np.max(impostor):.4f}")
        assert eer < 0.5, f"EER too high: {eer}"


def test_enrollment_store():
    from backend.models.enrollment_store import EnrollmentStore
    from backend.core.speaker_encoder import SpeakerEncoder
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        store = EnrollmentStore(base_dir=tmpdir)
        enc   = SpeakerEncoder("resemblyzer")

        embs = enc.embed_batch([speaker_a(i) for i in range(3)], SR)
        store.enroll("carol", embs, "resemblyzer")

        assert store.is_enrolled("carol", "resemblyzer")
        assert "carol" in store.list_users("resemblyzer")

        centroid = store.get_centroid("carol", "resemblyzer")
        assert centroid.shape == (256,)

        store.delete_user("carol", "resemblyzer")
        assert not store.is_enrolled("carol", "resemblyzer")
        print(f"  [OK] enroll / is_enrolled / list / get_centroid / delete all work")


# ------------------------------------------------------------------ #
#  Runner                                                             #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("\n=== Phase 4 Tests (Text-Independent Authentication) ===\n")

    print("1. Speaker Encoder")
    test_encoder()

    print("\n2. Enroll + Verify")
    test_enroll_and_verify()

    print("\n3. 1-to-N Identification")
    test_identify()

    print("\n4. EER Computation")
    test_eer()

    print("\n5. Enrollment Store CRUD")
    test_enrollment_store()

    print("\nAll Phase 4 tests PASSED.\n")
