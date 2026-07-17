"""
Integration tests for all FastAPI endpoints.
Uses FastAPI TestClient — no running server needed.

Endpoint coverage:
  GET  /health
  GET  /users
  POST /enroll/text-independent
  POST /auth/text-independent
  GET  /users/{username}
  DELETE /users/{username}
  POST /spoof/detect
"""
import os, sys, glob
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.conftest import TEST_USER, TI_WAVS, SPOOF_FAKE, DATA_DIR


# ── Health ───────────────────────────────────────────────────────
class TestHealth:
    def test_health_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_root_200(self, client):
        assert client.get("/").status_code == 200


# ── Users (before enroll) ────────────────────────────────────────
class TestUsersEmpty:
    def test_list_users_200(self, client):
        r = client.get("/users")
        assert r.status_code == 200
        assert "users" in r.json()
        assert isinstance(r.json()["users"], list)

    def test_unknown_user_404(self, client):
        r = client.get(f"/users/{TEST_USER}_nonexistent")
        assert r.status_code == 404


# ── Enrollment ───────────────────────────────────────────────────
class TestEnroll:
    def test_enroll_text_independent_success(self, client):
        files = [
            ("files", (os.path.basename(p), open(p, "rb"), "audio/wav"))
            for p in TI_WAVS[:5]
        ]
        r = client.post(
            "/enroll/text-independent",
            data={"username": TEST_USER},
            files=files,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["username"]         == TEST_USER
        assert body["mode"]             == "text_independent"
        assert body["samples_enrolled"] == 5

    def test_enroll_too_few_files_400(self, client):
        files = [("files", (os.path.basename(TI_WAVS[0]),
                             open(TI_WAVS[0], "rb"), "audio/wav"))]
        r = client.post(
            "/enroll/text-independent",
            data={"username": TEST_USER},
            files=files,
        )
        assert r.status_code == 400

    def test_enroll_missing_username_422(self, client):
        files = [
            ("files", (os.path.basename(p), open(p, "rb"), "audio/wav"))
            for p in TI_WAVS[:3]
        ]
        r = client.post("/enroll/text-independent", files=files)
        assert r.status_code == 422


# ── Authentication (requires enrolled user) ──────────────────────
class TestAuth:
    @pytest.fixture(autouse=True)
    def ensure_enrolled(self, client):
        """Re-enroll before each auth test in case previous test deleted."""
        files = [
            ("files", (os.path.basename(p), open(p, "rb"), "audio/wav"))
            for p in TI_WAVS[:5]
        ]
        client.post(
            "/enroll/text-independent",
            data={"username": TEST_USER},
            files=files,
        )

    def test_auth_same_speaker_accepted(self, client):
        with open(TI_WAVS[6], "rb") as f:
            r = client.post(
                "/auth/text-independent",
                data={"username": TEST_USER},
                files={"file": (os.path.basename(TI_WAVS[6]), f, "audio/wav")},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["accepted"]   == True
        assert body["decision"]   == "ACCEPTED"
        assert 0.0 <= body["spoof_score"] <= 1.0
        assert 0.0 <= body["voice_score"] <= 1.0

    def test_auth_unknown_user_404(self, client):
        with open(TI_WAVS[0], "rb") as f:
            r = client.post(
                "/auth/text-independent",
                data={"username": "nobody_enrolled_xyz"},
                files={"file": ("audio.wav", f, "audio/wav")},
            )
        assert r.status_code == 404

    def test_auth_response_has_latency(self, client):
        with open(TI_WAVS[7], "rb") as f:
            r = client.post(
                "/auth/text-independent",
                data={"username": TEST_USER},
                files={"file": ("audio.wav", f, "audio/wav")},
            )
        assert r.status_code == 200
        assert r.json()["latency_ms"] >= 0


# ── Spoof detection ──────────────────────────────────────────────
class TestSpoof:
    def test_real_audio_is_real(self, client):
        with open(TI_WAVS[0], "rb") as f:
            r = client.post(
                "/spoof/detect",
                files={"file": ("audio.wav", f, "audio/wav")},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["is_real"]      == True
        assert 0.0 <= body["score"] <= 1.0
        assert "attack_type"        in body
        assert body["latency_ms"]   >= 0

    def test_fake_audio_score_lower(self, client):
        with open(TI_WAVS[0], "rb") as r_file, \
             open(SPOOF_FAKE[0], "rb") as f_file:
            real_r = client.post("/spoof/detect",
                files={"file": ("real.wav", r_file, "audio/wav")})
            fake_r = client.post("/spoof/detect",
                files={"file": ("fake.wav", f_file, "audio/wav")})
        assert real_r.status_code == 200
        assert fake_r.status_code == 200
        real_score = real_r.json()["score"]
        fake_score = fake_r.json()["score"]
        assert real_score > fake_score, (
            f"Real score ({real_score:.4f}) should be > fake score ({fake_score:.4f})")


# ── User management ──────────────────────────────────────────────
class TestUserManagement:
    @pytest.fixture(autouse=True)
    def ensure_enrolled(self, client):
        files = [
            ("files", (os.path.basename(p), open(p, "rb"), "audio/wav"))
            for p in TI_WAVS[:5]
        ]
        client.post("/enroll/text-independent",
                    data={"username": TEST_USER}, files=files)

    def test_user_status_enrolled(self, client):
        r = client.get(f"/users/{TEST_USER}")
        assert r.status_code == 200
        body = r.json()
        assert body["username"]                  == TEST_USER
        assert body["enrolled_text_independent"] == True

    def test_user_appears_in_list(self, client):
        r = client.get("/users")
        assert r.status_code == 200
        assert TEST_USER in r.json()["users"]

    def test_delete_user(self, client):
        r = client.delete(f"/users/{TEST_USER}")
        assert r.status_code == 200
        assert r.json()["deleted"] == True

    def test_deleted_user_returns_404(self, client):
        client.delete(f"/users/{TEST_USER}")
        r = client.get(f"/users/{TEST_USER}")
        assert r.status_code == 404
