"""
Shared pytest fixtures for all test modules.
"""
import os, sys, glob
import numpy as np
import pytest
from starlette.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.api.app import app

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TI_WAVS   = sorted(glob.glob(f"{DATA_DIR}/raw/abhiram/text_independent/sample_*.wav"))
SPOOF_REAL = sorted(glob.glob(f"{DATA_DIR}/spoof_data/real/*.wav"))
SPOOF_FAKE = sorted(glob.glob(f"{DATA_DIR}/spoof_data/fake_generated/*.wav"))

assert len(TI_WAVS) >= 5,   f"Need ≥5 text-independent wavs, found {len(TI_WAVS)}"
assert len(SPOOF_REAL) >= 1, f"Need ≥1 real spoof wav, found {len(SPOOF_REAL)}"

TEST_USER = "_test_integration_user_"


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient — shared for whole test session."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="session")
def enroll_files():
    """5 wav file tuples ready for multipart upload."""
    return [
        ("files", (os.path.basename(p), open(p, "rb"), "audio/wav"))
        for p in TI_WAVS[:5]
    ]


@pytest.fixture(scope="session")
def single_wav():
    """One wav file for verify / spoof tests."""
    return TI_WAVS[5] if len(TI_WAVS) > 5 else TI_WAVS[0]


@pytest.fixture(scope="session")
def fake_wav():
    """One fake (synthesised) wav for spoof test."""
    return SPOOF_FAKE[0]


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_user(client):
    """Remove test user before and after the session."""
    client.delete(f"/users/{TEST_USER}")   # pre-clean
    yield
    client.delete(f"/users/{TEST_USER}")   # post-clean
