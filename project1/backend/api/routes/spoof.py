"""
Spoof / deepfake detection endpoint.

POST /spoof/detect — check if a WAV file is real voice or synthesised
"""
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.api.schemas      import SpoofResponse
from backend.api.dependencies import get_spoof_detector, load_audio_bytes

router = APIRouter(prefix="/spoof", tags=["Spoof Detection"])


@router.post("/detect", response_model=SpoofResponse)
async def detect_spoof(
    file: UploadFile = File(..., description="WAV audio to analyse"),
):
    raw = await file.read()
    try:
        audio = load_audio_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot read audio: {e}")

    try:
        detector = get_spoof_detector()
        result   = detector.detect(audio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spoof detection error: {e}")

    return SpoofResponse(
        is_real     = result.is_real,
        score       = result.score,
        attack_type = result.attack_type,
        latency_ms  = result.latency_ms,
    )
