"""
Authentication endpoints.

POST /auth/text-independent   — verify speaker identity
POST /auth/text-dependent     — verify passphrase + speaker identity
"""
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.api.schemas      import AuthResponse
from backend.api.dependencies import get_ti_pipeline, get_td_pipeline, load_audio_bytes

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/text-independent", response_model=AuthResponse)
async def auth_text_independent(
    username: str        = Form(..., description="Claimed username"),
    file:     UploadFile = File(..., description="WAV audio to verify"),
):
    raw = await file.read()
    try:
        audio = load_audio_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot read audio: {e}")

    pipeline = get_ti_pipeline()

    if not pipeline.is_enrolled(username):
        raise HTTPException(
            status_code=404,
            detail=f"User '{username}' is not enrolled. POST /enroll/text-independent first."
        )

    try:
        result = pipeline.authenticate(username, audio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication error: {e}")

    return AuthResponse(
        username    = username,
        decision    = result.final_decision,
        accepted    = result.accepted,
        spoof_score = result.spoof_result.score,
        voice_score = result.voice_result.score if result.voice_result else None,
        latency_ms  = result.total_ms,
    )


@router.post("/text-dependent", response_model=AuthResponse)
async def auth_text_dependent(
    username:   str        = Form(..., description="Claimed username"),
    passphrase: str        = Form("my voice is my password"),
    file:       UploadFile = File(..., description="WAV audio — user must say the passphrase"),
):
    raw = await file.read()
    try:
        audio = load_audio_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot read audio: {e}")

    pipeline = get_td_pipeline()

    if not pipeline.is_enrolled(username):
        raise HTTPException(
            status_code=404,
            detail=f"User '{username}' is not enrolled. POST /enroll/text-dependent first."
        )

    # Update passphrase in case it was customised per-user
    pipeline.passphrase = passphrase
    pipeline.text_verifier.passphrase = passphrase.lower().strip()

    try:
        result = pipeline.authenticate(username, audio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication error: {e}")

    return AuthResponse(
        username       = username,
        decision       = result.final_decision,
        accepted       = result.accepted,
        spoof_score    = result.spoof_result.score,
        voice_score    = result.voice_result.score if result.voice_result else None,
        text_score     = result.text_result.text_score if result.text_result else None,
        combined_score = result.combined_score,
        transcript     = result.text_result.transcript if result.text_result else None,
        latency_ms     = result.total_ms,
    )
