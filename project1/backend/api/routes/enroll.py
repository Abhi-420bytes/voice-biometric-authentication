"""
Enrollment endpoints.

POST /enroll/text-independent   — upload ≥3 WAV clips, build speaker centroid
POST /enroll/text-dependent     — upload ≥3 passphrase WAV clips
"""
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List

from backend.api.schemas      import EnrollResponse
from backend.api.dependencies import get_ti_pipeline, get_td_pipeline, load_audio_bytes

router = APIRouter(prefix="/enroll", tags=["Enrollment"])

MIN_SAMPLES = 3


@router.post("/text-independent", response_model=EnrollResponse)
async def enroll_text_independent(
    username: str = Form(..., description="Username to enroll"),
    files:    List[UploadFile] = File(..., description="WAV audio files (≥3)"),
):
    if len(files) < MIN_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_SAMPLES} audio files, got {len(files)}."
        )

    audios = []
    for f in files:
        raw = await f.read()
        try:
            audios.append(load_audio_bytes(raw))
        except Exception as e:
            raise HTTPException(status_code=422,
                                detail=f"Could not read '{f.filename}': {e}")

    try:
        pipeline = get_ti_pipeline()
        pipeline.enroll(username, audios)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {e}")

    return EnrollResponse(
        username         = username,
        mode             = "text_independent",
        samples_enrolled = len(audios),
        message          = f"User '{username}' enrolled for text-independent auth.",
    )


@router.post("/text-dependent", response_model=EnrollResponse)
async def enroll_text_dependent(
    username:   str  = Form(..., description="Username to enroll"),
    passphrase: str  = Form("my voice is my password",
                            description="Passphrase the user will speak"),
    files: List[UploadFile] = File(..., description="Passphrase WAV recordings (≥3)"),
):
    if len(files) < MIN_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_SAMPLES} audio files, got {len(files)}."
        )

    audios = []
    for f in files:
        raw = await f.read()
        try:
            audios.append(load_audio_bytes(raw))
        except Exception as e:
            raise HTTPException(status_code=422,
                                detail=f"Could not read '{f.filename}': {e}")

    try:
        pipeline = get_td_pipeline()
        pipeline.passphrase = passphrase
        pipeline.text_verifier.passphrase = passphrase.lower().strip()
        pipeline.enroll(username, audios)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {e}")

    return EnrollResponse(
        username         = username,
        mode             = "text_dependent",
        samples_enrolled = len(audios),
        message          = f"User '{username}' enrolled for text-dependent auth "
                           f"with passphrase '{passphrase}'.",
    )
