"""Pydantic models for all API request/response bodies."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


# ── Enrollment ──────────────────────────────────────────────────
class EnrollResponse(BaseModel):
    username:         str
    mode:             str        # "text_independent" | "text_dependent"
    samples_enrolled: int
    message:          str


# ── Authentication ───────────────────────────────────────────────
class AuthResponse(BaseModel):
    username:       str
    decision:       str          # ACCEPTED | REJECTED_SPOOF | REJECTED_VOICE | REJECTED_TEXT
    accepted:       bool
    spoof_score:    float
    voice_score:    Optional[float] = None
    text_score:     Optional[float] = None
    combined_score: Optional[float] = None
    transcript:     Optional[str]   = None
    latency_ms:     float


# ── Spoof detection ──────────────────────────────────────────────
class SpoofResponse(BaseModel):
    is_real:     bool
    score:       float
    attack_type: str
    latency_ms:  float


# ── User management ──────────────────────────────────────────────
class UserStatus(BaseModel):
    username:                   str
    enrolled_text_independent:  bool
    enrolled_text_dependent:    bool


class UserListResponse(BaseModel):
    users: list[str]
    count: int


class DeleteResponse(BaseModel):
    username: str
    deleted:  bool
    message:  str


# ── System ───────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status:  str
    version: str
    message: str
