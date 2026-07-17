"""
FastAPI application — Voice Biometric Authentication System.
"""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db.database           import init_db
from backend.api.schemas           import HealthResponse
from backend.api.routes.enroll     import router as enroll_router
from backend.api.routes.auth       import router as auth_router
from backend.api.routes.spoof      import router as spoof_router
from backend.api.routes.users      import router as users_router

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Voice Biometric Authentication API",
    description = (
        "Text-independent and text-dependent speaker verification "
        "with anti-spoofing / deepfake detection."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS (allow frontend dev server on port 5173 / 3000) ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:5173", "http://localhost:3000",
                         "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Startup ───────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()


# ── Routes ────────────────────────────────────────────────────────
app.include_router(enroll_router)
app.include_router(auth_router)
app.include_router(spoof_router)
app.include_router(users_router)


# ── Health check ──────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status  = "ok",
        version = "1.0.0",
        message = "Voice Biometric Authentication API is running.",
    )


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Voice Biometric Authentication API",
        "docs":    "/docs",
        "health":  "/health",
    }
