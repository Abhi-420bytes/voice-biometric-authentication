"""
Entry point — run the Voice Biometric Authentication API.

    python main.py
    python main.py --port 8080
    uvicorn main:app --reload
"""
import argparse, uvicorn

from backend.api.app import app          # noqa: F401 (needed for uvicorn import string)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",   default="127.0.0.1")
    ap.add_argument("--port",   type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="Auto-reload on file changes")
    args = ap.parse_args()

    uvicorn.run(
        "main:app",
        host    = args.host,
        port    = args.port,
        reload  = args.reload,
    )


if __name__ == "__main__":
    main()
