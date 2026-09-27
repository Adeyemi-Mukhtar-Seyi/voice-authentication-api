#!/usr/bin/env python3
"""Run the Voice Authentication API locally."""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from the project root .env file.
load_dotenv(BASE_DIR / ".env")


if __name__ == "__main__":
    os.makedirs(BASE_DIR / "static", exist_ok=True)
    os.makedirs(BASE_DIR / "voice_model", exist_ok=True)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )