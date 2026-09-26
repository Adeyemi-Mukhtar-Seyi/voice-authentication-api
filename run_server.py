#!/usr/bin/env python3
"""Run the Voice Authentication API locally."""

import os

import uvicorn


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs("voice_model", exist_ok=True)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
