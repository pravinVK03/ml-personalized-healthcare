"""Run the API for the React research interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the React research dashboard API.")
    parser.add_argument("--port", type=int, default=8503, help="Local API port (default: 8503).")
    args = parser.parse_args()
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    uvicorn.run("api.main:app", host="127.0.0.1", port=args.port)
