"""Run the API for the React research interface."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    uvicorn.run("api.main:app", host="127.0.0.1", port=8503)
