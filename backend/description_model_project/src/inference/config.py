"""
config.py
---------
Central configuration for the FastAPI backend, loaded from environment
variables (and an optional .env file — see .env.example).

Nothing here changes the recommendation algorithm; this module only decides
*where* the CSV/model/embeddings live and how the server is exposed, so the
same code can run locally instead of only on Kaggle.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # no-op if no .env file is present

# description_model_project/  (three levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where the raw dataset CSV lives. Must contain the columns the existing
# pipeline expects: song, artist, translated_genres, text, full_description.
DATA_CSV_PATH = Path(
    os.getenv("DATA_CSV_PATH", str(PROJECT_ROOT / "data" / "songs_with_descriptions_V3.csv"))
)

# Directory containing the pre-computed song_embeddings_1.npy / song_embeddings_2.npy
# (see src/training/embedder.py). This backend loads them — it does not generate them.
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", str(PROJECT_ROOT / "models")))

# Must match the SentenceTransformer model the embeddings were built with
# (embedder.DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2").
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

# Timeout (seconds) for outgoing requests to the external music API.
MUSIC_API_TIMEOUT = float(os.getenv("MUSIC_API_TIMEOUT", "5"))

# Comma-separated list of origins allowed to call this API (CORS). Defaults
# to common local-dev frontend ports (Live Server / http.server).
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500"
    ).split(",")
    if origin.strip()
]
