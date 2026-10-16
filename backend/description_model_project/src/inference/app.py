"""
app.py
------
FastAPI entry point for the Description Model music recommender.

This wires together, without changing either:
    - the existing semantic recommendation pipeline (recommender.py) — the
      trained SentenceTransformer + cosine-similarity retrieval logic,
      unchanged
    - the iTunes Search API metadata lookup (music_api.py)

into a single endpoint a separate HTML/CSS/JS frontend can call:

    POST /recommend  {"description": "..."}  ->  {"recommendations": [...]}

Expensive resources (SentenceTransformer model, dataset, embeddings) are
loaded exactly once at startup via the lifespan handler below, and reused
for every request — never reloaded per-call.

Note on scope: the original src/inference/app.py was a Gradio chatbot with
zero-shot intent classification, multi-turn session state and feedback
logging — a different UI built on top of the same recommender, not a
"description in, JSON out" API. It has been preserved unchanged as
gradio_app.py rather than merged into this file, since merging chatbot
session/intent logic into a stateless HTTP endpoint would have meant
redesigning it rather than adapting it.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import config, music_api
from src.inference.recommender import get_structured_recommendations
from src.preprocessing import preprocess_dataframe
from src.training import load_model, load_embeddings
from src.utils import build_genre_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("description_model.app")

# Populated once at startup by the lifespan handler below; read by /recommend.
RESOURCES: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading dataset from %s", config.DATA_CSV_PATH)
    if not config.DATA_CSV_PATH.exists():
        raise RuntimeError(
            f"Dataset CSV not found at {config.DATA_CSV_PATH}. "
            "Set DATA_CSV_PATH in .env or place the CSV there — see README.md."
        )
    raw_df = pd.read_csv(config.DATA_CSV_PATH)
    df = preprocess_dataframe(raw_df)

    logger.info("Loading pre-computed embeddings from %s", config.EMBEDDINGS_DIR)
    try:
        embeddings_1, embeddings_2 = load_embeddings(config.EMBEDDINGS_DIR)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"song_embeddings_1.npy / song_embeddings_2.npy not found in {config.EMBEDDINGS_DIR}. "
            "Generate them once with notebooks/main.ipynb (see README.md) — "
            "this backend loads existing embeddings, it does not (re)train or regenerate them."
        ) from exc

    if len(df) != embeddings_1.shape[0] or len(df) != embeddings_2.shape[0]:
        raise RuntimeError(
            f"Row-count mismatch: dataset has {len(df)} rows but embeddings have "
            f"{embeddings_1.shape[0]}/{embeddings_2.shape[0]} rows. The embeddings must have "
            "been generated from this exact CSV, in this row order."
        )

    logger.info("Loading SentenceTransformer model: %s", config.MODEL_NAME)
    model = load_model(config.MODEL_NAME)

    logger.info("Building genre vocabulary")
    genre_list = build_genre_list(df)

    RESOURCES.update(
        df=df,
        embeddings_1=embeddings_1,
        embeddings_2=embeddings_2,
        model=model,
        genre_list=genre_list,
    )
    logger.info("Startup complete — %d songs indexed.", len(df))

    yield

    RESOURCES.clear()


app = FastAPI(title="Description Model — Music Recommender API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    description: str = Field(..., description="Free-text description of the desired song.")
    method: Literal["combined", "lyrics", "description"] = Field(
        "combined",
        description=(
            "Which retrieval strategy to use: 'lyrics' (description_1 embeddings), "
            "'description' (description_2 embeddings), or 'combined' (average of both)."
        ),
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations to return.")


@app.get("/health")
def health():
    """Lightweight readiness check — useful while wiring up the frontend."""
    return {
        "status": "ok" if "df" in RESOURCES else "starting",
        "songs_indexed": len(RESOURCES["df"]) if "df" in RESOURCES else 0,
        "model": config.MODEL_NAME,
    }


@app.post("/recommend")
def recommend(payload: RecommendRequest):
    if "df" not in RESOURCES:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    records, error = get_structured_recommendations(
        payload.method,
        RESOURCES["df"],
        RESOURCES["embeddings_1"],
        RESOURCES["embeddings_2"],
        RESOURCES["model"],
        payload.description,
        RESOURCES["genre_list"],
        final_top=payload.top_k,
    )

    # The recommender only ever returns an error alongside an empty list for
    # input-validation problems (empty input, too short, non-English, an
    # unknown method, or no songs matching a detected genre) — all of which
    # are the caller's fault, not a server failure.
    if error:
        raise HTTPException(status_code=400, detail=error)

    recommendations = []
    for record in records:
        try:
            metadata = music_api.enrich_recommendation(record["song"], record["artist"])
        except Exception:
            # Belt-and-braces: music_api already catches its own failures,
            # but a single bad lookup must never take down the whole request.
            logger.exception(
                "Unexpected error enriching %r by %r", record["song"], record["artist"]
            )
            metadata = {
                "album": None,
                "cover": None,
                "itunes_genre": None,
                "duration_ms": None,
                "preview_url": None,
                "itunes_url": None,
                "release_date": None,
            }
        recommendations.append({**record, **metadata})

    return {"recommendations": recommendations}
