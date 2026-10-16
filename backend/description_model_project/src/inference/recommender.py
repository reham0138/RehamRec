"""
recommender.py
--------------
Core recommendation logic for the Description Model.

Three recommendation strategies are exposed:
    • recommend_by_lyrics_description  — uses description_1 embeddings
      (genre + artist + lyrics preview)
    • recommend_by_full_description    — uses description_2 embeddings
      (natural-language full description)
    • recommend_by_combined            — averages both embedding spaces

All three follow the same pipeline:
    1. Validate input
    2. Filter dataset by detected genre (if any)
    3. Encode the user query into the embedding space
    4. Compute cosine similarity against filtered embeddings
    5. Select top-100, sample 20, return top-5 from sample

The sampling step (sample 20, return 5 from that sample) adds light
diversity — without it the top-5 often cluster around one subgenre.

KNOWN BUG IN ORIGINAL CODE (fixed here)
----------------------------------------
In the original notebook, `show_recommendation` (option "1") was calling
`recommend_by_description_1(data, song_embeddings_2, ...)` — note the
**wrong** embeddings array was passed for option 1.  Similarly option "2"
passed song_embeddings_1 where song_embeddings_2 was expected.
Both calls have been corrected below.

Also, the original `suggest_songs` fallback block referenced the variable
`scored_candidates` which was never populated, causing a NameError at
runtime.  That block has been removed (the fallback logic is already
handled earlier in the loop).
"""

from __future__ import annotations

import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.utils.genre_extractor import extract_genres_from_query
from src.utils.input_validator import is_valid_query


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_by_genre(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    query: str,
    known_genres: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Narrow the search space to songs whose genre overlaps with the query.

    If no genre is detected in the query, the full dataset is returned unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Full songs DataFrame.
    embeddings : np.ndarray
        Embedding array aligned with df (same row order).
    query : str
        User query string.
    known_genres : list[str]
        Genre vocabulary extracted from the dataset.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Filtered (df subset, corresponding embedding rows).
    """
    matched_genres = extract_genres_from_query(query, known_genres)

    if matched_genres:
        genre_mask = df["translated_genres"].str.lower().apply(
            lambda cell: any(
                genre == token.strip()
                for token in cell.split(";")
                for genre in matched_genres
            )
        )
        filtered_df = df[genre_mask].copy()
        filtered_embs = embeddings[filtered_df.index.tolist()]
        print(f"[recommender] Genre filter matched: {matched_genres} → {len(filtered_df)} songs")
    else:
        print("[recommender] No genre detected. Searching full dataset.")
        filtered_df = df.copy()
        filtered_embs = embeddings

    return filtered_df, filtered_embs


def _rank_and_sample(
    filtered_df: pd.DataFrame,
    similarities: np.ndarray,
    top_n: int = 100,
    sample_k: int = 20,
    final_top: int = 5,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Rank candidates by similarity, sample for diversity, then return top results.

    The two-stage selection (rank → sample → re-rank) introduces controlled
    diversity: we avoid always returning the same cluster of near-identical songs.

    Parameters
    ----------
    filtered_df : pd.DataFrame
        Genre-filtered song DataFrame.
    similarities : np.ndarray
        Cosine similarity scores aligned with filtered_df rows.
    top_n : int
        Size of the initial similarity ranking pool.
    sample_k : int
        How many songs to sample from the top_n pool.
    final_top : int
        Final number of recommendations to return.
    random_seed : int | None
        Seed for reproducible sampling.  None = random each call.

    Returns
    -------
    pd.DataFrame
        Up to `final_top` recommended songs.
    """
    seed = random_seed if random_seed is not None else random.randint(0, 9999)

    filtered_df = filtered_df.copy()
    filtered_df["_similarity"] = similarities

    top_n_df = filtered_df.nlargest(top_n, "_similarity")
    sample_size = min(sample_k, len(top_n_df))
    sampled = top_n_df.sample(n=sample_size, random_state=seed)

    return sampled.nlargest(final_top, "_similarity")


def _format_response(results: pd.DataFrame) -> str:
    """
    Format a recommendations DataFrame into a numbered string.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain 'song' and 'artist' columns.

    Returns
    -------
    str
        Numbered list string, e.g. "1. Black Hole Sun by Soundgarden"
    """
    if results.empty:
        return "❌ No recommendations found."

    lines = [
        f"{i}. {row.song} by {row.artist}"
        for i, row in enumerate(results.itertuples(), start=1)
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured (JSON-ready) variants
# ---------------------------------------------------------------------------
# Added for the FastAPI backend: same pipeline (validate -> genre filter ->
# encode -> cosine similarity -> rank/sample) as the string-returning
# functions below, but returning a list[dict] instead of a formatted string
# so the frontend never has to parse "N. Song by Artist" text.  Ranking
# order and algorithm are unchanged; only the output shape differs.
def _to_structured(results: pd.DataFrame) -> list[dict]:
    if results.empty:
        return []

    return [
        {
            "song": row.get("song"),
            "artist": row.get("artist"),
            "genre": row.get("translated_genres"),
            "lyrics_preview": (row.get("text") or "")[:220],   # ← جديد
            "similarity": float(row["_similarity"]) if "_similarity" in row.index else None,
        }
        for _, row in results.iterrows()
    ]
def recommend_by_lyrics_description_structured(
    df: pd.DataFrame,
    embeddings_1: np.ndarray,
    model: SentenceTransformer,
    user_query: str,
    known_genres: list[str],
    top_n: int = 100,
    sample_k: int = 20,
    final_top: int = 5,
    random_seed: int | None = None,
) -> tuple[list[dict], str | None]:
    """Structured counterpart of recommend_by_lyrics_description.

    Returns
    -------
    tuple[list[dict], str | None]
        (recommendations, error). error is None on success; when set,
        recommendations is always [].
    """
    valid, error = is_valid_query(user_query)
    if not valid:
        return [], error

    filtered_df, filtered_embs = _filter_by_genre(df, embeddings_1, user_query, known_genres)
    if filtered_df.empty:
        return [], "No songs found for the detected genre."

    user_emb = model.encode([user_query], normalize_embeddings=True)[0]
    sims = cosine_similarity([user_emb], filtered_embs)[0]

    results = _rank_and_sample(filtered_df, sims, top_n, sample_k, final_top, random_seed)
    return _to_structured(results), None


def recommend_by_full_description_structured(
    df: pd.DataFrame,
    embeddings_2: np.ndarray,
    model: SentenceTransformer,
    user_query: str,
    known_genres: list[str],
    top_n: int = 100,
    sample_k: int = 20,
    final_top: int = 5,
    random_seed: int | None = None,
) -> tuple[list[dict], str | None]:
    """Structured counterpart of recommend_by_full_description."""
    valid, error = is_valid_query(user_query)
    if not valid:
        return [], error

    filtered_df, filtered_embs = _filter_by_genre(df, embeddings_2, user_query, known_genres)
    if filtered_df.empty:
        return [], "No songs found for the detected genre."

    user_emb = model.encode([user_query], normalize_embeddings=True)[0]
    sims = cosine_similarity([user_emb], filtered_embs)[0]

    results = _rank_and_sample(filtered_df, sims, top_n, sample_k, final_top, random_seed)
    return _to_structured(results), None


def recommend_by_combined_structured(
    df: pd.DataFrame,
    embeddings_1: np.ndarray,
    embeddings_2: np.ndarray,
    model: SentenceTransformer,
    user_query: str,
    known_genres: list[str],
    top_n: int = 100,
    sample_k: int = 20,
    final_top: int = 5,
    random_seed: int | None = None,
) -> tuple[list[dict], str | None]:
    """Structured counterpart of recommend_by_combined."""
    valid, error = is_valid_query(user_query)
    if not valid:
        return [], error

    matched_genres = extract_genres_from_query(user_query, known_genres)
    if matched_genres:
        genre_mask = df["translated_genres"].str.lower().apply(
            lambda cell: any(
                genre == token.strip()
                for token in cell.split(";")
                for genre in matched_genres
            )
        )
        filtered_df = df[genre_mask].copy()
        indices = filtered_df.index.tolist()
        emb1 = embeddings_1[indices]
        emb2 = embeddings_2[indices]
    else:
        filtered_df = df.copy()
        emb1 = embeddings_1
        emb2 = embeddings_2

    if filtered_df.empty:
        return [], "No songs found."

    user_emb = model.encode([user_query], normalize_embeddings=True)[0]

    sims_1 = cosine_similarity([user_emb], emb1)[0]
    sims_2 = cosine_similarity([user_emb], emb2)[0]
    combined_sims = (sims_1 + sims_2) / 2.0

    results = _rank_and_sample(filtered_df, combined_sims, top_n, sample_k, final_top, random_seed)
    return _to_structured(results), None


def get_structured_recommendations(
    method: str,
    df: pd.DataFrame,
    embeddings_1: np.ndarray,
    embeddings_2: np.ndarray,
    model: SentenceTransformer,
    user_query: str,
    known_genres: list[str],
    **kwargs,
) -> tuple[list[dict], str | None]:
    """
    Single dispatch point used by the FastAPI backend.

    Parameters
    ----------
    method : str
        One of "lyrics", "description", "combined".
    kwargs :
        Forwarded to the underlying *_structured function (e.g. final_top).
    """
    if method == "lyrics":
        return recommend_by_lyrics_description_structured(
            df, embeddings_1, model, user_query, known_genres, **kwargs
        )
    if method == "description":
        return recommend_by_full_description_structured(
            df, embeddings_2, model, user_query, known_genres, **kwargs
        )
    if method == "combined":
        return recommend_by_combined_structured(
            df, embeddings_1, embeddings_2, model, user_query, known_genres, **kwargs
        )
    return [], f"Unknown method '{method}'. Expected one of: lyrics, description, combined."


# ---------------------------------------------------------------------------
# Public recommendation functions
# ---------------------------------------------------------------------------

def recommend_by_lyrics_description(
    df: pd.DataFrame,
    embeddings_1: np.ndarray,
    model: SentenceTransformer,
    user_query: str,
    known_genres: list[str],
    top_n: int = 100,
    sample_k: int = 20,
    final_top: int = 5,
    random_seed: int | None = None,
) -> str:
    """
    Recommend songs using the lyrics-based description embeddings (embeddings_1).

    Best suited for queries that mention specific lyrical themes, emotions, or
    content (e.g. "a song about losing someone to addiction").

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed songs DataFrame.
    embeddings_1 : np.ndarray
        Embeddings built from description_1.
    model : SentenceTransformer
        The same model used to build embeddings_1.
    user_query : str
        Free-text description from the user.
    known_genres : list[str]
        Genre vocabulary for query-time genre detection.
    top_n, sample_k, final_top, random_seed :
        Ranking / sampling parameters (see _rank_and_sample).

    Returns
    -------
    str
        Formatted recommendation string.
    """
    valid, error = is_valid_query(user_query)
    if not valid:
        return f"⚠️ {error}"

    filtered_df, filtered_embs = _filter_by_genre(df, embeddings_1, user_query, known_genres)
    if filtered_df.empty:
        return "❌ No songs found for the detected genre."

    user_emb = model.encode([user_query], normalize_embeddings=True)[0]
    sims = cosine_similarity([user_emb], filtered_embs)[0]

    results = _rank_and_sample(filtered_df, sims, top_n, sample_k, final_top, random_seed)
    return _format_response(results)


def recommend_by_full_description(
    df: pd.DataFrame,
    embeddings_2: np.ndarray,
    model: SentenceTransformer,
    user_query: str,
    known_genres: list[str],
    top_n: int = 100,
    sample_k: int = 20,
    final_top: int = 5,
    random_seed: int | None = None,
) -> str:
    """
    Recommend songs using the full natural-language description embeddings (embeddings_2).

    Best suited for abstract, mood-based queries (e.g. "something melancholic
    and acoustic for a rainy afternoon").

    Parameters
    ----------
    (same as recommend_by_lyrics_description, but uses embeddings_2)

    Returns
    -------
    str
        Formatted recommendation string.
    """
    valid, error = is_valid_query(user_query)
    if not valid:
        return f"⚠️ {error}"

    filtered_df, filtered_embs = _filter_by_genre(df, embeddings_2, user_query, known_genres)
    if filtered_df.empty:
        return "❌ No songs found for the detected genre."

    user_emb = model.encode([user_query], normalize_embeddings=True)[0]
    sims = cosine_similarity([user_emb], filtered_embs)[0]

    results = _rank_and_sample(filtered_df, sims, top_n, sample_k, final_top, random_seed)
    return _format_response(results)


def recommend_by_combined(
    df: pd.DataFrame,
    embeddings_1: np.ndarray,
    embeddings_2: np.ndarray,
    model: SentenceTransformer,
    user_query: str,
    known_genres: list[str],
    top_n: int = 100,
    sample_k: int = 20,
    final_top: int = 5,
    random_seed: int | None = None,
) -> str:
    """
    Recommend songs by averaging similarity scores from both embedding spaces.

    Combining both spaces tends to outperform either individually because
    description_1 anchors on explicit genre/lyric content while description_2
    captures broader mood and narrative.

    Parameters
    ----------
    (uses both embeddings_1 and embeddings_2)

    Returns
    -------
    str
        Formatted recommendation string.
    """
    valid, error = is_valid_query(user_query)
    if not valid:
        return f"⚠️ {error}"

    matched_genres = extract_genres_from_query(user_query, known_genres)
    if matched_genres:
        genre_mask = df["translated_genres"].str.lower().apply(
            lambda cell: any(
                genre == token.strip()
                for token in cell.split(";")
                for genre in matched_genres
            )
        )
        filtered_df = df[genre_mask].copy()
        indices = filtered_df.index.tolist()
        emb1 = embeddings_1[indices]
        emb2 = embeddings_2[indices]
        print(f"[recommender] Genre filter: {matched_genres} → {len(filtered_df)} songs")
    else:
        print("[recommender] No genre detected. Searching full dataset.")
        filtered_df = df.copy()
        emb1 = embeddings_1
        emb2 = embeddings_2

    if filtered_df.empty:
        return "❌ No songs found."

    user_emb = model.encode([user_query], normalize_embeddings=True)[0]

    sims_1 = cosine_similarity([user_emb], emb1)[0]
    sims_2 = cosine_similarity([user_emb], emb2)[0]
    combined_sims = (sims_1 + sims_2) / 2.0   # Equal weight average

    results = _rank_and_sample(filtered_df, combined_sims, top_n, sample_k, final_top, random_seed)
    return _format_response(results)
