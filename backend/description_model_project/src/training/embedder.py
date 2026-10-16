"""
embedder.py
-----------
Generates and persists sentence embeddings for all songs in the dataset.

Two embedding spaces are created:
    • embeddings_1  — built from description_1 (genre + artist + lyrics preview)
    • embeddings_2  — built from description_2 (natural-language full description)

The dual-space approach lets the system compare query-to-lyric similarity AND
query-to-description similarity, then combine them for better retrieval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE: int = 64   # Increased from 32 — safe on Kaggle P100/T4
EMBEDDINGS_1_FILENAME: str = "song_embeddings_1.npy"
EMBEDDINGS_2_FILENAME: str = "song_embeddings_2.npy"


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Load a SentenceTransformer model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to 'all-MiniLM-L6-v2'
        which offers a good speed/quality trade-off for semantic search.

    Returns
    -------
    SentenceTransformer
        Loaded model instance.
    """
    print(f"[embedder] Loading sentence transformer: {model_name}")
    return SentenceTransformer(model_name)


def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of text strings into L2-normalised embeddings.

    Normalisation is critical here because we use cosine similarity downstream;
    with normalised embeddings, cosine similarity reduces to a simple dot product
    which is faster and numerically stable.

    Parameters
    ----------
    model : SentenceTransformer
        Loaded embedding model.
    texts : list[str]
        Strings to encode.
    batch_size : int
        Number of texts per GPU/CPU batch.
    show_progress : bool
        Display a tqdm progress bar.

    Returns
    -------
    np.ndarray
        Float32 array of shape (len(texts), embedding_dim).
    """
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )


def build_and_save_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    output_dir: Path | str = ".",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate both embedding arrays and save them to disk as .npy files.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.  Must contain 'description_1' and 'description_2'.
    model : SentenceTransformer
        Loaded embedding model.
    output_dir : Path | str
        Directory where the .npy files will be written.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (embeddings_1, embeddings_2)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[embedder] Generating embeddings from description_1 (lyrics + genre + artist)...")
    embeddings_1 = generate_embeddings(model, df["description_1"].tolist())

    print("[embedder] Generating embeddings from description_2 (full natural description)...")
    embeddings_2 = generate_embeddings(model, df["description_2"].tolist())

    path_1 = output_dir / EMBEDDINGS_1_FILENAME
    path_2 = output_dir / EMBEDDINGS_2_FILENAME
    np.save(path_1, embeddings_1)
    np.save(path_2, embeddings_2)
    print(f"[embedder] Saved embeddings_1 → {path_1}")
    print(f"[embedder] Saved embeddings_2 → {path_2}")

    return embeddings_1, embeddings_2


def load_embeddings(embeddings_dir: Path | str = ".") -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed embeddings from disk.

    Parameters
    ----------
    embeddings_dir : Path | str
        Directory that contains song_embeddings_1.npy and song_embeddings_2.npy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (embeddings_1, embeddings_2)
    """
    embeddings_dir = Path(embeddings_dir)
    embeddings_1 = np.load(embeddings_dir / EMBEDDINGS_1_FILENAME)
    embeddings_2 = np.load(embeddings_dir / EMBEDDINGS_2_FILENAME)
    print(f"[embedder] Loaded embeddings — shape: {embeddings_1.shape}")
    return embeddings_1, embeddings_2
