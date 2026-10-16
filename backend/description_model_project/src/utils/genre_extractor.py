"""
genre_extractor.py
------------------
Extracts genre mentions from free-text user queries.

Used at inference time to narrow the search space before computing
cosine similarity — dramatically reduces latency on large datasets.
"""

from __future__ import annotations
import pandas as pd


def build_genre_list(df: pd.DataFrame, genre_column: str = "translated_genres") -> list[str]:
    """
    Build a deduplicated list of all genres present in the dataset.

    Genres are stored as semicolon-separated strings in the DataFrame.
    This function splits each entry and collects unique values.

    Parameters
    ----------
    df : pd.DataFrame
        Songs DataFrame containing a genre column.
    genre_column : str
        Name of the column that holds genre strings.

    Returns
    -------
    list[str]
        Sorted list of unique genre strings (lowercased).
    """
    genre_set: set[str] = set()
    for entry in df[genre_column].dropna():
        genre_set.update(g.strip().lower() for g in entry.split(";"))
    return sorted(genre_set)


def extract_genres_from_query(query: str, known_genres: list[str]) -> list[str]:
    """
    Detect genre mentions in a natural-language user query.

    Uses simple substring matching — efficient and transparent.
    For example, a query "a grunge song about pain" will match "grunge"
    if "grunge" is in known_genres.

    Parameters
    ----------
    query : str
        Free-text description entered by the user.
    known_genres : list[str]
        Genre vocabulary built from the dataset (lowercased).

    Returns
    -------
    list[str]
        Genres found in the query.  Empty list if none detected.
    """
    query_lower = query.lower()
    return [genre for genre in known_genres if genre in query_lower]
