"""
text_cleaner.py
---------------
Handles lyrics cleaning and song description construction.

This module is responsible for:
- Removing noise tokens from raw lyrics (filler words, onomatopoeia, slang)
- Normalising text (lowercase, punctuation removal)
- Building two embedding-ready description formats:
    • description_1  — "{genre} song by {artist}. Lyrics: {truncated_lyrics}"
    • description_2  — the pre-generated natural-language 'full_description' column
"""

import re
import pandas as pd

# ---------------------------------------------------------------------------
# Noise vocabulary
# ---------------------------------------------------------------------------
# These tokens carry no semantic meaning for music recommendation and increase
# noise in the embedding space.  Add to this list if new noise tokens are
# discovered during EDA.
NOISE_TOKENS: set[str] = {
    "yeah", "ya", "na", "wan", "uh", "gon", "ima", "mm", "uhhuh", "bout",
    "em", "nigga", "niggas", "got", "ta", "lil", "ol", "hey", "oooh", "ooh",
    "oh", "yaka", "lalalala", "la", "da", "di", "yuh", "shawty", "oohooh",
    "shoorah", "mmmmmm", "ook", "bidibambambambam", "shh", "bro", "ho",
    "bambam", "shitll", "tonka",
}

# Maximum number of characters taken from cleaned lyrics when building
# description_1.  Keeping this short focuses the embedding on the song's
# opening mood/theme rather than its full narrative.
LYRICS_PREVIEW_LENGTH: int = 500


def clean_lyrics(text: str) -> str:
    """
    Clean a raw lyrics string.

    Steps
    -----
    1. Replace NaN / None with empty string.
    2. Strip stage directions written in square brackets (e.g. [Chorus]).
    3. Lowercase and collapse all non-word characters to spaces.
    4. Remove tokens that appear in NOISE_TOKENS.

    Parameters
    ----------
    text : str
        Raw lyrics string (may be NaN).

    Returns
    -------
    str
        Cleaned lyrics, or "" if the input was empty / NaN.
    """
    if pd.isna(text):
        return ""

    # Remove stage directions like [Chorus], [Verse 1], etc.
    text = re.sub(r"\[.*?\]", "", text)

    # Lowercase and replace all punctuation / special chars with a space
    text = re.sub(r"\W+", " ", text.lower())

    # Filter out noise tokens
    words = text.split()
    filtered = [w for w in words if w not in NOISE_TOKENS]

    return " ".join(filtered)


def build_description_1(row: pd.Series) -> str:
    """
    Build a short, structured description that blends genre, artist, and lyric content.

    Format: "A {genre} song by {artist}. Lyrics: {first N chars of cleaned lyrics}"

    This is the 'description_1' embedding source — it explicitly encodes genre
    and artist identity alongside lyric content so that genre-based queries map
    more reliably to relevant embeddings.

    Parameters
    ----------
    row : pd.Series
        A single row from the songs DataFrame.  Must contain the columns
        'translated_genres', 'artist', and 'text'.

    Returns
    -------
    str
        Formatted description string.
    """
    genre = row.get("translated_genres", "Unknown Genre")
    artist = row.get("artist", "Unknown Artist")
    lyrics_preview = row.get("text", "")[:LYRICS_PREVIEW_LENGTH]
    return f"A {genre} song by {artist}. Lyrics: {lyrics_preview}"


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to the raw songs DataFrame.

    - Cleans the 'text' (lyrics) column in-place.
    - Adds a 'description_1' column (genre + artist + lyrics preview).
    - Adds a 'description_2' column (alias for 'full_description').

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with new columns.
    """
    df = df.copy()
    df["text"] = df["text"].apply(clean_lyrics)
    df["description_1"] = df.apply(build_description_1, axis=1)
    df["description_2"] = df["full_description"]
    return df
