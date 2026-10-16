"""Inference sub-package — recommendation functions."""

from .recommender import (
    recommend_by_lyrics_description,
    recommend_by_full_description,
    recommend_by_combined,
    recommend_by_lyrics_description_structured,
    recommend_by_full_description_structured,
    recommend_by_combined_structured,
    get_structured_recommendations,
)

__all__ = [
    "recommend_by_lyrics_description",
    "recommend_by_full_description",
    "recommend_by_combined",
    "recommend_by_lyrics_description_structured",
    "recommend_by_full_description_structured",
    "recommend_by_combined_structured",
    "get_structured_recommendations",
]
