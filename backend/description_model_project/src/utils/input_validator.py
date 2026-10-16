"""
input_validator.py
------------------
Validates user-supplied query strings before they are passed to the
recommendation engine.

Checks performed:
    1. Non-empty / non-whitespace
    2. Minimum word count (≥ 3)
    3. Not entirely digits / symbols
    4. Written in English (via langdetect)
"""

from __future__ import annotations
import re
from langdetect import detect, LangDetectException

MIN_WORD_COUNT: int = 3


def is_valid_query(user_input: str) -> tuple[bool, str]:
    """
    Validate a user query string.

    Parameters
    ----------
    user_input : str
        Raw string entered by the user.

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message).
        error_message is "" when is_valid is True.
    """
    if not user_input.strip():
        return False, "Input is empty. Please describe what you're looking for."

    words = user_input.strip().split()
    if len(words) < MIN_WORD_COUNT:
        return False, (
            f"Please provide a more detailed description (at least {MIN_WORD_COUNT} words)."
        )

    if re.fullmatch(r"[\d\W_]+", user_input):
        return False, "Input appears to be numbers or symbols only."

    try:
        detected_lang = detect(user_input)
        if detected_lang != "en":
            return False, (
                f"Please write your request in English. "
                f"Detected language: {detected_lang}"
            )
    except LangDetectException:
        return False, "Unable to detect language. Try rephrasing your input."

    return True, ""
