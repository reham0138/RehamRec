"""
youtube_links.py
----------------
Appends YouTube search links to recommendation output strings.
"""

from __future__ import annotations
import urllib.parse


def add_youtube_links(recommendation_text: str) -> str:
    """
    Append a YouTube search URL to each 'song by artist' line in a
    recommendation string.

    Lines that do not contain ' by ' (e.g. headers) are passed through
    unchanged.

    Parameters
    ----------
    recommendation_text : str
        Multi-line string where each recommendation line follows the
        format "N. Song Title by Artist Name".

    Returns
    -------
    str
        Same text with '🔗 [Search on YouTube](url)' appended to each
        song line.
    """
    linked_lines: list[str] = []

    for line in recommendation_text.strip().split("\n"):
        stripped = line.strip()

        # Skip header lines (e.g. "Recommended Songs:")
        if stripped.lower().startswith("recommended songs"):
            linked_lines.append(line)
            continue

        if " by " in stripped.lower():
            encoded_query = urllib.parse.quote_plus(stripped)
            youtube_url = (
                f"https://www.youtube.com/results?search_query={encoded_query}"
            )
            linked_lines.append(f"{line}  🔗 [Search on YouTube]({youtube_url})")
        else:
            linked_lines.append(line)

    return "\n".join(linked_lines)
