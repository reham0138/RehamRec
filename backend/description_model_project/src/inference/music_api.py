"""
music_api.py
------------
Communication with the external music metadata API.

API choice: Apple's iTunes Search API (https://itunes.apple.com/search).
Verified (Sept 2026):
    - Free, public, no API key / no login required.
    - No published rate limit; the endpoint is commonly rate-limited by
      Apple at roughly ~20 requests/minute per client, so this module uses
      a short per-request timeout and never retries aggressively.
    - Returns: track name, artist name, collection (album) name, artwork
      URL, primary genre, track duration, a 30-second preview URL, the
      iTunes store page URL, and release date.
    - Does NOT return lyrics. No lyrics field is exposed by this module —
      it is not fabricated.

This module is intentionally the *only* place that talks to the external
API. It never touches the recommendation model or dataset.
"""

from __future__ import annotations

import logging

import requests

from src.inference import config

logger = logging.getLogger("description_model.music_api")

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"

# Every field this module can populate. enrich_recommendation() always
# returns a dict with exactly these keys, so callers never need to special
# case a lookup failure — missing data is just None.
_EMPTY_METADATA: dict = {
    "album": None,
    "cover": None,
    "itunes_genre": None,
    "duration_ms": None,
    "preview_url": None,
    "itunes_url": None,
    "release_date": None,
}


def search_song(song: str, artist: str, timeout: float | None = None) -> dict | None:
    """
    Query the iTunes Search API for a track matching `song` by `artist`.

    Parameters
    ----------
    song, artist : str
        Values as produced by the recommender (dataset 'song'/'artist' columns).
    timeout : float | None
        Request timeout in seconds. Defaults to config.MUSIC_API_TIMEOUT.

    Returns
    -------
    dict | None
        The first matching raw iTunes result, or None if nothing was found
        or the request failed for any reason (network error, timeout,
        rate limit, malformed response). Failures are logged, never raised.
    """
    term = f"{song or ''} {artist or ''}".strip()
    if not term:
        return None

    timeout = config.MUSIC_API_TIMEOUT if timeout is None else timeout
    params = {"term": term, "media": "music", "entity": "song", "limit": 1}

    try:
        response = requests.get(ITUNES_SEARCH_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.warning("iTunes API timed out for %r", term)
        return None
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        logger.warning("iTunes API returned HTTP %s for %r", status, term)
        return None
    except requests.exceptions.RequestException as exc:
        logger.warning("iTunes API request failed for %r: %s", term, exc)
        return None

    try:
        payload = response.json()
    except ValueError:
        logger.warning("iTunes API returned malformed JSON for %r", term)
        return None

    results = payload.get("results") or []
    if not results:
        logger.info("iTunes API found no match for %r", term)
        return None

    return results[0]


def enrich_recommendation(song: str, artist: str, timeout: float | None = None) -> dict:
    """
    Look up additional metadata for one recommended song.

    Never raises: any failure (not found, timeout, rate limit, malformed
    response) simply results in the corresponding fields being None so a
    single bad lookup can't break the rest of a /recommend response.

    Returns
    -------
    dict
        Always contains: album, cover, itunes_genre, duration_ms,
        preview_url, itunes_url, release_date.
    """
    result = search_song(song, artist, timeout=timeout)
    if not result:
        return dict(_EMPTY_METADATA)

    return {
        "album": result.get("collectionName"),
        "cover": result.get("artworkUrl100"),
        "itunes_genre": result.get("primaryGenreName"),
        "duration_ms": result.get("trackTimeMillis"),
        "preview_url": result.get("previewUrl"),
        "itunes_url": result.get("trackViewUrl"),
        "release_date": result.get("releaseDate"),
    }
