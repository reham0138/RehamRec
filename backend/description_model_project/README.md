# 🎵 Description Model — Music Recommender

A **semantic music recommendation system** that finds songs matching a
free-text natural language description, now exposed behind a **FastAPI**
backend so a separate HTML/CSS/JavaScript frontend can call it directly.
Part of a graduation project on AI-powered music recommendation.

---

## What It Does

A user types something like *"a dark grunge song about toxic love"* and the
system returns five songs ranked by semantic similarity, enriched with
metadata (album, cover art, genre, preview URL, etc.) from the iTunes
Search API. Three retrieval strategies are available:

| Method | Description | Best For |
|--------|-------------|----------|
| **Lyrics Embedding** (`lyrics`) | Uses `{genre} + {artist} + lyrics preview` embeddings | Queries with specific lyrical themes |
| **Description Matching** (`description`) | Uses pre-generated natural-language description embeddings | Mood/atmosphere queries |
| **Combined** (`combined`, default) | Averages both similarity scores | Best general-purpose results |

The ML pipeline itself — SentenceTransformer embeddings, genre pre-filter,
cosine-similarity ranking, top-100→sample-20→top-5 diversity sampling — is
**unchanged** from the original project. What's new is a clean interface
around it.

---

## Architecture

```
HTML / CSS / JavaScript  (your separate frontend)
        │  POST { "description": "..." }
        ▼
FastAPI backend  (src/inference/app.py)
        │
        ▼
Existing recommendation model  (src/inference/recommender.py)
        │  ranked list of {song, artist, genre, similarity}
        ▼
iTunes Search API  (src/inference/music_api.py)
        │  album, cover, preview URL, etc. — per song
        ▼
FastAPI → JSON  { "recommendations": [ {...}, {...} ] }
        ▼
Your frontend renders recommendation cards
```

---

## Dataset

- **Dataset:** [60k Songs with Descriptions](https://www.kaggle.com/datasets/rehammayea/60k-songs-with-description)
- **File:** `songs_with_descriptions_V3.csv` (hosted on Kaggle, not stored in this repo)
- **Required columns:** `song`, `artist`, `translated_genres`, `text` (lyrics), `full_description`

---

## Project Structure

```
description_model_project/
│
├── README.md
├── requirements.txt
├── download_assets.py   # Downloads the dataset CSV from Kaggle into data/
├── .env.example
├── .gitignore
│
├── src/
│   ├── inference/
│   │   ├── app.py            # FastAPI entry point — POST /recommend, GET /health
│   │   ├── recommender.py    # Core recommendation logic (unchanged) + JSON-ready variants
│   │   ├── music_api.py      # iTunes Search API integration
│   │   ├── config.py         # Env-var configuration (paths, model name, CORS)
│   │   └── gradio_app.py     # Original Gradio chatbot UI — kept as-is, optional
│   │
│   ├── preprocessing/
│   │   └── text_cleaner.py   # Lyrics cleaning, description builders (unchanged)
│   │
│   ├── training/
│   │   └── embedder.py       # Embedding generation & persistence (unchanged)
│   │
│   └── utils/
│       ├── genre_extractor.py
│       ├── input_validator.py
│       └── youtube_links.py  # Used by gradio_app.py only
│
├── data/            # songs_with_descriptions_V3.csv goes here (fetched by download_assets.py)
├── models/          # song_embeddings_1.npy / song_embeddings_2.npy (included in the repo)
└── notebooks/
    └── main.ipynb   # Generates the dataset preprocessing + embeddings (run on Kaggle)
```

---

## Run Instructions

### 1. Create/activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Get the dataset

- The precomputed embeddings (`models/song_embeddings_1.npy`,
  `models/song_embeddings_2.npy`) are already included in the repo.
- The dataset CSV is hosted on Kaggle. Download it into `data/` with:

```bash
python download_assets.py
```

  (Set the Kaggle dataset slug in `download_assets.py` or via the
  `KAGGLE_DATASET` environment variable. The CSV can also be downloaded
  manually from Kaggle and placed in `data/`.)
- The SentenceTransformer model (`all-MiniLM-L6-v2`) is downloaded
  automatically on the first run — nothing to do.

To regenerate the CSV/embeddings from scratch, run `notebooks/main.ipynb`
(on Kaggle). You can also point `DATA_CSV_PATH` / `EMBEDDINGS_DIR` in `.env`
at wherever you keep these files.

### 4. Configure environment

```bash
cp .env.example .env
# edit .env if your paths/ports differ from the defaults
```

### 5. Start FastAPI

```bash
uvicorn src.inference.app:app --reload
```

The server loads the dataset, model, and embeddings **once** at startup
(this takes a few seconds to a minute depending on dataset size) and then
serves requests from memory. Startup fails fast with a clear message if the
CSV or embeddings aren't where `.env` says they are.

### 6. Test the endpoint

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"description": "a calm romantic song for a rainy evening"}'
```

Or check readiness first: `curl http://127.0.0.1:8000/health`

### 7. Connect the HTML/JS frontend

Serve your frontend from a local dev server, e.g. VS Code "Live Server" or:

```bash
python -m http.server 5500
```

By default, CORS is configured to allow `http://127.0.0.1:5500` and
`http://localhost:5500` (see `CORS_ORIGINS` in `.env`).

---

## API Documentation

### `POST /recommend`

**Request**

```json
{
  "description": "I want a sad piano song for a lonely evening",
  "method": "combined",
  "top_k": 5
}
```

- `description` (string, required) — free-text description.
- `method` (string, optional, default `"combined"`) — one of `"lyrics"`, `"description"`, `"combined"`.
- `top_k` (integer, optional, default `5`, 1–20) — number of recommendations to return.

**Response — 200 OK**

```json
{
  "recommendations": [
    {
      "song": "Rain",
      "artist": "Soft Serenade",
      "genre": "jazz;romantic",
      "similarity": 0.74,
      "album": "Some Album",
      "cover": "https://.../100x100bb.jpg",
      "itunes_genre": "Jazz",
      "duration_ms": 210000,
      "preview_url": "https://.../preview.m4a",
      "itunes_url": "https://music.apple.com/...",
      "release_date": "2019-05-01T00:00:00Z"
    }
  ]
}
```

- `song`, `artist`, `genre`, `similarity` come from the ML model / dataset.
- `album`, `cover`, `itunes_genre`, `duration_ms`, `preview_url`,
  `itunes_url`, `release_date` come from the iTunes Search API and are
  `null` when that particular song couldn't be found there (the ML
  recommendations are still returned).
- **No `lyrics` field is included** — the iTunes Search API does not
  provide lyrics, so none is fabricated.

**Response — 400 Bad Request** (validation failure — empty input, fewer
than 3 words, non-English input, or no songs matched a detected genre):

```json
{ "detail": "Please provide a more detailed description (at least 3 words)." }
```

**Response — 422 Unprocessable Entity** — malformed request body (e.g. an
invalid `method` value; FastAPI/Pydantic validates this automatically).

**Response — 503 Service Unavailable** — server is still starting up.

### `GET /health`

Returns `{"status": "ok", "songs_indexed": <n>, "model": "<name>"}` — useful
while wiring up the frontend.

---

## Frontend Integration Example

```javascript
async function getRecommendations(description) {
  const response = await fetch("http://127.0.0.1:8000/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ description }),
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || "Request failed");
  }

  const data = await response.json();
  return data.recommendations; // array of song objects, ready to render
}
```

The frontend never needs to know anything about SentenceTransformer,
embeddings, FAISS, dataset paths, or Python module names — just this one
endpoint and its JSON shape.

---

## Music API

**Chosen API: [iTunes Search API](https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/)**
(`https://itunes.apple.com/search`).

Verified before use:
- Free, public, **no API key or account required** — nothing to put in `.env`.
- Returns track/artist/album name, artwork URL, primary genre, track
  duration, a 30-second preview URL, the store page URL, and release date.
- **Does not provide lyrics.** No lyrics field is exposed anywhere in the
  API response — this was verified rather than assumed.
- No officially published rate limit; in practice it's informally
  rate-limited by Apple, so requests use a short timeout (`MUSIC_API_TIMEOUT`,
  default 5s) and failures are handled gracefully rather than retried
  aggressively.

If a lookup fails for any reason (timeout, rate limit, not found, malformed
response), the affected song's metadata fields are simply `null` — the ML
recommendation itself is never dropped, and one failing lookup never affects
the others.

---

## Change Summary

**Created:**
- `src/inference/app.py` — new FastAPI entry point (`POST /recommend`, `GET /health`). Loads the model/dataset/embeddings once at startup via a `lifespan` handler.
- `src/inference/music_api.py` — iTunes Search API client, isolated from `app.py` as requested.
- `src/inference/config.py` — environment-variable configuration (paths, model name, CORS origins, API timeout).
- `.env.example`, `.gitignore` — environment/secret handling.
- `data/README.md`, `models/README.md` (+ `.gitkeep`) — placeholders explaining what local files go where, since the backend now also runs outside Kaggle.

**Modified:**
- `src/inference/recommender.py` — **the three original functions
  (`recommend_by_lyrics_description`, `recommend_by_full_description`,
  `recommend_by_combined`) are untouched, algorithm and all.** Added
  structured (`list[dict]`) counterparts of each, reusing the same private
  `_filter_by_genre` / `_rank_and_sample` helpers, plus a
  `get_structured_recommendations()` dispatcher used by the FastAPI
  endpoint. This was necessary because the original functions return a
  formatted string (`"1. Song by Artist"`), and the frontend must not have
  to parse that.
- `src/inference/__init__.py` — export the new structured functions alongside the originals.
- `requirements.txt` — added `fastapi`, `uvicorn[standard]`, `pydantic`, `requests`, `python-dotenv`. Every existing dependency was kept.

**Renamed (content unchanged):**
- `src/inference/app.py` → `src/inference/gradio_app.py`. The original
  `app.py` was a Gradio chatbot with zero-shot intent classification,
  multi-turn session state, and feedback logging — a different UI on top of
  the same recommender, not a stateless "description in, JSON out" endpoint.
  Rather than force chatbot/session logic into an HTTP API (which the task
  said not to do — "adapt it carefully", "don't over-engineer"), it was kept
  fully intact under a new filename as an optional, still-runnable
  alternative interface. `python -m src.inference.gradio_app` still works
  exactly as before, `gradio` stays in `requirements.txt` for it.

**Left unchanged:**
- `src/preprocessing/text_cleaner.py`, `src/training/embedder.py`,
  `src/utils/*.py` — no reason to touch these; the dataset columns,
  cleaning logic, and embedding generation are exactly as before.
- `notebooks/main.ipynb` — still the way to generate embeddings; not part
  of the runtime backend.

---

## Testing Performed

Since the real dataset/embeddings/model weights aren't included in this
repo (and downloading multi-GB model weights wasn't practical while
preparing this), the following was verified directly, end-to-end, using a
small synthetic dataset built with the *real* preprocessing/embedding code
and a lightweight stand-in for `SentenceTransformer.encode()`:

- All modules import successfully; no syntax errors.
- FastAPI app starts via its `lifespan` handler, loads dataset + embeddings + model once.
- `GET /health` returns the expected readiness payload.
- `POST /recommend` returns correct structured JSON for all three methods (`lyrics`, `description`, `combined`), preserving the original ranking logic.
- Input validation: empty description and too-short description both return `400` with the original validator's message; an invalid `method` value returns `422` via Pydantic.
- `top_k` is respected.
- CORS preflight (`OPTIONS`) succeeds for the configured origin.
- Startup fails with a clear, actionable error when the CSV or embeddings are missing, or when their row counts don't match.
- `music_api.py` was tested against simulated timeout, HTTP error (429), malformed JSON, and "not found" responses (outbound calls to `itunes.apple.com` aren't reachable from the sandbox this was built in) — in every case it returns the all-`null` metadata dict rather than raising.
- A full `/recommend` request was run with the metadata-enrichment step simulating one song failing outright (a raised exception) and one song returning no match — the endpoint still returned `200 OK` with all recommendations, exactly as required.

**What you still need to verify yourself:** an actual run against the real
60k-song dataset and real `sentence-transformers`/`torch` (not installed in
the environment this was prepared in), and a live call to
`https://itunes.apple.com/search` from your own network.

---

## Bugs Fixed from Original Code (carried over, unchanged)

| # | Original Bug | Fix Applied |
|---|-------------|-------------|
| 1 | `show_recommendation` option 1 passed `song_embeddings_2` instead of `song_embeddings_1` (and vice versa for option 2) | Corrected argument order in `gradio_app.py` |
| 2 | Fallback block in `suggest_songs` referenced undefined variable `scored_candidates` → `NameError` at runtime | Removed dead code; fallback is redundant with the loop logic |
| 3 | Module-level globals for session state (`last_input`, `last_recommendation`) were shared across concurrent Gradio users | Replaced with `gr.State()` per-session state components |

---

## License

For academic / educational use only. Dataset subject to original data provider's terms.
