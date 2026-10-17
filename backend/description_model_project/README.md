# 🎵 Description Model — Music Recommender

A **semantic music recommendation system** that finds songs matching a free-text natural language description. It is served through a **FastAPI** backend that the RehamRec HTML/CSS/JavaScript frontend calls directly.
Part of a graduation project on AI-powered music recommendation.

---

## What It Does

A user types something like *"a dark grunge song about toxic love"* and the system returns five songs ranked by semantic similarity, enriched with metadata (album, cover art, genre, preview URL, etc.) from the iTunes Search API. Three retrieval strategies are available:

| Method | Description | Best For |
|--------|-------------|----------|
| **Lyrics Embedding** (`lyrics`) | Uses `{genre} + {artist} + lyrics preview` embeddings | Queries with specific lyrical themes |
| **Description Matching** (`description`) | Uses pre-generated natural-language description embeddings | Mood/atmosphere queries |
| **Combined** (`combined`, default) | Averages both similarity scores | Best general-purpose results |

### How the model works

1. The query is validated (at least 3 words, English).
2. SentenceTransformer (`all-MiniLM-L6-v2`) encodes the query.
3. If a genre is detected in the query, songs are pre-filtered by genre.
4. Cosine similarity is computed against the precomputed song embeddings.
5. Top 100 candidates → sample of 20 → top 5, to keep results diverse.

---

## Architecture

```
HTML / CSS / JavaScript  (frontend)
        │  POST { "description": "..." }
        ▼
FastAPI backend  (src/inference/app.py)
        │
        ▼
Recommendation model  (src/inference/recommender.py)
        │  ranked list of {song, artist, genre, similarity}
        ▼
iTunes Search API  (src/inference/music_api.py)
        │  album, cover, preview URL, etc. — per song
        ▼
FastAPI → JSON  { "recommendations": [ {...}, {...} ] }
        ▼
Frontend renders recommendation cards
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
│   │   ├── recommender.py    # Core recommendation logic
│   │   ├── music_api.py      # iTunes Search API integration
│   │   ├── config.py         # Env-var configuration (paths, model name, CORS)
│   │   └── gradio_app.py     # Optional Gradio chatbot UI
│   │
│   ├── preprocessing/
│   │   └── text_cleaner.py   # Lyrics cleaning, description builders
│   │
│   ├── training/
│   │   └── embedder.py       # Embedding generation & persistence
│   │
│   └── utils/
│       ├── genre_extractor.py
│       ├── input_validator.py
│       └── youtube_links.py  # Used by gradio_app.py only
│
├── data/            # songs_with_descriptions_V3.csv goes here (fetched by download_assets.py)
├── models/          # song_embeddings_1.npy / song_embeddings_2.npy
└── notebooks/
    └── main.ipynb   # Dataset preprocessing + embedding generation (run on Kaggle)
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

- The precomputed embeddings (`models/song_embeddings_1.npy`, `models/song_embeddings_2.npy`) are included in the repo.
- The dataset CSV is hosted on Kaggle. Download it into `data/` with:

```bash
python download_assets.py
```

  (Set the Kaggle dataset slug in `download_assets.py` or via the `KAGGLE_DATASET` environment variable. The CSV can also be downloaded manually from Kaggle and placed in `data/`.)
- The SentenceTransformer model (`all-MiniLM-L6-v2`) is downloaded automatically on the first run.

To regenerate the CSV/embeddings from scratch, run `notebooks/main.ipynb` (on Kaggle). You can also point `DATA_CSV_PATH` / `EMBEDDINGS_DIR` in `.env` at wherever you keep these files.

### 4. Configure environment

```bash
cp .env.example .env      # Windows: copy .env.example .env
# edit .env if your paths/ports differ from the defaults
```

### 5. Start FastAPI

```bash
uvicorn src.inference.app:app --reload
```

The server loads the dataset, model, and embeddings **once** at startup (this can take from a few seconds to a minute) and then serves requests from memory. Startup fails with a clear message if the CSV or embeddings aren't where `.env` says they are.

### 6. Test the endpoint

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"description": "a calm romantic song for a rainy evening"}'
```

Or check readiness first: `curl http://127.0.0.1:8000/health`

### 7. Connect the frontend

Serve the `frontend/` folder from a local dev server, e.g. VS Code "Live Server" or:

```bash
cd ../../frontend
python -m http.server 5500
```

By default, CORS allows `http://127.0.0.1:5500` and `http://localhost:5500` (see `CORS_ORIGINS` in `.env`).

On Windows you can also start everything with `Run_RehamRec.bat` from the repository root.

### Optional: Gradio chatbot

An alternative chatbot interface (intent classification, multi-turn session state, feedback logging) is kept in `gradio_app.py`:

```bash
python -m src.inference.gradio_app
```

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

- `song`, `artist`, `genre`, `similarity` come from the model / dataset.
- `album`, `cover`, `itunes_genre`, `duration_ms`, `preview_url`, `itunes_url`, `release_date` come from the iTunes Search API and are `null` when that song couldn't be found there (the recommendation itself is still returned).
- No `lyrics` field is included, because the iTunes Search API does not provide lyrics.

**Response — 400 Bad Request** (empty input, fewer than 3 words, non-English input, or no songs matched a detected genre):

```json
{ "detail": "Please provide a more detailed description (at least 3 words)." }
```

**Response — 422 Unprocessable Entity** — malformed request body (e.g. an invalid `method` value).

**Response — 503 Service Unavailable** — server is still starting up.

### `GET /health`

Returns `{"status": "ok", "songs_indexed": <n>, "model": "<name>"}`.

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

The frontend only needs this one endpoint and its JSON shape.

---

## Music API

Song metadata comes from the **[iTunes Search API](https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/)** (`https://itunes.apple.com/search`):

- Free and public — **no API key or account required**.
- Returns track/artist/album name, artwork URL, primary genre, track duration, a 30-second preview URL, the store page URL, and release date.
- Does not provide lyrics.
- Has no officially published rate limit, so requests use a short timeout (`MUSIC_API_TIMEOUT`, default 5 seconds) and failures are handled gracefully instead of being retried aggressively.

If a lookup fails for any reason (timeout, rate limit, not found, malformed response), the affected song's metadata fields are `null`. The recommendation is never dropped, and one failing lookup doesn't affect the others.

---

## License

For academic / educational use only. Dataset subject to the original data provider's terms.
