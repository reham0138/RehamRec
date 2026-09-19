const API_URL = "http://127.0.0.1:8000";

// Short prompts used by the quick-search buttons. The backend validates
// natural-language queries as 3+ English words, so the buttons send a
// slightly richer description while keeping the visible labels unchanged.
const QUICK_PROMPTS = {
    happy: "a happy positive song",
    study: "a calm focused song for study",
    chill: "a calm relaxing song",
    workout: "an energetic song for workout"
};

const results = document.getElementById("results");
const query = document.getElementById("query");
const searchForm = document.getElementById("searchForm");
const surprise = document.getElementById("surprise");
const resultCount = document.getElementById("resultCount");
const emptyState = document.getElementById("emptyState");
const historyList = document.getElementById("historyList");
const clearHistory = document.getElementById("clearHistory");
const themeToggle = document.getElementById("themeToggle");
const menuToggle = document.getElementById("menuToggle");
const navLinks = document.getElementById("navLinks");

let history = JSON.parse(localStorage.getItem("rehamrec-history") || "[]");
let currentAudio = null;
let currentButton = null;

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatSimilarity(value) {
    if (typeof value !== "number" || Number.isNaN(value)) return "";
    return `${Math.round(value * 100)}% match`;
}

function stopCurrentAudio() {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }
    if (currentButton) {
        currentButton.classList.remove("playing");
        currentButton.textContent = "▶";
    }
    currentAudio = null;
    currentButton = null;
}

function togglePreview(song, button) {
    if (!song.preview_url) {
        if (song.itunes_url) window.open(song.itunes_url, "_blank", "noopener,noreferrer");
        return;
    }

    if (currentButton === button && currentAudio) {
        if (currentAudio.paused) {
            currentAudio.play().catch(() => {});
            button.classList.add("playing");
            button.textContent = "Ⅱ";
        } else {
            stopCurrentAudio();
        }
        return;
    }

    stopCurrentAudio();
    currentAudio = new Audio(song.preview_url);
    currentButton = button;

    currentAudio.addEventListener("ended", stopCurrentAudio, { once: true });
    currentAudio.play()
        .then(() => {
            button.classList.add("playing");
            button.textContent = "Ⅱ";
        })
        .catch(() => {
            stopCurrentAudio();
        });
}


async function fetchLyrics(song) {
    const params = new URLSearchParams({
        track_name: song.song || "",
        artist_name: song.artist || ""
    });

    if (song.album) params.set("album_name", song.album);
    if (typeof song.duration_ms === "number" && song.duration_ms > 0) {
        params.set("duration", String(Math.round(song.duration_ms / 1000)));
    }

    const response = await fetch(`https://lrclib.net/api/get?${params.toString()}`);
    if (response.status === 404) {
        throw new Error("Lyrics not found for this song.");
    }
    if (!response.ok) {
        throw new Error(`Lyrics service returned ${response.status}.`);
    }

    const payload = await response.json();

    if (payload.instrumental) {
        return "This track is instrumental.";
    }

    return payload.plainLyrics || payload.syncedLyrics?.replace(/\\[\\d{2}:\\d{2}(?:\\.\\d{1,3})?\\]/g, "").trim() ||
        "Lyrics are not available for this track.";
}

async function toggleLyrics(song, panel, button) {
    const isOpen = !panel.hidden;

    if (isOpen) {
        panel.hidden = true;
        button.textContent = "Lyrics";
        return;
    }

    panel.hidden = false;

    if (panel.dataset.loaded === "true") {
        button.textContent = "Hide lyrics";
        return;
    }

    button.disabled = true;
    button.textContent = "Loading...";

    try {
        const lyrics = await fetchLyrics(song);
        panel.innerHTML = `<div class="lyrics-title">Lyrics</div><pre>${escapeHtml(lyrics)}</pre>`;
        panel.dataset.loaded = "true";
        button.textContent = "Hide lyrics";
    } catch (error) {
        panel.innerHTML = `<div class="lyrics-error">${escapeHtml(error.message || "Lyrics are not available.")}</div>`;
        button.textContent = "Lyrics";
    } finally {
        button.disabled = false;
    }
}

function renderSongs(list) {
    results.innerHTML = "";

    list.forEach((song) => {
        const card = document.createElement("article");
        card.className = "song-card";

        const cover = song.cover
            ? `<img src="${escapeHtml(song.cover)}" alt="${escapeHtml(song.song)} cover" loading="lazy">`
            : `<span>♪</span><span>♫</span>`;

        const genre = song.itunes_genre || song.genre || "music";
        const match = formatSimilarity(song.similarity);
        const meta = [song.album, match].filter(Boolean).join(" · ");
        const buttonLabel = song.preview_url ? `Play ${song.song}` : `Open ${song.song}`;

        card.innerHTML = `
            <div class="cover ${song.cover ? "has-image" : ""}">
                ${cover}
            </div>
            <h3>${escapeHtml(song.song)}</h3>
            <p>${escapeHtml(song.artist)}</p>
            ${meta ? `<p class="song-meta">${escapeHtml(meta)}</p>` : ""}
            <div class="card-footer">
                <span class="tag">${escapeHtml(genre)}</span>
                <div class="card-actions">
                    <button class="lyrics-btn" type="button">Lyrics</button>
                    <button class="play-btn" type="button" aria-label="${escapeHtml(buttonLabel)}">▶</button>
                </div>
            </div>
            <div class="lyrics-panel" hidden></div>
        `;

        const playButton = card.querySelector(".play-btn");
        const lyricsButton = card.querySelector(".lyrics-btn");
        const lyricsPanel = card.querySelector(".lyrics-panel");

        playButton.addEventListener("click", () => togglePreview(song, playButton));
        lyricsButton.addEventListener("click", () => toggleLyrics(song, lyricsPanel, lyricsButton));
        results.appendChild(card);
    });

    resultCount.textContent = `${list.length} song${list.length === 1 ? "" : "s"}`;
    emptyState.hidden = list.length !== 0;
}

function renderLoading() {
    results.innerHTML = `
        <article class="song-card loading-card">
            <div class="cover"><span>♪</span><span>♫</span></div>
            <h3>Finding your music...</h3>
            <p>RehamRec is comparing your description with the music embeddings.</p>
        </article>
    `;
    resultCount.textContent = "loading...";
    emptyState.hidden = true;
}

function renderError(message) {
    results.innerHTML = "";
    emptyState.hidden = false;
    emptyState.textContent = message;
    resultCount.textContent = "0 songs";
}

async function getRecommendations(value) {
    const description = value.trim();

    if (!description) {
        return { recommendations: [], error: "Please describe the kind of music you want." };
    }

    try {
        const response = await fetch(`${API_URL}/recommend`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                description,
                method: "combined",
                top_k: 5
            })
        });

        let payload = {};
        try {
            payload = await response.json();
        } catch (_) {
            payload = {};
        }

        if (!response.ok) {
            return {
                recommendations: [],
                error: payload.detail || `The recommendation server returned ${response.status}.`
            };
        }

        return {
            recommendations: Array.isArray(payload.recommendations) ? payload.recommendations : [],
            error: null
        };
    } catch (error) {
        return {
            recommendations: [],
            error: "Cannot connect to RehamRec's AI server. Start the FastAPI backend on port 8000 and try again."
        };
    }
}

function saveHistory(value) {
    const clean = value.trim();
    if (!clean) return;

    history = [clean, ...history.filter(item => item.toLowerCase() !== clean.toLowerCase())].slice(0, 8);
    localStorage.setItem("rehamrec-history", JSON.stringify(history));
    renderHistory();
}

function renderHistory() {
    if (!history.length) {
        historyList.innerHTML = '<p class="muted">Your searches will appear here.</p>';
        return;
    }

    historyList.innerHTML = history.map(item =>
        `<button class="history-item" type="button" data-history="${escapeHtml(item)}">${escapeHtml(item)}</button>`
    ).join("");

    document.querySelectorAll("[data-history]").forEach(item => {
        item.addEventListener("click", () => {
            query.value = item.dataset.history;
            runSearch(item.dataset.history);
            document.getElementById("recommendations").scrollIntoView({ behavior: "smooth" });
        });
    });
}

async function runSearch(value) {
    const clean = value.trim();
    if (!clean) {
        renderError("Please describe the kind of music you want.");
        return;
    }

    saveHistory(clean);
    renderLoading();
    stopCurrentAudio();

    const { recommendations, error } = await getRecommendations(clean);

    if (error) {
        renderError(error);
        return;
    }

    renderSongs(recommendations);
}

searchForm.addEventListener("submit", event => {
    event.preventDefault();
    runSearch(query.value);
    document.getElementById("recommendations").scrollIntoView({ behavior: "smooth" });
});

surprise.addEventListener("click", () => {
    const prompts = Object.values(QUICK_PROMPTS);
    const randomPrompt = prompts[Math.floor(Math.random() * prompts.length)];
    query.value = randomPrompt;
    runSearch(randomPrompt);
    document.getElementById("recommendations").scrollIntoView({ behavior: "smooth" });
});

document.querySelectorAll(".quick-tags button").forEach(button => {
    button.addEventListener("click", () => {
        const prompt = QUICK_PROMPTS[button.dataset.query] || button.dataset.query;
        query.value = prompt;
        runSearch(prompt);
        document.getElementById("recommendations").scrollIntoView({ behavior: "smooth" });
    });
});

clearHistory.addEventListener("click", () => {
    history = [];
    localStorage.removeItem("rehamrec-history");
    renderHistory();
});

themeToggle.addEventListener("click", () => {
    document.body.classList.toggle("dark");
    const dark = document.body.classList.contains("dark");
    themeToggle.textContent = dark ? "light mode" : "dark mode";
    localStorage.setItem("rehamrec-theme", dark ? "dark" : "light");
});

menuToggle.addEventListener("click", () => {
    const open = navLinks.classList.toggle("open");
    menuToggle.setAttribute("aria-expanded", String(open));
    menuToggle.textContent = open ? "×" : "☰";
});

document.querySelectorAll(".links a").forEach(link => {
    link.addEventListener("click", () => {
        navLinks.classList.remove("open");
        menuToggle.setAttribute("aria-expanded", "false");
        menuToggle.textContent = "☰";
    });
});

if (localStorage.getItem("rehamrec-theme") === "dark") {
    document.body.classList.add("dark");
    themeToggle.textContent = "light mode";
}

document.getElementById("year").textContent = new Date().getFullYear();
renderHistory();
