"""
app.py
------
Gradio chatbot interface for the Description Model music recommender.

The chatbot handles two input modes:
    1. Song - Artist  (e.g. "Creep - Radiohead")
       → reserved for the Audio/Lyrics Model (Section 1 of the original project)
       → in this isolated version, a friendly message is returned.
    2. Free-text description  (e.g. "a dark grunge song about toxic love")
       → classified by a zero-shot BART model and routed to one of three
         recommendation strategies.

State management
----------------
Gradio Blocks does not natively support per-user server-side state across
callbacks without explicit state components.  We use gr.State() objects to
pass session data between event handlers rather than relying on module-level
globals (which would be shared across concurrent users).
"""

from __future__ import annotations

import traceback
import uuid
import os

import gradio as gr
import numpy as np
import pandas as pd
from transformers import pipeline

from src.inference.recommender import (
    recommend_by_lyrics_description,
    recommend_by_full_description,
    recommend_by_combined,
)
from src.utils.youtube_links import add_youtube_links

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEEDBACK_FILE: str = "outputs/feedback_dataset.csv"
ZERO_SHOT_MODEL: str = "facebook/bart-large-mnli"
INTENT_LABELS: list[str] = [
    "Greeting",
    "Song Description",
    "Opinion",
    "Question",
    "Song Feedback",
    "Chit-chat",
]


# ---------------------------------------------------------------------------
# Lazy-loaded zero-shot classifier
# ---------------------------------------------------------------------------
# Loaded once on first use so the app starts faster.
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        print("[app] Loading zero-shot classifier...")
        _classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
    return _classifier


# ---------------------------------------------------------------------------
# Feedback persistence
# ---------------------------------------------------------------------------

def save_feedback(
    user_id: str,
    input_text: str,
    recommendation: str,
    feedback_text: str,
) -> None:
    """Append a feedback row to the CSV log."""
    os.makedirs("outputs", exist_ok=True)
    new_row = pd.DataFrame([{
        "user_id": user_id,
        "input_text": input_text,
        "recommendation": recommendation,
        "feedback_text": feedback_text,
    }])
    if os.path.exists(FEEDBACK_FILE):
        existing = pd.read_csv(FEEDBACK_FILE)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(FEEDBACK_FILE, index=False)


# ---------------------------------------------------------------------------
# Chatbot response handlers
# ---------------------------------------------------------------------------

def chatbot_response(
    user_message: str,
    history: list,
    last_input_state: str | None,
    last_recommendation_state: str | None,
    session_id_state: str,
    df: pd.DataFrame,
    embeddings_1: np.ndarray,
    embeddings_2: np.ndarray,
    model,
    genre_list: list[str],
):
    """
    Primary message handler.

    Routes the user message to the appropriate logic branch and returns
    updated state to the Gradio components.
    """
    last_input_state = user_message
    history = history or []

    try:
        # --- Branch 1: Song-Artist lookup (reserved for Audio/Lyrics Model) ---
        if "-" in user_message and len(user_message.split("-")) >= 2:
            history.append((
                user_message,
                "🎵 Song-Artist lookup is part of the Audio/Lyrics Model. "
                "Please enter a free-text description to use the Description Model!",
            ))
            return (
                history,
                gr.update(visible=False),
                gr.update(visible=True),
                "",
                last_input_state,
                last_recommendation_state,
            )

        # --- Branch 2: Zero-shot intent classification ---
        classifier = get_classifier()
        result = classifier(user_message, INTENT_LABELS)
        intent = result["labels"][0]

        if intent == "Greeting":
            history.append((
                user_message,
                "Hello! 🎧 Describe the kind of song you're looking for and I'll find it for you. "
                "For example: *\"a grunge song about pain and toxic love\"*",
            ))
            return (
                history,
                gr.update(visible=False),
                gr.update(visible=False),
                "",
                last_input_state,
                last_recommendation_state,
            )

        if intent == "Question":
            history.append((
                user_message,
                "I don't have access to external information, but I can recommend music! "
                "Describe the mood or theme you're looking for.",
            ))
            return (
                history,
                gr.update(visible=False),
                gr.update(visible=False),
                "",
                last_input_state,
                last_recommendation_state,
            )

        if intent in ("Song Feedback", "Opinion"):
            save_feedback(
                session_id_state,
                last_input_state or "",
                last_recommendation_state or "No recommendation yet",
                user_message,
            )
            history.append((user_message, "Thanks for your feedback! 🙏"))
            return (
                history,
                gr.update(visible=False),
                gr.update(visible=False),
                "",
                last_input_state,
                last_recommendation_state,
            )

        if intent == "Chit-chat":
            history.append((
                user_message,
                "Nice to chat, but I'm best at finding music! "
                "Describe what you want to listen to. 🎵",
            ))
            return (
                history,
                gr.update(visible=False),
                gr.update(visible=False),
                "",
                last_input_state,
                last_recommendation_state,
            )

        # --- Branch 3: Song description → show method selection buttons ---
        history.append((
            user_message,
            "🎧 Please choose a recommendation method:\n"
            "1️⃣ **Lyrics Embedding** — searches by lyrical themes\n"
            "2️⃣ **Description Matching** — searches by natural language description\n"
            "3️⃣ **Combined** — averages both for best results",
        ))
        return (
            history,
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            last_input_state,
            last_recommendation_state,
        )

    except Exception:
        traceback.print_exc()
        history.append((user_message, "❌ An error occurred. Please try again."))
        return (
            history,
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            last_input_state,
            last_recommendation_state,
        )


def show_recommendation(
    option: str,
    history: list,
    last_input_state: str | None,
    last_recommendation_state: str | None,
    df: pd.DataFrame,
    embeddings_1: np.ndarray,
    embeddings_2: np.ndarray,
    model,
    genre_list: list[str],
):
    """Handle a recommendation method button click."""
    history = history or []

    if not last_input_state:
        return (
            history,
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            last_recommendation_state,
        )

    try:
        if option == "1":
            result = recommend_by_lyrics_description(
                df, embeddings_1, model, last_input_state, genre_list
            )
        elif option == "2":
            result = recommend_by_full_description(
                df, embeddings_2, model, last_input_state, genre_list
            )
        elif option == "3":
            result = recommend_by_combined(
                df, embeddings_1, embeddings_2, model, last_input_state, genre_list
            )
        else:
            result = "❌ Invalid option."

        result = add_youtube_links(result)
        last_recommendation_state = result
        history.append((f"Option {option} selected", result))
        return (
            history,
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            last_recommendation_state,
        )

    except Exception:
        traceback.print_exc()
        history.append((f"Option {option} selected", "❌ An error occurred. Please try again."))
        return (
            history,
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            last_recommendation_state,
        )


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app(
    df: pd.DataFrame,
    embeddings_1: np.ndarray,
    embeddings_2: np.ndarray,
    model,
    genre_list: list[str],
):
    """
    Construct and return the Gradio Blocks application.

    Parameters
    ----------
    df, embeddings_1, embeddings_2, model, genre_list :
        Pre-loaded data and model objects injected at startup.

    Returns
    -------
    gr.Blocks
        The assembled Gradio app (call .launch() to start it).
    """
    with gr.Blocks(theme="NoCrypt/miku") as demo:

        gr.HTML("""
        <div style='text-align:center; padding: 6px; border-radius: 10px; color:white;'>
          <h1 style='font-size: 2rem; font-family: Freestyle Script; margin: 3px;'>
            🎵 Music Recommender — Description Model
          </h1>
          <p style='font-size: 1.2rem; font-family: Georgia, serif; margin: 3px;'>
            Describe the kind of song you're looking for:<br>
            <em>"A dark grunge song about toxic love"</em>
          </p>
        </div>
        """)

        # Per-session state (avoids shared global mutation between users)
        session_id_state = gr.State(str(uuid.uuid4()))
        last_input_state = gr.State(None)
        last_recommendation_state = gr.State(None)

        chatbot = gr.Chatbot(label="🎧 Chat", bubble_full_width=False, height=450)
        user_input = gr.Textbox(
            placeholder="Describe the music you're looking for...",
            label="💬 Your Input",
            lines=1,
        )

        method_buttons = gr.Column(visible=False)
        with method_buttons:
            with gr.Row():
                btn1 = gr.Button("1️⃣ Lyrics Embedding")
                btn2 = gr.Button("2️⃣ Description Matching")
                btn3 = gr.Button("3️⃣ Combined (Recommended)")

        regenerate_btn = gr.Button("🔁 Try Another Recommendation", visible=False)

        # Wire up events
        common_outputs = [
            chatbot, method_buttons, regenerate_btn,
            user_input, last_input_state, last_recommendation_state,
        ]

        user_input.submit(
            fn=lambda msg, hist, li, lr, sid: chatbot_response(
                msg, hist, li, lr, sid, df, embeddings_1, embeddings_2, model, genre_list
            ),
            inputs=[user_input, chatbot, last_input_state, last_recommendation_state, session_id_state],
            outputs=common_outputs,
        )

        rec_outputs = [
            chatbot, method_buttons, regenerate_btn, user_input, last_recommendation_state,
        ]

        btn1.click(
            fn=lambda hist, li, lr: show_recommendation(
                "1", hist, li, lr, df, embeddings_1, embeddings_2, model, genre_list
            ),
            inputs=[chatbot, last_input_state, last_recommendation_state],
            outputs=rec_outputs,
        )
        btn2.click(
            fn=lambda hist, li, lr: show_recommendation(
                "2", hist, li, lr, df, embeddings_1, embeddings_2, model, genre_list
            ),
            inputs=[chatbot, last_input_state, last_recommendation_state],
            outputs=rec_outputs,
        )
        btn3.click(
            fn=lambda hist, li, lr: show_recommendation(
                "3", hist, li, lr, df, embeddings_1, embeddings_2, model, genre_list
            ),
            inputs=[chatbot, last_input_state, last_recommendation_state],
            outputs=rec_outputs,
        )

    return demo
