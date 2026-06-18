"""
GoodReads Book Recommender — Streamlit App
CF layer: Item-Based Collaborative Filtering (scikit-surprise)
         Best tuned config: pearson_baseline similarity, k=20
LLM layer: Gemini gemini-2.5-flash-lite re-ranking via google-genai
"""

import os
import random
import pandas as pd
import streamlit as st
from collections import defaultdict
from surprise import KNNBasic, Dataset, Reader
from pydantic import BaseModel, Field

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bookish: Find Your Next Read",
    page_icon="📚",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300&family=Lora:ital,wght@0,400;1,400;1,500&display=swap" rel="stylesheet">
<style>
    /* Global accent color */
    :root { --purple: #7C3AED; --purple-light: #EDE9FE; }

    /* Buttons */
    .stButton > button {
        background-color: #7C3AED !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        background-color: #6D28D9 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        color: #000000;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EDE9FE !important;
        color: #000000 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F5F3FF;
    }

    /* Book cards */
    .book-card {
        background: #F5F3FF;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
        border-left: 4px solid #7C3AED;
    }
    .book-title { font-weight: 700; font-size: 1rem; color: #4C1D95; }
    .book-author { color: #7C3AED; font-size: 0.85rem; }
    .book-reason { font-size: 0.9rem; margin-top: 6px; font-style: italic; }

    /* Metric labels */
    [data-testid="stMetricLabel"] { color: #7C3AED !important; font-weight: 600; }

    /* Mode radio captions */
    div[data-testid="stRadio"] div[data-testid="stCaptionContainer"] p {
        font-style: italic;
        color: #888 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Data & model loading ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    books   = pd.read_csv("Books.csv",   encoding="latin-1", on_bad_lines="skip")
    ratings = pd.read_csv("Ratings.csv", encoding="latin-1", on_bad_lines="skip")
    books["primary_author"] = books["authors"].str.split(",").str[0].str.strip()
    return books, ratings

@st.cache_resource
def train_model(_ratings):
    book_counts = _ratings.groupby("book_id")["rating"].count()
    filtered    = _ratings[_ratings["book_id"].isin(book_counts[book_counts >= 10].index)]
    reader   = Reader(rating_scale=(1, 5))
    data     = Dataset.load_from_df(filtered[["user_id", "book_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    model    = KNNBasic(k=20, min_k=2,
                        sim_options={"name": "pearson_baseline", "user_based": False},
                        verbose=False)
    model.fit(trainset)
    return model

books, ratings = load_data()
model          = train_model(ratings)

book_counts   = ratings.groupby("book_id")["rating"].count()
popular_books = set(book_counts[book_counts >= 20].index)
title_of      = dict(zip(books["book_id"], books["title"]))
author_of     = dict(zip(books["book_id"], books["primary_author"]))
avg_rat       = dict(zip(books["book_id"], books["average_rating"]))
year_of       = dict(zip(books["book_id"], books["original_publication_year"]))
image_of      = dict(zip(books["book_id"], books["image_url"]))

# ── CF helper ─────────────────────────────────────────────────────────────────
def get_cf_recs(user_id, top_n=40):
    seen       = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])
    candidates = popular_books - seen
    scored = [
        {
            "book_id":    bid,
            "title":      title_of.get(bid, "Unknown"),
            "author":     author_of.get(bid, "Unknown"),
            "avg_rating": avg_rat.get(bid, "N/A"),
            "year":       int(year_of[bid]) if pd.notna(year_of.get(bid)) else "N/A",
            "predicted":  round(model.predict(user_id, bid).est, 3),
            "image_url":  image_of.get(bid, ""),
        }
        for bid in candidates
    ]
    scored.sort(key=lambda x: -x["predicted"])
    return scored[:top_n]

# ── LLM schema & re-ranking ───────────────────────────────────────────────────
class BookPick(BaseModel):
    title:  str = Field(description="Exact book title from the candidate list.")
    author: str = Field(description="Author name as provided in the candidate list.")
    reason: str = Field(description="A brief explanation of why this book fits the reader's preference.")

SYSTEM_INSTRUCTION = (
    "You are a personal book concierge. You are given a list of book candidates "
    "already selected by a recommendation engine for a specific reader. "
    "Re-rank them based on how well they match the reader's stated preference, "
    "and return a short explanation for each. Use exact titles and authors from the list. "
    "Do not suggest books outside the provided list."
)

SURPRISE_INSTRUCTION = (
    "You are a contrarian book recommender with a talent for finding hidden gems. "
    "You are given a list of candidates pre-selected for this reader. "
    "Do NOT pick the most obvious or highest-predicted titles. Instead, find the unexpected match — "
    "the book on this list that fits the reader's request in a surprising or non-obvious way. "
    "Be bold. Explain why this overlooked pick is actually perfect for them. "
    "Use exact titles and authors from the list only."
)

ROAST_INSTRUCTION = (
    "You are a devastatingly sarcastic book critic who has seen this reader's history and has opinions. "
    "Roast their taste mercilessly — be specific, be funny, be mean in a loving way. "
    "Then, despite yourself, give them genuinely good recommendations from the candidate list. "
    "Each reason should start with the roast and end with why the book is actually perfect for them. "
    "Think Gordon Ramsay, but for books. Use exact titles and authors from the list only."
)

def llm_rerank(candidates, user_preference, api_key, top_n=5, mode="normal"):
    from google import genai
    client  = genai.Client(api_key=api_key)
    pool    = candidates[:40]
    random.shuffle(pool)
    catalog = "\n".join(
        f"- \"{c['title']}\" by {c['author']} ({c['year']}, avg rating {c['avg_rating']})"
        for c in pool
    )
    prompt = (
        f"Reader's preference: {user_preference}\n\n"
        f"Book candidates (pre-selected by a recommendation engine):\n{catalog}\n\n"
        f"Return the top {min(top_n + 5, len(pool))} books re-ranked by fit to the reader's preference."
    )
    instruction = {"normal": SYSTEM_INSTRUCTION, "surprise": SURPRISE_INSTRUCTION, "roast": ROAST_INSTRUCTION}[mode]
    temperature  = {"normal": 0.2, "surprise": 1.4, "roast": 0.9}[mode]
    config = {
        "system_instruction": instruction,
        "response_mime_type": "application/json",
        "response_schema":    list[BookPick],
        "temperature":        temperature,
        "max_output_tokens":  4096,
    }
    if mode == "normal":
        config["seed"] = 6604
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=config,
    )
    parsed = response.parsed
    if not parsed:
        return None
    return parsed[:top_n]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 bookish")
    st.markdown("*Life's too short to DNF. Your perfect book is waiting.*")
    st.markdown("---")
    user_rating_counts = ratings.groupby("user_id")["rating"].count()
    all_users = sorted(user_rating_counts.index.tolist(), key=lambda x: int(x))
    user_labels = {u: f"Reader #{u} ({user_rating_counts[u]} ratings)" for u in all_users}
    user_id = st.selectbox("Select a reader profile", all_users,
                           format_func=lambda u: user_labels[u],
                           help="Each profile represents a real reader's history from the GoodReads dataset.")
    st.markdown("---")
    top_n = st.slider("Number of recommendations to show", min_value=3, max_value=20, value=5)
    st.markdown("---")
    st.subheader("Ask Bookish Settings")
    st.info("✨ Ask Bookish is powered by Gemini AI. To use it, you'll need a free Gemini API key — a password that connects the app to Google's AI. Get yours at [aistudio.google.com](https://aistudio.google.com) and paste it below.", icon=None)
    api_key = st.text_input("Gemini API key (optional)", type="password",
                            help="Optional — the app has a built-in key. Enter your own to use a personal quota.")
    if not api_key:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    st.caption("*Never share your API key with anyone or paste it into a site you don't trust. Bookish only uses it for this session and never stores it.*")
    st.markdown("---")
    st.caption("Built & designed by Sammi 🌸")

# ── Pre-fetch CF recs ─────────────────────────────────────────────────────────
with st.spinner("Loading recommendations..."):
    cf_recs = get_cf_recs(user_id, top_n=100)

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
    <h1 style='font-size:3.2rem; color:#000000; margin-bottom:0; font-family: "Cormorant Garamond", serif; font-weight:300; letter-spacing:0.02em;'>📖 bookish</h1>
    <p style='font-size:1rem; color:rgb(0,0,0) !important; font-style:italic; margin-top:4px; letter-spacing:0.01em; font-family:"Lora", serif; font-weight:400;'>
        Life's too short to DNF. Your perfect book is waiting.
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📖 For You", "✨ Ask Bookish", "ℹ️ How It Works"])

# ── Tab 1: CF Recommendations ─────────────────────────────────────────────────
with tab1:
    st.header("Your Top Picks")
    st.caption(f"Recommended for Reader #{user_id} based on GoodReads rating history")

    cols = st.columns(5)
    for i, rec in enumerate(cf_recs[:top_n]):
        with cols[i % 5]:
            if rec["image_url"]:
                st.image(rec["image_url"], use_container_width=True)
            short = rec["title"].split("(")[0].split(":")[0].strip()
            st.markdown(f"**{i+1}. {short[:40]}**")
            st.caption(f"{rec['author']} ({rec['year']}) · ⭐ {rec['avg_rating']}")

# ── Tab 2: LLM Personalization ────────────────────────────────────────────────
with tab2:
    st.markdown("<h1 style='font-size:2rem; margin-bottom:0;'>Tell Us What You're Looking For</h1>", unsafe_allow_html=True)
    st.caption(f"Personalizing recommendations for **Reader #{user_id}** · Switch readers in the sidebar.")
    st.markdown("<p style='font-size:1rem; color:var(--color-text-secondary); margin-top:8px;'>Describe your mood, a book you loved, or what you're looking for today — Bookish will re-rank your recommendations accordingly.</p>", unsafe_allow_html=True)

    st.markdown("""
<div style='font-size:0.95rem; color:#555; margin-top:12px; margin-bottom:4px;'>
<strong>Try things like...</strong> &nbsp;
"I just got dumped and need a book that loves me back" &nbsp;·&nbsp;
"A twisty mystery for a rainy Sunday" &nbsp;·&nbsp;
"Like Gone Girl but no blood" &nbsp;·&nbsp;
"A hockey romance that would make my grandma blush"
</div>
""", unsafe_allow_html=True)

    preference = st.text_area(
        "",
        placeholder="e.g. A breezy beach read that requires zero brain cells...",
        height=100,
    )

    mode = st.radio(
        "Mode",
        options=["normal", "roast", "surprise"],
        format_func=lambda m: {
            "normal":  "✓ Match me.",
            "roast":   "🔥 Roast me.",
            "surprise": "🎲 Surprise me.",
        }[m],
        captions=[
            "The right book for exactly what you asked.",
            "We've seen your history, and we have thoughts. Here's what you actually need.",
            "The best book you'd never pick yourself.",
        ],
        label_visibility="collapsed",
    )

    if mode == "roast":
        st.caption("⚠️ *Not for the faint of heart.*")

    rerank_btn = st.button("✨ Find My Next Read",
                           disabled=(not api_key or not preference),
                           use_container_width=True)
    if preference and not api_key:
        st.warning("Add your Gemini API key in the sidebar to find your next read.", icon="🔑")

    if rerank_btn:
        spinner_msg = {"normal": "Finding your perfect books...", "surprise": "Finding your hidden gem...", "roast": "Preparing your roast..."}[mode]
        with st.spinner(spinner_msg):
            try:
                picks = llm_rerank(cf_recs, preference, api_key, top_n=top_n, mode=mode)
                if not picks:
                    st.error("The AI didn't return any picks. Try again or switch modes.")
                else:
                    st.session_state["llm_picks"]      = picks
                    st.session_state["llm_preference"] = preference
                    st.session_state["llm_user"]       = user_id
                    st.session_state["llm_mode"]       = mode
            except Exception as e:
                st.error(f"Something went wrong: {e}")

    if "llm_picks" in st.session_state:
        saved_mode = st.session_state.get("llm_mode", "normal")
        if saved_mode == "surprise":
            st.markdown(f"### 🎲 Surprise picks for: *\"{st.session_state['llm_preference']}\"*")
            st.caption("Intentionally unexpected — the AI was told to skip the obvious and find the hidden gem.")
        elif saved_mode == "roast":
            st.markdown(f"### 🔥 Roast picks for: *\"{st.session_state['llm_preference']}\"*")
            st.caption("The AI has thoughts about your taste. Don't take it personally.")
        else:
            st.markdown(f"### Top picks for: *\"{st.session_state['llm_preference']}\"*")
        image_lookup = {r["title"]: r["image_url"] for r in cf_recs}

        for i, p in enumerate(st.session_state["llm_picks"], 1):
            col_img, col_text = st.columns([1, 5])
            with col_img:
                url = image_lookup.get(p.title, "")
                if url:
                    st.image(url, width=70)
            with col_text:
                st.markdown(
                    f"<div class='book-card'>"
                    f"<div class='book-title'>{i}. {p.title}</div>"
                    f"<div class='book-author'>{p.author}</div>"
                    f"<div class='book-reason'>{p.reason}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab3:
    st.header("How Bookish Works")

    st.markdown("""
    **Bookish** combines two layers of intelligence to give you recommendations that are both
    personally relevant and thoughtfully explained.

    ---

    #### 📊 Layer 1: Collaborative Filtering
    The foundation of Bookish is a technique called **collaborative filtering,** the same
    approach used by Netflix and Spotify. Collaborative filtering works by finding patterns in user ratings and makes recommendations accordingly.

    Bookish uses **item-based** collaborative filtering: instead of finding users like you,
    it finds books that are similar to the ones you've already rated highly. If readers who loved *Gone Girl* also loved *The Silent Patient*, Bookish learns that connection and uses it to find your next read.

    ---

    #### ✨ Layer 2: AI Personalization with Ask Bookish
    Once we have your top candidates, **Gemini AI** re-ranks them based on whatever you tell it.
    Tell Bookish exactly what you're looking for: a cozy escape after a stressful week, more books
    like the one you just finished, something to devour on a long flight, or set boundaries around
    topics you'd rather avoid. The more specific you are, the better your picks will be.

    Ask Bookish never makes up books — it only re-ranks and explains titles already picked for you during the collaborative filtering stage.

    Bookish offers three modes, each with a different personality and level of creative risk:

    - **Match Me** uses a low temperature, meaning the AI stays focused and predictable. It picks the most obvious fit for what you asked and explains why clearly.
    - **Surprise Me** uses a high temperature, meaning the AI is more creative and unpredictable. Think of it as turning up the randomness dial. It deliberately avoids the safe picks and finds something unexpected that still fits your mood.
    - **Roast Me** sits in the middle: creative enough to have a personality, grounded enough to still give you genuinely good picks. It delivers the same caliber of recommendations as Match Me, just with a lot more attitude.

    ---

    #### 📚 The Data
    Bookish is trained on a dataset of **9,964 books** and **164,728 ratings** from actual GoodReads users, so every recommendation is rooted in what real readers read and reviewed.
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Books in catalog", "9,964")
    col2.metric("Ratings analyzed", "164,728")
    col3.metric("Readers in dataset", "1,192")
