import os
import pickle
import streamlit as st
import pandas as pd
from textblob import TextBlob
from groq import Groq
from functions.preprocess import clean_and_stem
import sqlite3
from datetime import datetime
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Scam Bridge Analytica",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------
# Global UI Style (Space Grey Glassmorphism)
# ----------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #f7f7f7;
        --bg-2: #ffffff;
        --glass: rgba(255, 255, 255, 0.95);
        --glass-2: rgba(255, 255, 255, 1);
        --border: rgba(0, 0, 0, 0.08);
        --text: #111111;
        --muted: #6b6f76;
        --accent: #0b5fff;
        --accent-2: #eef1f6;
        --accent-3: #1e6bff;
    }

    html, body, [class*="stApp"] {
        background: linear-gradient(180deg, #ffffff 0%, #f3f4f6 100%) fixed;
        color: var(--text);
        font-family: "SamsungOne", "Samsung One", "SamsungSharpSans", "Segoe UI", Arial, sans-serif;
    }

    /* Main container */
    .block-container {
        padding-top: 2.25rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f7f7f7 100%);
        border-right: 1px solid var(--border);
        box-shadow: none;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text);
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: var(--muted);
    }

    /* Glass panels */
    .glass {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
    }
    .card {
        margin: 1rem 0 1.25rem 0;
    }

    /* Title bar */
    .hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
    }
    .hero .title {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: 0.2px;
    }
    .hero .pill {
        background: rgba(11, 95, 255, 0.1);
        border: 1px solid rgba(11, 95, 255, 0.35);
        color: #0b5fff;
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        letter-spacing: 0.2px;
    }

    /* Sidebar nav emphasis */
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        justify-content: flex-start;
    }

    /* Headings */
    h1, h2, h3 {
        color: var(--text);
        letter-spacing: 0.2px;
    }
    .stCaption, .stMarkdown p, .stMarkdown li {
        color: var(--muted);
    }

    /* Inputs */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background: var(--glass-2);
        color: var(--text);
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.04);
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(180deg, #0b5fff 0%, #1e6bff 100%);
        color: #ffffff;
        border: 1px solid rgba(11, 95, 255, 0.7);
        border-radius: 12px;
        padding: 0.5rem 1rem;
        transition: transform 0.06s ease, box-shadow 0.2s ease;
    }
    .stButton button:hover {
        box-shadow: 0 8px 20px rgba(11, 95, 255, 0.25);
        transform: translateY(-1px);
    }

    /* Dataframes */
    .stDataFrame, .stTable {
        background: var(--glass-2);
        border-radius: 12px;
        border: 1px solid var(--border);
    }

    /* Divider */
    hr {
        border-color: var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------
# Groq API Setup
# ----------------------------------
GROQ_API_KEY_ENV = os.getenv("GROQ_API_KEY")

def get_active_groq_key():
    override = st.session_state.get("groq_api_key_override", "").strip()
    return override or GROQ_API_KEY_ENV

def get_groq_client():
    api_key = get_active_groq_key()
    if not api_key:
        return None
    return Groq(api_key=api_key)

# ----------------------------------
# Session State
# ----------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_loaded" not in st.session_state:
    st.session_state.chat_loaded = False

# ----------------------------------
# Local DB (SQLite) for Chat History
# ----------------------------------
DB_PATH = os.path.join("data", "chat_history.db")
KB_DIR = os.path.join("data", "knowledge_base")

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                user_msg TEXT NOT NULL,
                bot_msg TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                score REAL NOT NULL,
                sources TEXT
            )
            """
        )
        # Add sources column if upgrading from older schema
        cols = [row[1] for row in conn.execute("PRAGMA table_info(chat_history)").fetchall()]
        if "sources" not in cols:
            conn.execute("ALTER TABLE chat_history ADD COLUMN sources TEXT")
        conn.commit()

def load_recent_chats(limit=200):
    init_db()
    with get_db_connection() as conn:
        cur = conn.execute(
            """
            SELECT user_msg, bot_msg, sentiment, score, sources
            FROM chat_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    # Reverse to show oldest first in UI
    return list(reversed([
        {
            "user": r[0],
            "bot": r[1],
            "sentiment": r[2],
            "score": r[3],
            "sources": json.loads(r[4]) if r[4] else []
        }
        for r in rows
    ]))

def save_chat_to_db(user_msg, bot_msg, sentiment, score, sources, keep=200):
    init_db()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_history (created_at, user_msg, bot_msg, sentiment, score, sources)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (datetime.utcnow().isoformat(), user_msg, bot_msg, sentiment, score, json.dumps(sources)),
        )
        # Keep only the most recent `keep` records
        conn.execute(
            """
            DELETE FROM chat_history
            WHERE id NOT IN (
                SELECT id FROM chat_history
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (keep,),
        )
        conn.commit()

# ----------------------------------
# Helper Functions
# ----------------------------------
@st.cache_resource
def load_lr_artifacts():
    model_path = os.path.join("models", "final_model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer, None
    except FileNotFoundError:
        return None, None, "Logistic Regression artifacts not found. Run the training pipeline to create models/final_model.pkl and models/vectorizer.pkl."
    except Exception as exc:
        return None, None, f"Failed to load Logistic Regression artifacts: {exc}"


def analyze_sentiment_with_score(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)

    if polarity > 0:
        label = "Positive 😊"
    elif polarity < 0:
        label = "Negative 😠"
    else:
        label = "Neutral 😐"

    return label, polarity


def analyze_sentiment_lr(text, model, vectorizer):
    cleaned = clean_and_stem(text)
    if not cleaned.strip():
        return "Neutral 😐", 0.0

    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)
        confidence = float(proba.max())
    else:
        confidence = 1.0

    if prediction == "positive":
        label = "Positive 😊"
    elif prediction == "negative":
        label = "Negative 😠"
    else:
        label = "Neutral 😐"

    return label, round(confidence, 3)


def call_groq_llm(prompt):
    return call_groq_llm_with_system(
        "You are a factual, neutral political information assistant. Answer briefly and factually.",
        prompt
    )




def add_to_history(user_msg, bot_msg, sentiment, score, sources=None):
    st.session_state.chat_history.append({
        "user": user_msg,
        "bot": bot_msg,
        "sentiment": sentiment,
        "score": score,
        "sources": sources or []
    })

    # Keep last 200 chats in session
    st.session_state.chat_history = st.session_state.chat_history[-200:]
    # Persist to local DB
    save_chat_to_db(user_msg, bot_msg, sentiment, score, sources or [], keep=200)


def rate_limit_ok(max_per_minute):
    now = datetime.utcnow().timestamp()
    timestamps = st.session_state.setdefault("rate_limit_timestamps", [])
    timestamps = [t for t in timestamps if now - t < 60]
    if len(timestamps) >= max_per_minute:
        st.session_state.rate_limit_timestamps = timestamps
        return False
    timestamps.append(now)
    st.session_state.rate_limit_timestamps = timestamps
    return True


@st.cache_data
def load_kb_documents():
    if not os.path.isdir(KB_DIR):
        return []

    docs = []
    for name in sorted(os.listdir(KB_DIR)):
        if not name.lower().endswith((".txt", ".md")):
            continue
        path = os.path.join(KB_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                docs.append({"id": name, "text": text})
        except Exception:
            continue
    return docs


@st.cache_data
def build_kb_index(docs):
    if not docs:
        return None, None
    texts = [d["text"] for d in docs]
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve_context(query, docs, vectorizer, matrix, top_k=3):
    if not docs or vectorizer is None or matrix is None:
        return [], ""
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    sources = []
    context_parts = []
    for idx in top_idx:
        doc = docs[idx]
        sources.append(doc["id"])
        snippet = doc["text"][:800]
        context_parts.append(f"[{doc['id']}]\n{snippet}")
    return sources, "\n\n".join(context_parts)


def get_llm_response_with_rag(query, use_rag, top_k, use_cache):
    cache = st.session_state.setdefault("llm_cache", {})
    cache_key = f"{query}|rag={use_rag}|k={top_k}"
    if use_cache and cache_key in cache:
        return cache[cache_key]["response"], cache[cache_key]["sources"]

    sources = []
    context_block = ""
    if use_rag:
        docs = load_kb_documents()
        vectorizer, matrix = build_kb_index(docs)
        sources, context_block = retrieve_context(query, docs, vectorizer, matrix, top_k=top_k)

    system_prompt = (
        "You are a factual, neutral political information assistant. "
        "Answer briefly and factually. If context is provided, use it."
    )

    if use_rag and context_block:
        prompt = (
            "Use only the context below to answer. If the answer isn't in the context, say you don't know.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query}"
        )
    else:
        prompt = query

    response = call_groq_llm_with_system(system_prompt, prompt)
    if use_cache:
        cache[cache_key] = {"response": response, "sources": sources}
    return response, sources


def call_groq_llm_with_system(system_prompt, prompt):
    client = get_groq_client()
    if not client:
        return "⚠️ Groq chatbot disabled (API key not found)."

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "⚠️ Groq LLM temporarily unavailable."

# ----------------------------------
# Sidebar Debug (optional)
# ----------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.text_input(
        "Groq API Key (optional override)",
        type="password",
        key="groq_api_key_override",
        help="If set, this key overrides the environment variable for this session only."
    )

    if st.button("Clear Chat History"):
        init_db()
        with get_db_connection() as conn:
            conn.execute("DELETE FROM chat_history")
            conn.commit()
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

    st.markdown("### 📚 Chat Settings")
    use_rag = st.toggle("Use Knowledge Base (RAG)", value=True)
    top_k = st.slider("Top Sources", min_value=1, max_value=5, value=3)
    use_cache = st.toggle("Cache Answers", value=True)
    rate_limit = st.number_input("Max Requests / Minute", min_value=1, max_value=60, value=10, step=1)

    show_debug = st.toggle("Show Debug Info", value=False)
    if show_debug:
        st.json({
            "GROQ_API_KEY_FOUND": bool(get_active_groq_key()),
            "CHAT_HISTORY_LENGTH": len(st.session_state.chat_history),
            "RAG_ENABLED": use_rag,
            "KB_DOC_COUNT": len(load_kb_documents())
        })

# ----------------------------------
# MAIN UI
# ----------------------------------
st.markdown(
    """
    <div class="glass hero">
      <div class="title">Scam Bridge Analytica</div>
      <div class="pill">Sentiment & RAG Dashboard</div>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Choose a sentiment engine: TextBlob (lexicon-based) or Logistic Regression (trained model).")

model_choice = st.selectbox(
    "Sentiment Engine",
    ["TextBlob", "Logistic Regression"],
    index=0,
    help="TextBlob returns a polarity score (-1 to 1). Logistic Regression returns a confidence score (0 to 1)."
)
score_label = "Polarity Score" if model_choice == "TextBlob" else "Confidence"
st.caption("TextBlob uses a lexicon-based polarity score, while Logistic Regression uses a trained model confidence.")

lr_model, lr_vectorizer, lr_error = load_lr_artifacts()
if model_choice == "Logistic Regression" and lr_error:
    st.warning(lr_error)

# Load recent chat history from local DB (once per session)
if not st.session_state.chat_loaded:
    st.session_state.chat_history = load_recent_chats(limit=200)
    st.session_state.chat_loaded = True

# ======================================================
# 1. TEXT SENTIMENT ANALYSIS
# ======================================================
st.markdown('<div class="glass card">', unsafe_allow_html=True)
st.markdown("## 1. Text Sentiment Analysis")

text_input = st.text_area("Enter text to analyze sentiment")

if st.button("Analyze Sentiment"):
    if text_input.strip():
        if model_choice == "TextBlob":
            sentiment, score = analyze_sentiment_with_score(text_input)
            st.success(f"**Sentiment:** {sentiment}")
            st.info(f"**{score_label}:** {score}")
        else:
            if lr_error:
                st.error(lr_error)
            else:
                sentiment, score = analyze_sentiment_lr(
                    text_input,
                    lr_model,
                    lr_vectorizer
                )
                st.success(f"**Sentiment:** {sentiment}")
                st.info(f"**{score_label}:** {score}")

st.divider()
st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# 2. BULK CSV SENTIMENT ANALYSIS
# ======================================================
st.markdown('<div class="glass card">', unsafe_allow_html=True)
st.markdown("## 2. Bulk CSV Sentiment Analysis")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding="latin-1")
        st.success(f"Loaded {len(df)} rows successfully")

        text_column = st.selectbox(
            "Select text column for sentiment analysis",
            df.columns
        )

        if st.button("Analyze CSV Sentiment"):
            sentiments = []
            scores = []

            if model_choice == "TextBlob":
                for text in df[text_column].astype(str):
                    label, polarity = analyze_sentiment_with_score(text)
                    sentiments.append(label)
                    scores.append(polarity)
                df["Sentiment"] = sentiments
                df[score_label.replace(" ", "_")] = scores
            else:
                if lr_error:
                    st.error(lr_error)
                    st.stop()
                for text in df[text_column].astype(str):
                    label, confidence = analyze_sentiment_lr(
                        text,
                        lr_model,
                        lr_vectorizer
                    )
                    sentiments.append(label)
                    scores.append(confidence)
                df["Sentiment"] = sentiments
                df[score_label.replace(" ", "_")] = scores

            st.dataframe(df.head(20))
            st.markdown("### Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows Analyzed", len(df))
                st.metric(score_label, round(float(pd.Series(scores).mean()), 3) if scores else 0.0)
            with col2:
                sentiment_counts = pd.Series(sentiments).value_counts()
                st.bar_chart(sentiment_counts)

            st.download_button(
                "Download Result CSV",
                df.to_csv(index=False),
                "sentiment_output.csv",
                "text/csv"
            )

    except Exception as e:
        st.error("Error reading CSV file")

st.divider()
st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# 3. POLITICAL INFORMATION CHATBOT (GEMINI 2.5)
# ======================================================
st.markdown('<div class="glass card">', unsafe_allow_html=True)
st.markdown("## 3. Political Information Chatbot (LLM-Powered)")

# RAG status hint
if use_rag and not load_kb_documents():
    st.info("Knowledge base is empty. Add .txt or .md files to data/knowledge_base to enable RAG sources.")

# Display Chat History
for chat in st.session_state.chat_history:
    st.markdown(f"🔴 **User:** {chat['user']}")
    st.markdown(f"🟡 **Bot:** {chat['bot']}")
    st.markdown(f"🟢 **Sentiment:** {chat['sentiment']} | **Score:** {chat['score']}")
    if chat.get("sources"):
        st.markdown("**Sources:**")
        for src in chat["sources"]:
            st.markdown(f"- {src}")
    st.markdown("---")

user_query = st.text_input("Ask a political question:")

if st.button("Send"):
    if user_query.strip():
        if not rate_limit_ok(rate_limit):
            st.error("Rate limit exceeded. Please wait a minute and try again.")
        else:
            sentiment, score = analyze_sentiment_with_score(user_query)
            reply, sources = get_llm_response_with_rag(
                user_query,
                use_rag=use_rag,
                top_k=top_k,
                use_cache=use_cache
            )
            add_to_history(user_query, reply, sentiment, score, sources=sources)
            st.rerun()

st.divider()
st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# 4. MODEL PERFORMANCE (STATIC)
# ======================================================
st.markdown('<div class="glass card">', unsafe_allow_html=True)
st.markdown("## 4. Model Performance")

st.markdown("**Accuracy:** `0.836`")

st.code("""
precision    recall  f1-score   support

negative     0.84      0.96      0.89       359
positive     0.83      0.52      0.64       141

accuracy                         0.84       500
macro avg    0.83      0.74      0.77       500
weighted avg 0.84      0.84      0.82       500
""")
st.markdown("</div>", unsafe_allow_html=True)
