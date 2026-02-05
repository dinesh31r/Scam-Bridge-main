import os
import pickle
import streamlit as st
import pandas as pd
from textblob import TextBlob
from groq import Groq
from dotenv import load_dotenv
from functions.preprocess import clean_and_stem
import sqlite3
from datetime import datetime
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Scam Bridge Analytica",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load .env if present (e.g., GROQ_API_KEY)
load_dotenv()

# ----------------------------------
# Global UI Style (Space Grey Glassmorphism)
# ----------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #f4f7fb;
        --bg-2: #eef3fb;
        --panel: #ffffff;
        --panel-2: #f6f8fc;
        --border: rgba(100, 116, 139, 0.2);
        --text: #0f172a;
        --muted: #64748b;
        --accent: #5b76ff;
        --accent-2: #e8edff;
        --accent-3: #4f6dff;
        --success: #16a34a;
    }

    html, body, [class*="stApp"] {
        background: radial-gradient(1200px 500px at 70% -5%, rgba(91, 118, 255, 0.12), transparent 60%),
                    radial-gradient(1000px 600px at 20% 0%, rgba(16, 185, 129, 0.08), transparent 55%),
                    linear-gradient(180deg, #f8fbff 0%, #eef3fb 100%) fixed;
        color: var(--text);
        font-family: "Plus Jakarta Sans", "Manrope", "Segoe UI", Arial, sans-serif;
    }

    /* Main container */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 8px 0 24px rgba(15, 23, 42, 0.04);
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
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }
    .card {
        margin: 1rem 0 1.5rem 0;
    }

    /* Top bar */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.25rem;
        padding: 0.9rem 1.5rem;
        margin-bottom: 1.5rem;
        position: sticky;
        top: 0.5rem;
        z-index: 50;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .topbar .crumbs {
        font-size: 0.85rem;
        color: var(--muted);
        margin-bottom: 0.15rem;
    }
    .topbar .title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 0.2px;
        color: var(--text);
    }
    .top-actions {
        display: none;
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
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"],
    .stNumberInput input {
        background: var(--panel);
        color: var(--text);
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
        caret-color: var(--text);
    }
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder,
    .stNumberInput input::placeholder {
        color: rgba(100, 116, 139, 0.8);
    }

    /* Ensure labels and small text remain readable */
    label, .stMarkdown, .stCaption, .stSlider label, .stSelectbox label {
        color: var(--text);
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(180deg, #5b76ff 0%, #4f6dff 100%);
        color: #ffffff;
        border: 1px solid rgba(91, 118, 255, 0.7);
        border-radius: 12px;
        padding: 0.5rem 1rem;
        transition: transform 0.06s ease, box-shadow 0.2s ease;
    }
    .stButton button:hover {
        box-shadow: 0 10px 26px rgba(91, 118, 255, 0.25);
        transform: translateY(-1px);
    }

    /* Dataframes */
    .stDataFrame, .stTable {
        background: var(--panel);
        border-radius: 12px;
        border: 1px solid var(--border);
    }

    /* Divider */
    hr {
        border-color: var(--border);
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.25rem 0.2rem 0.75rem 0.2rem;
    }
    .sidebar-logo {
        width: 36px;
        height: 36px;
        border-radius: 12px;
        background: #0f172a;
        color: #ffffff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .sidebar-title {
        font-weight: 800;
        font-size: 1.2rem;
    }
    .sidebar-section {
        background: #eef2ff;
        border: 1px solid rgba(91, 118, 255, 0.2);
        border-radius: 10px;
        padding: 0.5rem 0.75rem;
        font-size: 0.85rem;
        color: #4f6dff;
        margin-bottom: 0.75rem;
    }
    .sidebar-nav {
        display: grid;
        gap: 0.35rem;
        margin-bottom: 1rem;
    }
    .sidebar-nav .nav-item {
        padding: 0.45rem 0.6rem;
        border-radius: 8px;
        color: var(--muted);
        font-size: 0.9rem;
    }
    .sidebar-nav .nav-item.active {
        color: var(--text);
        background: #f1f5ff;
        border: 1px solid rgba(91, 118, 255, 0.15);
    }

    .section-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.5rem;
    }

    .chat-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.9rem 1.1rem;
        margin: 0.6rem 0 0.9rem 0;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }
    .chat-meta {
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 0.4rem;
    }
    .chat-divider {
        height: 1px;
        background: rgba(148, 163, 184, 0.25);
        margin: 0.6rem 0;
        border: none;
    }

    /* Metric readability */
    .stMetric label, .stMetric div {
        color: var(--text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------
# Groq API Setup
# ----------------------------------
GROQ_API_KEY_ENV = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY_ENV = os.getenv("TAVILY_API_KEY")

def get_active_groq_key():
    override = st.session_state.get("groq_api_key_override", "").strip()
    return override or GROQ_API_KEY_ENV

def get_groq_client():
    api_key = get_active_groq_key()
    if not api_key:
        return None
    return Groq(api_key=api_key)

def get_tavily_client():
    if TavilyClient is None:
        return None
    if not TAVILY_API_KEY_ENV:
        return None
    return TavilyClient(api_key=TAVILY_API_KEY_ENV)

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


def retrieve_web_context(query, max_results=3):
    client = get_tavily_client()
    if not client:
        return [], ""
    try:
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
            include_images=False
        )
    except Exception:
        return [], ""

    results = response.get("results", []) if isinstance(response, dict) else []
    sources = []
    context_parts = []
    for r in results:
        title = r.get("title") or "Untitled Source"
        content = (r.get("content") or "").strip()
        if not content:
            continue
        sources.append(f"web: {title}")
        context_parts.append(f"[{title}]\n{content[:800]}")
    return sources, "\n\n".join(context_parts)


def get_llm_response_with_rag(query, use_rag, top_k, use_cache, use_web, web_k):
    cache = st.session_state.setdefault("llm_cache", {})
    cache_key = f"{query}|rag={use_rag}|k={top_k}|web={use_web}|wk={web_k}"
    if use_cache and cache_key in cache:
        return cache[cache_key]["response"], cache[cache_key]["sources"]

    sources = []
    context_block = ""
    if use_rag:
        docs = load_kb_documents()
        vectorizer, matrix = build_kb_index(docs)
        sources, context_block = retrieve_context(query, docs, vectorizer, matrix, top_k=top_k)

    if use_web:
        web_sources, web_context = retrieve_web_context(query, max_results=web_k)
        sources.extend(web_sources)
        if web_context:
            context_block = f"{context_block}\n\n{web_context}".strip()

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
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-logo">SB</div>
          <div class="sidebar-title">Scam-Bridge Analytica</div>
        </div>
        <div class="sidebar-section">Workspace</div>
        <div class="sidebar-nav">
          <div class="nav-item active">Overview</div>
        </div>
        """,
        unsafe_allow_html=True
    )
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
    use_web = st.toggle("Use Web Search (Tavily)", value=False)
    web_k = st.slider("Web Results", min_value=1, max_value=5, value=3)
    use_cache = st.toggle("Cache Answers", value=True)
    rate_limit = st.number_input("Max Requests / Minute", min_value=1, max_value=60, value=10, step=1)

    show_debug = st.toggle("Show Debug Info", value=False)
    if show_debug:
        st.json({
            "GROQ_API_KEY_FOUND": bool(get_active_groq_key()),
            "TAVILY_API_KEY_FOUND": bool(TAVILY_API_KEY_ENV),
            "CHAT_HISTORY_LENGTH": len(st.session_state.chat_history),
            "RAG_ENABLED": use_rag,
            "WEB_SEARCH_ENABLED": use_web,
            "KB_DOC_COUNT": len(load_kb_documents())
        })

# ----------------------------------
# MAIN UI
# ----------------------------------
st.markdown(
    """
    <div class="glass topbar">
      <div>
        <div class="crumbs">Pages / Overview</div>
        <div class="title">Overview</div>
      </div>
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
# 1 + 2. OVERVIEW CARDS (FULL WIDTH)
# ======================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
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

st.markdown('<div class="section-card">', unsafe_allow_html=True)
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

            st.markdown("### Results Table")
            st.dataframe(df, use_container_width=True, height=520)

            st.markdown("### Summary")
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2], gap="large")
            with col1:
                st.metric("Rows Analyzed", len(df))
                st.metric(score_label, round(float(pd.Series(scores).mean()), 3) if scores else 0.0)
            with col2:
                sentiment_counts = pd.Series(sentiments).value_counts()
                st.bar_chart(sentiment_counts, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### Export")
            st.download_button(
                "Download Result CSV",
                df.to_csv(index=False),
                "sentiment_output.csv",
                "text/csv"
            )

    except Exception:
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
if use_web and (TavilyClient is None or not TAVILY_API_KEY_ENV):
    st.info("Web search is enabled, but Tavily is not configured. Add TAVILY_API_KEY to your .env.")

# Display Chat History
for chat in st.session_state.chat_history:
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    st.markdown(f"🔴 **User:** {chat['user']}")
    st.markdown(f"🟡 **Bot:** {chat['bot']}")
    st.markdown(
        f'<div class="chat-meta">🟢 <strong>Sentiment:</strong> {chat["sentiment"]} | '
        f'<strong>Score:</strong> {chat["score"]}</div>',
        unsafe_allow_html=True
    )
    if chat.get("sources"):
        st.markdown('<hr class="chat-divider" />', unsafe_allow_html=True)
        st.markdown("**Sources:**")
        for src in chat["sources"]:
            st.markdown(f"- {src}")
    st.markdown("</div>", unsafe_allow_html=True)

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
                use_cache=use_cache,
                use_web=use_web,
                web_k=web_k
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
