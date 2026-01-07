import os
import streamlit as st
import pandas as pd
from textblob import TextBlob
from google import genai

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Scam Bridge Analytica",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------
# Gemini API Setup (2.5)
# ----------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ----------------------------------
# Session State
# ----------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------------
# Helper Functions
# ----------------------------------
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


def call_gemini_llm(prompt):
    if not gemini_client:
        return "⚠️ Gemini chatbot disabled (API key not found)."

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception:
        return "⚠️ Gemini LLM temporarily unavailable."


def add_to_history(user_msg, bot_msg, sentiment, score):
    st.session_state.chat_history.append({
        "user": user_msg,
        "bot": bot_msg,
        "sentiment": sentiment,
        "score": score
    })

    # Keep last 50 chats
    st.session_state.chat_history = st.session_state.chat_history[-50:]

# ----------------------------------
# Sidebar Debug
# ----------------------------------
with st.sidebar:
    st.markdown("### 🔧 Debug Info")
    st.json({
        "GEMINI_API_KEY_FOUND": bool(GEMINI_API_KEY),
        "CHAT_HISTORY_LENGTH": len(st.session_state.chat_history)
    })

# ----------------------------------
# MAIN UI
# ----------------------------------
st.title("🛡️ Scam Bridge Analytica")

# ======================================================
# 1. TEXT SENTIMENT ANALYSIS
# ======================================================
st.markdown("## 1. Text Sentiment Analysis")

text_input = st.text_area("Enter text to analyze sentiment")

if st.button("Analyze Sentiment"):
    if text_input.strip():
        sentiment, score = analyze_sentiment_with_score(text_input)
        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Polarity Score:** {score}")

st.divider()

# ======================================================
# 2. BULK CSV SENTIMENT ANALYSIS
# ======================================================
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

            for text in df[text_column].astype(str):
                label, polarity = analyze_sentiment_with_score(text)
                sentiments.append(label)
                scores.append(polarity)

            df["Sentiment"] = sentiments
            df["Polarity_Score"] = scores

            st.dataframe(df.head(20))
            st.download_button(
                "Download Result CSV",
                df.to_csv(index=False),
                "sentiment_output.csv",
                "text/csv"
            )

    except Exception as e:
        st.error("Error reading CSV file")

st.divider()

# ======================================================
# 3. POLITICAL INFORMATION CHATBOT (GEMINI 2.5)
# ======================================================
st.markdown("## 3. Political Information Chatbot (LLM-Powered)")

# Display Chat History
for chat in st.session_state.chat_history:
    st.markdown(f"🔴 **User:** {chat['user']}")
    st.markdown(f"🟡 **Bot:** {chat['bot']}")
    st.markdown(f"🟢 **Sentiment:** {chat['sentiment']} | **Score:** {chat['score']}")
    st.markdown("---")

user_query = st.text_input("Ask a political question:")

if st.button("Send"):
    if user_query.strip():
        sentiment, score = analyze_sentiment_with_score(user_query)
        reply = call_gemini_llm(
            f"You are a political information assistant. Answer factually:\n{user_query}"
        )
        add_to_history(user_query, reply, sentiment, score)
        st.rerun()

st.divider()

# ======================================================
# 4. MODEL PERFORMANCE (STATIC)
# ======================================================
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
