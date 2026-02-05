# 🗳️ Political Sentiment Analysis Hub (Scam Bridge Analytica)

A Streamlit web app for political sentiment analysis and contextual Q&A. It supports **two sentiment engines** (TextBlob and Logistic Regression), **bulk CSV analysis**, and a **Groq-powered chatbot** with optional **RAG (Retrieval-Augmented Generation)** over local knowledge-base files.

---

## ✨ Features

- **Real-Time Sentiment Analysis**
  - Choose between **TextBlob** (polarity score) and **Logistic Regression** (confidence score).

- **Bulk CSV Analysis**
  - Upload a CSV, select a text column, and download labeled output.

- **LLM Chatbot (Groq)**
  - Ask political questions; responses can be grounded in your local knowledge base.

- **RAG + Citations**
  - Sources are shown per response based on matching knowledge-base documents.

- **Local Chat History**
  - Stores up to 200 chats in a local SQLite database.

- **Config Panel (Sidebar)**
  - Groq API key override, cache toggle, RAG toggle, rate-limit control.

---

## 🧱 Tech Stack

- **Frontend**: Streamlit
- **ML**: Scikit-learn (Logistic Regression, TF-IDF), Pandas, NLTK
- **Chatbot**: Groq (LLM API)
- **RAG Index**: TF-IDF + cosine similarity
- **Storage**: SQLite (chat history)
- **Python**: 3.9+

---

## 📂 Project Structure (Key Files)

- `app.py` — Streamlit UI + app logic
- `functions/preprocess.py` — text cleaning
- `functions/model_trainer.py` — trains the Logistic Regression model
- `models/` — model artifacts (`final_model.pkl`, `vectorizer.pkl`)
- `data/` — CSV datasets + chat DB
- `data/knowledge_base/` — RAG source files (`.txt`, `.md`)

---

## 🚀 Quick Start

### 1) Create & activate venv
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) (Optional) Set Groq API Key
**Option A: Use a .env file (recommended)**
1. Create `.env` from the example:
```bash
cp .env.example .env
```
2. Put your key in `.env`:
```bash
GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

**Option B: Export in your shell**
```bash
export GROQ_API_KEY=your_key_here
```

### 4) Run the app
```bash
streamlit run app.py
```

App opens at: `http://localhost:8501`

---

## 🧠 Sentiment Engines

### TextBlob (Polarity)
- Uses lexicon-based scoring.
- Outputs **Polarity Score** in range `[-1, 1]`.

### Logistic Regression (Confidence)
- Uses a trained model and TF-IDF vectorizer.
- Outputs **Confidence** in range `[0, 1]`.

If you select Logistic Regression and artifacts are missing, you must train the model first (below).

---

## 🧪 Train the Logistic Regression Model

You must train the model once if `models/final_model.pkl` and `models/vectorizer.pkl` do not exist.

### Required input
Place your **raw tweets CSV** here:
```
data/political_tweets.csv
```

### 1) Auto-label tweets (Teacher Model)
```bash
python functions/label_data.py
```

### 2) Train the student model
```bash
python functions/model_trainer.py
```

This generates:
- `data/labeled_political_tweets.csv`
- `models/final_model.pkl`
- `models/vectorizer.pkl`

---

## 📚 RAG Knowledge Base

Add `.txt` or `.md` files here:
```
data/knowledge_base/
```

Example:
```bash
mkdir -p data/knowledge_base
printf "ECI election FAQ..." > data/knowledge_base/elections.txt
```

When **Use Knowledge Base (RAG)** is enabled in the sidebar:
- The app indexes your documents
- Retrieves top‑K relevant sources
- Passes them to Groq
- Displays source file names under each response

---

## ⚙️ Sidebar Controls

- **Groq API Key override** (session only)
- **Use Knowledge Base (RAG)**
- **Top Sources** (Top‑K docs)
- **Cache Answers**
- **Max Requests / Minute** (rate limiting)
- **Clear Chat History**

---

## 🗃️ Local Chat History

Chats are stored in:
```
data/chat_history.db
```

Up to the latest **200 chats** are retained. You can clear history from the sidebar.

---

## ✅ Tips & Troubleshooting

- **Missing TextBlob**:
  ```bash
  pip install textblob
  ```

- **Missing Groq**:
  ```bash
  pip install groq
  ```

- **Model not found**:
  Run the model training steps above.

- **RAG not working**:
  Ensure `data/knowledge_base/` has `.txt` or `.md` files.

---

## 🌟 Future Improvements (Ideas)

- Use embeddings-based retrieval (FAISS / sentence-transformers)
- Add citation links with snippet previews
- Add admin UI for knowledge-base uploads
- Deploy as a microservice (FastAPI + Streamlit frontend)

---

## 📄 License

Specify your license here (MIT / Apache / etc.)
