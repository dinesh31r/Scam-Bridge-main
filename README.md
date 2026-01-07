🗳️ Political Sentiment Analysis Hub (aka "Scam Bridge-Analytica")

This project is a full-stack, Python-based web application designed to analyze political sentiment from text data. It transforms raw, unstructured tweets into actionable insights using a custom-trained machine learning model and provides real-time contextual information through an integrated LLM chatbot.

The project is built using Streamlit, Scikit-learn, and the Google Gemini API.


Features

    📈 Real-Time Sentiment Analysis: A tool to classify the sentiment (Positive, Negative, or Neutral) of any text input in real-time using a trained Logistic Regression model.

    📁 Bulk CSV Analysis: A feature to upload an entire CSV file of tweets, analyze the sentiment for every row, and download the newly enriched dataset.

    🤖 AI-Powered Political Chatbot: An integrated chatbot (powered by the Google Gemini API) to answer factual, general-knowledge questions about politics, providing context to the sentiment analysis.

    📊 Model Performance Dashboard: A section displaying the model's performance metrics (Classification Report, Confusion Matrix) on a test set, providing transparency into its accuracy.

⚙️ How It Works: Architecture

This project uses a "Teacher-Student" ML pipeline and a modular application architecture.

1. The Machine Learning Pipeline

Because we started with no labeled data, we first created a pipeline to generate our own training set.

    "Teacher" Model (Auto-Labeling): The functions/label_data.py script uses a powerful Zero-Shot Classification model (from Hugging Face Transformers) to automatically label thousands of raw tweets as positive, negative, or neutral.

    "Student" Model (Final Classifier): The functions/model_trainer.py script trains a much faster, lightweight Logistic Regression model on the data labeled by the "teacher." This "student" model is the one deployed in the final app.

2. Application Architecture

    Frontend: A single-page Streamlit application (app.py).

    ML Backend: The trained Logistic Regression model (final_model.pkl) and the fitted TfidfVectorizer (vectorizer.pkl) are loaded into the app's memory using st.cache_resource for high-speed predictions.

    External API: The Google Gemini API is called for the chatbot feature, acting as an independent, external service.

🛠️ Technology Stack

    Frontend: Streamlit

    ML & Data Processing: Scikit-learn (Logistic Regression, TF-IDF), Pandas, NLTK

    Auto-Labeling (Teacher Model): Hugging Face transformers

    Chatbot: Google Gemini API (google-genai)

    Core: Python 3.9+

📂 Project Structure

🚀 How to Run This Project Locally

1. Prerequisites

    Python 3.9+

    A Google Gemini API Key.

2. Setup and Installation

    Clone the repository:

    Create and activate a virtual environment:

    Install dependencies:

    Set your Gemini API Key: (This key is only loaded in your current terminal session and is never saved in the code.)

3. Run the Project

You must first train the model, then run the app.

    Run the ML Pipeline (One-Time Setup):

        Place your raw, unlabeled tweet CSV in the data/ folder and name it political_tweets.csv.

        Step 1: Auto-label the data (The "Teacher")

        Step 2: Train the model (The "Student")

    (You should now have labeled_political_tweets.csv, final_model.pkl, and vectorizer.pkl)

    Run the Streamlit App: (Make sure you are in the same terminal where you set your API key)

The app will open in your browser at http://localhost:8501.

🌟 Future Improvements

    Microservice Architecture: Decouple the ML model from the Streamlit app by deploying it as a separate FastAPI service for true scalability.

    Improve Model: Replace Stemming with Lemmatization and implement N-grams (ngram_range=(1, 2)) to capture context like "not good."

    Chatbot RAG: Enhance the chatbot by implementing a Retrieval-Augmented Generation (RAG) pipeline to have it answer questions based on a trusted set of documents.
