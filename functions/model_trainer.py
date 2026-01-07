# Script to run the ML training pipeline

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preprocess import clean_and_stem # Import our cleaning function

# --- Configuration ---
LABELED_DATA_PATH = os.path.join("data", "labeled_political_tweets.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
TWEET_COLUMN = 'Tweet'
LABEL_COLUMN = 'sentiment'
TEST_SIZE = 0.2
RANDOM_STATE = 42

def train_and_save_model():
    """
    Executes the full ML pipeline: preprocessing, vectorization, training, and saving.
    """
    print("--- Starting Model Training Pipeline ---")

    # 1. Load Data
    try:
        df = pd.read_csv(LABELED_DATA_PATH)
        print(f"Loaded {len(df)} labeled tweets.")
    except FileNotFoundError:
        print(f"Error: Labeled data not found at {LABELED_DATA_PATH}. Run label_data.py first.")
        return
    
    # Basic data check
    if LABEL_COLUMN not in df.columns or TWEET_COLUMN not in df.columns:
        print(f"Error: Data must contain '{TWEET_COLUMN}' and '{LABEL_COLUMN}' columns.")
        return


    # 2. Preprocessing
    print("Applying text cleaning and stemming...")
    df['cleaned_tweet'] = df[TWEET_COLUMN].apply(clean_and_stem)

    # Drop rows where cleaning resulted in an empty string
    df = df[df['cleaned_tweet'].str.strip() != ''].reset_index(drop=True)

    # Drop rows with NaN in either cleaned_tweet or sentiment columns
    df = df.dropna(subset=['cleaned_tweet', LABEL_COLUMN]).reset_index(drop=True)
    print(f"Data remaining after cleaning and dropping NaNs: {len(df)} rows.")

    # 3. Split Data
    X = df['cleaned_tweet']
    y = df[LABEL_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # 4. Feature Extraction (Vectorization)
    print("Fitting TfidfVectorizer on training data...")
    # TfidfVectorizer is often better than CountVectorizer for document classification
    vectorizer = TfidfVectorizer(max_features=5000) # Use top 5000 unique terms
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Feature space size (Vocabulary): {X_train_vec.shape[1]}")

    # 5. Model Training (Logistic Regression - Fast and effective baseline)
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_vec, y_train)
    print("Model trained successfully.")

    # 6. Evaluation
    y_pred = model.predict(X_test_vec)
    print("\n--- Model Evaluation (Test Set) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save Artifacts
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {VECTORIZER_PATH}")
    
    print("\n--- Pipeline Complete. Ready for Streamlit (Phase 3) ---")


if __name__ == "__main__":
    train_and_save_model()