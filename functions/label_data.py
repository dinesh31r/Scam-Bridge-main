import pandas as pd
from transformers import pipeline
import time
import os
import psutil
import gc
from tqdm import tqdm

# --- Configuration ---
RAW_DATA_PATH = os.path.join("data", "political_tweets.csv")
LABELED_DATA_PATH = os.path.join("data", "labeled_political_tweets.csv")
TWEET_COLUMN = 'Tweet' # Column name in the CSV file
CANDIDATE_LABELS = ['positive', 'negative', 'neutral']
BATCH_SIZE = 16  # Reduced batch size to prevent overheating
CHECKPOINT_INTERVAL = 100  # Save progress every 100 tweets

def label_data():
    """
    Loads raw political tweets and labels them using a Zero-Shot Classification model.
    """
    print(f"Loading raw data from: {RAW_DATA_PATH}")

    # Check if the raw data file exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}. Please place your CSV there.")
        return

    try:
        df = pd.read_csv(RAW_DATA_PATH)
        # Drop rows where the tweet is missing (NaN)
        df.dropna(subset=[TWEET_COLUMN], inplace=True)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Check if the required tweet column exists
    if TWEET_COLUMN not in df.columns:
        print(f"Error: Column '{TWEET_COLUMN}' not found in the CSV.")
        return

    tweets = df[TWEET_COLUMN].tolist()
    print(f"Successfully loaded {len(tweets)} tweets for labeling.")

    # Check for existing progress
    if os.path.exists(LABELED_DATA_PATH):
        print("Found existing progress, loading...")
        labeled_df = pd.read_csv(LABELED_DATA_PATH)
        already_labeled = set(labeled_df.index)
    else:
        already_labeled = set()
        labeled_df = pd.DataFrame()

    # --- Initialize a lighter sentiment analysis model ---
    print("Initializing sentiment analysis pipeline...")
    try:
        # Use a smaller, more efficient model
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # Use CPU to prevent overheating
        )
    except Exception as e:
        print(f"Error initializing transformer pipeline. Check 'torch' and 'transformers' installation. Error: {e}")
        return
        
    # Monitor system resources
    def get_system_stats():
        cpu_percent = psutil.cpu_percent()
        memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        return f"CPU: {cpu_percent}%, Memory: {memory:.1f}MB"
    
    # --- Perform Labeling ---
    print(f"Starting to label {len(tweets)} tweets...")
    labeled_results = []
    start_time = time.time()

    # Create progress bar
    pbar = tqdm(total=len(tweets), desc="Labeling tweets")
    
    try:
        for i in range(0, len(tweets), BATCH_SIZE):
            if i % 100 == 0:  # Check system stats periodically
                print(f"\nSystem stats: {get_system_stats()}")
                
            batch = tweets[i:i + BATCH_SIZE]
            
            # Skip already labeled tweets
            if i in already_labeled:
                pbar.update(len(batch))
                continue
                
            # Process smaller batches to reduce memory usage
            results = []
            for tweet in batch:
                result = classifier(tweet[:512])[0]  # Limit tweet length
                # Convert sentiment to our labels
                label = 'positive' if result['label'] == 'POSITIVE' else 'negative'
                results.append(label)
            
            labeled_results.extend(results)
            
            # Update progress
            pbar.update(len(batch))
            
            # Save checkpoint periodically
            if i % CHECKPOINT_INTERVAL == 0:
                temp_df = pd.DataFrame({
                    TWEET_COLUMN: tweets[:len(labeled_results)],
                    'sentiment': labeled_results
                })
                temp_df.to_csv(LABELED_DATA_PATH, index=False)
                
            # Force garbage collection
            gc.collect()
            
            # Add a small delay to prevent overheating
            time.sleep(0.1)
    except Exception as e:
        print(f"Error during labeling: {e}")
    
    # --- Save Results ---
    df['sentiment'] = labeled_results
    
    # Drop the rows that were labeled as 'neutral' if we wanted a binary classifier
    # But since we want ternary, we keep all labels.
    
    df.to_csv(LABELED_DATA_PATH, index=False)
    
    print("\n--- Labeling Complete ---")
    print(f"Total labeled tweets: {len(df)}")
    print(f"Results saved to: {LABELED_DATA_PATH}")
    print("\nNext step: Run the training script.")

if __name__ == "__main__":
    label_data()