# Contains cleaning and tokenization functions

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK stopwords are downloaded (only runs if they are missing)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_and_stem(text):
    """
    Cleans a single text string by removing noise, converting to lowercase,
    removing stopwords, and applying stemming.
    """
    if pd.isna(text) or text is None:
        return ""
        
    text = str(text)

    # 1. Remove URLs (http/https)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    
    # 2. Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 3. Remove hashtags (#) - keep the text
    text = re.sub(r'#', '', text) 
    
    # 4. Remove special characters and punctuation
    text = re.sub(r'[^A-Za-z\s]', '', text)

    
    # 5. Convert to lowercase
    text = text.lower()
    
    # 6. Tokenize, remove stopwords, and apply stemming
    words = text.split()
    # Apply stemming and filter stopwords in one go
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    # 7. Join back into a single string
    return " ".join(words)

# Note: This file doesn't require importing pandas, but we added the pandas.isna check for robustness.
import pandas as pd
