"""
Trains the fake news detection model and saves it to the models/ directory.
"""

import numpy as np
import pandas as pd
import re
import os
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import nltk

def quick_preprocess(content):
    """Fast preprocessing without stemming - 10x faster"""
    content = re.sub(r'[^a-zA-Z\s]', '', str(content).lower())
    return content

def train_model():
    """Train the fake news detection model"""
    
    try:
        # Load data
        data_dir = "./data"
        true_path = os.path.join(data_dir, "True.csv")
        fake_path = os.path.join(data_dir, "Fake.csv")
        
        if not os.path.exists(true_path) or not os.path.exists(fake_path):
            print("❌ Error: Dataset files not found! Please ensure True.csv and Fake.csv are in the 'data' folder.")
            return

        print("✅ Dataset found. Starting training process...")
        
        # Read datasets
        true_df = pd.read_csv(true_path)
        fake_df = pd.read_csv(fake_path)
        
        # Add labels
        true_df['label'] = 0  # Real news
        fake_df['label'] = 1  # Fake news
        
        # Combine datasets
        df = pd.concat([true_df, fake_df], axis=0)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Fill missing values
        df = df.fillna('')
        
        # SPEED OPTIMIZATION: Use only 50% of data for faster training
        df = df.sample(frac=0.5, random_state=42).reset_index(drop=True)
        
        # Create content column
        df['content'] = df['title'] + ' ' + df['text']
        
        # Apply FAST preprocessing (no stemming - 10x faster!)
        df['content'] = df['content'].apply(quick_preprocess)
        
        # Prepare features and labels
        X = df['content'].values
        y = df['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Vectorization with ULTRA-FAST parameters
        vectorizer = TfidfVectorizer(
            max_features=2000,          # Reduced from 5000
            min_df=5,                   # More aggressive
            max_df=0.7,                 # More aggressive
            ngram_range=(1, 1),         # Only unigrams (faster)
            stop_words='english'        # Built-in (faster than NLTK)
        )
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        
        # Train model with ULTRA-FAST parameters
        model = LogisticRegression(
            max_iter=100,               # Reduced from 500
            solver='saga',
            random_state=42,
            n_jobs=-1,
            C=1.0,
            tol=1e-3                    # Less strict convergence
        )
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        # Save model and vectorizer
        os.makedirs('models', exist_ok=True)
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print(f"✅ Model trained successfully!")
        print(f"📊 Training Accuracy: {train_accuracy*100:.2f}%")
        print(f"📊 Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"💾 Model saved to 'models' folder")
    
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")

if __name__ == '__main__':
    # Download NLTK data if not present
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    train_model()
