"""
Fast Training Script for Fake News Detector
Optimized for speed with reduced training time
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk

# Download stopwords if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer
port_stem = PorterStemmer()

def stemming(content):
    """Preprocess and stem text content"""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content 
                      if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def train_optimized_model():
    """Train model with optimized parameters for speed"""
    
    print("=" * 60)
    print("🚀 FAST TRAINING MODE - Optimized for Speed")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load data
    print("\n📂 Loading datasets...")
    data_dir = "./data"
    true_path = os.path.join(data_dir, "True.csv")
    fake_path = os.path.join(data_dir, "Fake.csv")
    
    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        print("❌ Error: Dataset files not found!")
        return
    
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)
    print(f"✅ Loaded {len(true_df)} real news and {len(fake_df)} fake news articles")
    
    # Add labels
    true_df['label'] = 0  # Real news
    fake_df['label'] = 1  # Fake news
    
    # Combine and shuffle
    df = pd.concat([true_df, fake_df], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.fillna('')
    
    # Create content (title + text)
    print("\n🔄 Preprocessing text...")
    df['content'] = df['title'] + ' ' + df['text']
    
    # Apply stemming
    print("🔄 Stemming text (this may take a moment)...")
    df['content'] = df['content'].apply(stemming)
    
    # Prepare features
    X = df['content'].values
    y = df['label'].values
    
    # Split data
    print("\n✂️ Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Vectorization with optimized parameters
    print("\n🔢 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,      # Reduced features for speed
        min_df=2,               # Ignore rare words
        max_df=0.8,             # Ignore too common words
        ngram_range=(1, 2),     # Use unigrams and bigrams
        sublinear_tf=True       # Slightly better performance
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"✅ Feature matrix shape: {X_train_vec.shape}")
    
    # Train model with fast solver
    print("\n🎓 Training Logistic Regression model...")
    print("   Optimizer: SAGA (fast for large datasets)")
    print("   Parallel processing: Enabled (all CPU cores)")
    
    model = LogisticRegression(
        max_iter=500,           # Reduced iterations
        solver='saga',          # Fast solver for large datasets
        random_state=42,
        n_jobs=-1,              # Use all CPU cores
        C=1.0,                  # Regularization strength
        verbose=0
    )
    
    train_start = time.time()
    model.fit(X_train_vec, y_train)
    train_end = time.time()
    
    print(f"✅ Training completed in {train_end - train_start:.2f} seconds")
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    train_pred = model.predict(X_train_vec)
    test_pred = model.predict(X_test_vec)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\n{'='*60}")
    print("📈 MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Training Accuracy:   {train_accuracy*100:.2f}%")
    print(f"Test Accuracy:       {test_accuracy*100:.2f}%")
    print(f"{'='*60}")
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, test_pred, 
                                target_names=['Real News', 'Fake News']))
    
    # Save model
    print("\n💾 Saving model and vectorizer...")
    os.makedirs('models', exist_ok=True)
    
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("✅ Model saved to 'models/model.pkl'")
    print("✅ Vectorizer saved to 'models/vectorizer.pkl'")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"⏱️  TOTAL TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"{'='*60}")
    print("\n✅ Training complete! You can now use the model in the app.")
    
    # Save training info
    info = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': total_time,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X_train_vec.shape[1]
    }
    
    with open('models/training_info.pkl', 'wb') as f:
        pickle.dump(info, f)

if __name__ == "__main__":
    try:
        train_optimized_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
