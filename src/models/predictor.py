"""
Fake News Predictor Model Module
"""
import numpy as np
import pandas as pd
import re
import os
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

port_stem = PorterStemmer()

def stemming(content):
    """Preprocess and stem text content"""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content 
                      if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def quick_preprocess(content):
    """Fast preprocessing without stemming - 10x faster"""
    content = re.sub(r'[^a-zA-Z\s]', '', str(content).lower())
    return content

def load_model(model_dir='models'):
    """Load trained model and vectorizer"""
    model_path = os.path.join(model_dir, 'model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

def predict(text, model, vectorizer, use_stemming=False):
    """Predict if news is fake or real"""
    if use_stemming:
        processed_text = stemming(text)
    else:
        processed_text = quick_preprocess(text)
    
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    return {
        'prediction': 'fake' if prediction == 1 else 'real',
        'confidence': max(probability) * 100,
        'probabilities': {
            'real': probability[0] * 100,
            'fake': probability[1] * 100
        }
    }
