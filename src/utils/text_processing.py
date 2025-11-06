"""
Text processing utilities
"""
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

port_stem = PorterStemmer()

def clean_text(text):
    """Remove special characters and convert to lowercase"""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    return text

def remove_stopwords(text):
    """Remove common stopwords"""
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

def stem_text(text):
    """Apply Porter stemming"""
    words = text.split()
    words = [port_stem.stem(word) for word in words]
    return ' '.join(words)

def full_preprocess(text, use_stemming=True):
    """Complete text preprocessing pipeline"""
    text = clean_text(text)
    text = remove_stopwords(text)
    if use_stemming:
        text = stem_text(text)
    return text
