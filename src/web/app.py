"""
Fake News Predictor - Fancy Web UI
A beautiful Gradio interface for detecting fake news articles
"""

import gradio as gr
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

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer
port_stem = PorterStemmer()

# Global variables for model and vectorizer
model = None
vectorizer = None
model_accuracy = 0

def stemming(content):
    """Preprocess and stem text content"""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def quick_preprocess(content):
    """Fast preprocessing without stemming - 10x faster"""
    content = re.sub(r'[^a-zA-Z\s]', '', str(content).lower())
    return content

def train_model():
    """Train the fake news detection model"""
    global model, vectorizer, model_accuracy
    
    try:
        # Load data
        data_dir = "./data"
        true_path = os.path.join(data_dir, "True.csv")
        fake_path = os.path.join(data_dir, "Fake.csv")
        
        if not os.path.exists(true_path) or not os.path.exists(fake_path):
            return "❌ Error: Dataset files not found! Please ensure True.csv and Fake.csv are in the 'data' folder.", 0
        
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
        model_accuracy = test_accuracy
        
        # Save model and vectorizer
        os.makedirs('models', exist_ok=True)
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        return f"✅ Model trained successfully!\n📊 Training Accuracy: {train_accuracy*100:.2f}%\n📊 Test Accuracy: {test_accuracy*100:.2f}%\n💾 Model saved to 'models' folder", test_accuracy
    
    except Exception as e:
        return f"❌ Error during training: {str(e)}", 0

def load_model():
    """Load pre-trained model"""
    global model, vectorizer, model_accuracy
    
    try:
        if not os.path.exists('models/model.pkl') or not os.path.exists('models/vectorizer.pkl'):
            return "❌ No trained model found! Please train the model first."
        
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        return "✅ Model loaded successfully!"
    
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def predict_news(news_text):
    """Predict if news is fake or real"""
    global model, vectorizer
    
    if model is None or vectorizer is None:
        # Try to load model
        if os.path.exists('models/model.pkl') and os.path.exists('models/vectorizer.pkl'):
            load_model()
        else:
            return "⚠️ No model available", "Please train the model first!", 0.0, "Train the model using the 'Train Model' tab"
    
    try:
        # Preprocess text (fast mode)
        processed_text = quick_preprocess(news_text)
        
        # Vectorize
        text_vector = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        # Get confidence
        confidence = max(probability) * 100
        
        if prediction == 1:
            result = "🚨 FAKE NEWS"
            color = "red"
            message = "This article appears to be fake news. Be skeptical of its claims!"
            emoji = "🚫"
        else:
            result = "✅ REAL NEWS"
            color = "green"
            message = "This article appears to be legitimate news."
            emoji = "✓"
        
        # Create detailed output
        output_html = f"""
        <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, {'#ff6b6b' if prediction == 1 else '#51cf66'} 0%, {'#ff8787' if prediction == 1 else '#69db7c'} 100%);">
            <h2 style="color: white; text-align: center; margin: 0;">{emoji} {result} {emoji}</h2>
        </div>
        """
        
        confidence_html = f"""
        <div style="padding: 15px; border-radius: 10px; background: #f8f9fa; margin-top: 10px; color: #212529;">
            <h3 style="margin-top: 0; color: #212529;">📊 Prediction Details:</h3>
            <p style="color: #212529;"><strong style="color: #212529;">Confidence:</strong> {confidence:.2f}%</p>
            <p style="color: #212529;"><strong style="color: #212529;">Real News Probability:</strong> {probability[0]*100:.2f}%</p>
            <p style="color: #212529;"><strong style="color: #212529;">Fake News Probability:</strong> {probability[1]*100:.2f}%</p>
            <p style="margin-bottom: 0; color: #212529;"><strong style="color: #212529;">Analysis:</strong> {message}</p>
        </div>
        """
        
        return output_html, confidence_html, confidence, message
    
    except Exception as e:
        return "❌ Error", f"Error during prediction: {str(e)}", 0.0, "Please check your input and try again"

# Sample news articles for demonstration
sample_fake = """BREAKING: Scientists Discover That Earth is Actually Flat After All! 
In a shocking revelation that contradicts centuries of scientific consensus, a group of self-proclaimed researchers 
announced today that the Earth is indeed flat. The group claims to have conducted extensive experiments using 
spirit levels and rulers to prove their theory. NASA officials could not be reached for comment, presumably 
because they are too busy maintaining the global conspiracy to hide the truth."""

sample_real = """Climate Change Conference Reaches Historic Agreement on Emissions Reduction
World leaders gathered in Paris today to sign a landmark agreement aimed at reducing global carbon emissions 
by 50% over the next decade. The accord, which has been years in the making, represents a significant step 
forward in international cooperation on climate change. Scientists and environmental groups have praised 
the agreement while noting that implementation will be challenging."""

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="🔍 Fake News Detector") as app:
    
    gr.Markdown("""
    # 🔍 Fake News Predictor
    ### Powered by Machine Learning & Natural Language Processing
    
    Detect whether a news article is **FAKE** or **REAL** using advanced ML algorithms!
    """)
    
    with gr.Tabs():
        # Prediction Tab
        with gr.Tab("🎯 Detect Fake News"):
            gr.Markdown("### Paste a news article below to analyze its authenticity")
            
            with gr.Row():
                with gr.Column(scale=2):
                    news_input = gr.Textbox(
                        label="📰 News Article",
                        placeholder="Paste the news article text here...",
                        lines=10,
                        info="Enter the title and content of the news article"
                    )
                    
                    with gr.Row():
                        predict_btn = gr.Button("🔍 Analyze Article", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ Clear", size="lg")
                    
                    gr.Markdown("### Try these examples:")
                    with gr.Row():
                        example1_btn = gr.Button("📌 Example: Fake News", size="sm")
                        example2_btn = gr.Button("📌 Example: Real News", size="sm")
                
                with gr.Column(scale=1):
                    result_output = gr.HTML(label="🎯 Result")
                    details_output = gr.HTML(label="📊 Details")
                    confidence_gauge = gr.Number(label="Confidence Level", visible=False)
                    message_output = gr.Textbox(label="💡 Recommendation", lines=2)
            
            # Button actions
            predict_btn.click(
                fn=predict_news,
                inputs=[news_input],
                outputs=[result_output, details_output, confidence_gauge, message_output]
            )
            
            clear_btn.click(
                fn=lambda: ("", "", "", 0.0),
                outputs=[news_input, result_output, details_output, confidence_gauge]
            )
            
            example1_btn.click(lambda: sample_fake, outputs=[news_input])
            example2_btn.click(lambda: sample_real, outputs=[news_input])
        
        # Training Tab
        with gr.Tab("🎓 Train Model"):
            gr.Markdown("""
            ### Train the Fake News Detection Model
            
            **Instructions:**
            1. Ensure you have `True.csv` and `Fake.csv` files in the `data/` folder
            2. Click the "Train Model" button below
            3. Wait for training to complete (may take a few minutes)
            4. Once trained, you can use the prediction feature!
            
            **Dataset Requirements:**
            - Download from: [Kaggle - Fake News Detection](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
            - Place `True.csv` and `Fake.csv` in the `data/` folder
            """)
            
            train_btn = gr.Button("🚀 Train Model", variant="primary", size="lg")
            train_output = gr.Textbox(label="Training Status", lines=5)
            accuracy_output = gr.Number(label="Model Accuracy", visible=False)
            
            train_btn.click(
                fn=train_model,
                outputs=[train_output, accuracy_output]
            )
            
            gr.Markdown("---")
            
            load_btn = gr.Button("📂 Load Existing Model", variant="secondary")
            load_output = gr.Textbox(label="Load Status")
            
            load_btn.click(
                fn=load_model,
                outputs=[load_output]
            )
        
        # About Tab
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            # 📰 About This Project
            
            ## 🎯 What is This?
            This is a **Machine Learning-powered Fake News Detector** that analyzes news articles and predicts 
            whether they are likely to be fake or authentic.
            
            ## 🧠 How It Works
            1. **Natural Language Processing**: Text preprocessing including stemming and stopword removal
            2. **TF-IDF Vectorization**: Converts text into numerical features
            3. **Logistic Regression**: ML algorithm that classifies news as fake or real
            4. **Training**: Model learns from thousands of real and fake news articles
            
            ## 📊 Features
            - ✨ Beautiful and intuitive interface
            - 🎯 High accuracy prediction
            - 📈 Confidence scores for each prediction
            - 🚀 Fast and efficient analysis
            - 💾 Save and load trained models
            
            ## 🔧 Technology Stack
            - **Frontend**: Gradio
            - **ML Framework**: Scikit-learn
            - **NLP**: NLTK
            - **Language**: Python
            
            ## 📝 Dataset
            The model is trained on the Kaggle Fake News Detection dataset containing:
            - **Real News**: Authentic articles from reliable sources
            - **Fake News**: Known fake articles and misinformation
            
            ## ⚠️ Disclaimer
            This tool provides predictions based on patterns learned from training data. It should be used as 
            a supplementary tool and not as the sole method for verifying news authenticity. Always verify 
            important information from multiple reliable sources.
            
            ## 👨‍💻 Developer
            Built with ❤️ using Machine Learning and NLP
            
            ---
            **Version**: 1.0.0 | **Last Updated**: 2024
            """)
    
    gr.Markdown("""
    ---
    <div style="text-align: center; padding: 20px;">
        <p style="color: #666;">💡 <strong>Tip:</strong> For best results, include both the title and full article text</p>
        <p style="color: #666;">⚡ Made with <a href="https://gradio.app" target="_blank">Gradio</a></p>
    </div>
    """)

# Try to load model on startup
if os.path.exists('models/model.pkl') and os.path.exists('models/vectorizer.pkl'):
    load_model()
    print("✅ Model loaded successfully on startup!")
else:
    print("⚠️ No trained model found. Please train the model first.")

if __name__ == "__main__":
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
