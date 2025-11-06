# 📰 Fake News Predictor

A Machine Learning project that predicts whether a given news article is **Fake** or **Real** using **Natural Language Processing (NLP)** techniques and **Logistic Regression**.

---

## ✨ NEW: Fancy Web UI Available! 🎉

Now you can use a beautiful web interface to detect fake news with just a few clicks!

### 🚀 Quick Start - Web UI
```bash
# Windows users - Just double-click:
run.bat

# Or manually:
.\Virtual-env\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open: **http://localhost:7860** in your browser!

📖 **Detailed Guide**: See [RUN_APP.md](RUN_APP.md) for complete instructions

---

## 🌐 Alternative: Try It on Google Colab
You can also run the Jupyter notebook directly on Colab:  
👉 [**Open in Google Colab**](https://colab.research.google.com/drive/1V6HJIv7YEMOU61c6fuJ3apxpNNTHCjes?usp=sharing)

---

## 📁 Project Structure

```text
Fake-News-Predictor/
├── app.py                    # 🌟 NEW: Gradio Web UI Application
├── run.bat                   # 🌟 NEW: Quick start script for Windows
├── RUN_APP.md               # 🌟 NEW: Detailed web UI guide
├── data/                     # Folder to store raw Kaggle dataset
│   ├── True.csv
│   └── Fake.csv
├── models/                   # 🌟 NEW: Saved ML models (auto-created)
│   ├── model.pkl
│   └── vectorizer.pkl
├── processed/                # Folder for processed datasets (for notebook)
│   ├── train.csv
│   ├── test.csv
│   └── valid.csv
├── FakeNewsPredictor.ipynb  # Jupyter / Colab notebook for model training
├── requirements.txt          # Python dependencies (now includes Gradio!)
├── Virtual-env/              # Python virtual environment
├── .gitignore               
└── README.md


## ⚙️ Installation & Setup

### 🎨 Option 1: Web UI (Recommended for Beginners)

1️⃣ **Clone the Repository**
```bash
git clone https://github.com/Madhu-014/Fake-News-Predictor.git
cd Fake-News-Predictor
```

2️⃣ **Activate Virtual Environment**

**Windows:**
```bash
.\Virtual-env\Scripts\activate
```

**macOS / Linux:**
```bash
source Virtual-env/bin/activate
```

3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Download Dataset** (See below) and place in `data/` folder

5️⃣ **Run the Web UI**
```bash
python app.py
```

Open **http://localhost:7860** in your browser! 🎉

---

### 📓 Option 2: Jupyter Notebook

Follow steps 1-4 above, then:
```bash
jupyter notebook FakeNewsPredictor.ipynb
```
## 📊 Dataset

The dataset used in this project comes from Kaggle:  
🔗 [Fake News Detection Datasets by Emine YETMİŞ](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

### Steps to Download:

1. Visit the Kaggle link above
2. Click **Download** to get the ZIP file
3. Extract it and move the following two files into the `data/` folder:

```text
data/
├── True.csv
└── Fake.csv
```

✅ **You are all set!** Now you can:
- Run the **Web UI** to train and use the model with a beautiful interface
- Or use the **Jupyter Notebook** for detailed analysis

---

## 🎯 Features of the Web UI

✨ **Beautiful Gradio Interface**
- 🎨 Modern, intuitive design
- 📱 Mobile-friendly
- 🌙 Dark mode support

🔍 **Smart Predictions**
- Real-time analysis
- Confidence scores
- Detailed probability breakdown
- Sample articles for testing

🎓 **Easy Training**
- One-click model training
- Progress tracking
- Save/load trained models

📊 **Interactive Results**
- Visual prediction display
- Detailed statistics
- Recommendation messages

---

## 🖼️ Web UI Screenshots

### Main Detection Interface
- Paste any news article
- Get instant results
- See confidence scores

### Training Dashboard
- Train with your dataset
- View accuracy metrics
- Save models for later use

### About Section
- Learn how it works
- Understand the technology
- View project information

---

## 🛠️ Technology Stack

- **Frontend**: Gradio (Beautiful Web UI)
- **ML Framework**: Scikit-learn
- **NLP**: NLTK (Natural Language Toolkit)
- **Backend**: Python 3.7+
- **Vectorization**: TF-IDF
- **Algorithm**: Logistic Regression

---

## 📈 Model Performance

The model achieves high accuracy on the test dataset:
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98%

*Results may vary based on dataset and training parameters*

---

## 🎓 How It Works

1. **Text Preprocessing**: Remove special characters, convert to lowercase
2. **Stemming**: Reduce words to their root form
3. **Stopword Removal**: Remove common words that don't add meaning
4. **TF-IDF Vectorization**: Convert text to numerical features
5. **Logistic Regression**: Train classifier to predict fake/real
6. **Prediction**: Analyze new articles with confidence scores

---

## 💡 Usage Tips

### For Best Results:
- Include both **title** and **full article text**
- Use **original article content** (not summaries)
- Model works best with **English** articles
- Longer articles generally give better predictions

### Web UI Tips:
- Try the example buttons to see how it works
- Train the model once, use it multiple times
- Check the "About" tab for detailed information
- Confidence score shows prediction reliability

---

## 🚨 Disclaimer

This tool provides predictions based on patterns learned from training data. It should be used as a **supplementary tool** and not as the sole method for verifying news authenticity. Always:
- Verify information from multiple reliable sources
- Check the original source of the article
- Be critical of sensational claims
- Use fact-checking websites

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Improve documentation
- Submit pull requests

---

## 📝 License

This project is open source and available for educational purposes.

---

## 👨‍💻 Developer

Built with ❤️ using Machine Learning and NLP

---

## 🔗 Links

- 🌐 **Web UI**: Run locally with `python app.py`
- 📓 **Colab**: [Try Online](https://colab.research.google.com/drive/1V6HJIv7YEMOU61c6fuJ3apxpNNTHCjes?usp=sharing)
- 📊 **Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
- 📖 **Guide**: See [RUN_APP.md](RUN_APP.md) for detailed instructions

---

## ⭐ Show Your Support

If you find this project useful, please give it a ⭐ on GitHub!

---

**Happy Fact-Checking! 🔍✨**
