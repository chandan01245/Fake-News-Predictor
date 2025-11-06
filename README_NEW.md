# 📰 Fake News Predictor

A Machine Learning project that predicts whether a given news article is **Fake** or **Real** using **Natural Language Processing (NLP)** techniques and **Logistic Regression**.

---

## ✨ NEW: Restructured & Organized! 🎉

The project has been reorganized with a proper folder structure for better maintainability and scalability!

### 🚀 Quick Start

```bash
# Windows users - Run the restructuring:
restructure.bat

# Then create Python modules:
python post_restructure.py

# Run the web UI:
python src\web\app.py

# Or train the model:
python scripts\train_ultra_fast.py
```

---

## 📁 New Project Structure

```text
ML-project/
├── src/                      # 📦 Source code
│   ├── models/              # ML model code
│   │   ├── __init__.py
│   │   └── predictor.py     # Model loading & prediction
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   └── text_processing.py  # Text preprocessing
│   └── web/                 # Web interface
│       ├── __init__.py
│       └── app.py           # Gradio web application
│
├── scripts/                 # 🔧 Training & setup scripts
│   ├── train_fast.py        # Fast training with stemming
│   ├── train_ultra_fast.py  # Ultra-fast training (recommended)
│   ├── setup.bat            # Windows setup
│   └── run.bat              # Windows run script
│
├── docs/                    # 📚 Documentation
│   ├── GETTING_STARTED.md
│   ├── HOW_IT_WORKS.md
│   ├── RUN_APP.md
│   ├── TRAINING_OPTIMIZATIONS.md
│   └── ...
│
├── notebooks/               # 📓 Jupyter notebooks
│   └── FakeNewsPredictor.ipynb
│
├── tests/                   # 🧪 Test files
│   └── __init__.py
│
├── config/                  # ⚙️ Configuration
│   └── requirements.txt
│
├── data/                    # 📊 Dataset files
│   ├── True.csv
│   └── Fake.csv
│
├── models/                  # 🤖 Trained models
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── training_info.pkl
│
├── processed/               # 💾 Processed data
│
├── Virtual-env/             # 🐍 Python virtual environment
│
├── requirements.txt         # Python dependencies
├── README.md
└── restructure.bat         # Restructuring script
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/ML-project.git
cd ML-project
```

### 2️⃣ Activate Virtual Environment

**Windows:**
```bash
Virtual-env\Scripts\activate
```

**macOS / Linux:**
```bash
source Virtual-env/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download Dataset
Get the dataset from Kaggle:  
🔗 [Fake News Detection Datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

Extract and place `True.csv` and `Fake.csv` in the `data/` folder.

### 5️⃣ Train the Model
```bash
# Ultra-fast training (recommended)
python scripts\train_ultra_fast.py

# Or standard fast training
python scripts\train_fast.py
```

### 6️⃣ Run the Web Application
```bash
python src\web\app.py
```

Open **http://localhost:7860** in your browser! 🎉

---

## 🎯 Features

### 🌐 Beautiful Web UI
- Modern Gradio interface
- Real-time predictions
- Confidence scores
- Sample articles for testing

### 🤖 Smart ML Model
- High accuracy (~98%)
- Fast predictions
- TF-IDF vectorization
- Logistic Regression classifier

### ⚡ Optimized Training
- Ultra-fast mode: Train in ~60 seconds
- Fast mode: Train with full features
- Automatic model saving
- Progress tracking

### 📦 Clean Architecture
- Modular code structure
- Reusable components
- Easy to extend
- Well-documented

---

## 🛠️ Technology Stack

- **Frontend**: Gradio
- **ML Framework**: Scikit-learn
- **NLP**: NLTK
- **Language**: Python 3.7+
- **Vectorization**: TF-IDF
- **Algorithm**: Logistic Regression

---

## 📈 Model Performance

- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98%

*Using optimized training parameters*

---

## 🎓 How It Works

1. **Text Preprocessing**: Clean and normalize text
2. **Feature Extraction**: TF-IDF vectorization
3. **Model Training**: Logistic Regression classifier
4. **Prediction**: Analyze articles with confidence scores
5. **Web Interface**: Beautiful Gradio UI for easy interaction

---

## 💡 Project Benefits

### Before Restructuring ❌
- All files in root directory
- Hard to navigate
- Difficult to maintain
- Mixed concerns

### After Restructuring ✅
- Clean folder organization
- Clear separation of concerns
- Easy to find files
- Modular architecture
- Professional structure
- Easy to extend

---

## 📚 Documentation

All documentation is now organized in the `docs/` folder:

- **GETTING_STARTED.md** - Setup guide
- **HOW_IT_WORKS.md** - Technical details
- **RUN_APP.md** - Web UI guide
- **TRAINING_OPTIMIZATIONS.md** - Training tips

---

## 🔧 Development

### Adding New Features

1. **Model improvements** → `src/models/`
2. **Utilities** → `src/utils/`
3. **Web UI changes** → `src/web/`
4. **Training scripts** → `scripts/`
5. **Tests** → `tests/`

### Project Commands

```bash
# Run web app
python src\web\app.py

# Train model (ultra-fast)
python scripts\train_ultra_fast.py

# Train model (with stemming)
python scripts\train_fast.py

# Run tests
python -m pytest tests/

# Activate environment
Virtual-env\Scripts\activate
```

---

## 🚨 Disclaimer

This tool provides predictions based on patterns learned from training data. Use it as a supplementary tool, not as the sole method for verifying news authenticity. Always verify information from multiple reliable sources.

---

## 🤝 Contributing

Contributions welcome! Please:
- Follow the project structure
- Add tests for new features
- Update documentation
- Submit pull requests

---

## 📝 License

Open source - available for educational purposes.

---

## 👨‍💻 Developer

Built with ❤️ using Machine Learning and NLP

---

## ⭐ Show Your Support

If you find this project useful, please give it a ⭐ on GitHub!

---

**Happy Fact-Checking! 🔍✨**
