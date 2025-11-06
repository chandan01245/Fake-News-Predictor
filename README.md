## 📰 Fake News Predictor

A Machine Learning project that predicts whether a given news article is **Fake** or **Real** using Natural Language Processing (NLP) and Logistic Regression.

> NOTE: This `README.md` was merged with `README_NEW.md` on 2025-11-07 to combine reorganization notes and quick-start instructions. `README_NEW.md` is preserved for reference.

---

## 🚀 Quick Start (Windows - PowerShell)

1. Activate the virtual environment:

```powershell
.\Virtual-env\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the web app (choose the command that matches your layout):

```powershell
# If app.py is in the repository root
python app.py
# Or, for the reorganized layout
python src\web\app.py
```

Open http://localhost:7860 in your browser.

For macOS / Linux use:

```bash
source Virtual-env/bin/activate
pip install -r requirements.txt
python src/web/app.py
```

Detailed web UI instructions: see `docs/RUN_APP.md`.

---

## 📁 Project structure (overview)

```text
Fake-News-Predictor/
├── src/                # Source: web app, models, utils
├── scripts/            # Training & helper scripts
├── docs/               # Documentation (GETTING_STARTED, RUN_APP, etc.)
├── data/               # Raw datasets (True.csv, Fake.csv)
├── processed/          # Processed datasets
├── models/             # Saved model artifacts
├── notebooks/          # Jupyter notebooks
├── requirements.txt
├── README.md
└── README_NEW.md       # Kept for reference
```

---

## ⚙️ Installation & setup (summary)

1. Clone the repository:

```bash
git clone <repo-url>
cd <repo-dir>
```

2. Activate environment and install dependencies (see Quick Start above).

3. Download the dataset from Kaggle and place `True.csv` and `Fake.csv` into the `data/` directory.

4. Train the model (optional):

```powershell
python scripts\train_ultra_fast.py
python scripts\train_fast.py
```

---

## 🌐 Run on Google Colab

Open the training notebook in Colab:

[Open in Google Colab](https://colab.research.google.com/drive/1V6HJIv7YEMOU61c6fuJ3apxpNNTHCjes?usp=sharing)

---

## 🎯 Features

- Gradio Web UI with real-time predictions and confidence scores
- TF-IDF vectorization + Logistic Regression classifier
- Fast / ultra-fast training scripts
- Modular code layout for maintainability

---

## 🛠️ Technology stack

- Frontend: Gradio
- ML: scikit-learn
- NLP: NLTK
- Language: Python 3.7+

---

## 📈 Model performance (reported)

- Training accuracy: ~99%
- Test accuracy: ~98%

Results depend on dataset and training configuration.

---

## 🎓 How it works (high level)

1. Text preprocessing (cleaning, lowercasing)
2. Stopword removal / optional stemming
3. TF-IDF vectorization
4. Train Logistic Regression classifier
5. Serve predictions via Gradio web UI

---

## 🔧 Contributing

Contributions welcome. Please follow project structure, add tests, update docs, and submit pull requests.

---

## 📝 License

Open source — available for educational purposes.

---

## 👨‍💻 Developer

Built with ❤️ using Machine Learning and NLP

---

## 🔗 Useful links

- Docs: `docs/` folder (see `docs/RUN_APP.md`)
- Notebook (Colab): link above

---

**Happy Fact-Checking! 🔍✨**
