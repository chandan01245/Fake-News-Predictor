# 🚀 Quick Start Guide - Fake News Predictor Web UI

## 📋 Prerequisites

1. **Python 3.7+** installed
2. **Dataset files** (Download from Kaggle):
   - [Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
   - Extract and place `True.csv` and `Fake.csv` in the `data/` folder

## 🎯 Method 1: Quick Start (Windows)

### Step 1: Run the application
```bash
run.bat
```

That's it! The web interface will automatically open at `http://localhost:7860`

---

## 🔧 Method 2: Manual Setup

### Step 1: Activate Virtual Environment

**Windows:**
```bash
.\Virtual-env\Scripts\activate
```

**macOS/Linux:**
```bash
source Virtual-env/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python app.py
```

### Step 4: Open Your Browser
Navigate to: **http://localhost:7860**

---

## 📁 Project Structure

```
ML-project/
├── app.py                  # Main Gradio web application
├── run.bat                 # Quick start script (Windows)
├── requirements.txt        # Python dependencies
├── data/                   # Dataset folder (you need to add files here)
│   ├── True.csv           # Real news dataset
│   └── Fake.csv           # Fake news dataset
├── models/                 # Saved models (auto-created after training)
│   ├── model.pkl
│   └── vectorizer.pkl
└── README.md
```

---

## 🎓 How to Use the Web UI

### 1️⃣ **First Time Setup - Train the Model**

1. Go to the **"🎓 Train Model"** tab
2. Make sure `True.csv` and `Fake.csv` are in the `data/` folder
3. Click **"🚀 Train Model"** button
4. Wait for training to complete (2-5 minutes)
5. You'll see the accuracy results

### 2️⃣ **Detect Fake News**

1. Go to the **"🎯 Detect Fake News"** tab
2. Paste a news article in the text box
3. Click **"🔍 Analyze Article"**
4. View the results:
   - ✅ Real News or 🚨 Fake News prediction
   - Confidence score
   - Detailed probability breakdown

### 3️⃣ **Try Examples**

- Click **"📌 Example: Fake News"** or **"📌 Example: Real News"** to test with sample articles

---

## ✨ Features

- 🎨 **Beautiful UI**: Modern, intuitive Gradio interface
- 🚀 **Fast Analysis**: Get results in seconds
- 📊 **Detailed Insights**: Confidence scores and probability breakdowns
- 💾 **Model Persistence**: Train once, use anytime
- 📱 **Responsive Design**: Works on desktop and mobile
- 🌐 **Shareable**: Can create public links with `share=True`

---

## 🔍 Web Interface Preview

### Main Features:
1. **Detect Fake News Tab**: Analyze articles in real-time
2. **Train Model Tab**: Train your own model with custom data
3. **About Tab**: Learn about the technology and methodology

---

## ⚠️ Troubleshooting

### Issue: "No trained model found"
**Solution**: Go to the "Train Model" tab and train the model first

### Issue: "Dataset files not found"
**Solution**: Download `True.csv` and `Fake.csv` from Kaggle and place them in the `data/` folder

### Issue: Port already in use
**Solution**: Edit `app.py` and change `server_port=7860` to another port like `7861`

### Issue: Import errors
**Solution**: 
```bash
pip install --upgrade -r requirements.txt
```

---

## 🌐 Sharing Your App

To make your app accessible online, change line in `app.py`:
```python
app.launch(share=True)  # Creates a public URL
```

⚠️ **Note**: Share links expire after 72 hours

---

## 🛑 Stopping the Application

- Press `Ctrl + C` in the terminal
- Or close the command prompt window

---

## 💡 Tips for Best Results

1. **Include full articles**: Use both title and body text
2. **Original text**: Paste original article text (not summaries)
3. **English only**: Model is trained on English articles
4. **Complete sentences**: Fragments may not work well

---

## 🎯 Sample Articles to Test

### Fake News Example:
```
BREAKING: Scientists Discover That Earth is Actually Flat After All!
[Use the example button in the app]
```

### Real News Example:
```
Climate Change Conference Reaches Historic Agreement on Emissions Reduction
[Use the example button in the app]
```

---

## 📞 Need Help?

- Check the "About" tab in the web interface
- Review the error messages in the terminal
- Ensure all dataset files are in place
- Verify Python version: `python --version` (should be 3.7+)

---

## 🎉 Enjoy Detecting Fake News!

The web interface makes it easy to analyze news articles with just a few clicks. Happy fact-checking! 🔍✨
