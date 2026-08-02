<div align="center">
# 🧠 Sentiment Analysis AI

### AI-Powered Emotion Detection System using NLP & Machine Learning

<p>

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/NLP-TF--IDF-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Model-Linear%20SVM-purple?style=for-the-badge"/>

</p>

### 🚀 Live Demo

<a href="https://sentiment-analysis-mahadi.streamlit.app/">
<img src="https://img.shields.io/badge/Launch%20App-Streamlit-red?style=for-the-badge&logo=streamlit"/>
</a>

</div>

---

# 📌 About The Project

**Sentiment Analysis AI** is an NLP-based Machine Learning application that analyzes human-written text and predicts the underlying emotion.

The system uses a complete machine learning pipeline:

* 📝 Text preprocessing
* 🔢 TF-IDF feature extraction
* 🧠 Linear Support Vector Machine classifier
* 🏷️ Label encoding
* 🌐 Streamlit interactive dashboard

The application can classify text into multiple emotional categories in real-time.

---

# ✨ Features

## 🤖 Machine Learning Features

✅ Natural Language Processing pipeline
✅ TF-IDF based text representation
✅ Linear SVM multi-class classifier
✅ Emotion prediction from raw text
✅ Saved ML pipeline using Joblib
✅ Fast real-time inference

---

## 🎨 Application Features

✅ Modern Streamlit AI dashboard
✅ Interactive text prediction
✅ Emotion-based results
✅ Simple user interface
✅ Cloud deployment ready

---

# 🎭 Supported Emotions

| Emotion  | Emoji | Meaning                 |
| -------- | ----- | ----------------------- |
| Joy      | 😊    | Happiness, positivity   |
| Sadness  | 😢    | Sorrow, loneliness      |
| Anger    | 😡    | Frustration, irritation |
| Love     | ❤️    | Affection, care         |
| Fear     | 😨    | Anxiety, uncertainty    |
| Surprise | 😲    | Unexpected events       |

---

# 🏗️ Machine Learning Architecture

```
                User Text

                    ↓

            Text Preprocessing

                    ↓

             TF-IDF Vectorizer

                    ↓

            Linear SVM Classifier

                    ↓

              Label Encoder

                    ↓

          Predicted Emotion
```

---

# 🧠 Model Information

## TF-IDF Vectorizer

Converts text data into numerical features.

Configuration:

```python
ngram_range = (1,2)
min_df = 2
max_df = 0.95
```

---

## Linear SVM

Linear Support Vector Machine was selected because:

* Handles high-dimensional text data efficiently
* Performs well on NLP classification tasks
* Provides fast prediction
* Generalizes well on unseen data

---

# 📊 Dataset

Dataset:

**Kaggle Emotion Dataset for NLP**

Details:

| Property | Value                  |
| -------- | ---------------------- |
| Samples  | 16,000                 |
| Task     | Emotion Classification |
| Language | English                |
| Classes  | 6                      |

Example:

```
I feel extremely happy today ; joy

I am angry about this situation ; anger

I miss my family so much ; love
```

---

# 📁 Project Structure

```
Sentiment-Analysis/

│

├── app.py
├── emotion_svm_model.pkl
├── label_encoder.pkl
├── requirements.txt
├── train_model.ipynb
├── data/
│   └── train.txt
│
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/mahadiurrehman-pixel/Sentiment-Analysis.git

cd Sentiment-Analysis
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Run Application

```bash
streamlit run app.py
```

Application:

```
http://localhost:8501
```

---

# 📦 Saved Model Files

## emotion_svm_model.pkl

Contains:

* TF-IDF Vectorizer
* Trained Linear SVM model

---

## label_encoder.pkl

Maps numerical classes back to emotions.

Example:

```
0 → anger
1 → fear
2 → joy
3 → love
4 → sadness
5 → surprise
```

---

# 🧪 Example Predictions

### Input

```
I achieved my dream and I feel extremely happy today.
```

Output:

```
Emotion: Joy 😊
```

---

### Input

```
I feel lonely and everything is difficult today.
```

Output:

```
Emotion: Sadness 😢
```

---

# 🛠️ Tech Stack

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Core Programming        |
| Pandas       | Data Processing         |
| NumPy        | Numerical Computing     |
| Scikit-learn | Machine Learning        |
| TF-IDF       | Text Feature Extraction |
| LinearSVC    | Classification          |
| Joblib       | Model Saving            |
| Streamlit    | Web Application         |

---

# 🔮 Future Improvements

🚀 Upgrade to Transformer models:

* BERT
* RoBERTa
* DistilBERT

Additional features:

* REST API with FastAPI
* Docker deployment
* Multi-language emotion detection
* Real-time social media analysis
* Cloud-based ML service

---

# 🌐 Live Application

Try the deployed application:

🔗 https://sentiment-analysis-mahadi.streamlit.app/

---

# 👨‍💻 Developer

<div align="center">

## Mahadi Ur Rehman Siddiqui

AI / Machine Learning Engineer
Python Developer | NLP Enthusiast

Building intelligent systems using Artificial Intelligence and Machine Learning.

</div>

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

Made with ❤️ by **Mahadi Ur Rehman Siddiqui**

</div>
