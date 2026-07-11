# 🎬 IMDB Movie Review Sentiment Analysis

A Natural Language Processing (NLP) project that classifies IMDB movie reviews as **Positive** or **Negative** using Machine Learning.

---

## 📌 Overview

This project demonstrates a complete NLP pipeline including:

- Text Cleaning
- Stopword Removal
- Lemmatization
- Feature Extraction (CountVectorizer & TF-IDF)
- Model Training
- Model Evaluation
- Model Serialization using Pickle

The dataset contains **50,000 IMDB movie reviews** labeled as either positive or negative.

---

## 🚀 Features

- Clean and preprocess movie reviews
- Convert text into numerical features
- Train multiple Machine Learning models
- Compare model performance
- Save the trained model for future predictions

---

## 🛠 Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- Pickle

---

## 📂 Dataset

**Dataset:** IMDB Movie Reviews Dataset

- 50,000 reviews
- Binary Sentiment Classification
- Labels:
  - Positive
  - Negative

---

## 📋 Data Preprocessing

The following preprocessing steps were performed:

- Convert text to lowercase
- Remove unnecessary prefixes
- Remove punctuation
- Tokenization
- Stopword removal
- Lemmatization

---

## 🔤 Feature Extraction

Two text vectorization techniques were used:

### CountVectorizer

Converts text into a Bag-of-Words representation.

### TF-IDF Vectorizer

Assigns weights to words based on their importance in the document.

---

## 🤖 Machine Learning Models

The following models were trained:

- Logistic Regression
- Multinomial Naive Bayes

Each model was evaluated using:

- CountVectorizer
- TF-IDF Vectorizer

---

## 📊 Results

| Model | Feature Extraction | Accuracy |
|--------|-------------------|----------|
| Logistic Regression | CountVectorizer | **86.10%** |
| Logistic Regression | TF-IDF | **87.26%** |
| Multinomial Naive Bayes | CountVectorizer | **86.10%** |
| Multinomial Naive Bayes | TF-IDF | **87.26%** |

### Best Performing Model

✅ Logistic Regression + TF-IDF

Accuracy: **87.26%**

---

## 💾 Saved Files

The trained model and vectorizer are saved using Pickle.

```
model.pkl
countvec.pkl
```

These files can be loaded later to predict sentiment for new movie reviews.

---

## 📁 Project Structure

```
IMDB-Sentiment-Analysis/
│
├── IMDB Dataset.csv
├── sentiment_analysis.ipynb
├── model.pkl
├── countvec.pkl
├── README.md
└── requirements.txt
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/IMDB-Sentiment-Analysis.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the notebook

```bash
jupyter notebook
```

---

## 📌 Future Improvements

- Use GridSearchCV for hyperparameter tuning
- Train Support Vector Machine (SVM)
- Train Random Forest Classifier
- Implement Deep Learning (LSTM/BERT)
- Deploy using Flask or FastAPI
- Build a Streamlit Web Application

---

## 👨‍💻 Author

**Mahadi ur Rehman Siddiqui**

BSBIT Student | AI & Machine Learning Enthusiast | NLP Learner

---

⭐ If you found this project helpful, consider giving it a star!
