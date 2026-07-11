# 🎬 IMDB Movie Review Sentiment Analysis

A Natural Language Processing (NLP) project that classifies IMDB movie reviews into **Positive** and **Negative** sentiment using Machine Learning algorithms.

---

## 📌 Overview

This project implements an end-to-end NLP workflow for sentiment classification.

The pipeline includes:

- Text preprocessing
- Data cleaning
- Train-test splitting
- Feature extraction using CountVectorizer and TF-IDF
- Machine Learning model training
- Model evaluation
- Model serialization using Pickle

The dataset contains **50,000 IMDB movie reviews** with binary sentiment labels.

---

## 🚀 Features

- Clean raw movie review text
- Remove stopwords and unnecessary characters
- Apply lemmatization
- Convert text data into numerical vectors
- Train multiple ML models
- Compare model performance
- Save trained model and vectorizer for future predictions

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

Details:

- Total Reviews: **50,000**
- Classes:
  - ✅ Positive
  - ❌ Negative

---

# 🔄 NLP Pipeline

## 1. Text Preprocessing

The following preprocessing steps were applied:

- Convert text to lowercase
- Remove unnecessary prefixes
- Remove punctuation
- Remove stopwords
- Lemmatization

Example:

```
Original:
"This movie was amazing!"

Processed:
"movie amazing"
```

---

# 🔤 Feature Extraction

Two vectorization techniques were used:

## CountVectorizer

Converts text into a Bag-of-Words numerical representation.

## TF-IDF Vectorizer

Assigns importance weights to words based on their frequency and uniqueness.

TF-IDF was selected as the better performing feature extraction method.

---

# 🤖 Machine Learning Models

The following models were trained:

### 1. Logistic Regression

- CountVectorizer Features
- TF-IDF Features


### 2. Multinomial Naive Bayes

- CountVectorizer Features
- TF-IDF Features

---

# 📊 Model Performance

| Model | Feature Extraction | Accuracy |
|--------|-------------------|----------|
| Logistic Regression | CountVectorizer | 86.10% |
| Logistic Regression | TF-IDF | **87.26%** |
| Multinomial Naive Bayes | CountVectorizer | 86.10% |
| Multinomial Naive Bayes | TF-IDF | **87.26%** |

---

# 🏆 Best Model

**Logistic Regression + TF-IDF**

Accuracy:

```
87.26%
```

This model was selected and saved for future predictions.

---

# 💾 Saved Model Files

The trained model and TF-IDF vectorizer are saved using Pickle.

```
model.pkl
tfidf.pkl
```

These files can be loaded to classify new movie reviews without retraining the model.

---

# 📁 Project Structure

```
IMDB-Sentiment-Analysis/
│
├── IMDB Dataset.csv
├── sentiment_analysis.ipynb
├── model.pkl
├── tfidf.pkl
├── README.md
└── requirements.txt
```

---

# ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/IMDB-Sentiment-Analysis.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

---

# 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Add Support Vector Machine (SVM)
- Experiment with Random Forest
- Implement Deep Learning models:
  - LSTM
  - BERT
- Deploy model using:
  - Flask
  - FastAPI
  - Streamlit
- Create a real-time sentiment prediction web application

---


## 📄 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for more details.

---

# 👨‍💻 Author

**Mahadi ur Rehman Siddiqui**

BSBIT Student | AI & Machine Learning Enthusiast | NLP Learner

---

⭐ If you found this project useful, consider giving it a star!
