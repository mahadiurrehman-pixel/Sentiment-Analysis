<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Sentiment%20Analysis%20AI&fontSize=48&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Emotion%20Classification%20using%20NLP%20%26%20Machine%20Learning&descAlignY=58&descAlign=50" width="100%"/>

# 🧠 Sentiment Analysis AI

### Emotion Detection System using TF-IDF + Linear SVM

<p>

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/NLP-TF--IDF-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Model-Linear%20SVM-purple?style=for-the-badge"/>

</p>

</div>


---

# 📌 About The Project

**Sentiment Analysis AI** is an NLP-based machine learning application that detects the emotional tone of text.

The system uses:

- **TF-IDF Vectorization** for text feature extraction
- **Linear Support Vector Machine (LinearSVC)** for classification
- **Label Encoder** for emotion mapping
- **Streamlit** for an interactive AI dashboard


The application can analyze user-written text and classify it into different human emotions in real-time.

---

# ✨ Features

## 🤖 Machine Learning

✅ Text preprocessing pipeline  
✅ TF-IDF feature extraction  
✅ Linear SVM classifier  
✅ Multi-class emotion prediction  
✅ Saved ML pipeline using Joblib  
✅ Fast inference  


## 🎨 Streamlit Dashboard

✅ Modern AI-style interface  
✅ Text input prediction  
✅ Emotion result display  
✅ Emotion-based UI elements  
✅ Easy deployment ready  


---

# 🎭 Supported Emotions

| Emotion | Emoji |
|---|---|
| Joy | 😊 |
| Sadness | 😢 |
| Anger | 😡 |
| Love | ❤️ |
| Fear | 😨 |
| Surprise | 😲 |


---

# 🏗️ Machine Learning Pipeline


```
User Text

     ↓

Text Cleaning

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

# 🧠 Model Details


## TF-IDF Vectorizer

Used for converting text into numerical features.

Features:

- ngram_range = (1,2)
- min_df = 2
- max_df = 0.95


## Linear SVM

Linear Support Vector Machine is used because:

- Works well with high-dimensional text data
- Fast training
- Good generalization
- Effective for classification problems


---

# 📊 Dataset

Dataset used:

**Kaggle Emotion Dataset for NLP**

Dataset contains:

| Property | Value |
|-|-|
| Samples | 16,000 |
| Task | Emotion Classification |
| Language | English |
| Classes | 6 |


Example:


```
i didnt feel humiliated ; sadness

im feeling angry about this situation ; anger

i love spending time with my family ; love
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


### Clone Repository

```bash
git clone https://github.com/mahadi-ur-rehman-pixel/Sentiment-Analysis.git

cd Sentiment-Analysis
```


### Create Environment

```bash
python -m venv venv
```


Activate:


Windows:

```bash
venv\Scripts\activate
```


Linux:

```bash
source venv/bin/activate
```


Install dependencies:

```bash
pip install -r requirements.txt
```


---

# 🚀 Run Application


```bash
streamlit run app.py
```


Application will start at:


```
http://localhost:8501
```


---

# 📦 Model Files


### emotion_svm_model.pkl

Contains:

- TF-IDF Vectorizer
- Linear SVM Model


### label_encoder.pkl

Converts:

```
0 → anger

1 → fear

2 → joy

3 → love

4 → sadness

5 → surprise
```


---

# 🧪 Example Prediction


Input:

```
I just achieved my goal and I feel extremely happy.
```


Output:

```
Emotion: Joy 😊
```


---

Input:

```
I feel lonely and everything seems difficult today.
```


Output:

```
Emotion: Sadness 😢
```


---

# 🛠️ Tech Stack


| Technology | Purpose |
|-|-|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Scikit-learn | Machine Learning |
| TF-IDF | NLP Feature Extraction |
| LinearSVC | Classification |
| Joblib | Model Serialization |
| Streamlit | Web Application |


---

# 🔮 Future Improvements


- Transformer based model (BERT)
- REST API using FastAPI
- Docker deployment
- Multi-language emotion detection
- Real-time social media analysis
- Cloud deployment


---

# 👨‍💻 Developer


<div align="center">


## Mahadi Ur Rehman Siddiqui


AI / Machine Learning Engineer  
Python Developer | NLP Enthusiast


Building intelligent systems with Machine Learning and Artificial Intelligence.


</div>


---

<div align="center">

⭐ If you like this project, consider giving it a star!

Made with ❤️ by **Mahadi Ur Rehman Siddiqui**

</div>