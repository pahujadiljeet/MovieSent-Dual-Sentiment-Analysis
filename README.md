# 🎬 MovieSent – Dual Approach Sentiment Analysis  

MovieSent is an **NLP-based Sentiment Analysis Project** that combines the power of **Logistic Regression** (traditional ML) and **LSTM Neural Networks** (Deep Learning) to classify movie reviews as **Positive** or **Negative**.  

It provides an **interactive web interface** built with **Flask + HTML/CSS/Bootstrap**, where users can type a review, choose a model (Logistic Regression or LSTM), and instantly get sentiment predictions.  

---

## 📺 Demo Video  
▶️ Watch the full working demo on YouTube: [MovieSent Demo](https://youtu.be/89-loMRV1iI)  

---

## 📝 Abstract / Introduction  

Imagine a bustling film festival where organizers and critics are eager to gauge audience sentiment on the latest releases. With thousands of reviews pouring in from social media and review sites, manual analysis is impossible.  

**MovieSent – Dual Approach Sentiment Analysis** steps in as a powerful solution:  
- **Logistic Regression + TF-IDF** for quick classical ML predictions.  
- **LSTM Neural Network** for deep contextual understanding of reviews.  

This dual approach allows **real-time sentiment analysis**, helping studios, critics, and platforms make better decisions.  

---

## 🚀 Features  

✅ Dual-Model Approach: Logistic Regression & LSTM  
✅ Real-time Sentiment Analysis (Positive / Negative)  
✅ Web Interface with Flask + Bootstrap  
✅ Preprocessing with **NLTK + spaCy** (stopwords, lemmatization, cleaning)  
✅ Evaluation with Accuracy, Precision, Recall, F1-score  
✅ Model persistence using **Joblib & H5** formats  
✅ Supports 5000+ IMDB Movie Reviews dataset  

---

## 📂 Project Structure  

```bash
MovieSent/
│── app.py                        # Flask backend
│── requirements.txt              # Project dependencies
│── README.md                     # Documentation
│
├── templates/
│   └── index.html                # Frontend (HTML)
│
├── static/
│   └── style.css                 # Frontend styling
│
├── models/
│   ├── logistic_model.pkl        # Logistic Regression model
│   ├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
│   ├── lstm_model.h5             # Trained LSTM model
│   └── tokenizer.pkl             # Tokenizer for LSTM
│
├── notebooks/
│   └── MovieSent_Training.ipynb  # Model training code (Google Colab / Jupyter)
│
└── data/
    └── IMDB_Dataset.csv          # Movie reviews dataset (5000+ reviews)

## 🛠️ Tools & Libraries

  1. Python (pandas, numpy)
  2. NLTK, spaCy (text preprocessing)
  3. scikit-learn (Logistic Regression, TF-IDF)
  4. TensorFlow / Keras (LSTM)
  5. Flask (Backend API + Frontend rendering)
  6. Bootstrap / CSS (Frontend UI)
  7. Joblib & H5 (Model saving/loading)

