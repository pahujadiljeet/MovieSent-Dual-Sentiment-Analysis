from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load models
log_model = joblib.load("models/log_model.pkl")
tfidf = joblib.load("models/tfidf.pkl")

lstm_model = tf.keras.models.load_model("models/lstm_model.h5")
with open("models/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

MAX_SEQUENCE_LENGTH = 200  # same as used during LSTM training

def preprocess_text(text):
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    nltk.download('stopwords', quiet=True)
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    text = re.sub(r'<.*?>', '', text)           # remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)     # remove punctuation/numbers
    text = text.lower()
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    model_type = request.form['model']

    processed = preprocess_text(review)

    if model_type == "Logistic Regression":
        vect = tfidf.transform([processed]).toarray()
        pred = log_model.predict(vect)[0]
    else:  # LSTM
        seq = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred_prob = lstm_model.predict(padded)[0][0]
        pred = "positive" if pred_prob >= 0.6 else "negative"

    return render_template('index.html', review=review, prediction=pred)

if __name__ == '__main__':
    app.run(debug=True)
