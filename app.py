from flask import Flask, request, jsonify, render_template
import pickle, re
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


app = Flask(__name__)

# Load saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vect = pickle.load(f)

# Recreate preprocessing steps exactly as in training
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def data_processing(text):
    text = text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if w not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered_text]
    return " ".join(stemmed)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review = data.get('review', '')
        print("\n>>> Raw review:", review)  # Debug

        if not review.strip():
            return jsonify({'error': 'Empty input'}), 400

        processed = data_processing(review)
        print(">>> After preprocessing:", processed)  # Debug

        X = vect.transform([processed])
        print(">>> Vector shape:", X.shape)  # Debug

        pred = model.predict(X)[0]
        sentiment = "Positive" if pred == 1 else "Negative"
        print(">>> Predicted sentiment:", sentiment)

        return jsonify({'sentiment': sentiment})

    except Exception as e:
        print(">>> ERROR:", e)  # 👈  Show real issue in terminal
        return jsonify({'error': str(e)}), 500


import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
