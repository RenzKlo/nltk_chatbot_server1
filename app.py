from flask import Flask, request, jsonify
import pickle
from joblib import load

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load your trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
# Load your fitted TfidfVectorizer

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


def preprocess_text(text):
    # Remove newline and carriage return characters
    text = text.replace('\n', '').replace('\r', '')
    
    # Extract words and convert to lowercase
    words = re.findall(r'\b[A-Za-z]+\b', text)
    words = [word.lower() for word in words]
    
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Join words back into a single string
    processed_text = ' '.join(words)
    
    return processed_text

def predict_email(email_text, model, cv):
    # Preprocess the input email text
    processed_text = preprocess_text(email_text)
    
    # Vectorize the preprocessed text using CountVectorizer
    text_vector = cv.transform([processed_text])
    
    # Use the model to predict the label (spam or ham)
    prediction = model.predict(text_vector)
    
    # Return the predicted label
    return prediction[0]

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Server is connected'}), 200

@app.route('/detect', methods=['POST'])
def detect_spam():
    data = request.get_json()
    emailMessage = data['emailMessage']

    predicted_label = predict_email(emailMessage, model, vectorizer)
    print('label: ' ,predicted_label)
    # Return the result
    if  predicted_label == 'spam':
        return jsonify({'spamValue': 'spam'})
    else:
        return jsonify({'spamValue': 'not spam'})
    
if __name__ == "__main__":
    app.run(debug=True)
