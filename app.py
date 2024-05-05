from flask import Flask, request, jsonify
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

app = Flask(__name__)

# Load your trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your CountVectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess(message):
    # Convert to lowercase
    message = message.lower()

    # Remove punctuation
    message = message.translate(str.maketrans('', '', string.punctuation))

    # Tokenize into words
    words = word_tokenize(message)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    # Stem words
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]

    return ' '.join(words)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Server is connected'}), 200

@app.route('/detect', methods=['POST'])
def detect_spam():
    data = request.get_json()
    email = data['email']
    name = data['name']
    message = data['message']

    # Preprocess the message
    preprocessed_message = preprocess(message)

    # Transform the message
    transformed_message = vectorizer.transform([preprocessed_message])

    # Use the model to predict
    prediction = model.predict(transformed_message)

    # Return the result
    if prediction[0] == 1:
        return jsonify({'result': 'spam'})
    else:
        return jsonify({'result': 'not spam'})
    
    


if __name__ == "__main__":
    app.run(debug=True)