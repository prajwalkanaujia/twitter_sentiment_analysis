from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained sentiment analysis model
with open('model_pickle_new', 'rb') as f:
    loaded_model = pickle.load(f)

# Load the saved vectorizer
with open('vectorizer_pickle', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    data = request.json
    text = data['text']

    X = vectorizer.transform([text])
    # print(f"Predicted Sentiment: {loaded_model.predict(X)[0]}")

    return jsonify({'sentiment': loaded_model.predict(X)[0]})
    

if __name__ == '__main__':
    app.run(debug=True)