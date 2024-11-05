from flask import Flask, request, jsonify
from model import TextClassifier

app = Flask(__name__)
classifier = TextClassifier()

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text')
    prediction = classifier.predict(text)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
