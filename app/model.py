from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import joblib
import os

class TextClassifier:
    def __init__(self):
        if os.path.exists("vectorizer.pkl") and os.path.exists("classifier.pkl"):
            self.vectorizer = joblib.load("vectorizer.pkl")
            self.classifier = joblib.load("classifier.pkl")
        else:
            self.vectorizer = CountVectorizer()
            self.classifier = MultinomialNB()
            self.train()

    def train(self):
        texts, labels = self.generate_synthetic_data()
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        joblib.dump(self.vectorizer, 'vectorizer.pkl')
        joblib.dump(self.classifier, 'classifier.pkl')

    def generate_synthetic_data(self):
        texts = [
            "This is a positive text example",
            "Another happy and joyful text",
            "This is a negative example of text",
            "A sad and bad feeling example",
            "Positive emotions and joy",
            "Negative mood and sadness",
        ]
        labels = [1, 1, 0, 0, 1, 0]  # 1 - позитивний, 0 - негативний
        return texts, labels

    def predict(self, text):
        X = self.vectorizer.transform([text])
        return self.classifier.predict(X)[0]
