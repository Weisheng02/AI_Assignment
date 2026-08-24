import os
import random
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text, load_intents, extract_dataset

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "unrecognized_queries.json")

def log_unrecognized_query(query: str, confidence: float):
    """
    Continuous Learning Loop (Active Learning):
    Logs unrecognized or low-confidence queries for administrative review & data expansion.
    """
    logs = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except Exception:
            logs = []

    # Append new query if not duplicate
    if not any(item['query'] == query for item in logs):
        logs.append({
            "query": query,
            "confidence": confidence,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2)


class UniversityIntentClassifier:
    """
    Member 2 Machine Learning Classifier:
    Uses character-boundary TF-IDF (3-5 character n-grams) and balanced
    Logistic Regression (C=30.0). Character features are robust to short FAQ
    wording changes and minor spelling variation. A deployment confidence gate
    provides controlled fallback behaviour.
    """
    def __init__(self, confidence_threshold: float = 0.20):
        self.confidence_threshold = confidence_threshold
        self.pipeline = Pipeline([
            (
                'tfidf',
                TfidfVectorizer(
                    analyzer='char_wb',
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                ),
            ),
            (
                'clf',
                LogisticRegression(
                    C=30.0,
                    max_iter=2000,
                    random_state=42,
                    class_weight='balanced',
                ),
            ),
        ])
        self.responses_dict = {}
        self.patterns_dict = {}
        self.is_trained = False

    def train(self, dataset_path: str):
        intents_data = load_intents(dataset_path)
        patterns, tags, self.responses_dict = extract_dataset(intents_data)

        self.train_from_examples(patterns, tags, self.responses_dict)

    def train_from_examples(self, patterns, tags, responses_dict=None):
        """Train on an explicit set of examples.

        This entry point is used by evaluation code so the classifier, including
        its exact-match table, can only see the training partition. ``train``
        remains available for the deployed application and trains on the full
        dataset as before.
        """
        if len(patterns) != len(tags):
            raise ValueError("patterns and tags must contain the same number of items")
        if not patterns:
            raise ValueError("at least one training example is required")

        cleaned_patterns = []
        cleaned_tags = []
        self.patterns_dict = {}

        for pattern, tag in zip(patterns, tags):
            cleaned = clean_text(pattern)
            if not cleaned:
                continue
            cleaned_patterns.append(cleaned)
            cleaned_tags.append(tag)
            self.patterns_dict[cleaned] = tag

        if not cleaned_patterns:
            raise ValueError("training examples are empty after preprocessing")

        if responses_dict is not None:
            self.responses_dict = dict(responses_dict)

        # Store pattern mappings for fast exact matching
        self.pipeline.fit(cleaned_patterns, cleaned_tags)
        self.is_trained = True
        return self

    def predict_intent(self, text: str, log_if_fallback: bool = True):
        if not self.is_trained:
            raise RuntimeError("classifier must be trained before prediction")

        cleaned = clean_text(text)
        if not cleaned:
            return "fallback", 0.0

        if cleaned in self.patterns_dict:
            return self.patterns_dict[cleaned], 1.0

        probs = self.pipeline.predict_proba([cleaned])[0]
        max_idx = np.argmax(probs)
        confidence = probs[max_idx]
        predicted_tag = self.pipeline.classes_[max_idx]

        if confidence < self.confidence_threshold:
            if log_if_fallback:
                log_unrecognized_query(text, round(float(confidence), 4))
            return "fallback", confidence

        return predicted_tag, confidence

    def get_response(self, text: str, log_if_fallback: bool = True) -> dict:
        tag, confidence = self.predict_intent(text, log_if_fallback=log_if_fallback)

        if tag == "fallback" or tag not in self.responses_dict:
            response_text = "I'm sorry, I didn't quite understand your question. Could you please rephrase or ask about university courses, fees, admissions, or campus facilities?"
        else:
            response_text = random.choice(self.responses_dict[tag])

        return {
            "user_input": text,
            "predicted_tag": tag,
            "confidence": round(float(confidence), 4),
            "response": response_text
        }
