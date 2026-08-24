import os
import re
import json
import string
import nltk
from nltk.stem import WordNetLemmatizer

# Ensure local project paths are in NLTK search path if available
PROJECT_NLTK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")
MYENV_NLTK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "myenv", "nltk_data")
for d in [PROJECT_NLTK_DIR, MYENV_NLTK_DIR]:
    if os.path.exists(d) and d not in nltk.data.path:
        nltk.data.path.append(d)

lemmatizer = WordNetLemmatizer()

def _safe_lemmatize(word: str) -> str:
    try:
        return lemmatizer.lemmatize(word)
    except Exception:
        return word

def clean_text(text: str) -> str:
    """
    Cleans input text by converting to lowercase, removing punctuation, special characters, and extra spaces.
    Applies WordNet Lemmatization with safe fallback.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\n', ' ', text)
    # Preserve years, fee amounts, and alphanumeric course codes.  They carry
    # important meaning in university enquiries (for example, "2026" or
    # "A123") and should not be discarded merely because they contain digits.

    tokens = text.split()
    tokens = [_safe_lemmatize(word) for word in tokens]
    return " ".join(tokens)

def analyze_sentiment(text: str) -> dict:
    """
    Analyzes sentiment & urgency of student queries to flag priority administrative inquiries.
    """
    lowered = text.lower()
    negative_keywords = [
        'urgent', 'fail', 'angry', 'terrible', 'worst', 'emergency',
        'issue', 'problem', 'stuck', 'error', 'frustrated', 'frustration',
        'frustrating', 'upset', 'annoyed', 'worried', 'bad', 'hate', 'delay', 'slow'
    ]
    positive_keywords = [
        'thanks', 'thank you', 'great', 'awesome', 'good', 'love', 'nice',
        'helpful', 'excellent', 'happy', 'pleased', 'wonderful'
    ]

    neg_score = sum(1 for word in negative_keywords if re.search(r'\b' + re.escape(word) + r'\b', lowered))
    pos_score = sum(1 for word in positive_keywords if re.search(r'\b' + re.escape(word) + r'\b', lowered))

    if neg_score > pos_score:
        sentiment = "Negative / Urgent"
        emoji = "⚠️"
    elif pos_score > neg_score:
        sentiment = "Positive / Satisfied"
        emoji = "😊"
    else:
        sentiment = "Neutral Inquiry"
        emoji = "😐"

    return {"sentiment": sentiment, "emoji": emoji}

def load_intents(filepath: str) -> dict:
    """
    Loads intents dataset from JSON file with utf-8 encoding.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_dataset(intents_data: dict):
    """
    Extracts patterns (X) and tags/intents (y) along with response mappings.
    """
    patterns = []
    tags = []
    responses_dict = {}

    for intent in intents_data['intents']:
        tag = intent['tag']
        responses_dict[tag] = intent['responses']
        for pattern in intent['patterns']:
            cleaned = clean_text(pattern)
            if cleaned:
                patterns.append(cleaned)
                tags.append(tag)

    return patterns, tags, responses_dict
