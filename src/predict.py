# src/predict.py

import pickle
import joblib
import numpy as np
from pathlib import Path

from src.preprocessing import clean_text

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

model = joblib.load(MODELS_DIR / "ticket_classifier_v2.pkl")
vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer_v2.pkl")


def vectorize_ticket(description: str):
    """Returns the TF-IDF vector for a ticket description, for reuse in similarity search."""
    cleaned = clean_text(description)
    return vectorizer.transform([cleaned])


def classify_ticket(description: str):
    """
    Takes a raw ticket description, returns (predicted_category, confidence_pct).
    """
    ticket_vector = vectorize_ticket(description)

    prediction = model.predict(ticket_vector)[0]

    probabilities = model.predict_proba(ticket_vector)[0]
    confidence = float(np.max(probabilities) * 100)

    return prediction, confidence