# src/recommend.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text
from src.predict import vectorizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SIMILAR_TICKET_COLUMNS = [
    "Ticket_ID",
    "Ticket_Subject",
    "Issue_Category",
    "Priority_Level",
    "Resolution_Time_Hours",
    "Satisfaction_Score",
    "Similarity",
]


def load_ticket_corpus():
    """
    Loads the full ticket dataset and rebuilds Cleaned_Text (description only),
    then vectorizes it with the already-fitted v2 TF-IDF vectorizer.

    Returns (df, X) where X is the TF-IDF matrix for the whole dataset,
    row-aligned with df.
    """
    df = pd.read_csv(DATA_DIR / "customer_support_tickets.csv")

    cleaned_text = df["Ticket_Description"].apply(clean_text)
    X = vectorizer.transform(cleaned_text)

    return df, X


def recommend_resolution(ticket_vector, predicted_category, df, X, top_n=5):
    """
    Given an already-computed ticket_vector and its predicted category,
    finds the top_n most similar historical tickets within that category
    and derives a recommendation from them.
    """
    category_indices = df.index[df["Issue_Category"] == predicted_category].tolist()
    category_matrix = X[category_indices]

    similarities = cosine_similarity(ticket_vector, category_matrix)[0]

    top_positions = np.argsort(similarities)[::-1][:top_n]
    similar_indices = [category_indices[i] for i in top_positions]

    similar_tickets = df.loc[similar_indices].copy()
    similar_tickets["Similarity"] = similarities[top_positions]

    avg_resolution_time = similar_tickets["Resolution_Time_Hours"].mean()
    common_priority = similar_tickets["Priority_Level"].mode()[0]
    avg_satisfaction = similar_tickets["Satisfaction_Score"].mean()

    return {
        "similar_tickets": similar_tickets[SIMILAR_TICKET_COLUMNS],
        "suggested_priority": common_priority,
        "avg_resolution_time": avg_resolution_time,
        "avg_satisfaction": avg_satisfaction,
    }