# src/preprocessing.py

import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Lowercase -> strip non-letters -> remove stopwords.
    Used identically at training time and inference time.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)