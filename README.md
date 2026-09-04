# AI-Powered Customer Support Assistant

An AI-powered system that classifies customer support tickets by issue category and recommends a resolution path (priority, expected resolution time, satisfaction estimate) based on similar historical tickets — wrapped in an interactive Streamlit app.

## What it does

1. **Classifies** an incoming ticket description into one of five categories: Account, Billing, Fraud, General Inquiry, Technical
2. **Finds similar historical tickets** within the predicted category using cosine similarity on TF-IDF vectors
3. **Recommends** a suggested priority, expected resolution time, and average satisfaction score, derived from those similar tickets

## Dataset

Public "Customer Support Tickets" dataset from Kaggle — 20,000 tickets, 12 original columns, no missing values, no duplicate rows. Average ticket description length: 16.93 words. Most common category: Technical (5,918). Least common: Fraud (1,040).

## Pipeline
Ticket Description
↓
Text Cleaning (lowercase, remove non-letters) → Stopword Removal
↓
TF-IDF Vectorization (5,000 features, 1–2 word n-grams)
↓
Logistic Regression Classification → Category + Confidence
↓
Cosine Similarity Search (within predicted category)
↓
Recommendation (Priority, Resolution Time, Satisfaction) from similar tickets


Tokenization, stopword removal, and lemmatization were all implemented and tested during preprocessing. A lemmatized variant of the model was trained and evaluated head-to-head against the final (non-lemmatized) version on the same 10-ticket realistic test set: the non-lemmatized model scored 6/10 (60%) versus 5/10 (50%) for the lemmatized version. Lemmatization was excluded from the final model based on this result, in favor of the simpler pipeline that also performed slightly better.cy difference over stopword-removed text alone — kept for a simpler, more explainable pipeline.

## A note on model accuracy — data leakage investigation

Initial models (Logistic Regression, Naive Bayes, Random Forest, XGBoost) all reported 1.0000 accuracy, precision, recall, and F1 on a standard 80/20 train/test split. This is a red flag, not a genuine result, and was investigated rather than accepted at face value.

**Finding:** the dataset's `Ticket_Subject` and `Ticket_Description` fields are synthetically template-generated — a small, fixed pool of canned phrases per category with random filler words appended (e.g. every Fraud ticket subject follows a pattern like `"Phishing attempt - <random word>"`). A model can reach ~100% accuracy on a held-out split drawn from the same template distribution without learning anything that generalizes to realistically-phrased tickets.

**What was done about it:**
- Removed `Ticket_Subject` from the model input entirely (it was the most obvious leak — a near 1:1 mapping between canned subject phrases and category)
- Retrained on `Ticket_Description` alone — accuracy remained 1.0000 on the standard test split, confirming the deeper leak lives in the description templates too, which preprocessing can't fix (it's inherent to how the dataset was generated)
- Built a **hand-written realistic test set** (10 tickets phrased the way a real user would write them, 2 per category, not drawn from the dataset) as a separate, honest evaluation
- **Realistic-ticket accuracy: 6/10 (60%)** — this is the number that reflects true generalization, not the misleading 1.0000

This process — and the resulting 60% figure — is reported here deliberately, in place of the inflated training accuracy, as the more honest measure of model performance.

## Project structure
AI-Ticket-Classification/
├── data/
│ └── customer_support_tickets.csv
├── models/
│ ├── ticket_classifier_v2.pkl # final model (Description-only)
│ ├── tfidf_vectorizer_v2.pkl # final vectorizer
│ ├── ticket_classifier.pkl # v1 — Subject+Description (leaky, kept for comparison)
│ └── tfidf_vectorizer.pkl # v1 vectorizer
├── notebooks/
│ └── 01_EDA.ipynb # EDA, preprocessing, model training/comparison, leakage diagnosis
├── src/
│ ├── preprocessing.py # text cleaning
│ ├── predict.py # loads model/vectorizer, classifies a ticket
│ ├── recommend.py # similarity search + recommendation logic
│ └── app.py # Streamlit UI
└── requirements.txt


## Running the app

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Opens at `http://localhost:8501`.

## Tech stack

Python, pandas, scikit-learn (TF-IDF, Logistic Regression, cosine similarity), NLTK (stopwords), Streamlit