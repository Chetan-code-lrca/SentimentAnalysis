from __future__ import annotations

import pickle
from pathlib import Path

import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sentiment_utils import predict_sentiment

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_resources():
    """Load the trusted model artifacts and an already-installed NLTK corpus."""
    with (BASE_DIR / "svm_sentiment_model.pkl").open("rb") as model_file:
        model = pickle.load(model_file)
    with (BASE_DIR / "tfidf_vectorizer.pkl").open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    try:
        english_stop_words = set(stopwords.words("english"))
    except LookupError as exc:
        raise RuntimeError(
            "The NLTK English stopword corpus is missing. Install it once with "
            "'python -m nltk.downloader stopwords', then restart the app."
        ) from exc

    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise RuntimeError("The saved classifier must expose predict_proba() and classes_.")
    if not hasattr(vectorizer, "transform"):
        raise RuntimeError("The saved TF-IDF vectorizer must expose transform().")

    return model, vectorizer, english_stop_words, PorterStemmer()


st.set_page_config(page_title="Chetan's Sentiment Analyzer", page_icon="🧠")
st.title("🧠 Sentiment Analysis App")
st.markdown("### Built by Chetan Inaganti | NLP Developer")
st.info(
    "💼 Looking for custom NLP solutions? "
    "[LinkedIn](https://www.linkedin.com/in/chetan-inaganti-563a41322/)"
)

try:
    model, vectorizer, stop_words, stemmer = load_resources()
except (FileNotFoundError, pickle.UnpicklingError, RuntimeError) as exc:
    st.error(str(exc))
    st.stop()

st.write("## Classify a piece of text")
user_input = st.text_area("Enter text:")

if st.button("Predict"):
    try:
        prediction, positive_prob, negative_prob, neutral_prob = predict_sentiment(
            user_input,
            model=model,
            vectorizer=vectorizer,
            stop_words=stop_words,
            stemmer=stemmer,
        )
    except ValueError as exc:
        st.warning(str(exc))
    else:
        labels = {
            0: ("Negative", negative_prob),
            1: ("Positive", positive_prob),
            2: ("Neutral", neutral_prob),
        }
        sentiment, confidence = labels[prediction]
        st.write("### Prediction:", sentiment)
        st.write("### Confidence Score:", round(confidence, 2))

st.markdown("---")
st.caption(
    "Powered by scikit-learn, NLTK, and Streamlit. "
    "Predictions are model estimates, not definitive judgments."
)
