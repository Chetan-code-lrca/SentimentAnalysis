from __future__ import annotations

import streamlit as st

from sentiment_model import LABELS, load_artifacts, load_nltk_resources, predict_sentiment

st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠", layout="centered")
st.title("Sentiment Analysis")
st.write("Classify a short piece of text as positive, negative, or neutral using the saved TF-IDF/SVM model.")


@st.cache_resource
def get_nltk_resources():
    return load_nltk_resources(allow_download=True)


@st.cache_resource
def get_model_and_vectorizer():
    return load_artifacts()


try:
    stop_words, lemmatizer = get_nltk_resources()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

user_input = st.text_area(
    "Enter text to classify:",
    placeholder="I really enjoyed this product and would recommend it.",
)

if st.button("Predict sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            model, vectorizer = get_model_and_vectorizer()
            prediction, scores = predict_sentiment(
                user_input,
                model,
                vectorizer,
                stop_words,
                lemmatizer,
            )
        except ValueError as exc:
            st.warning(str(exc))
        except (OSError, AttributeError, ImportError, TypeError, RuntimeError) as exc:
            st.error(f"Could not load or run the saved model: {exc}")
        else:
            st.subheader(f"Prediction: {LABELS[prediction]}")
            st.metric("Confidence score", f"{scores[prediction]:.1%}")
            st.caption("Confidence is the model's probability estimate, not a guarantee of correctness.")

st.divider()
st.caption("Model: scikit-learn linear SVM with TF-IDF features.")
