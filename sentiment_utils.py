"""Preprocessing and prediction helpers for the saved sentiment model."""

from __future__ import annotations

import re
from typing import Any

from nltk.stem import PorterStemmer


def preprocess(
    text: str,
    stop_words: set[str],
    stemmer: PorterStemmer | None = None,
) -> str:
    """Match the notebook's lowercase, ASCII-word, stopword and stemming pipeline."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    active_stemmer = stemmer or PorterStemmer()
    words = re.sub(r"[^a-zA-Z]", " ", text.lower()).split()
    return " ".join(
        active_stemmer.stem(word)
        for word in words
        if word not in stop_words
    )


def predict_sentiment(
    text: str,
    *,
    model: Any,
    vectorizer: Any,
    stop_words: set[str],
    stemmer: PorterStemmer | None = None,
) -> tuple[int, float, float, float]:
    """Return class label, positive probability, negative probability, neutral probability."""
    cleaned = preprocess(text, stop_words, stemmer)
    if not cleaned:
        raise ValueError("Enter text containing at least one useful word.")

    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    classes = [int(label) for label in model.classes_]

    if len(classes) != len(probabilities):
        raise ValueError("Classifier class labels do not match its probability output.")

    probability_by_class = dict(zip(classes, probabilities))
    if not {0, 1, 2}.issubset(probability_by_class):
        raise ValueError("The saved model must include classes 0 (negative), 1 (positive), and 2 (neutral).")

    return (
        prediction,
        float(probability_by_class[1]),
        float(probability_by_class[0]),
        float(probability_by_class[2]),
    )
