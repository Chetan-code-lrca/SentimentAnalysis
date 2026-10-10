"""Preprocessing and inference helpers for the saved TF-IDF/SVM sentiment model."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

LABELS = {
    0: "Negative",
    1: "Positive",
    2: "Neutral",
}


def load_nltk_resources(*, allow_download: bool = True) -> tuple[set[str], WordNetLemmatizer]:
    """Load the same NLTK resources used in the training notebook.

    Resources are downloaded only when missing, not on every app start.
    """
    resources: list[str] = []
    try:
        loaded_stopwords = set(stopwords.words("english"))
    except LookupError:
        loaded_stopwords = set()
        resources.append("stopwords")

    lemmatizer = WordNetLemmatizer()
    try:
        lemmatizer.lemmatize("reviews")
    except LookupError:
        resources.append("wordnet")

    if resources:
        if not allow_download:
            resource_list = " ".join(resources)
            raise RuntimeError(
                "Missing NLTK data: "
                + ", ".join(resources)
                + ". Run python -m nltk.downloader "
                + resource_list
                + " once, then restart the app."
            )
        for resource in resources:
            if not nltk.download(resource, quiet=True):
                raise RuntimeError(
                    f"Could not download NLTK resource '{resource}'. "
                    "Check the network or install the NLTK data manually."
                )

        try:
            loaded_stopwords = set(stopwords.words("english"))
            lemmatizer.lemmatize("reviews")
        except LookupError as exc:
            raise RuntimeError(
                "NLTK data is still unavailable after download. "
                "Install the missing stopwords/wordnet resources and restart."
            ) from exc

    return loaded_stopwords, lemmatizer


def preprocess_text(
    text: str,
    stop_words: set[str],
    lemmatizer: Any,
) -> str:
    """Apply the notebook's lowercasing, alphabetic cleanup, stopword removal, and lemmatization."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = re.sub(r"[^a-zA-Z]", " ", text.lower())
    words = normalized.split()
    return " ".join(
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    )


def predict_sentiment(
    text: str,
    model: Any,
    vectorizer: Any,
    stop_words: set[str],
    lemmatizer: Any,
) -> tuple[int, dict[int, float]]:
    """Predict a sentiment and map probability scores by the model's actual class labels."""
    cleaned = preprocess_text(text, stop_words, lemmatizer)
    if not cleaned:
        raise ValueError("Enter text containing words to classify.")

    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])
    classes = [int(value) for value in model.classes_]
    probabilities = [float(value) for value in model.predict_proba(features)[0]]
    if len(classes) != len(probabilities):
        raise ValueError("The model returned a mismatched class/probability count.")
    if prediction not in classes:
        raise ValueError(f"The model predicted an unknown sentiment class: {prediction}.")

    scores = dict(zip(classes, probabilities, strict=True))
    return prediction, scores


def _load_pickle(path: Path, artifact_name: str) -> Any:
    """Load a trusted pickle artifact and turn corrupt-file failures into actionable errors."""
    try:
        with path.open("rb") as artifact_file:
            return pickle.load(artifact_file)
    except (
        EOFError,
        pickle.UnpicklingError,
        AttributeError,
        ImportError,
        ValueError,
        TypeError,
    ) as exc:
        raise RuntimeError(
            f"Could not load saved {artifact_name} at {path}; the file may be "
            "be empty, truncated, or incompatible with the installed library versions."
        ) from exc


def load_artifacts(model_dir: str | Path | None = None) -> tuple[Any, Any]:
    """Load the committed model and vectorizer relative to this source file."""
    directory = Path(model_dir).resolve() if model_dir is not None else Path(__file__).resolve().parent
    model_path = directory / "svm_sentiment_model.pkl"
    vectorizer_path = directory / "tfidf_vectorizer.pkl"

    if not model_path.is_file():
        raise FileNotFoundError(f"Saved sentiment model not found: {model_path}")
    if not vectorizer_path.is_file():
        raise FileNotFoundError(f"Saved TF-IDF vectorizer not found: {vectorizer_path}")

    model = _load_pickle(model_path, "sentiment model")
    vectorizer = _load_pickle(vectorizer_path, "TF-IDF vectorizer")

    for attribute in ("predict", "predict_proba", "classes_"):
        if not hasattr(model, attribute):
            raise TypeError(f"Saved model does not provide required attribute '{attribute}'.")
    if not hasattr(vectorizer, "transform"):
        raise TypeError("Saved vectorizer does not provide transform().")
    return model, vectorizer
