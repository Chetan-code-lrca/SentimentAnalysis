from __future__ import annotations

from pathlib import Path
import pickle

import pytest
from nltk.stem import PorterStemmer

from sentiment_utils import predict_sentiment, preprocess


class FakeVectorizer:
    def transform(self, texts):
        self.last_text = texts[0]
        return texts


class FakeModel:
    classes_ = [2, 0, 1]

    def predict(self, features):
        return [2]

    def predict_proba(self, features):
        return [[0.5, 0.2, 0.3]]


def test_preprocess_removes_stopwords_and_stems_words():
    result = preprocess(
        "The cats are running!",
        stop_words={"the", "are"},
        stemmer=PorterStemmer(),
    )

    assert result == "cat run"


def test_preprocess_rejects_non_string_input():
    with pytest.raises(TypeError, match="text must be a string"):
        preprocess(None, stop_words=set())  # type: ignore[arg-type]


def test_prediction_maps_probabilities_using_model_classes():
    vectorizer = FakeVectorizer()
    model = FakeModel()

    result = predict_sentiment(
        "A genuinely useful tool",
        model=model,
        vectorizer=vectorizer,
        stop_words={"a"},
        stemmer=PorterStemmer(),
    )

    assert vectorizer.last_text == "genuin use tool"
    assert result == (2, 0.3, 0.2, 0.5)


def test_prediction_rejects_empty_cleaned_text():
    with pytest.raises(ValueError, match="useful word"):
        predict_sentiment(
            "the !!!",
            model=FakeModel(),
            vectorizer=FakeVectorizer(),
            stop_words={"the"},
            stemmer=PorterStemmer(),
        )


def test_saved_model_and_vectorizer_are_compatible():
    root = Path(__file__).resolve().parents[1]
    with (root / "svm_sentiment_model.pkl").open("rb") as model_file:
        model = pickle.load(model_file)
    with (root / "tfidf_vectorizer.pkl").open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    features = vectorizer.transform(["this product is useful"])
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)

    assert features.shape[1] == model.n_features_in_
    assert len(prediction) == 1
    assert probabilities.shape == (1, len(model.classes_))
    assert set(int(label) for label in model.classes_) == {0, 1, 2}
