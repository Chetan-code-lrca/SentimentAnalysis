from __future__ import annotations

from pathlib import Path
import pickle

import pytest

from sentiment_utils import predict_sentiment, preprocess


class FakeLemmatizer:
    def lemmatize(self, word):
        return {"cats": "cat", "dogs": "dog"}.get(word, word)


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


def test_preprocess_matches_notebook_lemmatization_pipeline():
    result = preprocess(
        "The cats are dogs!",
        stop_words={"the", "are"},
        lemmatizer=FakeLemmatizer(),
    )

    assert result == "cat dog"


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
        lemmatizer=FakeLemmatizer(),
    )

    assert vectorizer.last_text == "genuinely useful tool"
    assert result == (2, 0.3, 0.2, 0.5)


def test_prediction_rejects_empty_cleaned_text():
    with pytest.raises(ValueError, match="useful word"):
        predict_sentiment(
            "the !!!",
            model=FakeModel(),
            vectorizer=FakeVectorizer(),
            stop_words={"the"},
            lemmatizer=FakeLemmatizer(),
        )


def test_prediction_rejects_extra_model_classes():
    model = FakeModel()
    model.classes_ = [0, 1, 2, 3]
    model.predict_proba = lambda features: [[0.1, 0.2, 0.3, 0.4]]

    with pytest.raises(ValueError, match="exactly classes 0"):
        predict_sentiment(
            "helpful words",
            model=model,
            vectorizer=FakeVectorizer(),
            stop_words=set(),
            lemmatizer=FakeLemmatizer(),
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
