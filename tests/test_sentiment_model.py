from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from sentiment_model import load_artifacts, predict_sentiment, preprocess_text


class FakeLemmatizer:
    def lemmatize(self, word: str) -> str:
        return {"reviews": "review", "products": "product"}.get(word, word)


class FakeVectorizer:
    def __init__(self) -> None:
        self.seen_text: list[str] | None = None

    def transform(self, texts: list[str]) -> list[str]:
        self.seen_text = texts
        return texts


class FakeModel:
    # Deliberately shuffled to verify model.classes_ controls probability mapping.
    classes_ = [2, 0, 1]

    def predict(self, features: list[str]) -> list[int]:
        return [0]

    def predict_proba(self, features: list[str]) -> list[list[float]]:
        return [[0.1, 0.7, 0.2]]


class SentimentHelperTests(unittest.TestCase):
    def test_preprocessing_matches_training_steps(self) -> None:
        cleaned = preprocess_text(
            "I LOVED these reviews & products! 123",
            stop_words={"i", "these"},
            lemmatizer=FakeLemmatizer(),
        )
        self.assertEqual(cleaned, "loved review product")

    def test_preprocessing_requires_string(self) -> None:
        with self.assertRaisesRegex(TypeError, "text must be a string"):
            preprocess_text(None, set(), FakeLemmatizer())  # type: ignore[arg-type]

    def test_inference_maps_probabilities_by_actual_class_label(self) -> None:
        vectorizer = FakeVectorizer()
        model = FakeModel()
        prediction, scores = predict_sentiment(
            "great products",
            model,
            vectorizer,
            stop_words=set(),
            lemmatizer=FakeLemmatizer(),
        )
        self.assertEqual(vectorizer.seen_text, ["great product"])
        self.assertEqual(prediction, 0)
        self.assertEqual(scores, {2: 0.1, 0: 0.7, 1: 0.2})

    def test_empty_after_preprocessing_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "containing words"):
            predict_sentiment(
                "the and a",
                FakeModel(),
                FakeVectorizer(),
                stop_words={"the", "and", "a"},
                lemmatizer=FakeLemmatizer(),
            )

    def test_committed_model_and_vectorizer_load_and_run(self) -> None:
        model, vectorizer = load_artifacts()
        features = vectorizer.transform(["this product is good"])
        predicted = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        classes = [int(value) for value in model.classes_]
        self.assertIn(predicted, {0, 1, 2})
        self.assertEqual(len(probabilities), len(classes))
        self.assertAlmostEqual(sum(float(value) for value in probabilities), 1.0, places=5)

    def test_missing_artifacts_fail_with_path_context(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Saved sentiment model not found"):
                load_artifacts(directory)


if __name__ == "__main__":
    unittest.main()
