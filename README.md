# Sentiment Analysis App

A Streamlit app that classifies text as **positive**, **negative**, or **neutral** using the repository's trained scikit-learn SVM and TF-IDF vectorizer.

## Correctness note

The training notebook uses lowercasing, alphabetic cleanup, NLTK English stopword removal, and WordNet lemmatization. The app now uses the same transformation before applying the saved vectorizer. Prediction confidence is mapped by the model's actual classes rather than assuming probability-column order.

## Run locally

Requirements: Python 3.10+.

```bash
git clone https://github.com/Chetan-code-lrca/SentimentAnalysis.git
cd SentimentAnalysis
python -m venv .venv
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and start the app:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

On first run, NLTK downloads the stopwords and wordnet data only if they are missing. If the machine is offline and resources are not cached, install them once when a connection is available:

```bash
python -m nltk.downloader stopwords wordnet
```

Open the local Streamlit URL printed by the command, normally http://localhost:8501.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover preprocessing parity, empty text, class-label/probability alignment, saved model artifact loading, and a real inference smoke check. GitHub Actions installs NLTK data, then runs the suite.

## Project files

- app.py — Streamlit UI
- sentiment_model.py — preprocessing, resource loading, model loading, and inference
- SVM.ipynb — notebook used for dataset preparation and SVM training
- svm_sentiment_model.pkl — trained classifier
- tfidf_vectorizer.pkl — fitted TF-IDF vectorizer

The notebook downloads the Twitter entity sentiment dataset from Kaggle through opendatasets; that download is needed only to repeat training. Normal app inference uses the committed model and vectorizer.

## Limitations

Sentiment is context-dependent; sarcasm, mixed feelings, domain-specific wording, and out-of-distribution text can produce incorrect predictions. Probability scores are estimates, not guarantees. Pickle artifacts must only be loaded from trusted sources.
