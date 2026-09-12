# Sentiment Analysis App

A Streamlit application that classifies input text as **positive**, **negative**, or **neutral** using a trained scikit-learn model and TF-IDF features.

## How it works

The application loads two serialized artifacts from the repository:

- `svm_sentiment_model.pkl` — trained SVM sentiment classifier
- `tfidf_vectorizer.pkl` — TF-IDF feature transformer used during training

User input is preprocessed by:

1. converting text to lowercase
2. removing non-alphabetic characters
3. splitting into words
4. removing English stopwords with NLTK
5. applying Porter stemming
6. transforming the result with the saved TF-IDF vectorizer
7. passing the features to the saved classifier

The UI displays the predicted sentiment and the corresponding model probability.

## Tech stack

- Python
- Streamlit
- scikit-learn
- NLTK
- PyTorch / Transformers dependencies included in the project environment
- pandas
- sentencepiece

## Run locally

### Requirements

Use a supported Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The application also needs the NLTK English stopword corpus. It is downloaded by the current app at startup.

Start Streamlit:

```bash
streamlit run app.py
```

## Repository structure

```text
SentimentAnalysis/
├── app.py
├── requirements.txt
├── svm_sentiment_model.pkl
├── tfidf_vectorizer.pkl
└── ...
```

## Notes

The model artifacts are loaded from local files; the application does not require an external model API or API key to make predictions.

The checked-in model and vectorizer must remain compatible with each other. They should be generated using the same preprocessing and feature-extraction pipeline used by the application.

The displayed probability is the classifier's probability estimate; it should not be interpreted as a guaranteed measure of correctness.

## Data and privacy

Do not commit private datasets, credentials, personal documents, private deployment configuration, or real user data to this public repository.

The app processes the text entered into the Streamlit interface for prediction. The repository itself does not define a database or user-account system.

## Known cleanup items

The dependency file has been corrected to use the standard `nltk` package name. The project also depends on `opendatasets` for dataset access in the accompanying notebook workflow.
