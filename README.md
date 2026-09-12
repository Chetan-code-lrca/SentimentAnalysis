# Sentiment Analysis App

A Streamlit application that classifies text as **positive**, **negative**, or **neutral** using a trained scikit-learn SVM and TF-IDF features.

## How it works

The app loads:

- `svm_sentiment_model.pkl` — trained SVM classifier
- `tfidf_vectorizer.pkl` — TF-IDF vectorizer used during training

Input text is processed by lowercasing, removing non-alphabetic characters, tokenizing, removing English stopwords with NLTK, applying Porter stemming, and transforming the result with the saved vectorizer before prediction.

The UI displays the predicted sentiment and the classifier's probability estimate.

## Tech stack

- Python
- Streamlit
- scikit-learn
- NLTK
- pandas
- Transformers / PyTorch dependencies
- sentencepiece
- opendatasets for the dataset notebook workflow

## Run locally

```bash
git clone https://github.com/Chetan-code-lrca/SentimentAnalysis.git
cd SentimentAnalysis
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, normally `http://localhost:8501`.

The application downloads the NLTK English stopword corpus at startup when needed.

## Repository structure

```text
SentimentAnalysis/
├── app.py
├── requirements.txt
├── svm_sentiment_model.pkl
├── tfidf_vectorizer.pkl
└── ...
```

## Model notes

The classifier and TF-IDF vectorizer must stay compatible with the same preprocessing pipeline used during training.

The probability shown by the UI is the model's probability estimate; it is not a guarantee that the prediction is correct.

## Data and privacy

The application predicts sentiment from text entered into the Streamlit interface and does not define a user-account or database system. Keep private datasets, personal documents, credentials, and other sensitive data out of the public repository.
