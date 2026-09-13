# Sentiment Analysis App

A small Streamlit app that classifies entered text as **positive**, **negative**, or **neutral** using a trained scikit-learn SVM and TF-IDF vectorizer.

## Features

- Text sentiment prediction
- Positive, negative, and neutral labels
- Confidence score from the trained classifier
- NLTK stopword removal and Porter stemming before prediction

## How it works

The application loads two trained artifacts from the repository:

```text
svm_sentiment_model.pkl
```

and

```text
tfidf_vectorizer.pkl
```

Before prediction, the input is lowercased, non-alphabetic characters are removed, the text is split into words, English stopwords are removed, and the remaining words are stemmed with the Porter stemmer. The processed text is then transformed with the saved TF-IDF vectorizer and passed to the SVM classifier.

The model output is mapped as follows:

```text
0 → Negative
1 → Positive
2 → Neutral
```

The confidence score shown in the UI is the probability returned by the classifier for the predicted class.

## Tech stack

- Python
- Streamlit
- scikit-learn
- NLTK
- pandas
- Transformers / PyTorch
- sentencepiece
- opendatasets

`transformers`, `torch`, `sentencepiece`, `pandas`, and `opendatasets` are included in the repository's dependency file for the broader project workflow; the Streamlit app itself uses the scikit-learn model, TF-IDF vectorizer, and NLTK preprocessing pipeline.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Chetan-code-lrca/SentimentAnalysis.git
cd SentimentAnalysis
```

### 2. Create a virtual environment

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Open the local Streamlit address shown in the terminal, normally:

```text
http://localhost:8501
```

On first startup, the application downloads the NLTK English stopword corpus when it is not already available in the environment.

## Project structure

```text
SentimentAnalysis/
├── app.py
├── requirements.txt
├── svm_sentiment_model.pkl
├── tfidf_vectorizer.pkl
└── ...
```

The two `.pkl` files must remain compatible with the preprocessing and feature pipeline used when the model was trained.

## Using the app

1. Start the Streamlit application.
2. Enter text in the input box.
3. Click **Predict**.
4. The app displays the predicted sentiment and confidence score.

For example:

```text
I really enjoyed this product and would recommend it.
```

The result is determined by the saved classifier; it is not a rule-based keyword lookup.

## Model notes

The confidence value is the model's probability estimate for the selected class. It should not be interpreted as a guarantee that the classification is correct.

The model should only be used with the same text preprocessing and TF-IDF feature assumptions used during training. Training or inference changes that alter the feature representation can make the saved model incompatible with the input.

## Limitations

Sentiment classification is sensitive to context, sarcasm, mixed opinions, domain-specific language, and text that differs from the training data. Predictions should be treated as model outputs rather than definitive judgments about the writer's intent.

## License

No license file is included in the repository.