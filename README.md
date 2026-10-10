# Sentiment Analysis App

A Streamlit demo that classifies text as positive, negative, or neutral using a saved scikit-learn SVM classifier and TF-IDF vectorizer.

The repository contains the trained artifacts (`svm_sentiment_model.pkl` and `tfidf_vectorizer.pkl`) and the notebook (`SVM.ipynb`) used for the broader data collection and training workflow.

## Current app behavior

- Preprocesses input with lowercasing, non-letter removal, NLTK English stopword removal, and Porter stemming.
- Produces the saved classifier's predicted class and class probabilities.
- Maps classes as 0 = Negative, 1 = Positive, and 2 = Neutral.
- Rejects empty text and text that becomes empty after preprocessing.
- Loads model files relative to the repository, independent of the directory used to start Streamlit.

The classifier confidence is a model probability estimate, not a guarantee of correctness. Sentiment analysis can fail on sarcasm, context-dependent statements, mixed opinions, and out-of-domain text.

## Requirements

- Python 3.11 or newer
- The saved model and vectorizer in the repository root
- NLTK's English stopword corpus installed once in the Python environment

## Run locally

Create and activate a virtual environment, then install requirements:

    python -m venv .venv
    # Linux / macOS
    source .venv/bin/activate
    # Windows PowerShell
    .venv\Scripts\Activate.ps1

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m nltk.downloader stopwords
    streamlit run app.py

The app does not download NLTK resources when it starts. After the corpus has been installed, inference can run without a network connection.

## Notebook workflow

`SVM.ipynb` demonstrates the broader workflow, including dataset acquisition, exploratory analysis, preprocessing, train/test split, TF-IDF vectorization, SVM training, evaluation, and artifact serialization. Dataset acquisition through `opendatasets` requires network access and the appropriate dataset credentials; it is not required to run the saved-model app.

The Streamlit app does not use Transformers, PyTorch, or SentencePiece, so those heavy dependencies are not installed by the app's requirements. Pandas, Matplotlib, scikit-learn, NLTK, and opendatasets support the notebook workflow.

## Tests and CI

Run the tests with:

    python -m pytest -q

GitHub Actions checks Python syntax and tests the preprocessing helpers and the compatibility of the saved model/vectorizer artifacts. CI does not download the training dataset or require external credentials.

## Artifact trust

Pickle files can execute code during deserialization. Only load model artifacts from this trusted repository or another source whose origin you have verified. Do not replace the pickle files with untrusted uploads.

## Limitations

- The current app is a small demonstration, not a production API.
- The saved model's quality depends on the training data and preprocessing used when it was created.
- The repository does not include a license file; confirm licensing before redistributing the code, notebook, model artifacts, or data.
