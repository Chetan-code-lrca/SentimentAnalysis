import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Add this at the top of app.py
st.set_page_config(page_title="Chetan's Sentiment Analyzer", page_icon="🧠")

# ... inside the main section ...
st.title("🧠 Sentiment Analysis App")
st.markdown("### Built by Chetan Inaganti | NLP Developer")
st.info("💼 Looking for custom NLP solutions? [https://www.linkedin.com/in/chetan-inaganti-563a41322/](https://www.linkedin.com/in/chetan-inaganti-563a41322/)")

# ... at the bottom of the app ...
st.markdown("---")
st.caption("Powered by HuggingFace Transformers & Streamlit | © 2026 Chetan Inaganti")


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load model and vectorizer
model = pickle.load(open("svm_sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def preprocess(text):
    # Lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Tokenization
    words = text.split()

    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    return " ".join(words)

def predict_sentiment(text):

    # Convert text to TF-IDF features
    cleaned = preprocess(text)
    text_tfidf = vectorizer.transform([cleaned]).toarray()

    # Predict sentiment
    prediction = model.predict(text_tfidf)[0]

    # Get probabilities
    probs = model.predict_proba(text_tfidf)[0]

    positive_prob = probs[1]
    negative_prob = probs[0]
    neutral_prob = probs[2]

    return prediction, positive_prob, negative_prob, neutral_prob

st.write("## Sentiment Analysis App")

user_input = st.text_area("Enter text:")

if st.button("Predict"):

    prediction, pos_prob, neg_prob, neutral_prob = predict_sentiment(user_input)

    if prediction == 1:
        sentiment = "Positive"
        prob = pos_prob
    elif prediction == 0:
        sentiment = "Negative"
        prob = neg_prob
    else:
        sentiment = "Neutral"
        prob = neutral_prob

    st.write("### Prediction:", sentiment)
    st.write("### Confidence Score:", round(prob, 2))
