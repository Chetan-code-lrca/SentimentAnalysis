from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model
with open("svm_sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def predict_sentiment(text):

    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)

    if prediction[0] == 1:
        return "Positive"
    else:
        return "Negative"


@app.route("/predict", methods=["POST"])
def predict():

    text = request.json["text"]
    result = predict_sentiment(text)

    return jsonify({"sentiment": result})


if __name__ == "__main__":
    app.run(debug=True)
