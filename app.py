from flask import Flask, request, jsonify
import pandas as pd
from skops.io import load, get_untrusted_types

app = Flask(__name__)

trusted = get_untrusted_types(file="models/model.skops")
model = load("models/model.skops", trusted=trusted)


@app.route("/")
def home():
    return jsonify({
        "message": "IRIS Prediction API - Version 2 (GitHub Actions CD)"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = pd.DataFrame(
        [data["features"]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    )

    prediction = model.predict(features)

    return jsonify({
        "prediction": int(prediction[0])
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
