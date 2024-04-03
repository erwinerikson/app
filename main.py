"""File Testing"""

import os
import json
import tensorflow as tf
from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def app_go():
    """Page Home"""
    return "Twitter Sentiment Analysis"

MODEL_PATH = "./model/serving_model/1711820293"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    """Page Predict"""
    request_json = request.json
    tweet = request_json.get("data")
    if type(tweet) == str:
        label = model.predict([tweet])
        prediction = tf.argmax(label[0]).numpy()
        class_labels = ['Negative', 'Positive']

        response_json = {
            "tweet": tweet,
            "result": class_labels[prediction]
        }
    else:
        response_json = {
            "tweet": tweet,
            "result": "Kesalahan memasukkan data"
        }

    return json.dumps(response_json)

if __name__ == '__main__':
    #app.run(host='127.0.0.1', port=5000)
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 5000)))
