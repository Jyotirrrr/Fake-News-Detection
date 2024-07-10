import re

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the models and vectorizer
vectorization = joblib.load('vectorization_model.pkl')
LR = joblib.load('LR_model.pkl')
rfc = joblib.load('rfc_model.pkl')
gbc = joblib.load('gbc_model.pkl')

def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

def output_label(n):
    if n == 0:
        return "It is a Fake News"
    elif n == 1:
        return "It is a Genuine News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_gbc = gbc.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)
    
    result = {
        "LR Prediction": output_label(pred_LR[0]),
        "GBC Prediction": output_label(pred_gbc[0]),
        "RFC Prediction": output_label(pred_rfc[0])
    }
    return result

@app.route('/')
def home():
    return render_template('design.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    result = manual_testing(news)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
