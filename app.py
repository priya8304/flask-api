
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model1.pkl')
scaler = joblib.load('scaler1.pkl')
model = joblib.load('model2.pkl')
scaler = joblib.load('scaler2.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
