
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load('models/xgb_credit_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    probability = model.predict_proba(df)[:, 1][0]
    score = int(np.clip(
        600 - 20/np.log(2) * np.log(50) + 20/np.log(2) * np.log((1 - probability)/(probability + 1e-10)),
        300, 900
    ))
    risk_category = (
        'Very High' if score < 500 else
        'High' if score < 600 else
        'Medium' if score < 700 else
        'Low' if score < 800 else
        'Very Low'
    )

    return jsonify({
        'credit_score': score,
        'default_probability': round(float(probability), 4),
        'risk_category': risk_category,
        'decision': 'Approve' if score >= 650 else 'Review' if score >= 550 else 'Decline'
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
