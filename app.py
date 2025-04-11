from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load trained artifacts
model = joblib.load('model/xgb_model.pkl')
scaler = joblib.load('model/scaler.pkl')
optimal_threshold = joblib.load('model/threshold.pkl')

# Columns you scaled in training
scale_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# Restore Gender encoding exactly as done in training
gender_encoder = LabelEncoder()
gender_encoder.classes_ = np.array(['Female', 'Male'])

# Feature list from training
features = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Geography_France', 'Geography_Germany', 'Geography_Spain']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df_input = pd.DataFrame([data])

    # Encode 'Gender' exactly as training
    df_input['Gender'] = gender_encoder.transform(df_input['Gender'])

    # One-hot encode 'Geography' exactly as training (drop_first=False)
    df_input = pd.get_dummies(df_input, columns=['Geography'], drop_first=False)

    # Add any missing geography columns explicitly with zero values
    for col in ['Geography_France', 'Geography_Germany', 'Geography_Spain']:
        if col not in df_input:
            df_input[col] = 0

    # Ensure the column order matches exactly the training features
    df_input = df_input[features]

    # Scale numeric columns using trained scaler
    df_input_scaled = df_input.copy()
    df_input_scaled[scale_cols] = scaler.transform(df_input[scale_cols])

    # Make predictions
    probas = model.predict_proba(df_input_scaled)[:, 1]

    # Apply custom threshold
    prediction = (probas >= optimal_threshold).astype(int)

    return jsonify({
        'churn_probability': float(probas[0]),
        'churn_prediction': int(prediction[0]),
        'threshold_used': float(optimal_threshold)
    })

if __name__ == '__main__':
    app.run(debug=True)
