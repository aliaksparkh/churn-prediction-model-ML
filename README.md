# Customer Churn Prediction Project
<p align="center">
  <img src="https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/cust_churn.png" width="400">
</p>

This repository presents an end-to-end machine learning pipeline to **predict customer churn** using an **XGBoost classifier**, featuring data preprocessing, exploratory analysis, model evaluation, and deployment via a Flask API.

## 🎯 Objective

- Accurately predict **customer churn** to help businesses reduce revenue loss and retain clients.
- Develop and deploy a robust, scalable, and accessible prediction model as an API.

## 📁 Project Structure

- [`model/`](https://github.com/aliaksparkh/churn-prediction-model-ML/tree/main/model) — Directory containing saved model artifacts:
  - [`xgb_model.pkl`](https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/model/xgb_model.pkl) — Trained XGBoost model
  - [`scaler.pkl`](https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/model/scaler.pkl) — StandardScaler used for feature scaling
  - [`threshold.pkl`](https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/model/threshold.pkl) — Optimized threshold for classification

- [`app.py`](https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/app.py) — Flask API script for serving the model

- [`requirements.txt`](https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/requirements.txt) — Python dependencies for the project

- [`churn-prediction.ipynb`](https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/Customer_Churn_ML_Models.ipynb) — Jupyter notebook with exploratory analysis, preprocessing, and model training


## ⚙️ Model Development

- **Preprocessing** steps included:
  - Dropping irrelevant columns (`CustomerId`, `Surname`)
  - Encoding categorical variables (`Geography`, `Gender`)
  - Feature scaling using `StandardScaler`

- **Model Training**:
  - A total of **7 classification models** were trained and evaluated:
    - Logistic Regression
    - K-Nearest Neighbors
    - Decision Tree
    - Random Forest
    - Support Vector Machine
    - Gradient Boosting
    - **XGBoost** (selected as the final model based on performance)
  
- Evaluation metrics used:
  - Accuracy, Precision, Recall, F1-score
  - ROC AUC
  - Confusion matrix
  - Optimal threshold adjustment to improve performance

 ## 🚀 Deployment

- The final model is deployed using **Flask**, enabling real-time predictions through a RESTful API.
- The API accepts customer data in JSON format and returns a churn prediction.

## ▶️ How to Run

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   
2. **Start the Flask API** 

   python app.py

4. **Make predictions**

Send a POST request to http://localhost:5000/predict with customer data in JSON format.

**Example using cURL**:
  
curl -X POST http://127.0.0.1:5000/predict \
-H 'Content-Type: application/json' \
-d '{"CreditScore": 600, "Age": 40, "Tenure": 5, "Balance": 50000, "NumOfProducts": 2, "EstimatedSalary": 100000, "Geography": "France", "Gender": "Male", "HasCrCard": 1, "IsActiveMember": 1}'

Sample API Response:
{
  "churn_probability": 0.08523,
  "churn_prediction": 0,
  "threshold_used": 0.37769
}
