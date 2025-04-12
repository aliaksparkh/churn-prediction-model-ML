# 🧠 Customer Churn Prediction

<p align="center">
  <img src="https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/cust_churn.png" width="400">
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Made with Python">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="MIT License">
  </a>
</p>

This repository presents a complete machine learning pipeline for **customer churn prediction** using an **XGBoost classifier**. The project includes data preprocessing, exploratory data analysis, model development, evaluation, and deployment via a Flask API.

---

## 🎯 Objective

- Develop a predictive model to **identify customers likely to churn**, enabling businesses to implement proactive retention strategies.
- Build and deploy a **robust, scalable, and production-ready API** for real-time inference.

---

## 📁 Project Structure

- [`model/`](https://github.com/aliaksparkh/churn-prediction-model-ML/tree/main/model)  
  Contains saved model artifacts:
  - [`xgb_model.pkl`](https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/model/xgb_model.pkl) — Trained XGBoost classifier  
  - [`scaler.pkl`](https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/model/scaler.pkl) — `StandardScaler` object for feature normalization  
  - [`threshold.pkl`](https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/model/threshold.pkl) — Optimized decision threshold for classification  

- [`app.py`](https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/app.py)  
  Flask application serving the model via a REST API

- [`requirements.txt`](https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/requirements.txt)  
  List of required Python packages

- [`Customer_Churn_ML_Models.ipynb`](https://github.com/aliaksparkh/churn-prediction-model-ML/blob/main/Customer_Churn_ML_Models.ipynb)  
  Jupyter Notebook detailing exploratory analysis, preprocessing steps, model training, and evaluation

---

## ⚙️ Model Development

### 🔄 Preprocessing

- Removal of irrelevant features (`CustomerId`, `Surname`)  
- Encoding categorical variables (`Geography`, `Gender`)  
- Outlier detection and treatment  
- Feature scaling using `StandardScaler`  
- Class imbalance handled with **SMOTE**  

### 📊 Exploratory Data Analysis

- **Gender**: Female customers show higher churn (25.1%) than males (16.5%)  
- **Credit Card**: Churn rate nearly identical regardless of credit card ownership  
- **Activity**: Inactive members have significantly higher churn (26.9%) compared to active ones (14.3%)  
- **Geography**: Customers from Germany churn the most (32%), nearly double the rate of those from France and Spain (~16%)  

### 🧪 Model Training & Evaluation

Seven classification algorithms were evaluated:

- Logistic Regression  
- K-Nearest Neighbors  
- Decision Tree  
- Random Forest  
- Support Vector Machine  
- Gaussian Naive Bayes  
- **XGBoost** — selected as the final model based on superior performance

**Evaluation Metrics**:
- Accuracy: **0.849**
- ROC AUC: **0.857**
- Cross-validated ROC AUC: **0.961 ± 0.003**
- Additional metrics: Precision, Recall, F1-score, Confusion Matrix  
- **Optimal threshold** tuning (F1-maximizing threshold: **0.38**) to balance precision and recall

---

## 🚀 Deployment

The trained model is deployed using **Flask**, allowing real-time predictions through a **RESTful API**. The API accepts customer data in JSON format and returns churn probability and prediction based on the optimized threshold.

---

## ▶️ How to Run

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   
2. **Start the Flask API** 

   <pre> ```bash python app.py ``` </pre>
  
3. **Make predictions**

    Send a POST request to http://localhost:5000/predict with customer data in JSON format.

  **Example using cURL**:
  <pre> ```bash
 curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "CreditScore": 600,
        "Age": 40,
        "Tenure": 5,
        "Balance": 50000,
        "NumOfProducts": 2,
        "EstimatedSalary": 100000,
        "Geography": "France",
        "Gender": "Male",
        "HasCrCard": 1,
        "IsActiveMember": 1
      }'
  </pre>
  
  Sample API Response:
<pre> ```bash
  {
  "churn_probability": 0.08523,
  "churn_prediction": 0,
  "threshold_used": 0.37769
  }
</pre>

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this project with proper attribution.
