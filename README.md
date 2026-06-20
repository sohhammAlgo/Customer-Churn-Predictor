# 📉 Telco Customer Churn Prediction

## 📌 Overview

Customer churn is a major challenge for subscription-based businesses. This project predicts whether a telecom customer is likely to leave the service based on account information, billing details, and customer demographics.

The project demonstrates a complete Machine Learning lifecycle:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model Training
* Model Evaluation
* Model Persistence using Joblib
* FastAPI Deployment

---

# 🎯 Problem Statement

Telecom companies lose revenue when customers discontinue their services. Identifying customers at risk of churn enables businesses to take proactive retention measures.

The objective of this project is to build a Machine Learning model capable of predicting customer churn using historical customer data.

---

# 📂 Dataset

**Dataset:** Telco Customer Churn Dataset

## Features Used

| Feature          | Description                          |
| ---------------- | ------------------------------------ |
| tenure           | Number of months customer stayed     |
| MonthlyCharges   | Monthly subscription charges         |
| TotalCharges     | Total amount charged                 |
| Partner          | Whether customer has a partner       |
| Dependents       | Whether customer has dependents      |
| PaperlessBilling | Whether paperless billing is enabled |

## Target Variable

| Column | Description                                 |
| ------ | ------------------------------------------- |
| Churn  | 1 = Customer Churned, 0 = Customer Retained |

---

# ⚙️ Project Workflow

## 1. Data Preprocessing

### Tasks Performed

* Converted `TotalCharges` to numeric values
* Handled missing values using median imputation
* Encoded categorical features
* Selected relevant features for training

Example:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)
```

---

## 2. Exploratory Data Analysis (EDA)

Performed:

* Dataset inspection
* Missing value analysis
* Histograms
* Correlation analysis
* Churn distribution analysis
* Feature importance analysis

---

## 3. Model Training

### Algorithm Used

```text
Random Forest Classifier
```

### Configuration

```python
RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)
```

---

## 4. Model Evaluation

Metrics Used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

Example:

```python
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
```

---

## 5. Model Persistence

The trained model is stored using Joblib.

```python
joblib.dump(
    model,
    "model/churn_model.pkl"
)
```

Load later:

```python
model = joblib.load(
    "model/churn_model.pkl"
)
```

---

# 🚀 FastAPI Deployment

The model is deployed using FastAPI and provides real-time churn predictions.

## Endpoint

```http
POST /predict
```

### Sample Request

```json
{
  "tenure": 12,
  "MonthlyCharges": 80,
  "TotalCharges": 1000,
  "Partner": 1,
  "Dependents": 0,
  "PaperlessBilling": 1
}
```

### Sample Response

```json
{
  "prediction": 1,
  "churn_probability": 0.6333,
  "result": "Customer Likely To Churn"
}
```

---

# 📁 Project Structure

```text
telco-churn-prediction/
│
├── app.py
├── train.py
├── requirements.txt
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── model/
│   └── churn_model.pkl
│
├── notebooks/
│   └── churn_analysis.ipynb
│
└── README.md
```

---

# 🛠 Installation

## Clone Repository

```bash
git clone <repository-url>
cd telco-churn-prediction
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training the Model

Run:

```bash
python train.py
```

Output:

```text
model/churn_model.pkl
```

---

# 🌐 Running FastAPI

Start the API server:

```bash
uvicorn app:app --reload
```

Server URL:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 📊 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* FastAPI
* Uvicorn
* Joblib

---

# 📈 Future Improvements

* Hyperparameter Tuning using GridSearchCV
* Feature Engineering
* XGBoost Integration
* Docker Containerization
* CI/CD Pipeline
* Cloud Deployment (AWS, Azure, GCP)

---

# 🏆 Results

The Random Forest model successfully predicts customer churn and exposes predictions through a FastAPI endpoint.

### Achievements

* End-to-End Machine Learning Pipeline
* Automated Model Persistence
* REST API Deployment
* Production-Ready Project Structure
* Real-Time Customer Churn Prediction

---