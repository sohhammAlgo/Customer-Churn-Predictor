from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0.0"
)

# Load trained model
model = joblib.load("model/churn_model.pkl")


class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Partner: int
    Dependents: int
    PaperlessBilling: int


@app.get("/")
def home():
    return {
        "message": "Telco Churn Prediction API Running"
    }


@app.post("/predict")
def predict(data: CustomerData):

    features = pd.DataFrame(
        [[
            data.tenure,
            data.MonthlyCharges,
            data.TotalCharges,
            data.Partner,
            data.Dependents,
            data.PaperlessBilling
        ]],
        columns=[
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Partner",
            "Dependents",
            "PaperlessBilling"
        ]
    )

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "result": (
            "Customer Likely To Churn"
            if prediction == 1
            else "Customer Not Likely To Churn"
        )
    }