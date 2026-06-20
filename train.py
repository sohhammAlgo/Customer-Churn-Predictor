import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def load_data(PATH):
    df = pd.read_csv(PATH)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    return df

def preprocess_data(df):
    df['Churn'] = df['Churn'].map({
        "No": 0,
        "Yes": 1
    })

    binary_cols = [
        "Partner",
        "Dependents",
        "PaperlessBilling"
    ]
    for col in binary_cols:
        df[col] = df[col].astype(str).map({
            "No": 0,
            "Yes": 1
        })

    return df

def select_features(df):
    X = df[[
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Partner",
        "Dependents",
        "PaperlessBilling"
    ]]

    y = df["Churn"]

    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        "model/churn_model.pkl"
    )

    print("Model saved successfully!")

    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nModel Performance")
    print("-" * 50)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC Score:", roc_auc)

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.3f}"
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def show_feature_importance(model, X):
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance")
    print("-" * 50)
    print(importance_df)

def main():
    os.makedirs("model", exist_ok=True)
    DATA_PATH = "data/Telco-Customer-Churn.csv"

    print("Loading dataset...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    df = preprocess_data(df)

    X, y = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training Random Forest...")
    model = train_model(X_train, y_train)

    evaluate_model(
        model,
        X_test,
        y_test
    )

    show_feature_importance(
        model,
        X
    )

if __name__ == "__main__":
    main()