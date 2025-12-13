from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import json  # <--- Added this to help clean up the data

app = FastAPI(
    title="Credit score  API",
    description="Predict high or low credit risk for a customer",
    version="1.0.0"
)

# ---------------------------------------------------
# 1. Load trained model + determine required columns
# ---------------------------------------------------
try:
    model = joblib.load("models/model.pkl")
    print("Loaded RandomForest model.")
except Exception as e:
    print("Error loading model:", e)
    model = None

# Load feature names from processed dataset
try:
    training_df = pd.read_csv("Data/Processed/customer_risk_data_final.csv")
    feature_columns = training_df.drop(columns=["is_high_risk", "CustomerId"]).columns.tolist()
    print("Loaded training feature names.")
except:
    feature_columns = []
    print("Warning: Could not load training features.")


# ---------------------------------------------------
# 2. Flexible Pydantic model
# ---------------------------------------------------
class CustomerFeatures(BaseModel):
    """Flexible: API accepts ANY incoming features."""

    class Config:
        extra = "allow"  # allow arbitrary fields


@app.get("/")
def home():
    return {"message": "Credit Risk API Running. Use /predict to score customers."}


# ---------------------------------------------------
# 3. Prediction
# ---------------------------------------------------
@app.post("/predict")
def predict_risk(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Convert to dict then DataFrame
    data = pd.DataFrame([features.dict()])

    # ---------------------------------------------------
    # Align incoming features with model’s expected columns
    # ---------------------------------------------------
    missing_cols = [col for col in feature_columns if col not in data.columns]
    for col in missing_cols:
        data[col] = 0  # default value

    # Remove unknown columns
    data = data[feature_columns]

    # ---------------------------------------------------
    # Predict
    # ---------------------------------------------------
    try:
        # Ensure data columns order matches training features
        data = data[feature_columns]

        # Raw model outputs
        pred = model.predict(data)[0]
        proba = model.predict_proba(data)[0]  # array of probabilities

        # --- FIX 1: Convert numpy classes to standard Python integers ---
        classes = [int(c) for c in model.classes_]

        proba_by_class = {str(cls): float(proba[i]) for i, cls in enumerate(classes)}

        # Find index of class '1' (High Risk). If '1' not present, fallback
        if 1 in classes:
            idx_high = classes.index(1)
            high_risk_prob = float(proba[idx_high])
        else:
            # no explicit 1 class (rare) -> fall back to second column or max
            high_risk_prob = float(proba.max())

        # Convert to a more realistic credit score (logistic mapping)
        score = 300 + 550 / (1 + (2.718281828459045 ** (8 * (high_risk_prob - 0.5))))
        credit_score = int(score)

        # --- FIX 2: Convert features to clean standard Python dict ---
        # We use json.loads(to_json()) to scrub all numpy types automatically
        clean_features = json.loads(data.iloc[0].to_json())

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model prediction failed: {str(e)}")

    return {
        "risk": int(pred),
        "risk_probability": high_risk_prob,
        "probabilities": proba_by_class,
        "classes": classes,
        "scored_features": clean_features,
        "label": "High Risk" if pred == 1 else "Low Risk",
        "credit_score": int(float(credit_score))
    }