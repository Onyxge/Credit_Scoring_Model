import os
import joblib
import mlflow.sklearn
import pandas as pd
import concurrent.futures  # <--- Standard library for handling timeouts
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CreditRequest, CreditResponse

app = FastAPI(
    title="Credit Risk Scoring API",
    version="2.0.0"
)

# -------------------------------
# CONFIGURATION
# -------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REMOTE_MODEL_URI = "models:/CreditRiskModel/1"
LOCAL_MODEL_PATH = "models/model.pkl"
LOCAL_SCALER_PATH = "models/rfm_scaler.pkl"
MLFLOW_TIMEOUT_SECONDS = 20  # <--- Maximum wait time for MLflow

model = None
scaler = None


# -------------------------------
# HELPER: Wrapper to load model
# -------------------------------
def load_remote_model():
    """Helper function to run inside a thread"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(REMOTE_MODEL_URI)


# -------------------------------
# LOAD RESOURCES
# -------------------------------
print("--- Starting Service Initialization ---")

# 1. Load Scaler
if os.path.exists(LOCAL_SCALER_PATH):
    try:
        scaler = joblib.load(LOCAL_SCALER_PATH)
        print(f"✅ Scaler loaded from {LOCAL_SCALER_PATH}")
    except Exception as e:
        print(f"⚠️  Found scaler but failed to load: {e}")
else:
    print(f"⚠️  Warning: No scaler found.")

# 2. Load Model (Hybrid with Timeout)
try:
    print(f"📡 Attempting to load from MLflow Registry (Timeout: {MLFLOW_TIMEOUT_SECONDS}s)...")

    # We use a ThreadPoolExecutor to run the load function with a strict deadline
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_remote_model)
        model = future.result(timeout=MLFLOW_TIMEOUT_SECONDS)

    print("✅ Success: Loaded model from MLflow Registry!")

except (concurrent.futures.TimeoutError, Exception) as e:
    # This block catches BOTH the Timeout and any Connection Errors
    error_msg = "Timed out" if isinstance(e, concurrent.futures.TimeoutError) else str(e)
    print(f"⚠️  MLflow load failed ({error_msg}).")
    print("🔄 Switching to Local Fallback...")

    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            model = joblib.load(LOCAL_MODEL_PATH)
            print(f"✅ Success: Loaded model from local fallback: {LOCAL_MODEL_PATH}")
        except Exception as local_e:
            raise RuntimeError(f"❌ CRITICAL: Local load failed. Error: {local_e}")
    else:
        raise RuntimeError(f"❌ CRITICAL: MLflow failed and local file not found.")



@app.get("/")
def health():
    # Helper to see which model is currently active
    return {
        "status": "API running",
        "model_loaded": "Yes" if model else "No"
    }


@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # 1. Create DataFrame
        df = pd.DataFrame([request.dict()])

        # 2. Apply Scaler (If available)
        if scaler:
            # Important: Ensure columns match what scaler expects
            input_data = scaler.transform(df)
        else:
            input_data = df

        # 3. Predict
        pred = int(model.predict(input_data)[0])

        # 4. Get Probability (Safely)
        try:
            # Some models return [prob_0, prob_1], we want prob_1 (High Risk)
            prob = float(model.predict_proba(input_data)[0][1])
        except (AttributeError, IndexError):
            # If model doesn't support probability, default to 0.0 or 1.0 based on prediction
            prob = float(pred)

        # 5. Calculate Custom Credit Score
        credit_score = int(850 - prob * 550)

        return {
            "risk": pred,
            "risk_probability": prob,
            "label": "High Risk" if pred == 1 else "Low Risk",
            "credit_score": credit_score
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))