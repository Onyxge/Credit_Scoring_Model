from fastapi.testclient import TestClient
from src.api.main import app  # Adjust this import if your main.py is elsewhere

client = TestClient(app)

def test_read_root():
    """Test if the API root / returns 200 OK"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Risk API Running. Use /predict to score customers."}

def test_predict_endpoint():
    """Test a sample prediction to ensure model works"""
    payload = {
        "age": 30,
        "income": 50000,
        "loan_amount": 10000,
        "loan_tenure_months": 12,
        "avg_dpd_per_delinquency": 0,
        "delinquency_ratio": 0,
        "credit_utilization_ratio": 0.3,
        "num_open_accounts": 2,
        "residence_type": "Owned",
        "loan_purpose": "Education",
        "loan_type": "Unsecured"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "risk_probability" in response.json()