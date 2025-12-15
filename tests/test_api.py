from fastapi.testclient import TestClient
from src.api.main import app  # Adjust this import if your main.py is elsewhere

client = TestClient(app)

def test_read_root():
    """Test if the API root / returns 200 OK"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Risk API Running. Use /predict to score customers."}
def test_predict_endpoint():
    payload = {
        "Recency": 10,
        "Frequency": 25,
        "Monetary": 50000,
        "Cat_financial_services": 1,
        "Cat_airtime": 0,
        "Cat_utility_bill": 0,
        "Ch_ChannelId_1": 1,
        "Ch_ChannelId_2": 0,
        "Ch_ChannelId_3": 0,
        "Avg_Tx_Hour": 14,
        "Std_Tx_Hour": 3,
        "Channel_Diversity": 0.6,
        "Category_Diversity": 0.4,
        "Engagement_Score": 0.7
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "risk" in data
    assert "risk_probability" in data
    assert "credit_score" in data
