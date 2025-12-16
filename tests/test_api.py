from tests.conftest import client


def test_predict_endpoint():
    payload = {
        "Recency": 15,
        "Frequency": 10,
        "Monetary": 30000,
        "Channel_Diversity": 0.4,
        "Category_Diversity": 0.3,
        "Engagement_Score": 0.2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["label"] == "Low Risk"
    assert 300 <= data["credit_score"] <= 850
from tests.conftest import client


def test_predict_endpoint():
    payload = {
        "Recency": 15,
        "Frequency": 10,
        "Monetary": 30000,
        "Channel_Diversity": 0.4,
        "Category_Diversity": 0.3,
        "Engagement_Score": 0.2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["label"] == "Low Risk"
    assert 300 <= data["credit_score"] <= 850
