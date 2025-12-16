import numpy as np
from fastapi.testclient import TestClient
from src.api.main import app, get_model, get_scaler


class MockModel:
    def predict(self, X):
        return np.array([0])

    def predict_proba(self, X):
        return np.array([[0.85, 0.15]])


class MockScaler:
    def transform(self, X):
        return X  # passthrough


def override_model():
    return MockModel()


def override_scaler():
    return MockScaler()


app.dependency_overrides[get_model] = override_model
app.dependency_overrides[get_scaler] = override_scaler

client = TestClient(app)
