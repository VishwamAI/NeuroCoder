import pytest
import torch
from src.models.model_training import AdvancedNeuroCoder
from src.api import app
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def model():
    return AdvancedNeuroCoder(vocab_size=10000)

def test_api_generate_code(client, model):
    input_data = {
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "task": "generate"
    }
    response = client.post("/generate-code", json=input_data)
    assert response.status_code == 200
    assert "output" in response.json()

def test_api_feedback(client):
    feedback_data = {
        "code_id": "1234",
        "rating": 5,
        "comments": "Great code generation!"
    }
    response = client.post("/feedback", json=feedback_data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_api_benchmarks(client):
    response = client.get("/benchmarks")
    assert response.status_code == 200
    assert "benchmarks" in response.json()
    assert "comparisons" in response.json()
