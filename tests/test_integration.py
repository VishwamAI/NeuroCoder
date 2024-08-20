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
        "input_ids": [[1, 2, 3, 4]],  # Add batch dimension
        "attention_mask": [[1, 1, 1, 1]],  # Add batch dimension
        "task": "generate"
    }
    response = client.post("/generate-code", json=input_data)
    assert response.status_code == 200
    assert "token_output" in response.json()
    assert "task_output" in response.json()

    token_output = response.json()["token_output"]
    task_output = response.json()["task_output"]

    assert isinstance(token_output, list)
    assert isinstance(task_output, list)
    assert len(token_output) == 1  # Batch size
    assert len(token_output[0]) == 4  # Sequence length
    assert len(task_output) == 1  # Batch size

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
