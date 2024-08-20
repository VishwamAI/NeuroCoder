import pytest
import torch
from src.models.advanced_architecture import AdvancedNeuroCoder

@pytest.fixture
def model():
    return AdvancedNeuroCoder(vocab_size=10000)

def test_model_initialization(model):
    assert model is not None
    assert model.embedding.num_embeddings == 10000

def test_forward_pass(model):
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.tensor([[1, 1, 1, 1]])
    token_output, task_output = model(input_ids, attention_mask)
    assert token_output is not None and task_output is not None
    assert token_output.shape == (1, 4, 10000)  # (batch_size, sequence_length, vocab_size)
    assert task_output.shape == (1, 3)  # (batch_size, num_tasks)

def test_model_output_range(model):
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.tensor([[1, 1, 1, 1]])
    token_output, task_output = model(input_ids, attention_mask)
    assert torch.all(token_output >= 0) and torch.all(token_output <= 1)  # Assuming token_output is probabilities
    assert torch.all(task_output >= 0) and torch.all(task_output <= 1)  # Assuming task_output is probabilities

def test_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model should have parameters"

def test_model_training_mode(model):
    model.train()
    assert model.training == True
    model.eval()
    assert model.training == False
