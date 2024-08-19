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
    output = model(input_ids, attention_mask)
    assert output is not None
    assert output.shape == (1, 4, 10000)  # (batch_size, sequence_length, vocab_size)

def test_model_output_range(model):
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.tensor([[1, 1, 1, 1]])
    output = model(input_ids, attention_mask)
    assert torch.all(output >= 0) and torch.all(output <= 1)  # Assuming output is probabilities

def test_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model should have parameters"

def test_model_training_mode(model):
    model.train()
    assert model.training == True
    model.eval()
    assert model.training == False
