import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, GPT2Config, GPT2Model
import numpy as np
from typing import List, Dict, Any, Union
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.models.advanced_architecture import AdvancedNeuroCoder
from bayes_opt import BayesianOptimization
from torch.nn.utils.rnn import pad_sequence

from src.models.advanced_architecture import AdvancedNeuroCoder

# AdvancedNeuroCoder is now imported and will be used instead of the previous NeuroCoder class

def load_datasets():
    # Mock implementation for testing purposes
    sequence_length = 100
    task_to_label = {'code_generation': 0, 'bug_fixing': 1}
    train_data = [
        {'input_ids': torch.randint(0, 10000, (sequence_length,)), 'attention_mask': torch.ones(sequence_length), 'labels': torch.randint(0, 10000, (sequence_length,)), 'task': 'code_generation', 'task_labels': torch.tensor(task_to_label['code_generation'])},
        {'input_ids': torch.randint(0, 10000, (sequence_length,)), 'attention_mask': torch.ones(sequence_length), 'labels': torch.randint(0, 10000, (sequence_length,)), 'task': 'bug_fixing', 'task_labels': torch.tensor(task_to_label['bug_fixing'])}
    ]
    val_data = [
        {'input_ids': torch.randint(0, 10000, (sequence_length,)), 'attention_mask': torch.ones(sequence_length), 'labels': torch.randint(0, 10000, (sequence_length,)), 'task': 'code_generation', 'task_labels': torch.tensor(task_to_label['code_generation'])},
        {'input_ids': torch.randint(0, 10000, (sequence_length,)), 'attention_mask': torch.ones(sequence_length), 'labels': torch.randint(0, 10000, (sequence_length,)), 'task': 'bug_fixing', 'task_labels': torch.tensor(task_to_label['bug_fixing'])}
    ]
    return train_data, val_data

def generate_synthetic_data():
    # Mock implementation for testing purposes
    sequence_length = 100
    task_to_label = {'edge_case': 2, 'uncommon_scenario': 3}  # Continuing from previous task labels
    return [
        {'input_ids': torch.randint(0, 10000, (sequence_length,)), 'attention_mask': torch.ones(sequence_length), 'labels': torch.randint(0, 10000, (sequence_length,)), 'task': 'edge_case', 'task_labels': torch.tensor(task_to_label['edge_case'])},
        {'input_ids': torch.randint(0, 10000, (sequence_length,)), 'attention_mask': torch.ones(sequence_length), 'labels': torch.randint(0, 10000, (sequence_length,)), 'task': 'uncommon_scenario', 'task_labels': torch.tensor(task_to_label['uncommon_scenario'])}
    ]

def train_model(model: AdvancedNeuroCoder, train_loader: DataLoader, val_loader: DataLoader, config: Dict[str, Any]):
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=config['total_steps'])
    token_criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Use -100 as padding index
    task_criterion = nn.CrossEntropyLoss()

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            task_labels = batch['task_labels'].to(model.device)

            try:
                # Ensure input tensors have the correct shape
                input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
                attention_mask = attention_mask.unsqueeze(0) if attention_mask.dim() == 1 else attention_mask
                labels = labels.unsqueeze(0) if labels.dim() == 1 else labels
                task_labels = task_labels.unsqueeze(0) if task_labels.dim() == 1 else task_labels

                # Log input shapes for debugging
                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}: Input shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}")

                token_output, task_output = model(input_ids, attention_mask)

                # Ensure token_output and labels have the same shape
                if token_output.shape[1] != labels.shape[1]:
                    min_len = min(token_output.shape[1], labels.shape[1])
                    token_output = token_output[:, :min_len, :]
                    labels = labels[:, :min_len]

                # Mask out padding tokens
                mask = (labels != -100).float()
                token_loss = token_criterion(token_output.contiguous().view(-1, token_output.size(-1)), labels.contiguous().view(-1))
                token_loss = (token_loss * mask.view(-1)).sum() / mask.sum()

                task_loss = task_criterion(task_output, task_labels.squeeze())
                loss = token_loss + task_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                # Log detailed information for debugging
                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}: Token Loss: {token_loss.item():.4f}, Task Loss: {task_loss.item():.4f}")
                    print(f"Token Output Shape: {token_output.shape}, Labels Shape: {labels.shape}")
                    print(f"Task Output Shape: {task_output.shape}, Task Labels Shape: {task_labels.shape}")

            except RuntimeError as e:
                print(f"Error during training (Batch {batch_idx}): {e}")
                print(f"Input shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}")
                print(f"Labels shape: {labels.shape}, Task labels shape: {task_labels.shape}")
                continue

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)
                task_labels = batch['task_labels'].to(model.device)

                try:
                    # Ensure input tensors have the correct shape
                    input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
                    attention_mask = attention_mask.unsqueeze(0) if attention_mask.dim() == 1 else attention_mask
                    labels = labels.unsqueeze(0) if labels.dim() == 1 else labels
                    task_labels = task_labels.unsqueeze(0) if task_labels.dim() == 1 else task_labels

                    token_output, task_output = model(input_ids, attention_mask)

                    # Ensure token_output and labels have the same shape
                    if token_output.shape[1] != labels.shape[1]:
                        min_len = min(token_output.shape[1], labels.shape[1])
                        token_output = token_output[:, :min_len, :]
                        labels = labels[:, :min_len]

                    # Mask out padding tokens
                    mask = (labels != -100).float()
                    token_loss = token_criterion(token_output.contiguous().view(-1, token_output.size(-1)), labels.contiguous().view(-1))
                    token_loss = (token_loss * mask.view(-1)).sum() / mask.sum()

                    task_loss = task_criterion(task_output, task_labels.squeeze())
                    loss = token_loss + task_loss

                    total_val_loss += loss.item()
                except RuntimeError as e:
                    print(f"Error during validation (Batch {batch_idx}): {e}")
                    print(f"Input shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}")
                    print(f"Labels shape: {labels.shape}, Task labels shape: {task_labels.shape}")
                    continue

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    return model

def calculate_rewards(actions, labels):
    # Implement reward calculation based on the task and performance
    # This is a placeholder implementation
    return torch.tensor([1.0 if a == l else -1.0 for a, l in zip(actions, labels)])

def hyperparameter_optimization(model: AdvancedNeuroCoder, train_data: List, val_data: List) -> Dict[str, Any]:
    from bayes_opt import BayesianOptimization

    def objective(learning_rate, weight_decay, warmup_steps, num_epochs):
        config = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'warmup_steps': int(warmup_steps),
            'num_epochs': int(num_epochs),
            'total_steps': len(train_data) * int(num_epochs),
            'max_grad_norm': 1.0
        }
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)

        train_model(model, train_loader, val_loader, config)

        # Return negative validation loss as we want to maximize this objective
        return -model.eval_loss(val_loader)

    pbounds = {
        'learning_rate': (1e-5, 1e-3),
        'weight_decay': (0.0, 0.1),
        'warmup_steps': (100, 2000),
        'num_epochs': (5, 20)
    }

    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=1)
    optimizer.maximize(init_points=5, n_iter=20)

    best_params = optimizer.max['params']
    return {
        'learning_rate': best_params['learning_rate'],
        'weight_decay': best_params['weight_decay'],
        'warmup_steps': int(best_params['warmup_steps']),
        'num_epochs': int(best_params['num_epochs']),
        'total_steps': len(train_data) * int(best_params['num_epochs']),
        'max_grad_norm': 1.0
    }

def continuous_learning(model: AdvancedNeuroCoder, new_data: List[Dict[str, torch.Tensor]]):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    for batch in new_data:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        task = batch['task']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

        # Add regularization to preserve existing knowledge
        for param, old_param in zip(model.parameters(), model.old_params):
            loss += 0.001 * torch.sum((param - old_param) ** 2)

        loss.backward()
        optimizer.step()

    # Update old parameters
    model.old_params = [param.clone().detach() for param in model.parameters()]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedNeuroCoder(vocab_size=10000).to(device)  # Adjust vocab_size as needed
    train_data, val_data = load_datasets()
    synthetic_data = generate_synthetic_data()

    # Combine real and synthetic data
    train_data += synthetic_data

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'total_steps': 100000,
        'num_epochs': 10,
        'max_grad_norm': 1.0,
        'device': device
    }

    # Hyperparameter optimization
    optimized_config = hyperparameter_optimization(model, train_data, val_data)
    config.update(optimized_config)

    # Train the model
    train_model(model, train_loader, val_loader, config)

    # Continuous learning
    new_data = load_datasets()  # Load new data periodically
    continuous_learning(model, new_data)

    # Save the trained model
    torch.save(model.state_dict(), 'neurocoder_model.pth')
