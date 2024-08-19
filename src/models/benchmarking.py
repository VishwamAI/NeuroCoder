import time
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from src.models.advanced_architecture import AdvancedNeuroCoder
from torch.utils.data import DataLoader
from tqdm import tqdm

def benchmark_accuracy(model: AdvancedNeuroCoder, test_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Benchmarking accuracy"):
            outputs = model(batch['input_ids'], batch['attention_mask'])
            _, predicted = torch.max(outputs.data, 1)
            total += batch['labels'].size(0)
            correct += (predicted == batch['labels']).sum().item()
    return correct / total

def benchmark_speed(model: AdvancedNeuroCoder, test_loader: DataLoader) -> float:
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Benchmarking speed"):
            _ = model(batch['input_ids'], batch['attention_mask'])
    end_time = time.time()
    return end_time - start_time

def benchmark_efficiency(model: AdvancedNeuroCoder) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_benchmarks(model: AdvancedNeuroCoder, test_loader: DataLoader) -> Dict[str, float]:
    accuracy = benchmark_accuracy(model, test_loader)
    speed = benchmark_speed(model, test_loader)
    efficiency = benchmark_efficiency(model)

    return {
        'accuracy': accuracy,
        'speed': speed,
        'efficiency': efficiency
    }

def compare_to_industry_standards(benchmark_results: Dict[str, float]) -> Dict[str, str]:
    industry_standards = {
        'accuracy': 0.85,
        'speed': 10.0,  # seconds
        'efficiency': 1000000  # parameters
    }

    comparisons = {}
    for metric, value in benchmark_results.items():
        if metric == 'accuracy':
            comparisons[metric] = 'Above' if value > industry_standards[metric] else 'Below'
        elif metric == 'speed':
            comparisons[metric] = 'Faster' if value < industry_standards[metric] else 'Slower'
        elif metric == 'efficiency':
            comparisons[metric] = 'More efficient' if value < industry_standards[metric] else 'Less efficient'

    return comparisons

def visualize_benchmarks(benchmark_results: Dict[str, float]):
    metrics = list(benchmark_results.keys())
    values = list(benchmark_results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values)
    plt.title('NeuroCoder Benchmark Results')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.savefig('benchmark_results.png')
    plt.close()

def interactive_mode(model: AdvancedNeuroCoder):
    print("Welcome to NeuroCoder Interactive Mode!")
    while True:
        user_input = input("Enter your code snippet (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Process user input and generate code
        input_ids = torch.tensor([model.tokenizer.encode(user_input)])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = model(input_ids, attention_mask)

        generated_code = model.tokenizer.decode(output[0])
        print("Generated Code:")
        print(generated_code)

        # Get user feedback
        feedback = input("Was this output helpful? (yes/no): ")
        if feedback.lower() == 'no':
            print("We're sorry the output wasn't helpful. Your feedback will be used to improve the model.")
            # Here you would typically log the feedback for later analysis

def error_analysis(model: AdvancedNeuroCoder, test_loader: DataLoader) -> List[Dict[str, Any]]:
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Analyzing errors"):
            outputs = model(batch['input_ids'], batch['attention_mask'])
            _, predicted = torch.max(outputs.data, 1)

            # Find incorrect predictions
            incorrect_mask = predicted != batch['labels']
            for i in range(len(incorrect_mask)):
                if incorrect_mask[i]:
                    errors.append({
                        'input': model.tokenizer.decode(batch['input_ids'][i]),
                        'expected': model.tokenizer.decode(batch['labels'][i]),
                        'predicted': model.tokenizer.decode(predicted[i])
                    })
    return errors

if __name__ == "__main__":
    model = AdvancedNeuroCoder(vocab_size=10000)  # Adjust vocab_size as needed
    model.load_state_dict(torch.load('neurocoder_model.pth'))

    # Load test data
    from src.models.model_training import load_datasets
    _, test_data = load_datasets()
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Run benchmarks
    benchmark_results = run_benchmarks(model, test_loader)
    comparisons = compare_to_industry_standards(benchmark_results)

    print("Benchmark Results:")
    for metric, value in benchmark_results.items():
        print(f"{metric.capitalize()}: {value}")

    print("\nComparison to Industry Standards:")
    for metric, comparison in comparisons.items():
        print(f"{metric.capitalize()}: {comparison} than industry standard")

    # Visualize benchmarks
    visualize_benchmarks(benchmark_results)

    # Interactive mode
    interactive_mode(model)

    # Error analysis
    errors = error_analysis(model, test_loader)
    print(f"\nFound {len(errors)} errors. First 5 errors:")
    for error in errors[:5]:
        print(f"Input: {error['input']}")
        print(f"Expected: {error['expected']}")
        print(f"Predicted: {error['predicted']}")
        print()
