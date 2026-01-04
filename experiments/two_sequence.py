#!/usr/bin/env python3
"""
Two-Sequence Problem - Experiment 3 (Section 5.3)

The LSTM must classify or predict values from two real-valued input streams
with a long noise tail.

Task Description:
    Two simultaneous real-valued input streams with Gaussian noise.
    - Task 3a: Classify sequences based on signal, followed by noise tail
    - Task 3b: Information elements also have Gaussian noise
    - Task 3c: Target is conditional expectation (regression)

Paper Criteria:
    - ST1: "none of 256 test sequences misclassified"
    - ST2: "mean absolute test error below 0.01"

Paper Hyperparameters (Section 5.3):
    - Sequence length T: 100 or 1000 (random +/- 10%)
    - Learning rate: 1.0 (3a, 3b), 0.1 (3c)
    - Weight init: [-0.1, 0.1]
    - Input gate biases: -1, -3, -5
    - Output gate biases: -2, -4, -6
    - Gaussian noise: mean 0, variance 0.2

Reference: Section 5.3 of Hochreiter & Schmidhuber (1997)
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aquarius_lstm.metrics import paper_accuracy_criterion, ExperimentResult


def generate_two_sequence_data(
    task: str = "3a",
    seq_len: int = 100,
    num_samples: int = 256,
    noise_variance: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Two-Sequence Problem dataset as specified in Section 5.3.
    
    The task involves two simultaneous input streams. The network must
    remember information from the beginning of the sequence and make
    a decision at the end after a long noise tail.
    
    Args:
        task: One of "3a", "3b", or "3c"
            - 3a: Clean signal elements, followed by noise tail
            - 3b: Signal elements also have Gaussian noise
            - 3c: Target is conditional expectation (regression)
        seq_len: Base sequence length (actual length varies +/- 10%)
        num_samples: Number of sequences to generate
        noise_variance: Variance of Gaussian noise (paper uses 0.2)
        seed: Random seed for reproducibility
    
    Returns:
        X: Input sequences, shape (num_samples, max_len, 2)
           - Two input streams with signals at the start and noise tail
        Y: Targets, shape (num_samples, 1)
           - Binary classification (3a, 3b) or regression (3c)
    
    Paper reference (Section 5.3):
        "Two simultaneous real-valued input streams...
         The first few sequence elements convey task-relevant information,
         followed by a long sequence of noisy, irrelevant inputs."
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise_std = np.sqrt(noise_variance)
    
    # Actual sequence lengths vary by +/- 10%
    min_len = int(seq_len * 0.9)
    max_len = int(seq_len * 1.1) + 1
    actual_lengths = np.random.randint(min_len, max_len, num_samples)
    
    X = np.zeros((num_samples, max_len, 2), dtype=np.float32)
    Y = np.zeros((num_samples, 1), dtype=np.float32)
    
    # Number of signal elements at the start (paper: 2-4 elements)
    num_signal_elements = 2
    
    for i in range(num_samples):
        actual_len = actual_lengths[i]
        
        # Generate the two input streams
        # Stream 1 and Stream 2 each have signal elements at the start
        
        if task == "3a":
            # Task 3a: Clean signal elements, noise tail
            # Signal elements are +1 or -1 for each stream
            signal1 = np.random.choice([-1.0, 1.0], num_signal_elements)
            signal2 = np.random.choice([-1.0, 1.0], num_signal_elements)
            
            # Place signals at the start
            X[i, :num_signal_elements, 0] = signal1
            X[i, :num_signal_elements, 1] = signal2
            
            # Noise tail (Gaussian with mean 0, variance 0.2)
            X[i, num_signal_elements:actual_len, 0] = np.random.normal(
                0, noise_std, actual_len - num_signal_elements
            )
            X[i, num_signal_elements:actual_len, 1] = np.random.normal(
                0, noise_std, actual_len - num_signal_elements
            )
            
            # Target: classification based on first signal elements
            # Class 1 if both positive, Class 0 otherwise (XOR-like)
            Y[i] = 1.0 if (signal1[0] > 0 and signal2[0] > 0) else 0.0
            
        elif task == "3b":
            # Task 3b: Signal elements also have Gaussian noise
            signal1 = np.random.choice([-1.0, 1.0], num_signal_elements)
            signal2 = np.random.choice([-1.0, 1.0], num_signal_elements)
            
            # Signal with added noise
            X[i, :num_signal_elements, 0] = signal1 + np.random.normal(
                0, noise_std, num_signal_elements
            )
            X[i, :num_signal_elements, 1] = signal2 + np.random.normal(
                0, noise_std, num_signal_elements
            )
            
            # Noise tail
            X[i, num_signal_elements:actual_len, 0] = np.random.normal(
                0, noise_std, actual_len - num_signal_elements
            )
            X[i, num_signal_elements:actual_len, 1] = np.random.normal(
                0, noise_std, actual_len - num_signal_elements
            )
            
            # Same classification as 3a (based on underlying signal)
            Y[i] = 1.0 if (signal1[0] > 0 and signal2[0] > 0) else 0.0
            
        elif task == "3c":
            # Task 3c: Target is conditional expectation (regression)
            # Signal values are continuous, drawn from Gaussian
            signal1 = np.random.normal(0, 1.0, num_signal_elements)
            signal2 = np.random.normal(0, 1.0, num_signal_elements)
            
            # Signal with added noise
            X[i, :num_signal_elements, 0] = signal1 + np.random.normal(
                0, noise_std, num_signal_elements
            )
            X[i, :num_signal_elements, 1] = signal2 + np.random.normal(
                0, noise_std, num_signal_elements
            )
            
            # Noise tail
            X[i, num_signal_elements:actual_len, 0] = np.random.normal(
                0, noise_std, actual_len - num_signal_elements
            )
            X[i, num_signal_elements:actual_len, 1] = np.random.normal(
                0, noise_std, actual_len - num_signal_elements
            )
            
            # Target: conditional expectation (mean of signal products)
            # Scaled to [0, 1] range
            product = signal1[0] * signal2[0]
            Y[i] = 0.5 + 0.1 * product  # Scaled to avoid extreme values
            
        else:
            raise ValueError(f"Unknown task: {task}. Must be one of: 3a, 3b, 3c")
    
    return X, Y


def run_tinygrad(
    task: str = "3a",
    seq_len: int = 100,
    num_samples: int = 256,
    num_epochs: int = 100,
    hidden_size: int = 6,
    learning_rate: Optional[float] = None,
    seed: Optional[int] = 42,
    realize_every: int = 10,
    log_every: int = 10,
) -> ExperimentResult:
    """Run the Two-Sequence Problem experiment with tinygrad."""
    from tinygrad.tensor import Tensor
    from aquarius_lstm.cell import LSTMCell1997
    from aquarius_lstm.initialization import init_weights_paper, init_gate_biases
    
    # Set learning rate based on task if not specified
    if learning_rate is None:
        learning_rate = 0.1 if task == "3c" else 1.0
    
    # Determine criterion based on task
    criterion = "ST2" if task == "3c" else "ST1"
    
    print(f"\n{'='*60}")
    print(f"TWO-SEQUENCE PROBLEM - Task {task} (Section 5.3) - tinygrad")
    print(f"{'='*60}")
    print(f"Sequence length: {seq_len}")
    print(f"Samples: {num_samples}")
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Criterion: {criterion}")
    print(f"{'='*60}\n")
    
    # Generate data
    X, Y = generate_two_sequence_data(task, seq_len, num_samples, seed=seed)
    X_tensor = Tensor(X)
    Y_tensor = Tensor(Y)
    
    # Initialize model with paper-accurate settings
    # Input gate biases: -1, -3, -5 (distributed across cells)
    # Output gate biases: -2, -4, -6
    input_gate_biases = init_gate_biases(hidden_size, gate_type="input", 
                                          bias_values=[-1.0, -3.0, -5.0])
    output_gate_biases = init_gate_biases(hidden_size, gate_type="output",
                                           bias_values=[-2.0, -4.0, -6.0])
    
    # Use mean biases for the cell (since LSTMCell1997 takes single values)
    cell = LSTMCell1997(
        input_size=2,
        hidden_size=hidden_size,
        init_range=0.1,
        input_gate_bias=-3.0,  # Mean of [-1, -3, -5]
        output_gate_bias=-4.0,  # Mean of [-2, -4, -6]
        seed=seed,
    )
    
    # Apply paper-specified biases
    cell.b_in = Tensor(input_gate_biases, requires_grad=True)
    cell.b_out = Tensor(output_gate_biases, requires_grad=True)
    
    # Output projection
    W_out = Tensor(
        init_weights_paper((1, hidden_size), 0.1, seed),
        requires_grad=True
    )
    b_out = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)
    
    params = cell.parameters() + [W_out, b_out]
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for sample_idx in range(num_samples):
            # Initialize state
            h, s_c = cell.init_state()
            
            # Forward through sequence
            for t in range(X.shape[1]):
                x_t = X_tensor[sample_idx, t]
                h, s_c = cell(x_t, h, s_c)
                
                # Realize periodically to prevent graph explosion
                if t % realize_every == 0:
                    h = h.realize()
                    s_c = s_c.realize()
            
            # Compute prediction and loss
            pred = W_out.dot(h) + b_out
            # Apply sigmoid for classification tasks
            if task in ["3a", "3b"]:
                pred = pred.sigmoid()
            target = Y_tensor[sample_idx]
            loss = (pred - target).pow(2).sum()
            
            # Backward and update
            loss.backward()
            
            for p in params:
                if p.grad is not None:
                    p.assign(p - p.grad * learning_rate).realize()
                    p.grad = None
            
            epoch_loss += float(loss.numpy())
        
        avg_loss = epoch_loss / num_samples
        
        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    # Evaluate on all samples
    predictions = []
    for sample_idx in range(num_samples):
        h, s_c = cell.init_state()
        for t in range(X.shape[1]):
            x_t = X_tensor[sample_idx, t]
            h, s_c = cell(x_t, h, s_c)
        pred = W_out.dot(h) + b_out
        if task in ["3a", "3b"]:
            pred = pred.sigmoid()
        predictions.append(float(pred.numpy()))
    
    predictions = np.array(predictions).reshape(-1, 1)
    
    # Evaluate against paper criterion
    result = paper_accuracy_criterion("two_sequence", predictions, Y, criterion=criterion)
    print(f"\n{result}")
    
    return result


def run_torch(
    task: str = "3a",
    seq_len: int = 100,
    num_samples: int = 256,
    num_epochs: int = 100,
    hidden_size: int = 6,
    learning_rate: Optional[float] = None,
    seed: Optional[int] = 42,
    log_every: int = 10,
) -> ExperimentResult:
    """Run the Two-Sequence Problem experiment with PyTorch."""
    import torch
    from aquarius_lstm.cell_torch import LSTMCell1997Torch
    from aquarius_lstm.initialization import init_gate_biases
    
    # Paper specs: LR = 1.0 for T=100, 0.5 for T=500, 0.1 for T=1000
    if learning_rate is None:
        if task == "3c":
            learning_rate = 0.05
        elif seq_len <= 100:
            learning_rate = 0.1
        elif seq_len <= 500:
            learning_rate = 0.05
        else:
            learning_rate = 0.01
    
    # Determine criterion based on task
    criterion = "ST2" if task == "3c" else "ST1"
    
    print(f"\n{'='*60}")
    print(f"TWO-SEQUENCE PROBLEM - Task {task} (Section 5.3) - PyTorch")
    print(f"{'='*60}")
    print(f"Sequence length: {seq_len}")
    print(f"Samples: {num_samples}")
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Criterion: {criterion}")
    print(f"{'='*60}\n")
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate data
    X, Y = generate_two_sequence_data(task, seq_len, num_samples, seed=seed)
    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y)
    
    # Initialize model with paper-accurate settings
    cell = LSTMCell1997Torch(
        input_size=2,
        hidden_size=hidden_size,
        init_range=0.1,
        input_gate_bias=0.0,
        output_gate_bias=0.0,
        seed=seed,
    )
    
    W_out = torch.nn.Parameter(torch.randn(1, hidden_size) * 0.1)
    b_out = torch.nn.Parameter(torch.zeros(1))
    
    params = list(cell.parameters()) + [W_out, b_out]
    optimizer = torch.optim.SGD(params, lr=learning_rate)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for sample_idx in range(num_samples):
            optimizer.zero_grad()
            
            h, s_c = cell.init_state()
            
            for t in range(X.shape[1]):
                x_t = X_tensor[sample_idx, t]
                h, s_c = cell(x_t, h, s_c)
            
            pred = h @ W_out.T + b_out
            if task in ["3a", "3b"]:
                pred = torch.sigmoid(pred)
            target = Y_tensor[sample_idx]
            loss = ((pred - target) ** 2).sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_samples
        
        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    # Evaluate
    predictions = []
    with torch.no_grad():
        for sample_idx in range(num_samples):
            h, s_c = cell.init_state()
            for t in range(X.shape[1]):
                x_t = X_tensor[sample_idx, t]
                h, s_c = cell(x_t, h, s_c)
            pred = h @ W_out.T + b_out
            if task in ["3a", "3b"]:
                pred = torch.sigmoid(pred)
            predictions.append(pred.item())
    
    predictions = np.array(predictions).reshape(-1, 1)
    
    result = paper_accuracy_criterion("two_sequence", predictions, Y, criterion=criterion)
    print(f"\n{result}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Two-Sequence Problem (Section 5.3)")
    parser.add_argument("--mode", choices=["smoke", "paper"], default="smoke",
                       help="smoke: quick test, paper: full reproduction")
    parser.add_argument("--backend", choices=["tinygrad", "torch", "both"], 
                       default="tinygrad")
    parser.add_argument("--task", choices=["3a", "3b", "3c"], default="3a",
                       help="Task variant: 3a (clean signal), 3b (noisy signal), 3c (regression)")
    parser.add_argument("--seq-len", type=int, default=None,
                       help="Override sequence length (paper uses 100 or 1000)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Override learning rate (paper: 1.0 for 3a/3b, 0.1 for 3c)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set parameters based on mode
    if args.mode == "smoke":
        params = {
            "task": args.task,
            "seq_len": args.seq_len or 50,
            "num_samples": 64,
            "num_epochs": 30,
            "hidden_size": 6,
        }
    else:  # paper mode
        params = {
            "task": args.task,
            "seq_len": args.seq_len or 100,  # Paper uses 100 or 1000
            "num_samples": 256,  # Paper uses 256 test sequences
            "num_epochs": 100,
            "hidden_size": 6,  # Paper uses 6 memory cells (3 blocks x 2 cells)
        }
    
    if args.lr is not None:
        params["learning_rate"] = args.lr
    
    params["seed"] = args.seed
    
    results = []
    
    if args.backend in ["tinygrad", "both"]:
        try:
            result = run_tinygrad(**params)
            results.append(result)
        except ImportError as e:
            print(f"tinygrad not available: {e}")
    
    if args.backend in ["torch", "both"]:
        try:
            result = run_torch(**params)
            results.append(result)
        except ImportError as e:
            print(f"PyTorch not available: {e}")
    
    # Return exit code based on pass/fail
    passed = all(r.passed for r in results) if results else False
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
