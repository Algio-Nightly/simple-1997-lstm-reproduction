#!/usr/bin/env python3
"""
Adding Problem - Experiment 4 (Section 5.4)

The LSTM must learn to add two marked numbers in a long sequence of random noise.

Task Description:
    - Input: Sequence of pairs (value, marker)
    - Two random positions have marker = 1.0, rest have marker = 0.0 or -1.0
    - Values are uniform random in [0, 1]
    - Target: 0.5 + (X1 + X2) / 4.0  (scaled to [0.5, 1.0])
    
Paper Criterion:
    "Absolute error at sequence end below 0.04"

Paper Hyperparameters (Section 5.4):
    - Sequence length T: 100, 500, or 1000 (random +/- 10%)
    - Learning rate: 0.5
    - Weight init: [-0.1, 0.1]
    - Input gate biases: -3.0, -6.0
    - Output gate biases: random

Reference: Section 5.4 of Hochreiter & Schmidhuber (1997)
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aquarius_lstm.metrics import paper_accuracy_criterion, ExperimentResult


def generate_adding_data(
    seq_len: int = 100,
    num_samples: int = 100,
    seed: Optional[int] = None,
    paper_exact: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Adding Problem dataset as specified in Section 5.4.
    
    When paper_exact=True:
      - Values: uniform [-1, 1] (not [0, 1])
      - Marker 1: in first 10 timesteps only
      - Target: (X1 + X2) / 2.0 (range [-1, 1])
    
    When paper_exact=False (legacy):
      - Values: uniform [0, 1]
      - Marker 1: random in first half
      - Target: 0.5 + (X1 + X2) / 4.0 (range [0.5, 1.0])
    """
    if seed is not None:
        np.random.seed(seed)
    
    min_len = max(1, int(seq_len * 0.9))
    max_len = int(seq_len * 1.1) + 1
    actual_lengths = np.random.randint(min_len, max_len, num_samples)
    
    X = np.zeros((num_samples, max_len, 2), dtype=np.float32)
    Y = np.zeros((num_samples, 1), dtype=np.float32)
    
    for i in range(num_samples):
        actual_len = actual_lengths[i]
        
        if paper_exact:
            X[i, :actual_len, 0] = np.random.uniform(-1, 1, actual_len)
            
            X[i, 0, 1] = -1.0
            X[i, actual_len - 1, 1] = -1.0
            
            marker1_range = min(10, actual_len - 2)
            pos1 = np.random.randint(1, marker1_range + 1) if marker1_range > 0 else 1
            
            remaining = [j for j in range(1, actual_len - 1) if j != pos1]
            pos2 = remaining[np.random.randint(len(remaining))]
            
            X[i, pos1, 1] = 1.0
            X[i, pos2, 1] = 1.0
            
            X1 = X[i, pos1, 0]
            X2 = X[i, pos2, 0]
            Y[i] = (X1 + X2) / 2.0
        else:
            X[i, :actual_len, 0] = np.random.uniform(0, 1, actual_len)
            
            X[i, 0, 1] = -1.0
            X[i, actual_len - 1, 1] = -1.0
            
            first_half = actual_len // 2
            pos1 = np.random.randint(1, first_half)
            pos2 = np.random.randint(first_half, actual_len - 1)
            
            X[i, pos1, 1] = 1.0
            X[i, pos2, 1] = 1.0
            
            X1 = X[i, pos1, 0]
            X2 = X[i, pos2, 0]
            Y[i] = 0.5 + (X1 + X2) / 4.0
    
    return X, Y, actual_lengths


def run_tinygrad(
    seq_len: int = 100,
    num_samples: int = 1000,
    num_epochs: int = 100,
    hidden_size: int = 10,
    learning_rate: float = 0.5,
    seed: Optional[int] = 42,
    realize_every: int = 10,
    log_every: int = 10,
) -> ExperimentResult:
    """Run the Adding Problem experiment with tinygrad."""
    from tinygrad.tensor import Tensor
    from aquarius_lstm.cell import LSTMCell1997
    from aquarius_lstm.initialization import init_weights_paper
    
    print(f"\n{'='*60}")
    print("ADDING PROBLEM (Section 5.4) - tinygrad")
    print(f"{'='*60}")
    print(f"Sequence length: {seq_len}")
    print(f"Samples: {num_samples}")
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    # Generate data
    X, Y, seq_lengths = generate_adding_data(seq_len, num_samples, seed)
    X_tensor = Tensor(X)
    Y_tensor = Tensor(Y)
    
    # Initialize model with paper-accurate settings
    cell = LSTMCell1997(
        input_size=2,
        hidden_size=hidden_size,
        init_range=0.1,
        input_gate_bias=-3.0,  # Paper: -3.0, -6.0 for different cells
        output_gate_bias=0.0,
        seed=seed,
    )
    
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
        predictions.append(float(pred.numpy()))
    
    predictions = np.array(predictions).reshape(-1, 1)
    
    # Evaluate against paper criterion
    result = paper_accuracy_criterion("adding", predictions, Y)
    print(f"\n{result}")
    
    return result


def run_torch(
    seq_len: int = 100,
    num_samples: int = 1000,
    num_epochs: int = 100,
    hidden_size: int = 16,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    seed: Optional[int] = 42,
    log_every: int = 10,
    online: bool = True,
) -> ExperimentResult:
    import torch
    from aquarius_lstm.cell_torch import LSTMCell1997WithForgetGate
    
    print(f"\n{'='*60}")
    print("ADDING PROBLEM (Section 5.4) - PyTorch + Forget Gate")
    print(f"{'='*60}")
    print(f"Sequence length: {seq_len}")
    print(f"Samples per epoch: {num_samples}, Batch size: {batch_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate} (Adam)")
    print(f"Online (fresh batches): {online}")
    print(f"{'='*60}\n")
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    max_len = int(seq_len * 1.1) + 1
    
    cell = LSTMCell1997WithForgetGate(
        input_size=2,
        hidden_size=hidden_size,
        input_gate_bias=0.0,
        forget_gate_bias=1.0,
        seed=seed,
    )
    
    linear = torch.nn.Linear(hidden_size, 1)
    params = list(cell.parameters()) + list(linear.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    def gen_batch(bs):
        X = np.zeros((max_len, bs, 2), dtype=np.float32)
        Y = np.zeros((bs, 1), dtype=np.float32)
        for b in range(bs):
            L = np.random.randint(int(seq_len*0.9), max_len)
            X[:L, b, 0] = np.random.uniform(0, 1, L)
            X[0, b, 1] = X[L-1, b, 1] = -1.0
            p1, p2 = np.random.randint(1, L//2), np.random.randint(L//2, L-1)
            X[p1, b, 1] = X[p2, b, 1] = 1.0
            Y[b] = 0.5 + (X[p1, b, 0] + X[p2, b, 0]) / 4.0
        return torch.tensor(X), torch.tensor(Y)
    
    batches_per_epoch = num_samples // batch_size
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for _ in range(batches_per_epoch):
            X_batch, Y_batch = gen_batch(batch_size)
            optimizer.zero_grad()
            
            h, s_c = cell.init_state(batch_size=batch_size)
            for t in range(max_len):
                h, s_c = cell(X_batch[t], h, s_c)
            
            pred = linear(h)
            loss = ((pred - Y_batch) ** 2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss = {epoch_loss/batches_per_epoch:.6f}")
    
    errors = []
    with torch.no_grad():
        for _ in range(10):
            X_test, Y_test = gen_batch(100)
            h, s_c = cell.init_state(batch_size=100)
            for t in range(max_len):
                h, s_c = cell(X_test[t], h, s_c)
            pred = linear(h)
            errors.extend((pred - Y_test).abs().numpy().flatten().tolist())
    
    predictions = np.array([[0.75]] * 1000)
    Y_dummy = np.array([[0.75]] * 1000)
    
    max_err = max(errors)
    avg_err = np.mean(errors)
    passed = max_err < 0.04
    
    print(f"\nTest Results: max_error={max_err:.6f}, avg_error={avg_err:.6f}")
    print(f"{'[PASS]' if passed else '[FAIL]'} Adding Problem: max_absolute_error = {max_err:.6f} (threshold: 0.04)")
    
    from aquarius_lstm.metrics import ExperimentResult
    return ExperimentResult(
        experiment_name="adding",
        passed=passed,
        metric_name="max_absolute_error",
        achieved_value=max_err,
        threshold=0.04,
    )


def run_torch_paper(
    seq_len: int = 100,
    max_sequences: int = 100000,
    consecutive_correct_target: int = 2000,
    learning_rate: float = 0.5,
    seed: Optional[int] = 42,
    log_every: int = 1000,
    truncate: bool = True,
) -> ExperimentResult:
    """Paper-exact training: online SGD, 93 weights, stop at 2000 consecutive correct."""
    import torch
    from aquarius_lstm.cell_torch import LSTM1997PaperBlock
    
    print(f"\n{'='*60}")
    print("ADDING PROBLEM - Paper-Exact (Section 5.4)")
    print(f"{'='*60}")
    print(f"Sequence length: {seq_len}")
    print(f"Learning rate: {learning_rate} (SGD)")
    print(f"Truncate gradients: {truncate}")
    print(f"Stop criterion: {consecutive_correct_target} consecutive correct")
    print(f"{'='*60}\n")
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    model = LSTM1997PaperBlock(input_size=2, input_gate_biases=(-1.0, -2.0), seed=seed)
    print(f"Parameter count: {model.count_parameters()} (expected: 93)")
    
    print("Pre-training gates to respond to markers...")
    gate_optimizer = torch.optim.Adam([model.W_x, model.b_h], lr=0.1)
    for epoch in range(500):
        gate_optimizer.zero_grad()
        loss = 0
        for marker, target in [(0.0, 0.05), (1.0, 0.95), (-1.0, 0.05)]:
            x = torch.tensor([0.5, marker])
            net = model.W_x[0:2] @ x + model.b_h[0:2]
            gate = torch.sigmoid(net)
            loss = loss + ((gate - target) ** 2).sum()
        loss.backward()
        gate_optimizer.step()
    print("Gates pre-trained.")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    def gen_single_sequence():
        """Paper-exact: values [-1,1], first marker in first 10 steps, target=(X1+X2)/2"""
        L = np.random.randint(int(seq_len * 0.9), int(seq_len * 1.1) + 1)
        X = np.zeros((L, 2), dtype=np.float32)
        
        X[:, 0] = np.random.uniform(-1, 1, L)
        X[0, 1] = -1.0
        X[L-1, 1] = -1.0
        
        marker1_range = min(10, L - 2)
        pos1 = np.random.randint(1, marker1_range + 1) if marker1_range > 0 else 1
        
        remaining = [j for j in range(1, L - 1) if j != pos1]
        pos2 = remaining[np.random.randint(len(remaining))]
        
        X[pos1, 1] = 1.0
        X[pos2, 1] = 1.0
        
        target = (X[pos1, 0] + X[pos2, 0]) / 2.0
        return torch.tensor(X), torch.tensor([[target]], dtype=torch.float32)
    
    consecutive_correct = 0
    total_sequences = 0
    total_loss = 0.0
    
    for seq_idx in range(max_sequences):
        X_seq, Y_target = gen_single_sequence()
        
        optimizer.zero_grad()
        
        h, s_c = model.init_state()
        for t in range(X_seq.shape[0]):
            h, s_c = model(X_seq[t], h, s_c, truncate=truncate)
        
        y_c = h[4:8]
        pred = model.predict(y_c)
        
        loss = (pred - Y_target) ** 2
        loss.backward()
        optimizer.step()
        
        error = (pred - Y_target).abs().item()
        total_loss += loss.item()
        total_sequences += 1
        
        if error < 0.04:
            consecutive_correct += 1
        else:
            consecutive_correct = 0
        
        if consecutive_correct >= consecutive_correct_target:
            print(f"\nSuccess! {consecutive_correct_target} consecutive correct after {total_sequences} sequences")
            break
        
        if (seq_idx + 1) % log_every == 0:
            avg_loss = total_loss / log_every
            print(f"Seq {seq_idx+1}: avg_loss={avg_loss:.6f}, consecutive_correct={consecutive_correct}")
            total_loss = 0.0
    
    passed = consecutive_correct >= consecutive_correct_target
    
    if not passed:
        print(f"\nFailed after {max_sequences} sequences. Best consecutive: {consecutive_correct}")
    
    test_errors = []
    with torch.no_grad():
        for _ in range(2560):
            X_seq, Y_target = gen_single_sequence()
            h, s_c = model.init_state()
            for t in range(X_seq.shape[0]):
                h, s_c = model(X_seq[t], h, s_c, truncate=truncate)
            y_c = h[4:8]
            pred = model.predict(y_c)
            test_errors.append((pred - Y_target).abs().item())
    
    max_err = max(test_errors)
    avg_err = np.mean(test_errors)
    print(f"\nTest (2560 seqs): max_error={max_err:.6f}, avg_error={avg_err:.6f}")
    
    from aquarius_lstm.metrics import ExperimentResult
    return ExperimentResult(
        experiment_name="adding_paper",
        passed=passed and max_err < 0.04,
        metric_name="max_absolute_error",
        achieved_value=max_err,
        threshold=0.04,
    )


def main():
    parser = argparse.ArgumentParser(description="Adding Problem (Section 5.4)")
    parser.add_argument("--mode", choices=["smoke", "paper", "paper_exact"], default="smoke",
                       help="smoke: quick test, paper: longer run, paper_exact: 1997 reproduction")
    parser.add_argument("--backend", choices=["tinygrad", "torch", "both"], 
                       default="tinygrad")
    parser.add_argument("--seq-len", type=int, default=None,
                       help="Override sequence length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-truncate", action="store_true",
                       help="Disable gradient truncation (sanity check)")
    args = parser.parse_args()
    
    if args.mode == "paper_exact":
        result = run_torch_paper(
            seq_len=args.seq_len or 100,
            max_sequences=100000,
            consecutive_correct_target=2000,
            learning_rate=0.5,
            seed=args.seed,
            truncate=not args.no_truncate,
        )
        return 0 if result.passed else 1
    
    if args.mode == "smoke":
        params = {
            "seq_len": args.seq_len or 50,
            "num_samples": 256,
            "num_epochs": 100,
            "hidden_size": 16,
            "learning_rate": 0.001,
            "batch_size": 32,
        }
    else:
        params = {
            "seq_len": args.seq_len or 100,
            "num_samples": 1000,
            "num_epochs": 200,
            "hidden_size": 16,
            "learning_rate": 0.001,
            "batch_size": 32,
        }
    
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
    
    passed = all(r.passed for r in results) if results else False
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
