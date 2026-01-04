#!/usr/bin/env python3
"""
Temporal Order Problem - Experiment 6 (Section 5.6)

Classify sequences based on the temporal order of special symbols.

Task 6a (2 symbols): Classify into 4 classes (XX, XY, YX, YY)
Task 6b (3 symbols): Classify into 8 classes (XXX through YYY)

Task Description:
    - Input: Sequence with distractor symbols and 2 (or 3) relevant symbols X/Y
    - Relevant symbols appear at random positions
    - Target: Class determined by order of relevant symbols
    
Paper Criterion:
    "Final absolute error of all output units below 0.3"

Paper Hyperparameters (Section 5.6):
    - Sequence length: 100-110 steps
    - Learning rate: 0.5 (2 symbols), 0.1 (3 symbols)
    - Input gate biases: -2.0, -4.0 (2 sym) or -2, -4, -6 (3 sym)

Reference: Section 5.6 of Hochreiter & Schmidhuber (1997)
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aquarius_lstm.metrics import paper_accuracy_criterion, ExperimentResult


# Symbol encoding (one-hot): B, E, X, Y, and 4 distractor symbols
# Total: 8 input units
SYMBOLS = ['B', 'X', 'Y', 'E', 'd1', 'd2', 'd3', 'd4']
SYMBOL_TO_IDX = {s: i for i, s in enumerate(SYMBOLS)}


def one_hot_encode(symbol: str, num_symbols: int = 8) -> np.ndarray:
    """One-hot encode a symbol."""
    vec = np.zeros(num_symbols, dtype=np.float32)
    vec[SYMBOL_TO_IDX[symbol]] = 1.0
    return vec


def generate_temporal_order_data(
    seq_len: int = 100,
    num_samples: int = 256,
    num_relevant: int = 2,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Temporal Order dataset per Section 5.6.
    
    Paper specifies:
    - Sequence length: 100-110 steps
    - E at start, B at end (trigger symbol)
    - Relevant symbols at t1 in [10,20] and t2 in [50,60]
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_classes = 2 ** num_relevant
    distractors = ['d1', 'd2', 'd3', 'd4']
    
    actual_lengths = np.random.randint(seq_len, int(seq_len * 1.1) + 1, num_samples)
    max_len = int(seq_len * 1.1) + 1
    
    X = np.zeros((num_samples, max_len, 8), dtype=np.float32)
    Y = np.zeros((num_samples, num_classes), dtype=np.float32)
    
    for i in range(num_samples):
        actual_len = actual_lengths[i]
        
        if num_relevant == 2:
            t1 = np.random.randint(10, 21)
            t2 = np.random.randint(50, 61)
            relevant_positions = [t1, t2]
        else:
            t1 = np.random.randint(10, 21)
            t2 = np.random.randint(33, 44)
            t3 = np.random.randint(66, 77)
            relevant_positions = [t1, t2, t3]
        
        relevant_symbols = np.random.choice(['X', 'Y'], num_relevant)
        
        for t in range(actual_len):
            if t == 0:
                X[i, t] = one_hot_encode('E')
            elif t == actual_len - 1:
                X[i, t] = one_hot_encode('B')
            elif t in relevant_positions:
                idx = relevant_positions.index(t)
                X[i, t] = one_hot_encode(relevant_symbols[idx])
            else:
                X[i, t] = one_hot_encode(np.random.choice(distractors))
        
        class_idx = 0
        for k, sym in enumerate(relevant_symbols):
            if sym == 'Y':
                class_idx += 2 ** (num_relevant - 1 - k)
        
        Y[i, class_idx] = 1.0
    
    return X, Y


def run_tinygrad(
    seq_len: int = 100,
    num_samples: int = 256,
    num_epochs: int = 100,
    num_relevant: int = 2,
    hidden_size: int = 10,
    learning_rate: float = 0.5,
    seed: Optional[int] = 42,
    realize_every: int = 10,
    log_every: int = 10,
) -> ExperimentResult:
    """Run Temporal Order with tinygrad."""
    from tinygrad.tensor import Tensor
    from aquarius_lstm.cell import LSTMCell1997
    from aquarius_lstm.initialization import init_weights_paper
    
    num_classes = 2 ** num_relevant
    experiment_name = f"temporal_order" if num_relevant == 2 else "temporal_order_3"
    
    print(f"\n{'='*60}")
    print(f"TEMPORAL ORDER ({num_relevant} symbols) - tinygrad")
    print(f"{'='*60}")
    print(f"Sequence length: {seq_len}")
    print(f"Classes: {num_classes}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    X, Y = generate_temporal_order_data(seq_len, num_samples, num_relevant, seed)
    X_tensor = Tensor(X)
    Y_tensor = Tensor(Y)
    
    input_gate_bias = -2.0 if num_relevant == 2 else -2.0
    
    cell = LSTMCell1997(
        input_size=8,
        hidden_size=hidden_size,
        init_range=0.1,
        input_gate_bias=input_gate_bias,
        seed=seed,
    )
    
    W_out = Tensor(
        init_weights_paper((num_classes, hidden_size), 0.1, seed),
        requires_grad=True
    )
    b_out = Tensor(np.zeros(num_classes, dtype=np.float32), requires_grad=True)
    
    params = cell.parameters() + [W_out, b_out]
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        
        for sample_idx in range(num_samples):
            h, s_c = cell.init_state()
            
            for t in range(X.shape[1]):
                x_t = X_tensor[sample_idx, t]
                h, s_c = cell(x_t, h, s_c)
                if t % realize_every == 0:
                    h = h.realize()
                    s_c = s_c.realize()
            
            # Softmax output
            logits = W_out.dot(h) + b_out
            pred = logits.softmax()
            target = Y_tensor[sample_idx]
            
            # Cross-entropy loss
            loss = -(target * (pred + 1e-7).log()).sum()
            
            loss.backward()
            
            for p in params:
                if p.grad is not None:
                    p.assign(p - p.grad * learning_rate).realize()
                    p.grad = None
            
            epoch_loss += float(loss.numpy())
            
            # Check accuracy
            pred_class = int(pred.numpy().argmax())
            true_class = int(target.numpy().argmax())
            if pred_class == true_class:
                correct += 1
        
        if (epoch + 1) % log_every == 0:
            acc = correct / num_samples
            print(f"Epoch {epoch+1:3d}: Loss={epoch_loss/num_samples:.4f}, Acc={acc:.2%}")
    
    # Final evaluation
    predictions = []
    for sample_idx in range(num_samples):
        h, s_c = cell.init_state()
        for t in range(X.shape[1]):
            x_t = X_tensor[sample_idx, t]
            h, s_c = cell(x_t, h, s_c)
        logits = W_out.dot(h) + b_out
        pred = logits.softmax()
        predictions.append(pred.numpy())
    
    predictions = np.array(predictions)
    
    result = paper_accuracy_criterion(experiment_name, predictions, Y)
    print(f"\n{result}")
    
    return result


def run_torch(
    seq_len: int = 100,
    num_samples: int = 256,
    num_epochs: int = 100,
    num_relevant: int = 2,
    hidden_size: int = 4,
    learning_rate: float = 0.5,
    seed: Optional[int] = 42,
    log_every: int = 10,
) -> ExperimentResult:
    """Run Temporal Order with PyTorch."""
    import torch
    import torch.nn.functional as F
    from aquarius_lstm.cell_torch import LSTMCell1997Torch
    
    num_classes = 2 ** num_relevant
    experiment_name = f"temporal_order" if num_relevant == 2 else "temporal_order_3"
    
    print(f"\n{'='*60}")
    print(f"TEMPORAL ORDER ({num_relevant} symbols) - PyTorch")
    print(f"{'='*60}")
    
    if seed is not None:
        torch.manual_seed(seed)
    
    X, Y = generate_temporal_order_data(seq_len, num_samples, num_relevant, seed)
    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y)
    Y_labels = torch.argmax(Y_tensor, dim=1)
    
    input_gate_bias = -2.0 if num_relevant == 2 else -2.0
    
    cell = LSTMCell1997Torch(
        input_size=8,
        hidden_size=hidden_size,
        init_range=0.1,
        input_gate_bias=input_gate_bias,
        seed=seed,
    )
    
    with torch.no_grad():
        if num_relevant == 2:
            cell.b_in[:2] = -2.0
            cell.b_in[2:] = -4.0
        else:
            cell.b_in[:2] = -2.0
            cell.b_in[2:4] = -4.0
            cell.b_in[4:] = -6.0
    
    W_out = torch.nn.Parameter(torch.randn(num_classes, hidden_size) * 0.1)
    b_out = torch.nn.Parameter(torch.zeros(num_classes))
    
    params = list(cell.parameters()) + [W_out, b_out]
    optimizer = torch.optim.SGD(params, lr=learning_rate)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        
        for sample_idx in range(num_samples):
            optimizer.zero_grad()
            
            h, s_c = cell.init_state()
            
            for t in range(X.shape[1]):
                x_t = X_tensor[sample_idx, t]
                h, s_c = cell(x_t, h, s_c)
            
            logits = h @ W_out.T + b_out
            loss = F.cross_entropy(logits.unsqueeze(0), Y_labels[sample_idx:sample_idx+1])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if logits.argmax().item() == Y_labels[sample_idx].item():
                correct += 1
        
        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1:3d}: Loss={epoch_loss/num_samples:.4f}, Acc={correct/num_samples:.2%}")
    
    predictions = []
    with torch.no_grad():
        for sample_idx in range(num_samples):
            h, s_c = cell.init_state()
            for t in range(X.shape[1]):
                x_t = X_tensor[sample_idx, t]
                h, s_c = cell(x_t, h, s_c)
            logits = h @ W_out.T + b_out
            pred = F.softmax(logits, dim=0)
            predictions.append(pred.numpy())
    
    predictions = np.array(predictions)
    
    result = paper_accuracy_criterion(experiment_name, predictions, Y)
    print(f"\n{result}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Temporal Order (Section 5.6)")
    parser.add_argument("--mode", choices=["smoke", "paper"], default="smoke")
    parser.add_argument("--backend", choices=["tinygrad", "torch", "both"], default="tinygrad")
    parser.add_argument("--num-symbols", type=int, choices=[2, 3], default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.mode == "smoke":
        params = {
            "seq_len": 100,
            "num_samples": 64,
            "num_epochs": 50,
            "num_relevant": args.num_symbols,
            "hidden_size": 4,
            "learning_rate": 0.5 if args.num_symbols == 2 else 0.1,
        }
    else:
        params = {
            "seq_len": 100,
            "num_samples": 256,
            "num_epochs": 150,
            "num_relevant": args.num_symbols,
            "hidden_size": 4,
            "learning_rate": 0.5 if args.num_symbols == 2 else 0.1,
        }
    
    params["seed"] = args.seed
    
    results = []
    
    if args.backend in ["tinygrad", "both"]:
        try:
            results.append(run_tinygrad(**params))
        except ImportError as e:
            print(f"tinygrad not available: {e}")
    
    if args.backend in ["torch", "both"]:
        try:
            results.append(run_torch(**params))
        except ImportError as e:
            print(f"PyTorch not available: {e}")
    
    passed = all(r.passed for r in results) if results else False
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
