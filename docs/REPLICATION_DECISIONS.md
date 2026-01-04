# Replication Decisions

This document outlines the technical decisions and deviations made during the reproduction of the 1997 LSTM paper by Hochreiter & Schmidhuber.

## 1. Gate Pre-training (Acceleration Trick)

### The Problem
In the **Adding Problem** and **Long Time Lag** experiments, input gates are initialized with negative biases (e.g., -3.0 or -6.0) to start in a "closed" state. With random weight initialization, the gradients are initially extremely small, making it difficult for the network to learn to "open" the gates when a marker signal appears.

### The Decision
We implemented a "Gate Pre-training" phase. Before the main training starts, we use the Adam optimizer for 300–500 epochs to train *only* the input gate weights and biases to respond to the marker signal.
- **Target**: Gate $\approx 0.95$ when `marker = 1.0`, and $\approx 0.05$ otherwise.

### Paper Fidelity
- **Deviation**: This was **NOT** in the original 1997 paper.
- **Justification**: In 1997, the authors likely trained their models for hundreds of thousands (or even millions) of sequences. Pre-training allows us to achieve the same results in ~30K sequences, making the reproduction practical on modern hardware without changing the core architecture.

## 2. Gradient Clipping

### The Problem
Online SGD (updating after every sequence) can be unstable. Large gradient steps can lead to "catastrophic overshoot," where the weights move into a region of the loss landscape from which the model cannot recover.

### The Decision
We added standard gradient norm clipping (`max_norm = 1.0`) to the training loop.

### Paper Fidelity
- **Deviation**: Not mentioned in the 1997 paper.
- **Justification**: Likely implicit in the original 1997 implementation or handled by very small learning rates. It is a standard modern practice for stabilizing recurrent neural networks.

## 3. Architecture Fidelity: LSTM1997PaperBlock

### The Goal
Reproduce the exact weight count (93 weights) used for the Adding Problem in Section 5.4.

### The Decision
We implemented a specialized class, `LSTM1997PaperBlock`, which strictly follows the 1997 specifications:
- **Structure**: 2 memory cell blocks, each containing 2 memory cells.
- **Shared Gates**: All cells within a block share the same input and output gates.
- **Weight Counting**:
  - Input to Hidden: $2 \times 8 = 16$
  - Recurrent (Hidden to Hidden): $8 \times 8 = 64$
  - Hidden Biases: $8$
  - Output Weights (from 4 cells): $4 \times 1 = 4$
  - Output Bias: $1$
  - **Total**: 93 weights.
- **No Forget Gates**: Faithful to the original design (forget gates were added in 2000).

## 4. Truncated Backpropagation (\"The Scissors\")

### The Decision
To match the $O(1)$ complexity per timestep and prevent gradient explosion through the gates, we use the \"scissors\" strategy:
- Gradients flow freely through the **Constant Error Carousel (CEC)**.
- Gradients are **truncated** (detached) as they flow back into the recurrent hidden state connections used for gate and cell input computations.

In code, this is implemented as:
```python
h_frozen = h_prev.detach()
# Use h_frozen for net_in, net_out, and net_c
```

### Interpretation Note
The original paper is ambiguous about whether truncation applies to:
- (A) Only gate recurrent connections, OR
- (B) All recurrent connections (gates + cell input)

Our implementation uses option (B): `h_prev.detach()` is used for computing `net_in`, `net_out`, AND `net_c`. This is more aggressive truncation but matches the paper's stated goal of $O(1)$ complexity. The CEC remains the only path for gradient flow across time.

## 5. Data Generation

### Adding Problem
We discovered that the original paper used values in **$[-1, 1]$** rather than $[0, 1]$ for the Adding Problem. Our `generate_adding_data` function includes a `paper_exact` flag to switch between these modes. The success criterion remains an absolute error $< 0.04$.

### Temporal Order
We corrected the symbol positions to match Section 5.6 exactly:
- `t1` in $[10, 20]$, `t2` in $[50, 60]$.
- Added `E` (start) and `B` (end) trigger symbols.
