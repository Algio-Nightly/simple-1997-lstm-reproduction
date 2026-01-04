"""
Aquarius LSTM 1997 - Faithful reproduction of the original LSTM paper.

This package implements the Long Short-Term Memory architecture exactly as
described in Hochreiter & Schmidhuber (1997), including:

- The Constant Error Carousel (CEC) with weight 1.0 self-connection
- Input and Output gates (no forget gate in the original)
- Truncated backpropagation for O(1) complexity per timestep
- Paper-accurate activation functions g(x) and h(x)

Paper: https://www.bioinf.jku.at/publications/older/2604.pdf

Usage:
    # Tinygrad implementation
    from aquarius_lstm import LSTMCell1997
    
    # PyTorch implementation
    from aquarius_lstm import LSTMCell1997Torch
"""

__version__ = "0.1.0"
__author__ = "Aban Hasan"
__paper__ = "Hochreiter & Schmidhuber (1997) - Long Short-Term Memory"

from .activations import sigmoid, g_squash, h_squash
from .cell import LSTMCell1997
from .cell_torch import LSTMCell1997Torch, LSTMCell1997WithForgetGate
from .block import LSTMBlock1997
from .initialization import init_weights_paper, init_gate_biases
from .metrics import paper_accuracy_criterion

__all__ = [
    "LSTMCell1997",
    "LSTMCell1997Torch",
    "LSTMCell1997WithForgetGate",
    "LSTMBlock1997",
    "sigmoid",
    "g_squash",
    "h_squash",
    "init_weights_paper",
    "init_gate_biases",
    "paper_accuracy_criterion",
]
