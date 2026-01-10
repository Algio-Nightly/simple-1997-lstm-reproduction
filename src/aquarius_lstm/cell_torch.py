"""
Original 1997 LSTM Memory Cell implementation in PyTorch.

This is a verification implementation that mirrors the tinygrad version
exactly. Use this to cross-check results and for compatibility with
PyTorch-based workflows.

See cell.py for the canonical implementation and detailed documentation.
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import numpy as np

from .activations import g_squash_torch, h_squash_torch
from .initialization import init_weights_paper, init_gate_biases


class LSTMCell1997Torch(nn.Module):
    """
    PyTorch implementation of the 1997 LSTM Memory Cell.
    
    Implements the exact same equations as the tinygrad version for
    cross-validation. See cell.py for detailed documentation.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        init_range: float = 0.1,
        input_gate_bias: float = -2.0,
        output_gate_bias: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Input Gate
        self.W_in = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, input_size), init_range)
        ))
        self.U_in = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, hidden_size), init_range)
        ))
        self.b_in = nn.Parameter(torch.full(
            (hidden_size,), input_gate_bias, dtype=torch.float32
        ))
        
        # Output Gate
        self.W_out = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, input_size), init_range)
        ))
        self.U_out = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, hidden_size), init_range)
        ))
        self.b_out = nn.Parameter(torch.full(
            (hidden_size,), output_gate_bias, dtype=torch.float32
        ))
        
        # Cell Input
        self.W_c = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, input_size), init_range)
        ))
        self.U_c = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, hidden_size), init_range)
        ))
        self.b_c = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
    
    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        s_c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ====================================================================
        # THE SCISSORS (Truncated Backpropagation)
        # ====================================================================
        # We DETACH the recurrent signal before it enters the gate computations.
        # This enforces O(1) complexity per timestep because we don't backprop
        # through the gate's recurrent connections across time.
        #
        # CRITICAL: We do NOT detach s_c_prev - the CEC gradient tunnel stays open!
        # This is the key insight of the 1997 paper.
        # ====================================================================
        h_frozen = h_prev.detach()
        
        is_batched = x_t.dim() == 2
        
        if is_batched:
            net_in = x_t @ self.W_in.T + h_frozen @ self.U_in.T + self.b_in
            net_out = x_t @ self.W_out.T + h_frozen @ self.U_out.T + self.b_out
            net_c = x_t @ self.W_c.T + h_frozen @ self.U_c.T + self.b_c
        else:
            net_in = self.W_in @ x_t + self.U_in @ h_frozen + self.b_in
            net_out = self.W_out @ x_t + self.U_out @ h_frozen + self.b_out
            net_c = self.W_c @ x_t + self.U_c @ h_frozen + self.b_c
        
        y_in = torch.sigmoid(net_in)
        y_out = torch.sigmoid(net_out)
        g_val = g_squash_torch(net_c)
        
        s_c_t = s_c_prev + (y_in * g_val)
        
        h_s_c = h_squash_torch(s_c_t)
        h_t = y_out * h_s_c
        
        return h_t, s_c_t
    
    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self.W_in.device
        
        if batch_size == 1:
            h_0 = torch.zeros(self.hidden_size, device=device)
            s_c_0 = torch.zeros(self.hidden_size, device=device)
        else:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            s_c_0 = torch.zeros(batch_size, self.hidden_size, device=device)
        
        return h_0, s_c_0


class LSTMCell1997WithForgetGate(nn.Module):
    """
    EXPERIMENTAL: 1997 LSTM with forget gate (Gers et al. 1999 extension).
    
    WARNING: This is NOT the paper-faithful 1997 architecture. Forget gates
    were added in 1999/2000. Use LSTM1997PaperBlock for paper reproduction.
    
    The forget gate prevents unbounded cell state accumulation by allowing
    the network to learn when to reset/decay stored information.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        init_range: float = 0.1,
        input_gate_bias: float = 0.0,
        forget_gate_bias: float = 1.0,
        output_gate_bias: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.W_in = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, input_size), init_range)
        ))
        self.U_in = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, hidden_size), init_range)
        ))
        self.b_in = nn.Parameter(torch.full(
            (hidden_size,), input_gate_bias, dtype=torch.float32
        ))
        
        self.W_f = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, input_size), init_range)
        ))
        self.U_f = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, hidden_size), init_range)
        ))
        self.b_f = nn.Parameter(torch.full(
            (hidden_size,), forget_gate_bias, dtype=torch.float32
        ))
        
        self.W_out = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, input_size), init_range)
        ))
        self.U_out = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, hidden_size), init_range)
        ))
        self.b_out = nn.Parameter(torch.full(
            (hidden_size,), output_gate_bias, dtype=torch.float32
        ))
        
        self.W_c = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, input_size), init_range)
        ))
        self.U_c = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, hidden_size), init_range)
        ))
        self.b_c = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
    
    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        s_c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_batched = x_t.dim() == 2
        
        if is_batched:
            net_in = x_t @ self.W_in.T + h_prev @ self.U_in.T + self.b_in
            net_f = x_t @ self.W_f.T + h_prev @ self.U_f.T + self.b_f
            net_out = x_t @ self.W_out.T + h_prev @ self.U_out.T + self.b_out
            net_c = x_t @ self.W_c.T + h_prev @ self.U_c.T + self.b_c
        else:
            net_in = self.W_in @ x_t + self.U_in @ h_prev + self.b_in
            net_f = self.W_f @ x_t + self.U_f @ h_prev + self.b_f
            net_out = self.W_out @ x_t + self.U_out @ h_prev + self.b_out
            net_c = self.W_c @ x_t + self.U_c @ h_prev + self.b_c
        
        y_in = torch.sigmoid(net_in)
        y_f = torch.sigmoid(net_f)
        y_out = torch.sigmoid(net_out)
        g_val = g_squash_torch(net_c)
        
        s_c_t = y_f * s_c_prev + y_in * g_val
        
        h_s_c = h_squash_torch(s_c_t)
        h_t = y_out * h_s_c
        
        return h_t, s_c_t
    
    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self.W_in.device
        
        if batch_size == 1:
            h_0 = torch.zeros(self.hidden_size, device=device)
            s_c_0 = torch.zeros(self.hidden_size, device=device)
        else:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            s_c_0 = torch.zeros(batch_size, self.hidden_size, device=device)
        
        return h_0, s_c_0


class LSTMModel1997Torch(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        **cell_kwargs,
    ):
        super().__init__()
        
        self.cell = LSTMCell1997Torch(input_size, hidden_size, **cell_kwargs)
        self.output_proj = nn.Linear(hidden_size, output_size, bias=True)
        
        with torch.no_grad():
            self.output_proj.weight.copy_(torch.tensor(
                init_weights_paper((output_size, hidden_size), 0.1)
            ))
            self.output_proj.bias.zero_()
    
    def forward_sequence(
        self,
        x_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = x_seq.shape[0]
        batch_size = x_seq.shape[1] if x_seq.dim() == 3 else 1
        
        h, s_c = self.cell.init_state(batch_size, device=x_seq.device)
        
        for t in range(seq_len):
            x_t = x_seq[t]
            h, s_c = self.cell(x_t, h, s_c)
        
        pred = self.output_proj(h)
        
        return pred, h, s_c
    
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        pred, _, _ = self.forward_sequence(x_seq)
        return pred


class LSTM1997PaperBlock(nn.Module):
    """
    Paper-exact LSTM architecture for Adding Problem (Section 5.4).
    
    Architecture: 2 blocks x 2 cells = 4 cells total, shared gates.
    Hidden state h(t) = [y_in1, y_in2, y_out1, y_out2, y_c1, y_c2, y_c3, y_c4] (8 units)
    
    Weight count = 93:
      - W_x: input->hidden = 2x8 = 16
      - W_h: hidden->hidden = 8x8 = 64
      - b_h: hidden biases = 8
      - W_o: cells->output = 4x1 = 4
      - b_o: output bias = 1
    
    NO forget gate (1997 original).
    """
    
    def __init__(
        self,
        input_size: int = 2,
        init_range: float = 0.1,
        input_gate_biases: Tuple[float, float] = (-3.0, -6.0),
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = 8
        self.n_cells = 4
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.W_x = nn.Parameter(torch.tensor(
            init_weights_paper((8, input_size), init_range)
        ))
        self.W_h = nn.Parameter(torch.tensor(
            init_weights_paper((8, 8), init_range)
        ))
        
        b_h = np.zeros(8, dtype=np.float32)
        b_h[0] = input_gate_biases[0]
        b_h[1] = input_gate_biases[1]
        b_h[2] = np.random.uniform(-init_range, init_range)
        b_h[3] = np.random.uniform(-init_range, init_range)
        # All weights (including the bias weights) are randomly initialized in the range [-0.1, 0.1]
        b_h[4:8] = np.random.uniform(-init_range, init_range)
        
        self.b_h = nn.Parameter(torch.tensor(b_h))
        
        self.W_o = nn.Parameter(torch.tensor(
            init_weights_paper((1, 4), init_range)
        ))
        self.b_o = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    
    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        s_c_prev: torch.Tensor,
        truncate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_frozen = h_prev.detach() if truncate else h_prev
        
        is_batched = x_t.dim() == 2
        
        if is_batched:
            net = x_t @ self.W_x.T + h_frozen @ self.W_h.T + self.b_h
        else:
            net = self.W_x @ x_t + self.W_h @ h_frozen + self.b_h
        
        y_in = torch.sigmoid(net[..., 0:2])
        y_out = torch.sigmoid(net[..., 2:4])
        g_c = g_squash_torch(net[..., 4:8])
        
        if is_batched:
            y_in_exp = torch.stack([y_in[:, 0], y_in[:, 0], y_in[:, 1], y_in[:, 1]], dim=1)
            y_out_exp = torch.stack([y_out[:, 0], y_out[:, 0], y_out[:, 1], y_out[:, 1]], dim=1)
        else:
            y_in_exp = torch.stack([y_in[0], y_in[0], y_in[1], y_in[1]])
            y_out_exp = torch.stack([y_out[0], y_out[0], y_out[1], y_out[1]])
        
        s_c_t = s_c_prev + y_in_exp * g_c
        
        y_c = y_out_exp * h_squash_torch(s_c_t)
        
        if is_batched:
            h_t = torch.cat([y_in, y_out, y_c], dim=1)
        else:
            h_t = torch.cat([y_in, y_out, y_c])
        
        return h_t, s_c_t
    
    def predict(self, y_c: torch.Tensor) -> torch.Tensor:
        if y_c.dim() == 2:
            return y_c @ self.W_o.T + self.b_o
        return self.W_o @ y_c + self.b_o
    
    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self.W_x.device
        
        if batch_size == 1:
            h_0 = torch.zeros(8, device=device)
            s_c_0 = torch.zeros(4, device=device)
        else:
            h_0 = torch.zeros(batch_size, 8, device=device)
            s_c_0 = torch.zeros(batch_size, 4, device=device)
        
        return h_0, s_c_0
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
