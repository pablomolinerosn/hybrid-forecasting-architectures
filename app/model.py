"""
ModeloDCRNN standalone para inferencia (servir el modelo, no entrenarlo).

Arquitectura idéntica a src/models/dcrnn/entrenar_dcrnn.py::ModeloDCRNN: recorre
los seq_len pasos temporales con una celda DCRNN arrastrando el estado
oculto H, y proyecta el H final a una predicción escalar por nodo (el
target 'horizon' horas después del último paso observado).
"""

import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import DCRNN


class ModeloDCRNN(nn.Module):
    """forward(xb) con xb=(B, N, L, F) -> (B, N) [target escalado]."""

    def __init__(self, in_channels, hidden, K, edge_index, edge_weight):
        super().__init__()
        self.recurrent = DCRNN(in_channels=in_channels, out_channels=hidden, K=K)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 1))
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

    def forward(self, xb):
        B, N, L, Fd = xb.shape
        x = xb.reshape(B * N, L, Fd)
        ei = torch.cat([self.edge_index + k * N for k in range(B)], dim=1)
        ew = self.edge_weight.repeat(B)
        H = None
        for t in range(L):
            H = self.recurrent(x[:, t, :], ei, ew, H)
        out = self.head(H).view(B, N)
        return out
