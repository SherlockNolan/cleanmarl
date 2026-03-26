"""
Critic network for value function estimation.
"""

import torch.nn as nn
import torch

class Critic(nn.Module):
    """Critic network that estimates state value V(s)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        """
        Args:
            input_dim: Dimension of input state
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x