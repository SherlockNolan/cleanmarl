"""
Actor network for continuous action spaces.
Uses a Gaussian policy with learnable log_std.
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Actor network that outputs a Gaussian distribution over actions.
    Mean is bounded to [-1, 1] via tanh activation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int) -> None:
        """
        Args:
            input_dim: Dimension of input observations
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            output_dim: Dimension of action space
        """
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def act(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_layer(x)
        mean = torch.tanh(mean)

        std = torch.exp(self.log_std).expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        action = distribution.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = distribution.log_prob(action).sum(dim=-1)

        return action, log_prob

    def get_log_prob(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of given actions."""
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_layer(x)
        mean = torch.tanh(mean)

        std = torch.exp(self.log_std).expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        log_prob = distribution.log_prob(actions).sum(dim=-1)

        return log_prob

    def get_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute entropy of the policy distribution."""
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_layer(x)

        std = torch.exp(self.log_std).expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        entropy = distribution.entropy().sum(dim=-1)

        return entropy