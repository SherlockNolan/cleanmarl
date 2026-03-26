"""
Multi-Agent Proximal Policy Optimization (MAPPO) algorithm.

Reference: https://arxiv.org/abs/2103.01955
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from cleanmarl.models import Actor, Critic


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO algorithm."""
    actor_hidden_dim: int = 32
    actor_num_layers: int = 1
    critic_hidden_dim: int = 64
    critic_num_layers: int = 1
    optimizer: str = "Adam"
    learning_rate_actor: float = 0.0008
    learning_rate_critic: float = 0.0008
    gamma: float = 0.99
    td_lambda: float = 0.95
    normalize_advantage: bool = False
    normalize_return: bool = False
    epochs: int = 3
    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    clip_gradients: float = -1
    device: str = "cpu"


class RolloutBuffer:
    """Buffer for storing rollout data from multiple episodes."""

    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_space: int,
        state_space: int,
        action_space: int,
        normalize_reward: bool = False,
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.normalize_reward = normalize_reward
        self.device = device
        self.episodes = [None] * buffer_size
        self.pos = 0

    def add(self, episode: dict):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values)).float().to(self.device)
        self.episodes[self.pos] = episode
        self.pos += 1

    def get_batch(self):
        self.pos = 0
        lengths = [len(episode["obs"]) for episode in self.episodes if episode is not None]
        if not lengths:
            return None
        max_length = max(lengths)
        obs = torch.zeros(
            (self.buffer_size, max_length, self.num_agents, self.obs_space)
        ).to(self.device)
        actions = torch.zeros((self.buffer_size, max_length, self.num_agents, self.action_space)).to(
            self.device
        )
        log_probs = torch.zeros((self.buffer_size, max_length, self.num_agents)).to(
            self.device
        )
        reward = torch.zeros((self.buffer_size, max_length)).to(self.device)
        states = torch.zeros((self.buffer_size, max_length, self.state_space)).to(
            self.device
        )
        done = torch.zeros((self.buffer_size, max_length)).to(self.device)
        mask = torch.zeros(self.buffer_size, max_length, dtype=torch.bool).to(
            self.device
        )
        for i in range(self.buffer_size):
            if self.episodes[i] is None:
                continue
            length = lengths[i]
            obs[i, :length] = self.episodes[i]["obs"]
            actions[i, :length] = self.episodes[i]["actions"]
            log_probs[i, :length] = self.episodes[i]["log_prob"]
            reward[i, :length] = self.episodes[i]["reward"]
            states[i, :length] = self.episodes[i]["states"]
            done[i, :length] = self.episodes[i]["done"]
            mask[i, :length] = 1
        if self.normalize_reward:
            mu = torch.mean(reward[mask])
            std = torch.std(reward[mask])
            reward[mask.bool()] = (reward[mask] - mu) / (std + 1e-6)
        self.episodes = [None] * self.buffer_size
        return (
            obs.float(),
            actions.float(),
            log_probs.float(),
            reward.float(),
            states.float(),
            done.float(),
            mask,
        )

    def reset(self):
        self.pos = 0
        self.episodes = [None] * self.buffer_size


def _norm_d(grads, d):
    """Compute norm of gradients."""
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.tensor(norms), d)
    return total_norm_d


class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) algorithm.

    Uses centralized training with decentralized execution.
    Critics have access to global state during training.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        n_agents: int,
        config: MAPPOConfig,
    ):
        """
        Initialize MAPPO algorithm.

        Args:
            obs_dim: Observation dimension per agent
            action_dim: Action dimension per agent
            state_dim: Global state dimension
            n_agents: Number of agents
            config: MAPPO configuration
        """
        self.config = config
        self.n_agents = n_agents
        self.device = torch.device(config.device)

        self.actor = Actor(
            input_dim=obs_dim,
            hidden_dim=config.actor_hidden_dim,
            num_layers=config.actor_num_layers,
            output_dim=action_dim,
        ).to(self.device)

        self.critic = Critic(
            input_dim=state_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
        ).to(self.device)

        Optimizer = getattr(optim, config.optimizer)
        self.actor_optimizer = Optimizer(
            self.actor.parameters(), lr=config.learning_rate_actor
        )
        self.critic_optimizer = Optimizer(
            self.critic.parameters(), lr=config.learning_rate_critic
        )

    def get_action(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get actions and log probs for given observations."""
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            actions, log_probs = self.actor.act(obs_tensor)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def evaluate_actions(
        self,
        b_obs: torch.Tensor,
        b_actions: torch.Tensor,
        b_log_probs: torch.Tensor,
        b_states: torch.Tensor,
        b_reward: torch.Tensor,
        b_done: torch.Tensor,
        b_mask: torch.Tensor,
    ) -> dict:
        """
        Evaluate a batch of experiences and compute losses.

        Returns a dictionary with training metrics.
        """
        config = self.config
        n_agents = self.n_agents
        device = self.device

        # Compute TD(λ) returns and advantages
        return_lambda = torch.zeros(
            (b_actions.size(0), b_actions.size(1), n_agents)
        ).float().to(device)
        advantages = torch.zeros(
            (b_actions.size(0), b_actions.size(1), n_agents)
        ).float().to(device)

        with torch.no_grad():
            for ep_idx in range(return_lambda.size(0)):
                ep_len = b_mask[ep_idx].sum()
                last_return_lambda = 0
                last_advantage = 0
                for t in reversed(range(int(ep_len))):
                    if t == int(ep_len - 1):
                        next_value = 0
                    else:
                        next_value = self.critic(x=b_states[ep_idx, t + 1])
                    return_lambda[ep_idx, t] = last_return_lambda = b_reward[
                        ep_idx, t
                    ] + config.gamma * (
                        config.td_lambda * last_return_lambda
                        + (1 - config.td_lambda) * next_value
                    )
                    advantages[ep_idx, t] = return_lambda[ep_idx, t] - self.critic(
                        x=b_states[ep_idx, t]
                    )

        if config.normalize_advantage:
            adv_mu = advantages.mean(dim=-1)[b_mask].mean()
            adv_std = advantages.mean(dim=-1)[b_mask].std()
            advantages = (advantages - adv_mu) / adv_std

        if config.normalize_return:
            ret_mu = return_lambda.mean(dim=-1)[b_mask].mean()
            ret_std = return_lambda.mean(dim=-1)[b_mask].std()
            return_lambda = (return_lambda - ret_mu) / ret_std

        # Training loop
        actor_losses = []
        critic_losses = []
        entropies = []
        kl_divergences = []
        actor_gradients = []
        critic_gradients = []
        clipped_ratios = []

        for _ in range(config.epochs):
            actor_loss = 0
            critic_loss = 0
            entropy_sum = 0
            kl_divergence = 0
            clipped_ratio = 0

            for t in range(b_obs.size(1)):
                # Policy gradient loss
                current_logprob = self.actor.get_log_prob(b_obs[:, t], b_actions[:, t])
                log_ratio = current_logprob - b_log_probs[:, t]
                ratio = torch.exp(log_ratio)

                pg_loss1 = advantages[:, t] * ratio
                pg_loss2 = advantages[:, t] * torch.clamp(
                    ratio, 1 - config.ppo_clip, 1 + config.ppo_clip
                )
                pg_loss = (
                    torch.min(pg_loss1[b_mask[:, t]], pg_loss2[b_mask[:, t]])
                    .mean(dim=-1)
                    .sum()
                )

                # Entropy bonus
                entropy_loss = self.actor.get_entropy(b_obs[:, t])[b_mask[:, t]].mean(dim=-1).sum()
                entropy_sum += entropy_loss
                actor_loss += -pg_loss - config.entropy_coef * entropy_loss

                # Value loss
                current_values = self.critic(x=b_states[:, t]).expand(-1, n_agents)
                value_loss = F.mse_loss(
                    current_values[b_mask[:, t]], return_lambda[:, t][b_mask[:, t]]
                ) * (b_mask[:, t].sum())
                critic_loss += value_loss

                # Track KL divergence
                b_kl_divergence = (
                    ((ratio - 1) - log_ratio)[b_mask[:, t]].mean(dim=-1).sum()
                )
                kl_divergence += b_kl_divergence

                clipped_ratio += (
                    ((ratio - 1.0).abs() > config.ppo_clip)[b_mask[:, t]]
                    .float()
                    .mean(dim=-1)
                    .sum()
                )

            actor_loss /= b_mask.sum()
            critic_loss /= b_mask.sum()
            entropy_sum /= b_mask.sum()
            kl_divergence /= b_mask.sum()
            clipped_ratio /= b_mask.sum()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            actor_gradient = _norm_d([p.grad for p in self.actor.parameters()], 2)
            critic_gradient = _norm_d([p.grad for p in self.critic.parameters()], 2)

            if config.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=config.clip_gradients
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=config.clip_gradients
                )

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy_sum.item())
            kl_divergences.append(kl_divergence.item())
            actor_gradients.append(actor_gradient)
            critic_gradients.append(critic_gradient)
            clipped_ratios.append(clipped_ratio.cpu().item())

        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy": np.mean(entropies),
            "kl_divergence": np.mean(kl_divergences),
            "actor_gradient": np.mean(actor_gradients),
            "critic_gradient": np.mean(critic_gradients),
            "clipped_ratio": np.mean(clipped_ratios),
        }

    def save(self, path: str):
        """Save model checkpoints."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])