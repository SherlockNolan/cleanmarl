"""
Model architectures for multi-agent reinforcement learning.

This module contains neural network definitions for actors and critics.
"""

from .actor import Actor
from .critic import Critic

__all__ = ["Actor", "Critic"]