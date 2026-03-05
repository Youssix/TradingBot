"""Abstract base class for all RL trading agents (continuous action space)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.learning.replay_buffer import Transition


class BaseAgent(ABC):
    """Base interface for continuous-action RL agents.

    All agents output a float action in [-1, +1]:
      -1 = full sell, 0 = hold, +1 = full buy
    Position sizing is encoded in the action magnitude.
    """

    @abstractmethod
    def select_action(self, state: list[float]) -> tuple[float, dict]:
        """Select an action given the current state.

        Returns:
            (action_float, info_dict) where action_float is in [-1, +1]
            and info_dict contains agent-specific metadata like
            {"mean": ..., "std": ..., "log_prob": ..., "q_value": ...}.
        """

    @abstractmethod
    def store_transition(self, transition: Transition) -> None:
        """Store a transition for training."""

    @abstractmethod
    def train_step(self) -> float | None:
        """Perform one training step.

        Returns:
            Loss value as float, or None if not enough data.
        """

    @abstractmethod
    def end_episode(self, episode_reward: float) -> None:
        """Called at the end of each episode (trade close)."""

    @abstractmethod
    def save_checkpoint(self) -> bytes | None:
        """Serialize agent state to bytes."""

    @abstractmethod
    def load_checkpoint(self, data: bytes) -> None:
        """Load agent state from bytes."""

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Return agent statistics."""

    @property
    @abstractmethod
    def epsilon(self) -> float:
        """Current exploration parameter."""

    @property
    @abstractmethod
    def episode(self) -> int:
        """Number of completed episodes."""

    @property
    @abstractmethod
    def total_reward(self) -> float:
        """Cumulative reward."""

    @property
    @abstractmethod
    def training(self) -> bool:
        """Whether the agent is in training mode."""

    @training.setter
    @abstractmethod
    def training(self, val: bool) -> None: ...

    @property
    @abstractmethod
    def buffer_size(self) -> int:
        """Number of transitions stored."""
