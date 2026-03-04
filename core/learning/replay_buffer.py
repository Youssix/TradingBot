"""Experience replay buffer for stable DQN training."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Transition:
    """Single state-action-reward-next_state transition."""
    state: list[float]
    action: int  # 0=HOLD, 1=BUY, 2=SELL
    reward: float
    next_state: list[float]
    done: bool = False


class ReplayBuffer:
    """Fixed-size circular buffer for experience replay.

    Stores (state, action, reward, next_state, done) transitions
    and supports uniform random sampling for DQN training.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._capacity = capacity

    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample a random batch of transitions."""
        batch_size = min(batch_size, len(self._buffer))
        return random.sample(list(self._buffer), batch_size)

    def sample_arrays(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample and return as numpy arrays for training."""
        batch = self.sample(batch_size)
        return {
            "states": np.array([t.state for t in batch], dtype=np.float32),
            "actions": np.array([t.action for t in batch], dtype=np.int64),
            "rewards": np.array([t.reward for t in batch], dtype=np.float32),
            "next_states": np.array([t.next_state for t in batch], dtype=np.float32),
            "dones": np.array([t.done for t in batch], dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity

    def clear(self) -> None:
        self._buffer.clear()

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize buffer contents."""
        return [
            {
                "state": t.state,
                "action": t.action,
                "reward": t.reward,
                "next_state": t.next_state,
                "done": t.done,
            }
            for t in self._buffer
        ]
