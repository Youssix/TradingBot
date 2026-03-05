"""Experience replay buffers: uniform and prioritized (SumTree-based PER)."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Transition:
    """Single state-action-reward-next_state transition."""
    state: list[float]
    action: float  # continuous: [-1, +1] where sign=direction, magnitude=conviction
    reward: float
    next_state: list[float]
    done: bool = False


# ---------------------------------------------------------------------------
# Uniform Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular buffer for experience replay.

    Stores (state, action, reward, next_state, done) transitions
    and supports uniform random sampling for off-policy training.
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
            "actions": np.array([t.action for t in batch], dtype=np.float32),
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


# ---------------------------------------------------------------------------
# SumTree for efficient proportional sampling
# ---------------------------------------------------------------------------

class SumTree:
    """Binary tree where each leaf stores a priority and parent nodes store sums.

    Supports O(log n) update and proportional sampling.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: list[Transition | None] = [None] * capacity
        self._write_idx = 0
        self._size = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self._tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._tree):
            return idx

        if value <= self._tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self._tree[left])

    @property
    def total(self) -> float:
        return float(self._tree[0])

    @property
    def max_priority(self) -> float:
        leaf_start = self._capacity - 1
        n = min(self._size, self._capacity)
        if n == 0:
            return 1.0
        return float(np.max(self._tree[leaf_start:leaf_start + n]))

    @property
    def min_priority(self) -> float:
        leaf_start = self._capacity - 1
        n = min(self._size, self._capacity)
        if n == 0:
            return 1.0
        priorities = self._tree[leaf_start:leaf_start + n]
        nonzero = priorities[priorities > 0]
        return float(np.min(nonzero)) if len(nonzero) > 0 else 1.0

    def add(self, priority: float, data: Transition) -> None:
        idx = self._write_idx + self._capacity - 1
        self._data[self._write_idx] = data
        self.update(idx, priority)
        self._write_idx = (self._write_idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self._tree[idx]
        self._tree[idx] = priority
        self._propagate(idx, change)

    def get(self, value: float) -> tuple[int, float, Transition]:
        idx = self._retrieve(0, value)
        data_idx = idx - self._capacity + 1
        return idx, float(self._tree[idx]), self._data[data_idx]  # type: ignore[return-value]

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Prioritized Experience Replay Buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay using SumTree for proportional sampling.

    Supports importance-sampling weight correction and priority updates.
    High-error transitions are sampled more frequently.

    Args:
        capacity: Max number of transitions.
        alpha: How much prioritization (0 = uniform, 1 = full priority).
        beta_start: Initial importance-sampling exponent.
        beta_frames: Number of frames to anneal beta from beta_start to 1.0.
        epsilon: Small constant added to priorities to avoid zero-priority.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-6,
    ) -> None:
        self._tree = SumTree(capacity)
        self._capacity = capacity
        self._alpha = alpha
        self._beta_start = beta_start
        self._beta_frames = beta_frames
        self._epsilon = epsilon
        self._frame = 0
        self._max_priority = 1.0

    @property
    def beta(self) -> float:
        """Current importance-sampling exponent (anneals toward 1.0)."""
        fraction = min(self._frame / max(self._beta_frames, 1), 1.0)
        return self._beta_start + fraction * (1.0 - self._beta_start)

    @property
    def mean_priority(self) -> float:
        """Average priority in the buffer (for logging)."""
        n = len(self._tree)
        if n == 0:
            return 0.0
        return self._tree.total / n

    def push(self, transition: Transition) -> None:
        """Add transition with max priority (ensures it gets sampled at least once)."""
        priority = self._max_priority ** self._alpha
        self._tree.add(priority, transition)

    def sample(self, batch_size: int) -> tuple[
        dict[str, np.ndarray], np.ndarray, list[int]
    ]:
        """Sample a prioritized batch.

        Returns:
            (batch_arrays, importance_weights, tree_indices)
        """
        self._frame += 1
        n = len(self._tree)
        batch_size = min(batch_size, n)

        indices: list[int] = []
        priorities: list[float] = []
        transitions: list[Transition] = []

        segment = self._tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = random.uniform(lo, hi)
            idx, priority, data = self._tree.get(value)
            if data is None:
                # Fallback: retry with random value
                value = random.uniform(0, self._tree.total)
                idx, priority, data = self._tree.get(value)
            indices.append(idx)
            priorities.append(priority)
            transitions.append(data)

        # Importance-sampling weights
        beta = self.beta
        total = self._tree.total
        min_prob = self._tree.min_priority / total if total > 0 else 1e-6
        min_prob = max(min_prob, 1e-8)
        max_weight = (min_prob * n) ** (-beta)

        weights = np.zeros(batch_size, dtype=np.float32)
        for i, p in enumerate(priorities):
            prob = p / total if total > 0 else 1.0 / n
            prob = max(prob, 1e-8)
            weight = (prob * n) ** (-beta)
            weights[i] = weight / max_weight

        batch = {
            "states": np.array([t.state for t in transitions], dtype=np.float32),
            "actions": np.array([t.action for t in transitions], dtype=np.float32),
            "rewards": np.array([t.reward for t in transitions], dtype=np.float32),
            "next_states": np.array([t.next_state for t in transitions], dtype=np.float32),
            "dones": np.array([t.done for t in transitions], dtype=np.float32),
        }

        return batch, weights, indices

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
        """Update priorities using absolute TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + self._epsilon) ** self._alpha
            self._max_priority = max(self._max_priority, priority)
            self._tree.update(idx, priority)

    def sample_arrays(self, batch_size: int) -> dict[str, np.ndarray]:
        """Uniform-compatible interface: sample without weights/indices."""
        batch, _, _ = self.sample(batch_size)
        return batch

    def __len__(self) -> int:
        return len(self._tree)

    @property
    def capacity(self) -> int:
        return self._capacity

    def clear(self) -> None:
        self._tree = SumTree(self._capacity)
        self._frame = 0
        self._max_priority = 1.0

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize buffer contents (for debugging)."""
        result = []
        for i in range(len(self._tree)):
            data = self._tree._data[i]
            if data is not None:
                result.append({
                    "state": data.state,
                    "action": data.action,
                    "reward": data.reward,
                    "next_state": data.next_state,
                    "done": data.done,
                })
        return result
