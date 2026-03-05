"""Transformer state encoder with rolling state buffer for RL agents.

Defaults: seq_len=64, embed_dim=128, 3 layers, 4 heads, ff_dim=256.
"""

from __future__ import annotations

from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class StateBuffer:
    """Rolling buffer of state vectors for sequence-based encoding."""

    def __init__(self, seq_len: int = 64, state_dim: int = 23) -> None:
        self._seq_len = seq_len
        self._state_dim = state_dim
        self._buffer: deque[list[float]] = deque(maxlen=seq_len)

    def push(self, state: list[float]) -> None:
        self._buffer.append(state)

    def get_sequence(self) -> np.ndarray:
        """Return zero-padded sequence of shape (seq_len, state_dim)."""
        seq = np.zeros((self._seq_len, self._state_dim), dtype=np.float32)
        data = list(self._buffer)
        n = len(data)
        if n > 0:
            # Right-align: most recent at the end
            seq[-n:] = np.array(data, dtype=np.float32)
        return seq

    def get_mask(self) -> np.ndarray:
        """Return boolean mask: True where data exists, False where padding."""
        mask = np.zeros(self._seq_len, dtype=bool)
        n = len(self._buffer)
        if n > 0:
            mask[-n:] = True
        return mask

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()


if TORCH_AVAILABLE:
    class TransformerStateEncoder(nn.Module):
        """Transformer encoder for state sequences.

        Input: [B, seq_len, state_dim]
        Output: [B, embed_dim] (masked mean pooling)

        Defaults: seq_len=64, embed_dim=128, 3 layers, 4 heads.
        """

        def __init__(
            self,
            state_dim: int = 23,
            embed_dim: int = 128,
            nhead: int = 4,
            num_layers: int = 3,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
            seq_len: int = 64,
            device: torch.device | str | None = None,
        ) -> None:
            super().__init__()
            self._state_dim = state_dim
            self._embed_dim = embed_dim
            self._seq_len = seq_len

            # Linear projection from state_dim to embed_dim
            self.input_proj = nn.Linear(state_dim, embed_dim)

            # Learnable positional encoding
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

            # LayerNorm before transformer
            self.input_norm = nn.LayerNorm(embed_dim)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-norm for better training stability
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Output projection
            self.output_norm = nn.LayerNorm(embed_dim)

        def forward(
            self, x: torch.Tensor, mask: torch.Tensor | None = None
        ) -> torch.Tensor:
            """Encode state sequence.

            Args:
                x: [B, seq_len, state_dim]
                mask: [B, seq_len] boolean mask (True = valid, False = padding)

            Returns:
                [B, embed_dim] — masked mean pooling of transformer output.
            """
            B, S, _ = x.shape

            # Project and add positional encoding
            h = self.input_proj(x) + self.pos_embedding[:, :S, :]
            h = self.input_norm(h)

            # Transformer expects src_key_padding_mask where True = ignore
            padding_mask = None
            if mask is not None:
                padding_mask = ~mask  # invert: True = padding = ignore

            h = self.transformer(h, src_key_padding_mask=padding_mask)

            # Masked mean pooling
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()  # [B, S, 1]
                h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                h = h.mean(dim=1)

            return self.output_norm(h)

        @property
        def output_dim(self) -> int:
            return self._embed_dim
