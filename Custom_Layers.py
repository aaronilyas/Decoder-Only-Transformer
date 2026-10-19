import math

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class WordEmbedding(Module):
    """Learnable token lookup table of shape (vocab_size, embedding_dim)."""

    weight: Parameter

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        if vocab_size < 1 or embedding_dim < 1:
            raise ValueError("vocab_size and embedding_dim must be positive")

        self.weight = Parameter(torch.randn(vocab_size, embedding_dim))

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.dtype.is_floating_point or token_ids.dtype == torch.bool:
            raise TypeError("token_ids must be an integer tensor of token indices")
        # Indexing wraps negatives. One combined reduction syncs at most once.
        vocab_size = self.weight.shape[0]
        out_of_range = (token_ids < 0) | (token_ids >= vocab_size)
        if torch.any(out_of_range):
            raise IndexError(f"token ids must be in [0, {vocab_size})")
        return self.weight[token_ids.long()]


def _sinusoidal_positional_encoding(max_seq_len: int, embedding_dim: int) -> Tensor:
    """Sinusoidal positions, including a trailing sine when embedding_dim is odd.

    PE[pos, 2k] = sin(pos / 10000^(2k/d))
    PE[pos, 2k+1] = cos(pos / 10000^(2k/d))
    """
    position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    two_k = torch.arange(0, embedding_dim, 2, dtype=torch.float32)
    div_term = torch.exp(two_k * (-math.log(10000.0) / embedding_dim))
    angles = position * div_term

    encoding = torch.zeros(max_seq_len, embedding_dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    odd_columns = embedding_dim // 2
    if odd_columns:
        encoding[:, 1::2] = torch.cos(angles[:, :odd_columns])
    return encoding


class PositionalWordEmbedding(Module):
    """Adds a fixed sinusoidal positional encoding to token embeddings."""

    positional_encoding: Tensor

    def __init__(self, embedding_dim: int, max_seq_len: int) -> None:
        super().__init__()
        if embedding_dim < 1 or max_seq_len < 1:
            raise ValueError("embedding_dim and max_seq_len must be positive")

        encoding = _sinusoidal_positional_encoding(max_seq_len, embedding_dim)
        self.register_buffer("positional_encoding", encoding)

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim < 2:
            raise ValueError("embeddings must have shape (..., seq_len, embedding_dim)")

        positional = self.positional_encoding
        seq_len = embeddings.shape[-2]
        embedding_dim = embeddings.shape[-1]
        if embedding_dim != positional.shape[1]:
            raise ValueError(
                f"expected embedding dim {positional.shape[1]}, got {embedding_dim}"
            )
        if seq_len > positional.shape[0]:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {positional.shape[0]}"
            )

        encoding = positional[:seq_len].to(dtype=embeddings.dtype, device=embeddings.device)
        return embeddings + encoding
