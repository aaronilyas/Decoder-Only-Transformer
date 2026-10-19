import math

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class WordEmbedding_Layer(Module):
    """Learnable token lookup table of shape (vocab_size, embedding_dim)."""

    weight_matrix: Parameter

    def __init__(self, number_of_word_embeddings: int, size_of_each_word_embedding: int) -> None:
        super().__init__()
        if number_of_word_embeddings < 1 or size_of_each_word_embedding < 1:
            raise ValueError("vocabulary size and embedding dimension must be positive")

        self.weight_matrix = Parameter(
            torch.randn(number_of_word_embeddings, size_of_each_word_embedding)
        )

    def get_word_embedding_matrix(self) -> Tensor:
        return self.weight_matrix

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.dtype.is_floating_point or token_ids.dtype == torch.bool:
            raise TypeError("token_ids must be an integer tensor of token indices")
        return self.weight_matrix[token_ids.long()]


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


class Position_Encoded_Word_Embedding_Layer(Module):
    """Adds a precomputed sinusoidal positional encoding to token embeddings."""

    position_encoded_word_embedding_matrix: Tensor

    def __init__(self, size_of_each_word_embedding: int, max_seq_len: int) -> None:
        super().__init__()
        if size_of_each_word_embedding < 1 or max_seq_len < 1:
            raise ValueError("embedding dimension and max_seq_len must be positive")

        encoding = _sinusoidal_positional_encoding(max_seq_len, size_of_each_word_embedding)
        self.register_buffer("position_encoded_word_embedding_matrix", encoding)

    def get_position_encoded_word_embedding_matrix(self) -> Tensor:
        return self.position_encoded_word_embedding_matrix

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim < 2:
            raise ValueError("embeddings must have shape (..., seq_len, embedding_dim)")

        positional = self.position_encoded_word_embedding_matrix
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
