"""Spec for WordEmbedding and PositionalWordEmbedding.

Implement both classes in ``Custom_Layers.py``. These tests are the contract.

WordEmbedding(vocab_size, embedding_dim)
    Learnable lookup table. ``weight`` is an ``nn.Parameter`` of shape
    ``(vocab_size, embedding_dim)``. ``forward`` takes integer token ids of
    any leading shape and returns those rows. Token ids outside
    ``[0, vocab_size)`` raise ``IndexError`` or ``ValueError``. Non-integer
    ids raise ``TypeError``. ``vocab_size`` and ``embedding_dim`` must be > 0.

PositionalWordEmbedding(embedding_dim, max_seq_len)
    Fixed sinusoidal positions, registered as the buffer
    ``positional_encoding`` of shape ``(max_seq_len, embedding_dim)``:

        PE[pos, 2k]   = sin(pos / 10000 ** (2k / embedding_dim))
        PE[pos, 2k+1] = cos(pos / 10000 ** (2k / embedding_dim))

    The buffer is not a parameter. ``forward`` takes embeddings of shape
    ``(..., seq_len, embedding_dim)`` and returns
    ``embeddings + positional_encoding[:seq_len]`` without modifying the
    input. ``seq_len`` above ``max_seq_len``, or a mismatched trailing
    dimension, raises ``ValueError``. ``embedding_dim`` and ``max_seq_len``
    must be > 0.
"""

import math

import pytest
import torch

from Custom_Layers import PositionalWordEmbedding, WordEmbedding


def _sinusoid(position: int, index: int, embedding_dim: int) -> float:
    even_index = (index // 2) * 2
    angle = position / (10000.0 ** (even_index / embedding_dim))
    if index % 2 == 0:
        return math.sin(angle)
    return math.cos(angle)


def test_word_embedding_returns_the_table_row_for_each_token_id():
    vocab_size = 11
    embedding_dim = 8
    layer = WordEmbedding(vocab_size, embedding_dim)

    assert isinstance(layer.weight, torch.nn.Parameter)
    assert layer.weight.shape == (vocab_size, embedding_dim)
    assert layer.weight.requires_grad

    token_ids = torch.tensor([[1, 4, 0], [7, 4, 2]])
    embeddings = layer(token_ids)

    assert embeddings.shape == (2, 3, embedding_dim)
    assert torch.equal(embeddings[0, 0], layer.weight[1])
    assert torch.equal(embeddings[0, 1], layer.weight[4])
    assert torch.equal(embeddings[1, 0], layer.weight[7])
    assert torch.equal(embeddings[0, 1], embeddings[1, 1])

    sequence = torch.tensor([3, 0, 3])
    assert layer(sequence).shape == (3, embedding_dim)
    assert torch.equal(layer(sequence)[2], layer.weight[3])


def test_word_embedding_gradients_accumulate_only_on_tokens_that_appeared():
    vocab_size = 11
    embedding_dim = 8
    layer = WordEmbedding(vocab_size, embedding_dim)
    token_ids = torch.tensor([[1, 4, 0], [7, 4, 2]])

    layer(token_ids).sum().backward()

    expected = torch.zeros(vocab_size, embedding_dim)
    for token_id in token_ids.flatten().tolist():
        expected[token_id] += 1
    torch.testing.assert_close(layer.weight.grad, expected)


@pytest.mark.parametrize("embedding_dim", [1, 8])
def test_positional_encoding_matches_the_sinusoid_and_is_not_learned(embedding_dim):
    max_seq_len = 6
    layer = PositionalWordEmbedding(embedding_dim, max_seq_len)
    positional = layer.positional_encoding

    assert positional.shape == (max_seq_len, embedding_dim)
    assert positional.requires_grad is False
    assert list(layer.parameters()) == []
    assert "positional_encoding" in dict(layer.named_buffers())

    expected = torch.tensor(
        [
            [_sinusoid(position, index, embedding_dim) for index in range(embedding_dim)]
            for position in range(max_seq_len)
        ],
        dtype=positional.dtype,
    )
    torch.testing.assert_close(positional, expected, atol=1e-5, rtol=1e-4)

    # Position 0 is the identity offset: sin(0) = 0 on even columns, cos(0) = 1 on odd ones.
    assert positional[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    if embedding_dim > 1:
        assert positional[0, 1].item() == pytest.approx(1.0, abs=1e-6)
        assert not torch.equal(positional[0], positional[1])


def test_positional_forward_adds_the_sequence_index_without_changing_the_input():
    vocab_size = 20
    embedding_dim = 8
    seq_len = 4
    words = WordEmbedding(vocab_size, embedding_dim)
    positions = PositionalWordEmbedding(embedding_dim, max_seq_len=9)

    token_ids = torch.tensor([[5, 5, 1, 5]])
    embeddings = words(token_ids)
    before = embeddings.detach().clone()
    encoded = positions(embeddings)

    assert encoded.shape == embeddings.shape
    torch.testing.assert_close(
        encoded,
        embeddings + positions.positional_encoding[:seq_len],
    )
    assert torch.equal(embeddings, before)

    # Same word at two indexes differs by the position rows only.
    torch.testing.assert_close(
        encoded[0, 0] - encoded[0, 1],
        positions.positional_encoding[0] - positions.positional_encoding[1],
    )
    # Different words at one index differ by their table rows only.
    torch.testing.assert_close(
        encoded[0, 1] - encoded[0, 2],
        words.weight[5] - words.weight[1],
    )

    encoded.sum().backward()
    assert words.weight.grad is not None
    assert words.weight.grad[5].abs().sum() > 0
    assert positions.positional_encoding.grad is None


def test_layers_reject_invalid_sizes_ids_and_sequence_lengths():
    with pytest.raises(ValueError):
        WordEmbedding(0, 4)
    with pytest.raises(ValueError):
        WordEmbedding(4, 0)
    with pytest.raises(ValueError):
        PositionalWordEmbedding(0, 4)
    with pytest.raises(ValueError):
        PositionalWordEmbedding(4, 0)

    layer = WordEmbedding(5, 4)
    with pytest.raises(TypeError):
        layer(torch.tensor([1.0, 2.0]))
    with pytest.raises((IndexError, ValueError)):
        layer(torch.tensor([-1, 0]))
    with pytest.raises((IndexError, ValueError)):
        layer(torch.tensor([5]))

    positions = PositionalWordEmbedding(4, max_seq_len=3)
    with pytest.raises(ValueError):
        positions(torch.randn(2, 4, 4))
    with pytest.raises(ValueError):
        positions(torch.randn(2, 3, 5))
