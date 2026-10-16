import math

import pytest
import torch

from custom_layers import Position_Encoded_Word_Embedding_Layer, WordEmbedding_Layer


def _reference_positional_encoding(position: int, index: int, embedding_dim: int) -> float:
    even_index = (index // 2) * 2
    angle = position / (10000.0 ** (even_index / embedding_dim))
    if index % 2 == 0:
        return math.sin(angle)
    return math.cos(angle)


def test_word_embedding_lookup_shape_and_grad():
    vocab_size = 11
    embedding_dim = 8
    layer = WordEmbedding_Layer(vocab_size, embedding_dim)

    table = layer.get_word_embedding_matrix()
    assert table.shape == (vocab_size, embedding_dim)
    assert table is layer.weight_matrix
    assert layer.weight_matrix.requires_grad

    token_ids = torch.tensor([[1, 4, 0], [7, 4, 2]])
    embeddings = layer(token_ids)
    assert embeddings.shape == (2, 3, embedding_dim)
    assert torch.equal(embeddings[0, 1], layer.weight_matrix[4])
    assert torch.equal(embeddings[1, 0], layer.weight_matrix[7])

    ranked_ids = torch.randint(0, vocab_size, (2, 3, 5))
    assert layer(ranked_ids).shape == (2, 3, 5, embedding_dim)

    embeddings.sum().backward()
    used = {1, 4, 0, 7, 2}
    for token_id in range(vocab_size):
        row_grad = layer.weight_matrix.grad[token_id]
        if token_id in used:
            assert row_grad.abs().sum() > 0
        else:
            assert torch.equal(row_grad, torch.zeros(embedding_dim))


@pytest.mark.parametrize("embedding_dim", [1, 7, 8])
def test_positional_encoding_matches_formula(embedding_dim):
    max_seq_len = 10
    layer = Position_Encoded_Word_Embedding_Layer(embedding_dim, max_seq_len)
    positional = layer.get_position_encoded_word_embedding_matrix()

    assert positional.shape == (max_seq_len, embedding_dim)
    assert positional.requires_grad is False
    assert list(layer.parameters()) == []
    assert positional is layer.position_encoded_word_embedding_matrix

    expected = torch.empty(max_seq_len, embedding_dim)
    for position in range(max_seq_len):
        for index in range(embedding_dim):
            expected[position, index] = _reference_positional_encoding(
                position, index, embedding_dim
            )
    torch.testing.assert_close(positional, expected, atol=1e-5, rtol=1e-4)

    for position, index in ((0, 0), (3, embedding_dim - 1)):
        actual = positional[position, index].item()
        reference = _reference_positional_encoding(position, index, embedding_dim)
        assert abs(actual - reference) < 1e-5


def test_positional_forward_adds_encoding_out_of_place():
    vocab_size = 20
    embedding_dim = 7
    seq_len = 5
    words = WordEmbedding_Layer(vocab_size, embedding_dim)
    positions = Position_Encoded_Word_Embedding_Layer(embedding_dim, max_seq_len=9)

    token_ids = torch.randint(0, vocab_size, (4, seq_len))
    embeddings = words(token_ids)
    before = embeddings.detach().clone()
    encoded = positions(embeddings)

    assert encoded.shape == (4, seq_len, embedding_dim)
    torch.testing.assert_close(
        encoded,
        embeddings + positions.position_encoded_word_embedding_matrix[:seq_len],
    )
    assert torch.equal(embeddings, before)

    unbatched = words(token_ids[0])
    assert positions(unbatched).shape == (seq_len, embedding_dim)

    encoded.sum().backward()
    assert words.weight_matrix.grad is not None
    assert positions.position_encoded_word_embedding_matrix.grad is None

    with pytest.raises(ValueError):
        positions(torch.randn(2, 10, embedding_dim))
