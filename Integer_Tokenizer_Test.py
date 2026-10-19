"""Spec for an integer tokenizer.

Implement ``build_vocab``, ``tokenize``, and ``detokenize`` in
``Integer_Tokenizer.py``. These tests are the contract. ``tokenize`` is the
function the rest of the model calls. It returns token ids that can index
``WordEmbedding.weight``.

Splitting
---------
Read the text from left to right and take the longest match of:

    <pad>
    <unk>
    <|prompt_start|>
    <|prompt_end|>
    <|response_start|>
    <|response_end|>
    a word of letters, optionally with one internal apostrophe (You're, I'm)
    a run of digits (11 is one token)
    any other single non-whitespace character (., ?, ×, +, -, ', !)

Whitespace, including newlines, is not a token. It only separates tokens.

Ids
---
``build_vocab(texts)`` returns a ``dict[str, int]`` inserted in id order.
These six strings always occupy ids 0 through 5, even when the texts never
contain them:

    0 <pad>
    1 <unk>
    2 <|prompt_start|>
    3 <|prompt_end|>
    4 <|response_start|>
    5 <|response_end|>

Every other token that the splitter finds in ``texts`` gets the next id,
sorted by Python's default string order. The same token always gets one id.
Input order does not matter. Ids are exactly ``range(len(vocab))``.

``tokenize(text, vocab)`` maps each token through that dict. A token that is
absent maps to ``1`` (``<unk>``). ``detokenize(token_ids, vocab)`` maps ids
back to strings and joins them with a single space. An empty token list
becomes ``""``.
"""

import pytest

from Integer_Tokenizer import build_vocab, detokenize, tokenize

_TEXTS = [
    "<|prompt_start|> Ava has 7 red apples. <|prompt_end|>",
    "<|response_start|> 7 <|response_end|>",
    "What is 11 × 11?",
    "You're welcome.",
    "I'm fine.",
    "'light'",
    "how",
    "How many",
]

_VOCAB_IN_ID_ORDER = [
    "<pad>",
    "<unk>",
    "<|prompt_start|>",
    "<|prompt_end|>",
    "<|response_start|>",
    "<|response_end|>",
    "'",
    ".",
    "11",
    "7",
    "?",
    "Ava",
    "How",
    "I'm",
    "What",
    "You're",
    "apples",
    "fine",
    "has",
    "how",
    "is",
    "light",
    "many",
    "red",
    "welcome",
    "×",
]


def test_build_vocab_reserves_special_ids_then_sorts_the_rest():
    vocab = build_vocab(_TEXTS)

    assert list(vocab) == _VOCAB_IN_ID_ORDER
    assert list(vocab.values()) == list(range(len(vocab)))
    assert build_vocab(list(reversed(_TEXTS))) == vocab
    assert list(build_vocab([])) == _VOCAB_IN_ID_ORDER[:6]

    with pytest.raises(TypeError):
        build_vocab("Ava has 7")
    with pytest.raises(TypeError):
        build_vocab(["Ava", 7])


def test_tokenize_keeps_specials_numbers_and_apostrophe_words_whole():
    vocab = build_vocab(_TEXTS)

    assert tokenize("<|prompt_start|> Ava has 7.", vocab) == [2, 11, 18, 9, 7]
    assert tokenize("7", vocab) == [9]
    assert tokenize("<|response_start|> 7 <|response_end|>", vocab) == [4, 9, 5]
    assert tokenize("11", vocab) == [8]
    assert tokenize("What is 11 × 11?", vocab) == [14, 20, 8, 25, 8, 10]
    assert tokenize("<|prompt_end|><|response_start|>", vocab) == [3, 4]
    assert tokenize("You're", vocab) == [15]
    assert tokenize("I'm", vocab) == [13]
    assert tokenize("'light'", vocab) == [6, 21, 6]
    assert tokenize("How", vocab) != tokenize("how", vocab)

    ids = tokenize("Ava has 7.", vocab)
    assert all(type(token_id) is int for token_id in ids)
    assert all(0 <= token_id < len(vocab) for token_id in ids)
    assert 0 not in ids


def test_detokenize_rejoins_with_single_spaces():
    vocab = build_vocab(_TEXTS)
    messy = "<|prompt_start|>\nWhat   is\t11 × 11?\n<|prompt_end|><|response_start|>\n7\n<|response_end|>"
    ids = tokenize(messy, vocab)

    assert detokenize(ids, vocab) == (
        "<|prompt_start|> What is 11 × 11 ? <|prompt_end|> <|response_start|> 7 <|response_end|>"
    )
    assert tokenize(detokenize(ids, vocab), vocab) == ids
    assert tokenize("", vocab) == []
    assert tokenize(" \n\t", vocab) == []
    assert detokenize([], vocab) == ""


def test_unknown_tokens_and_bad_inputs():
    vocab = build_vocab(_TEXTS)

    assert tokenize("Ava zebra", vocab) == [11, 1]
    assert detokenize([11, 1], vocab) == "Ava <unk>"
    assert tokenize("<unk>", vocab) == [1]
    assert tokenize("<pad>", vocab) == [0]

    with pytest.raises(TypeError):
        tokenize(None, vocab)
    with pytest.raises(TypeError):
        detokenize([1.5], vocab)
    with pytest.raises((IndexError, ValueError)):
        detokenize([len(vocab)], vocab)
    with pytest.raises((IndexError, ValueError)):
        detokenize([-1], vocab)
