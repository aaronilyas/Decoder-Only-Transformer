from typing import Dict, List, Set
from torch.nn import Parameter
from torch import Tensor, rand_like, sin, cos, zeros_like
from torch.nn import Module
class word_embedding_layer(Module):
    vocab_dict: Dict
    weight_matrix: Parameter

    def __init__(self, vocab_set: Set, dimension_of_word_embedding_vectors: int) -> None:
        super().__init__()

        self.weight_matrix = Parameter(rand_like(Tensor(len(vocab_set),dimension_of_word_embedding_vectors)))
        self.vocab_dict = {}

        for word in vocab_set:
            word_input_tensor = zeros_like(Tensor(len(vocab_set), 1))
            word_input_tensor[0] = 1
            self.vocab_dict[str(word)] = word_input_tensor.T @ self.weight_matrix


    def feed_forward(self,word_input: Tensor) -> Tensor:
        return word_input.T @ self.weight_matrix

    def __str__(self) -> str:
        return str(self.vocab_dict)

    def get_keys_from_dict(self)->str:
        keys = ""
        for word in self.vocab_dict.keys():
            keys += word + " "

        return keys

    def get_values_from_dict(self)->int:
        values: List = []
        for value in self.vocab_dict.values():
            values.append(value)

        return len(values[0])
