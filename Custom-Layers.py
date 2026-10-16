from torch import Module, Tensor, rand_like, sin, cos
from torch.nn import Parameter
class WordEmbedding_Layer(Module):

    weight_matrix: Parameter
    vocab_matrix: Parameter

    def __init__(self, number_of_word_embeddings: int, size_of_each_word_embedding: int) -> None:
        super().__init__()
        input_tensor_for_weights = Tensor(size_of_each_word_embedding,number_of_word_embeddings)
        input_tensor_for_weights = rand_like(input_tensor_for_weights)

        self.weight_matrix = Parameter(input_tensor_for_weights)
        self.weight_matrix.requires_grad = True

        self.vocab_matrix = Parameter(Tensor(number_of_word_embeddings,size_of_each_word_embedding))
        self.vocab_matrix.requires_grad = False

    def get_word_embedding_matrix(self) -> Tensor:
        return self.vocab_matrix.matmul(self.weight_matrix)
    

class Position_Encoded_Word_Embedding_Layer(Module):
    
    position_encoded_word_embedding_matrix: Tensor
    word_embedding_matrix: Tensor

    def __init__(self, word_embedding_matrix) -> None:
        super().__init__()
        self.word_embedding_matrix = word_embedding_matrix
        rows, columns = word_embedding_matrix.shape
        
        for i in range(rows):
            for j in range(columns):
                if columns % 2 is 0:
                    self.word_embedding_matrix[i][j] = self.word_embedding_matrix[i][j] + sin(Tensor(j))
                else:
                    self.word_embedding_matrix[i][j] = self.word_embedding_matrix[i][j] + cos(Tensor(j))

        self.position_encoded_word_embedding_matrix = self.word_embedding_matrix

    def get_position_encoded_word_embedding_matrix(self)-> Tensor:
        return self.position_encoded_word_embedding_matrix

                



        
    
