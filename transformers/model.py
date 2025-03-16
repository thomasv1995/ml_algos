import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    # Embeddings layers, used to map a sentence to a list of embeddings vectors

    def __init__(self, d_model: int, vector_size: int):
        super().__init__()
        self.d_model = d_model # number of embeddings (usually 512)
        self.vocab_size = vector_size
        # Pytorch embeddings layer, maps the token ID to its associated embeddings vector
        self.embedding = nn.Embedding(vocab_size, d_model) 
    
    def forward(self, x):
        # get the embeddings of a given input
        return self.embedding(x) * math.sqrt(self.d_model) # part 3.4 in paper

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model =d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(1000, even_id/self.model)
        position = torch.arange(max_sequence_length, dtype=torch.float).reshape(max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        # interleave the odd and even positional encodings
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        # provides vectors of positional encodings for each word (in this case, 10 words)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE