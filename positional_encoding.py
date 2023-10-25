import torch

class positional_encodings() :
    def __init__(self, time_steps, embedding_size) :
        self.time_step = time_steps
        self.embedding_size = embedding_size
        
    def __call__(self) :
        denominator = [i-1 if i%2 else i for i in range(self.embedding_size)]
        denominator = torch.tensor(denominator, dtype=torch.float64)/self.embedding_size
        denominator = pow(1e4, denominator)
        
        numerator = torch.arange(self.time_step, dtype=torch.float64)
        
        numerator = numerator.view(-1, 1)
        denominator = denominator.view(1, -1)
        
        embedding_matrix = numerator @ denominator
        embedding_matrix[:, 0::2] = torch.sin(embedding_matrix[:, 0::2])
        embedding_matrix[:, 1::2] = torch.cos(embedding_matrix[:, 1::2])
        
        return embedding_matrix