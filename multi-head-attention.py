import torch

class MultiHeadAttention(torch.nn.Module) :
    def __init__(self, embedding_size, n_attention_heads) :
        super(MultiHeadAttention, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embedding_size, n_attention_heads)
    
    def __call__(self, x) :
        output, _ = self.attention(x, x, x)
        return output