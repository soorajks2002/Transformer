import torch

class MultiHeadAttention(torch.nn.Module) :
    def __init__(self, embedding_size, n_attention_heads) :
        super(MultiHeadAttention, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embedding_size, n_attention_heads)
        self.layer_normalization = torch.nn.LayerNorm(embedding_size)
        
    def __call__(self, x) :
        output, _ = self.attention(x, x, x)
        output = output.view(-1, output.size(2))
        output = self.layer_normalization(output)
        output = output.view(x.size(0), x.size(1), x.size(2))
        output += x
        return output