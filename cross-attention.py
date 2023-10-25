import torch

class CrossAttention(torch.nn.Module) :
    def __init__(self, embedding_size, n_attention_heads) :
        super(CrossAttention, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embedding_size, n_attention_heads)
        
    def __call__(self, encoder_x, decoder_x) :
        out,_ = self.attention(decoder_x, encoder_x, encoder_x)
        return out