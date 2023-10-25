import torch

class CrossAttention(torch.nn.Module) :
    def __init__(self, embedding_size, n_attention_heads) :
        super(CrossAttention, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embedding_size, n_attention_heads)
        self.layer_normalization = torch.nn.LayerNorm(embedding_size)
        
    def __call__(self, encoder_x, decoder_x) :
        output,_ = self.attention(decoder_x, encoder_x, encoder_x)
        output = output.view(-1, output.size(2))
        output = self.layer_normalization(output)
        output = output.view(decoder_x.size(0), decoder_x.size(1), decoder_x.size(2))
        output += decoder_x
        
        return output