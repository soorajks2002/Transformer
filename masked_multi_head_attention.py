import torch

class MaskedMultiHeadAttention(torch.nn.Module) :
    def __init__(self, embedding_size, n_attention_heads) :
        super(MaskedMultiHeadAttention, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embedding_size, n_attention_heads)
        self.layer_normalization = torch.nn.LayerNorm(embedding_size)
        
    def __call__(self, x) :
        output = x.permute(1,0,2)
        
        mask = torch.tril(torch.ones(output.size(0), output.size(0)), diagonal=0)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        output, _ = self.attention(output, output, output, attn_mask=mask)
        output = output.permute(1,0,2)
        
        output = output.reshape(-1, output.size(2))
        output = self.layer_normalization(output)
        output = output.view(x.size(0), x.size(1), x.size(2))
        output += x
        
        return output