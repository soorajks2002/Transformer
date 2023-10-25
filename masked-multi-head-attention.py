import torch

class MultiHeadAttention(torch.nn.Module) :
    def __init__(self, embedding_size, n_attention_heads) :
        super(MultiHeadAttention, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embedding_size, n_attention_heads)
    
    def __call__(self, x) :
        x = x.permute(1,0,2)
        
        mask = torch.tril(torch.ones(x.size(0), x.size(0)), diagonal=0)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        output, _ = self.attention(x, x, x, attn_mask=mask)
        
        return output.permute(1,0,2)