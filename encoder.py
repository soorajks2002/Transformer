import torch
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward

class EncoderBlock() :
    def __init__(self, context_length, embedding_size, n_attention_heads) :
        self.attention_block = MultiHeadAttention(embedding_size, n_attention_heads)
        self.feed_forward_block = FeedForward(context_length*embedding_size)
        
    def __call__(self, x) :
        out = self.attention_block(x)
        out = self.feed_forward_block(x)
        
        return out