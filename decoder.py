import torch
from masked_multi_head_attention import MaskedMultiHeadAttention
from cross_attention import CrossAttention
from feed_forward import FeedForward

class DecoderBlock() :
    def __init__(self, context_length, embedding_size, n_attention_heads) :
        self.masked_attention_block = MaskedMultiHeadAttention(embedding_size, n_attention_heads)
        self.cross_attention_block = CrossAttention(embedding_size, n_attention_heads)
        self.feed_forward_block = FeedForward(context_length*embedding_size)
        
    def __call__(self, encoder_x, decoder_x) :
        decoder_x = self.masked_attention_block(decoder_x)
        out = self.cross_attention_block(encoder_x, decoder_x)
        out = self.feed_forward_block(out)
        
        return out