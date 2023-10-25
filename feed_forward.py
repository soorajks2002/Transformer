import torch

class FeedForward(torch.nn.Module) :
    def __init__(self, input_size) :
        super(FeedForward, self).__init__()
        self.flatten_layer = torch.nn.Flatten()
        self.linear_layer = torch.nn.Linear(input_size, input_size)
        self.activation_layer = torch.nn.LeakyReLU()
        self.normalization_layer = torch.nn.LayerNorm(input_size)
        
    def __call__(self, x) :
        x = self.flatten_layer(x)
        out = self.linear_layer(x)
        out = self.activation_layer(out)
        out = self.normalization_layer(out)
        
        out += x
        
        return out