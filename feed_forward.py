import torch

class FeedForward(torch.nn.Module) :
    def __init__(self, input_size) :
        super(FeedForward, self).__init__()
        self.flatten_layer = torch.nn.Flatten()
        self.linear_layer = torch.nn.Linear(input_size, input_size)
        self.activation_layer = torch.nn.LeakyReLU()
        self.normalization_layer = torch.nn.LayerNorm(input_size)
        
    def __call__(self, x) :
        out = self.flatten_layer(x)
        out = self.linear_layer(out)
        out = self.activation_layer(out)
        out = self.normalization_layer(out)
        out = out.view(x.size(0), x.size(1), x.size(2))
        out += x
        
        return out