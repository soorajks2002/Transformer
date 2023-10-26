import torch
from encoder import EncoderBlock
from decoder import DecoderBlock
from data_loader import DataLoader

epochs = 45
learning_rate = 1e-3
batch_size = 100
dataset_path = ""
context_length = 10
embedding_size = 48
number_attention_heads = 10


class Transformers(torch.nn.Module) :
    def __init__(self) :
        super(Transformers, self).__init__()


dataloader = DataLoader(dataset_path, batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(learning_rate)
