import torch

class Dataset(torch.utils.data.Dataset) :
    def __init__(self, path) :
        pass
    
    def __getitem__(self, index) :
        pass
    
    def __len__(self) :
        pass

def get_dataloader(path, batch_size) : 
    dataset = Dataset(path)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataLoader