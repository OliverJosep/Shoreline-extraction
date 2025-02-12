from torch.utils.data import Dataset

class BaseFormes(Dataset):
    def __init__(self):
        super().__init__()
        self.len = 0

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        raise NotImplementedError("This method should be implemented in subclasses")