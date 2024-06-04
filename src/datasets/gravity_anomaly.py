from torch.utils.data import Dataset

class GravityAnomalyDataset(Dataset):
    def __init__(self, data_root, split):
        super().__init__()


    def __len__(self):
        return 10
    

    def __getitem__(self, index):
        return (), ()