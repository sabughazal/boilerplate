import os
import torch
import numpy as np
from torch.utils.data import Dataset

class AcmeDataset(Dataset):
    def __init__(self, data_root, split):
        super().__init__()
        self.samples = []
        # assert os.path.exists(data_root), "Invalid data root path!"
        assert split in ["train", "eval", "test"], "Invalid split!"

        # placeholder dataset
        for _ in range(2000 * (["test", "eval", "train"].index(split)+1)):
            self.samples.append((
                np.random.random((1, 28, 28)),
                np.random.randint(0, 10, size=(1,))
            ))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        sample, label = self.samples[index]
        sample = sample.flatten()

        sample = torch.from_numpy(sample).type(torch.float32)
        label = torch.from_numpy(label).type(torch.float32)

        return sample, label



if __name__ == "__main__":
    import os
    import sys
    # from torch.utils.data import DataLoader

    if not os.path.exists(sys.argv[1]):
        raise ValueError("Invalid path!")

    data_root = sys.argv[1]
    ds = AcmeDataset(data_root, "train")
    # dl = DataLoader(dataset=ds, batch_size=4, shuffle=True)
    # for inputs, labels in dl:
    #     print(inputs.shape)
    #     break
    # print(len(dl.dataset))
    print(ds[0])
