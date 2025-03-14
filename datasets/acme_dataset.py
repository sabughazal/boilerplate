import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from bp_utils import calc_multi_class_weights



class AcmeDataset(Dataset):
    def __init__(self, data_root, split, logger=None):
        super().__init__()
        self.samples = []
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.class_counts = [0] * len(self.classes)

        # assert os.path.exists(data_root), "Invalid data root path!"
        assert split in ["train", "eval", "test"], "Invalid split!"

        # placeholder dataset
        for _ in range(2000 * (["test", "eval", "train"].index(split)+1)):
            cls_label = np.random.randint(0, 10, size=(1,))
            self.class_counts[cls_label.item()] += 1

            self.samples.append((
                np.random.random((1, 28, 28)),
                cls_label,
            ))

        random.shuffle(self.samples)
        print(f"Dataset: {split}")
        print(f"Will be extracting a total of {self.__len__():,} ascans.")
        print(*[f"  class '{objcls}' has {count:,} samples." for objcls, count in zip(self.classes, self.class_counts)], sep='\n')
        print("Class weights:")
        class_weights = calc_multi_class_weights(self.class_counts)
        print(*[f"  class '{objcls}' has weight {w:.4f}." for objcls, w in zip(self.classes, class_weights)], sep='\n')
        print("weights", class_weights.tolist())
        print()


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
