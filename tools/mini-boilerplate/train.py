import os
import torch
import random
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np




SEED = 711
DEVICE = "cuda:0"
NUM_EPOCHS = 100
BATCH_SIZE = 128
DATA_ROOT = "C:\\Users\\sultan.abughazal\\Documents\\Datasets\\ugvgpr-dataset"


# define a model
#
class ModelType(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# define a dataset
#
class DatasetType(Dataset):
    def __init__(self, data_root, split):
        super().__init__()
        assert split in ["train", "eval"], "Invalid split!"

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return index, index


def get_inline_args():
    parser = argparse.ArgumentParser(description="Short training script.")
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        required=True,
        help="The path to the output checkpoint.")
    parser.add_argument(
        '--train',
        action="store_true",
        help="A flag to run training.")

    return parser.parse_args()

def main():
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    # define a loss function
    model = ModelType()
    model = model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)

    # train

    ds = DatasetType(DATA_ROOT, "train")
    data_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True)

    outputs = []
    for epoch in range(NUM_EPOCHS):
        for input, target in data_loader:
            input = input.to(DEVICE)
            target = target.to(DEVICE)

            output = model(input)
            loss = criterion(output, input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch:{epoch+1:0>3}, Loss:{loss.item():.8f}')



if __name__ == "__main__":
    main()
