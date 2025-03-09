import torch.nn as nn
import torch.nn.functional as F

class AcmeModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.fc(x)
        return x, F.softmax(x, dim=1)
