import torch
import torch.nn as nn
from torch import Tensor

class MLP(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        output_size: int = 10
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
