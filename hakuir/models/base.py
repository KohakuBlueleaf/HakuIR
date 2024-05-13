import torch
import torch.nn as nn


class HakuIRModel(nn.Module):
    def __init__(self):
        super(HakuIRModel, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
