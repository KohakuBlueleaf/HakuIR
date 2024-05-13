import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BasicLoss(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor, reduction: str):
        raise NotImplementedError
