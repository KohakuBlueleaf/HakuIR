from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

from ...models import HakuIRModel
from ..loss import BasicLoss


class IRTrainer(pl.LightningModule):
    def __init__(
        self,
        ir_model: HakuIRModel,
        loss_func: BasicLoss,
        learning_rate: float = 1e-4,
        optimizer: type = optim.AdamW,
        optimizer_config: dict[str, Any] = {
            'weight_decay': 1e-2,
            'betas': (0.9, 0.999),
        },
        lr_scheduler: type = optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_config: dict[str, Any] = {
            'T_max': 300_000,
            'eta_min': 1e-7,
        },
        warmup_steps: int = 0,
        **kwargs
    ):
        super().__init__()