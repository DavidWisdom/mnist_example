from typing import List

import torch
from torchmetrics.classification import Accuracy
import pytorch_lightning as pl
from models import MLP

class LitMLP(pl.LightningModule):
    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        milestones: List[int] = [5],
        gamma: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma
        self.model = MLP()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass', num_classes=10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc(logits, y)
        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc(logits, y)
        self.log("val/loss", loss, on_epoch=True)
        self.log("val/acc", acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        acc = self.acc(logits, y)
        self.log("test/acc", acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]
