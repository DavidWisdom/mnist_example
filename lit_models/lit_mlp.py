from typing import List

import torch
from torchmetrics.classification import Accuracy
import pytorch_lightning as pl
from models import MLP  # 假设您的 MLP 模型在此文件中

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

        # 设置量化配置
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc(logits, y)
        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 非量化模型评估
        self.model.eval()
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc(logits, y)
        self.log("val/loss_non_quant", loss, on_epoch=True)
        self.log("val/acc_non_quant", acc, on_epoch=True)

        # 量化模型评估
        # 准备量化
        model_quant = self.model
        model_quant.eval()
        torch.quantization.prepare(model_quant, inplace=True)  # 准备量化
        logits_quant = model_quant(x)
        loss_quant = self.loss_fn(logits_quant, y)
        acc_quant = self.acc(logits_quant, y)
        self.log("val/loss_quant", loss_quant, on_epoch=True)
        self.log("val/acc_quant", acc_quant, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # 非量化模型评估
        self.model.eval()
        x, y = batch
        logits = self.model(x)
        acc = self.acc(logits, y)
        self.log("test/acc_non_quant", acc, on_epoch=True)

        # 量化模型评估
        model_quant = self.model
        model_quant.eval()
        torch.quantization.prepare(model_quant, inplace=True)  # 准备量化
        logits_quant = model_quant(x)
        acc_quant = self.acc(logits_quant, y)
        self.log("test/acc_quant", acc_quant, on_epoch=True)

    def on_validation_epoch_end(self):
        # 在验证结束后转换为量化模型
        torch.quantization.convert(self.model, inplace=True)  # 转换为量化模型

    def on_test_epoch_end(self):
        # 在测试结束后转换为量化模型
        torch.quantization.convert(self.model, inplace=True)  # 转换为量化模型

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)
        return [optimizer], [scheduler]
