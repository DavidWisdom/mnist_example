from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torchvision import transforms
from .custom_dataset import MNISTDataset

class MNISTDataModule(LightningModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """
        数据加载器
        Args:
            data_dir (str): 数据存储目录
            batch_size (int): 批量大小
            num_workers (int): 数据加载的工作线程数
            pin_memory (bool): 是否使用页锁定内存
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        """
        设置数据集
        Args:
            stage (Optional[str]): 当前阶段（train, val, test 或 None）
        """
        # 定义预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # 均值和标准差
        ])

        # 加载训练、验证和测试数据集
        self.train_dataset = MNISTDataset(self.data_dir, train=True, transform=transform)
        self.val_dataset = MNISTDataset(self.data_dir, train=False, transform=transform)
        self.test_dataset = MNISTDataset(self.data_dir, train=False, transform=transform)

    def collate_fn(self, batch):
        """
        自定义的批处理函数
        """
        images, labels = zip(*batch)
        images = torch.stack(images)  # 将图像堆叠成一个批次
        labels = torch.tensor(labels)  # 将标签转为张量
        return images, labels

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=True
        )
