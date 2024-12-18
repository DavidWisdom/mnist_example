import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

class MNISTDataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None):
        """
        MNIST 数据集加载器
        Args:
            root (str): 数据集的根目录
            train (bool): 是否加载训练数据
            transform (callable): 数据预处理方法
        """
        self.dataset = MNIST(root=root, train=train, download=True)
        self.transform = transform

    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        获取指定索引的数据样本
        """
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
