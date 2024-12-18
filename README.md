# mnist_example

`mnist_example` 是一个基于 PyTorch Lightning 构建的深度学习项目，旨在对 MNIST 数据集进行分类任务。该项目包括数据加载、模型定义、训练和测试等功能，目的是为研究人员和开发人员提供一个简洁且易于扩展的框架，用于快速启动基于 MNIST 数据集的图像分类任务。

## 项目结构

```
mnist_example/
│
├── conf/                           # 配置文件
│   └── config.yaml                 # 配置文件，包含训练参数等
│
├── data/                           # 数据模块
│   └── custom_datamodule.py        # 自定义的数据模块
│   └── custom_dataset.py           # 自定义数据集类
│
├── models/                         # 模型定义
│   └── mlp.py                      # 简单的多层感知器模型
│
├── lit_models/                     # PyTorch Lightning 模型
│   └── lit_mlp.py                  # 基于 PyTorch Lightning 的多层感知器模型封装
│
├── scripts/                        # 训练脚本
│   └── run_experiment.py           # 用于训练和评估模型的主脚本
│
└── README.md                       # 项目的说明文档
```

## 安装

### 克隆仓库

首先，克隆该项目仓库：

```bash
git clone https://github.com/your-username/mnist_example.git
cd mnist_example
```

### 创建虚拟环境

建议使用虚拟环境来管理项目依赖：

```bash
python -m venv venv
source venv/bin/activate  # Windows 用户使用 venv\Scripts\activate
```

### 安装依赖

安装项目的所有依赖：

```bash
pip install -r requirements.txt
```

## 配置

项目的配置文件位于 `conf/config.yaml`，你可以在此文件中配置训练参数、数据加载设置、模型超参数等。

### 示例配置

```yaml
seed: 1234

trainer:
  overfit_batches: 0.0
  check_val_every_n_epoch: 1
  max_epochs: 25
  min_epochs: 1
  num_sanity_val_steps: 0

data:
  batch_size: 16
  num_workers: 4
  pin_memory: true

lit_model:
  lr: 0.001
  weight_decay: 0.0005
  milestones: [5]
  gamma: 0.1
```

## 使用方法

### 训练模型

要开始训练模型，请运行以下命令：

```bash
python scripts/run_experiment.py
```

此命令会加载配置文件中的设置，初始化数据模块和模型，开始训练并在训练过程中保存检查点。

### 测试模型

训练完成后，你可以使用以下命令对测试集进行评估：

```bash
python scripts/run_experiment.py --test
```

### 使用 Hydra 配置

该项目使用 `Hydra` 进行配置管理。你可以通过更改 `config.yaml` 文件中的参数，或在命令行中覆盖配置来调整训练过程。例如：

```bash
python scripts/run_experiment.py trainer.max_epochs=50
```

## 代码结构

### 数据模块

`data/` 目录包含自定义的数据模块和数据集类。`custom_datamodule.py` 负责组织数据加载和预处理，`custom_dataset.py` 实现了自定义的 Dataset 类。

### 模型定义

`models/` 目录包含基础的模型定义，例如 `mlp.py`，这是一个多层感知器（MLP）模型，它用于对 MNIST 数据进行分类。

### PyTorch Lightning 模型

`lit_models/` 目录包含 PyTorch Lightning 模型封装。`lit_mlp.py` 是一个 PyTorch Lightning 封装的多层感知器模型，其中实现了训练、验证、测试步骤等，使得训练过程更加简洁和高效。

### 脚本

`scripts/run_experiment.py` 是主训练脚本，负责加载配置、初始化数据模块和模型，并开始训练和评估。通过 `Hydra` 配置管理，训练过程中的各种超参数和设置都可以通过配置文件或命令行进行灵活调整。

## 贡献

欢迎任何对该项目的贡献！如果你想为该项目做出贡献，请按照以下步骤进行：

1. Fork 该项目
2. 创建一个新的分支 (`git checkout -b feature-branch`)
3. 提交你的更改 (`git commit -am 'Add new feature'`)
4. 推送到分支 (`git push origin feature-branch`)
5. 提交 pull request

## 联系方式

- 维护者：[DavidWisdom](https://github.com/DavidWisdom)
- 项目地址：[https://github.com/DavidWisdom/mnist_example](https://github.com/DavidWisdom/mnist_example)

## 许可证

该项目采用 [MIT License](LICENSE) 开源。

---

### 说明：
1. **项目描述**：简洁描述了项目的功能和目的，方便开发者快速理解。
2. **项目结构**：展示了项目的文件夹结构，并标明了每个文件的作用。
3. **安装和配置**：详细说明了如何安装依赖以及如何配置项目。
4. **使用方法**：提供了清晰的说明，告诉用户如何开始训练和测试模型。
5. **贡献**：鼓励外部开发者参与贡献，并给出明确的贡献步骤。
6. **联系方式与许可证**：提供了联系作者的方式，以及开源许可信息。

通过以上格式，`README.md` 会更加易于理解和使用，可以帮助其他开发者快速上手并贡献代码。