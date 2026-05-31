# 环境配置说明

本项目推荐使用 Conda 管理环境。当前仓库不要求在本机完成完整训练；如果电脑没有合适的 NVIDIA GPU，可以只完成数据预处理、脚本检查和小规模调试。

## 推荐环境

- Python 3.10
- Conda 环境名：`wafer`
- PyTorch + torchvision
- CUDA 版 PyTorch 用于完整训练
- 推荐硬件：NVIDIA GPU，显存 6GB 以上

## 创建基础环境

```bash
conda env create -f environment.yml
conda activate wafer
```

`environment.yml` 不绑定当前机器的绝对路径，也不固定 PyTorch CUDA 构建。创建环境后，根据实际机器安装 PyTorch。

## 安装 PyTorch

CUDA 12.1 示例：

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU 调试环境：

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

如果 CUDA 版本不同，请使用 PyTorch 官网生成的安装命令。

## 安装 Kaggle CLI

`environment.yml` 已包含 `kaggle`。如果当前环境没有安装，可执行：

```bash
pip install kaggle
```

准备 Kaggle API token：

- Windows: `%USERPROFILE%\.kaggle\kaggle.json`
- Linux/macOS: `~/.kaggle/kaggle.json`

## 验证环境

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m py_compile analyze_pkl.py data_loader.py visualize_metrics.py
```

`torch.cuda.is_available()` 为 `False` 时仍可做 CPU 调试，但不建议完整训练。

## 数据准备

Windows PowerShell：

```powershell
.\scripts\download_dataset.ps1
python data_loader.py
```

Linux/macOS：

```bash
bash scripts/download_dataset.sh
python data_loader.py
```

## 训练

```bash
cd resnet
python train.py
```

训练输出包括：

- `resnet/resNet34.pth`
- `resnet/training_metrics/metrics.json`
- `resnet/test_report.json`

这些文件默认不进入 Git 仓库。
