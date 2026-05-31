# course-wafer-project

基于迁移学习与可解释 AI 的晶圆表面缺陷图谱识别项目。项目使用 WM-811K 晶圆图数据集，将晶圆缺陷图谱划分为 9 类，并以 ResNet-34 完成分类建模，后续通过 Grad-CAM 生成热力图辅助解释模型关注区域和可能的工艺来源。

> 当前仓库定位为“可复现课程版”。本地机器不要求完整训练；仓库提供清晰的环境、数据下载、预处理、训练、评估和 Grad-CAM 流程，完整训练建议在支持 CUDA 的 GPU 环境中执行。

## 项目目标

- 解析 Kaggle WM-811K / LSWMD 原始数据，生成 PyTorch `ImageFolder` 格式数据集。
- 使用 ImageNet 预训练 ResNet-34 进行 9 类晶圆缺陷分类。
- 输出训练指标、测试集分类报告、混淆矩阵等评估结果。
- 在训练完成后使用 Grad-CAM 生成缺陷热力图，为工艺/设备溯源分析提供视觉依据。

## 数据来源

- 数据集：WM-811K Wafer Map
- Kaggle 地址：https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
- 原始文件：`LSWMD.pkl`

数据和模型权重不提交到 GitHub。请按下文命令下载并在本地生成：

```text
archive/LSWMD.pkl          # Kaggle 原始数据，本地文件，不入库
dataset/train|val|test/    # 预处理生成的图片数据，不入库
resnet/resNet34.pth        # 训练得到的模型权重，不入库
```

## 目录结构

```text
.
├── README.md
├── environment.yml
├── env_setup.md
├── goal.md
├── analyze_pkl.py
├── data_loader.py
├── visualize_metrics.py
├── scripts/
│   ├── download_dataset.ps1
│   └── download_dataset.sh
├── resnet/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── batch_predict.py
│   └── load_weights.py
├── grad_cam/
│   └── Grad-CAM 相关代码
└── archive/
    └── 下载官网数据集.txt
```

本地工作区中可能存在 `WaferMap/` 和 `WM-811K_semiconductor_wafer_map_pattern_classified/` 参考目录。它们是外部参考项目/数据衍生材料，当前 `.gitignore` 默认不上传，避免嵌套 Git 仓库以错误 submodule 形式进入本仓库。

## 环境需求

推荐训练环境：

- Python 3.10
- Conda 环境名：`wafer`
- PyTorch + torchvision
- NVIDIA GPU，显存建议 6GB 以上
- CUDA 版 PyTorch 推荐用于完整训练

可选调试环境：

- 无 GPU 时可以使用 CPU 版 PyTorch
- CPU 环境适合运行代码检查、数据预处理、小批量调试
- 不建议在 CPU 环境完整训练全部 30 个 epoch

创建环境：

```bash
conda env create -f environment.yml
conda activate wafer
```

如果需要 CUDA 版 PyTorch，请按本机 CUDA 版本参考 PyTorch 官网命令安装：

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果只做 CPU 调试：

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 数据下载与预处理

1. 准备 Kaggle API token。

   将 `kaggle.json` 放到默认位置：

   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`

2. 下载数据。

   Windows PowerShell：

   ```powershell
   .\scripts\download_dataset.ps1
   ```

   Linux/macOS：

   ```bash
   bash scripts/download_dataset.sh
   ```

3. 生成训练、验证、测试图片集。

   ```bash
   python data_loader.py
   ```

生成后的数据结构：

```text
dataset/
├── train/0..8/
├── val/0..8/
└── test/0..8/
```

类别映射：

```text
0 Center
1 Donut
2 Edge-Loc
3 Edge-Ring
4 Loc
5 Random
6 Scratch
7 Near-full
8 none
```

4. 生成数据集摘要和样例图。

   ```bash
   python scripts/summarize_dataset.py --dataset-dir dataset --output-dir reports
   ```

   输出文件：

   ```text
   reports/dataset_summary.json
   reports/dataset_counts.csv
   reports/dataset_sample_gallery.png
   ```

## 训练与评估

进入 ResNet 目录执行训练：

```bash
cd resnet
python train.py
```

训练脚本会：

- 加载 `../dataset/train` 和 `../dataset/val`
- 使用 ResNet-34，输出 9 类结果
- 使用类别权重缓解样本不均衡
- 保存最优权重到 `resnet/resNet34.pth`
- 保存训练指标到 `resnet/training_metrics/metrics.json`
- 在训练结束后评估 `../dataset/test` 并生成 `resnet/test_report.json`

当前仓库不要求在本机完成完整训练。如果机器没有合适 GPU，只需确认脚本可编译、数据路径正确，并在 GPU 环境中继续训练。

如果已经有训练好的权重，可单独运行测试集评估：

```bash
cd resnet
python evaluate.py --data-dir ../dataset/test --weights resNet34.pth
```

评估会输出：

```text
resnet/test_report.json
resnet/confusion_matrix.json
```

## 预测

单张图片预测：

```bash
cd resnet
python predict.py --image-path ../dataset/test/0/0000000.png --weights resNet34.pth
```

批量预测：

```bash
cd resnet
python batch_predict.py --data-path ../dataset/test --weights resNet34.pth --output-csv batch_predictions.csv
```

## Grad-CAM

Grad-CAM 需要先获得训练后的权重，例如：

```text
resnet/resNet34.pth
```

建议流程：

1. 完成 ResNet 训练并保存权重。
2. 从 9 个类别中各选择 1 张代表性测试图片。
3. 使用 ResNet 的最后卷积层生成热力图。
4. 将热力图与原晶圆图叠加，分析模型关注区域是否与缺陷空间分布一致。

命令示例：

```bash
cd grad_cam
python resnet_grad_cam.py --image-path ../dataset/test/6/0000000.png --weights ../resnet/resNet34.pth --output outputs/scratch_grad_cam.png
```

工艺机理溯源说明见 [docs/mechanism_traceability.md](docs/mechanism_traceability.md)。

## 非训练工作流

Windows PowerShell 可运行：

```powershell
.\scripts\run_non_training_workflow.ps1
```

如果 Windows 执行策略禁止脚本运行，可只对本次命令临时放行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_non_training_workflow.ps1
```

该脚本会完成：

- Python 静态编译检查
- 已生成 `dataset/` 的样本统计与样例图生成
- 检查是否存在训练权重，并提示评估/Grad-CAM 后续命令

如果没有 `resnet/resNet34.pth`，评估和 Grad-CAM 会跳过，但对应代码入口已经准备好。

## 已有基线结果

当前工作区已有一次训练产物记录，作为课程版基线参考：

- 测试集 accuracy：约 `0.7430`
- macro F1：约 `0.7160`
- weighted F1：约 `0.7564`

这些结果来自本地已有 `resnet/test_report.json`，轻量摘要见 [docs/baseline_results.md](docs/baseline_results.md)。完整复现需要按本 README 重新下载数据、预处理并训练。

## 代码检查

不执行完整训练时，可先做静态编译检查：

```bash
python -m py_compile analyze_pkl.py data_loader.py visualize_metrics.py
python -m py_compile resnet/model.py resnet/train.py resnet/predict.py resnet/batch_predict.py resnet/load_weights.py
python -m py_compile grad_cam/main_cnn.py grad_cam/main_vit.py grad_cam/main_swin.py grad_cam/utils.py grad_cam/vit_model.py grad_cam/swin_model.py
```

## 说明

- `dataset/`、`archive/*.pkl`、模型权重和训练输出均被 `.gitignore` 排除。
- 私有仓库也不建议提交 Kaggle 原始数据和模型权重，避免仓库膨胀和数据许可问题。
- 本项目用于课程实践与实验复现，不直接作为生产质检系统。
