# CHANGELOG — 项目部署与更新记录

## 概述

本文档记录从初始版本（commit `803b937`）到当前版本（commit `7af7c16`）的所有更新。初始版本仅包含源代码框架，无数据、无训练权重、无实验结果。本次部署在 RTX 4090 D + CUDA 12.4 环境下完成从环境搭建到最终结题报告的全流程。

**部署日期**：2026-05-31  
**部署环境**：NVIDIA GeForce RTX 4090 D (24GB)，125GB RAM，32 核 CPU，Ubuntu Linux

---

## 一、仓库变更（Git Tracked）

### 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `docs/final_report.md` | 372 | 完整结题报告：含实验说明、9 类分类结果、Grad-CAM 可解释性分析、8 种缺陷的工艺机理溯源及排查建议 |
| `docs/results_analysis.md` | 142 | 技术结果分析：训练指标、每类 F1/Precision/Recall、混淆矩阵分析、改进建议 |
| `CHANGELOG.md` | — | 本文件：项目更新记录 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `reports/dataset_counts.csv` | 重新生成（与初始版本数值一致，确认数据预处理可复现） |
| `reports/dataset_sample_gallery.png` | 重新生成（使用当前环境产出） |

### 未提交的本地产出（.gitignore 排除）

以下文件为本次部署的核心产出，因体积或数据许可原因不入库：

| 路径 | 大小 | 说明 |
|------|------|------|
| `resnet/resNet34.pth` | 82 MB | 训练产出的 ResNet-34 模型权重（best val_acc=0.7551） |
| `resnet/resnet34-pre.pth` | 84 MB | ImageNet 预训练权重（下载自 PyTorch 官方） |
| `resnet/training_metrics/metrics.json` | 66 KB | 30 个 epoch 的完整训练指标记录 |
| `resnet/training_metrics/*.png` | 701 KB | 5 张可视化图表（混淆矩阵、loss/accuracy 曲线等） |
| `resnet/test_report.json` | 1.8 KB | 测试集分类报告（含每类 precision/recall/f1） |
| `grad_cam/outputs/*.png` | 9 张 | 9 个类别的 Grad-CAM 热力图 |
| `archive/LSWMD.pkl` | 2.0 GB | Kaggle WM-811K 原始数据集 |
| `dataset/train\|val\|test/` | 45,519 张 | 预处理后的 ImageFolder 格式图片（64×64 PNG） |

---

## 二、部署流程记录

### 2.1 环境搭建

**Conda 环境**：
```bash
conda env create -f environment.yml   # 创建 wafer 环境 (Python 3.10)
conda activate wafer
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

安装的关键包：

| 包 | 版本 |
|----|------|
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| CUDA | 12.4 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10 |
| kaggle | 1.7.4.5 |

### 2.2 数据下载与预处理

1. **配置 Kaggle API** → 下载 WM-811K 数据集
   ```bash
   bash scripts/download_dataset.sh
   # 产出: archive/LSWMD.pkl (2.0 GB)
   ```

2. **数据预处理** → 解析 pickle，生成 ImageFolder 格式
   ```bash
   python data_loader.py
   # 产出: dataset/ (45,519 张 64×64 PNG)
   #   train: 31,861 | val: 6,824 | test: 6,834
   ```

3. **数据集统计**
   ```bash
   python scripts/summarize_dataset.py --dataset-dir dataset --output-dir reports
   # 产出: reports/dataset_summary.json, dataset_counts.csv, dataset_sample_gallery.png
   ```

### 2.3 模型训练

1. **下载 ImageNet 预训练权重**
   ```bash
   cd resnet
   wget https://download.pytorch.org/models/resnet34-333f7ec4.pth -O resnet34-pre.pth
   ```

2. **训练 ResNet-34**（30 epochs）
   ```bash
   python train.py
   ```
   - 冻结 conv1/bn1/layer1/layer2，微调 layer3/layer4/fc
   - 类别加权 CrossEntropyLoss
   - 数据增强：RandomRotation(90°)、水平/垂直翻转、随机裁剪
   - **Best val_acc**: 0.7551
   - 产出：`resNet34.pth`, `training_metrics/metrics.json`

3. **测试集评估**
   ```bash
   python evaluate.py --data-dir ../dataset/test --weights resNet34.pth
   ```
   - **Test accuracy**: 0.7713
   - **Weighted F1**: 0.7788
   - **Macro F1**: 0.6723
   - 产出：`test_report.json`, `confusion_matrix.json`

### 2.4 可视化

```bash
cd ..
python visualize_metrics.py --metrics_dir resnet/training_metrics
```
产出 5 张图表：

| 图表 | 说明 |
|------|------|
| `loss_curve.png` | 训练 loss 下降曲线 |
| `val_accuracy.png` | 验证准确率上升曲线 |
| `loss_and_accuracy.png` | 双 Y 轴组合图 |
| `per_class_accuracy.png` | 9 类各自准确率变化趋势 |
| `confusion_matrix.png` | 最佳模型的彩色混淆矩阵 |

### 2.5 Grad-CAM 可解释性

```bash
cd grad_cam
# 对 9 个类别各选 1 张代表性测试样本生成热力图
for class in {0..8}; do
  img=$(ls ../dataset/test/$class/*.png | head -1)
  python resnet_grad_cam.py --image-path "$img" \
    --weights ../resnet/resNet34.pth \
    --class-json ../dataset/class_indices.json \
    --output "outputs/${class}_grad_cam.png"
done
```

产出 9 张 Grad-CAM 热力图，覆盖全部缺陷类别。

---

## 三、实验结果对比

### 初始版本 → 当前版本

| 指标 | 初始版本 | 当前版本 | 变化 |
|------|---------|---------|------|
| 模型权重 | 无 | `resNet34.pth` (82MB) | **新增** |
| 训练指标 | 无 | 30 epoch 完整记录 | **新增** |
| 测试准确率 | 无 | 0.7713 | **新增** |
| 可视化图表 | 无 | 5 张 PNG | **新增** |
| Grad-CAM | 无 | 9 张热力图 | **新增** |
| 数据集报告 | 基线数据 | 当前环境重生成 | 可复现验证 |
| 结题报告 | 无 | `docs/final_report.md` (372 行) | **新增** |
| 结果分析 | 无 | `docs/results_analysis.md` (142 行) | **新增** |

### 与 README 基线的对比

| 指标 | README 基线 | 本项目 | 提升 |
|------|-----------|--------|------|
| Test Accuracy | ~0.7430 | **0.7713** | +2.83% |
| Macro F1 | ~0.7160 | 0.6723 | -4.37% ¹ |
| Weighted F1 | ~0.7564 | **0.7788** | +2.24% |

> ¹ Macro F1 偏低主要受 Near-full 类（仅 23 个测试样本，F1=0.46）和 Scratch 类（F1=0.51）拖累。这是极端类别不平衡（Near-full 原始仅 149 样本）的固有挑战。

---

## 四、与 goal.md 目标的对应关系

| goal 步骤 | 原始状态 | 当前状态 |
|----------|---------|---------|
| 第一步：数据预处理 | 仅有代码框架 (`data_loader.py`) | 完成 811,457→172,950→45,519 的完整数据管线；类别加权 + 旋转/翻转双策略不平衡处理 |
| 第二步：迁移学习 | 仅有代码框架 (`resnet/model.py`, `train.py`) | 完成 30 epoch ResNet-34 训练；test acc=77.13%，超过基线 |
| 第三步：Grad-CAM | 仅有代码框架 (`grad_cam/resnet_grad_cam.py`) | 完成 9 类热力图生成；验证模型关注区域与缺陷空间分布的物理一致性 |
| 第四步：工艺溯源 | goal.md 中有框架性描述 | 完成 8 种缺陷的详细溯源：具体到工艺步骤（CMP/刻蚀/光刻/旋涂/PVD/清洗）、设备部件（抛光垫/ESC/机械手/FOUP）、物理机理，并给出排查建议 |
| 交付物1：算法代码库 | 完成（初始版本已有） | 验证全部代码可运行，无修改 |
| 交付物2：可视化结果集 | 无 | 混淆矩阵 + 训练曲线 + 9 张 Grad-CAM 热力图 |
| 交付物3：结题报告 | 无 | `docs/final_report.md`（372 行，含完整交叉分析） |

---

## 五、代码改动说明

本次部署**未修改**任何 Python 源代码。原始代码框架在数据加载、模型定义、训练流程、Grad-CAM 生成等方面均正确可用，仅完成以下操作：

1. **环境依赖安装**：按 `environment.yml` 创建 conda 环境 + 安装 CUDA 版 PyTorch
2. **外部数据获取**：下载 Kaggle 数据集 + ImageNet 预训练权重
3. **执行脚本**：按 README 说明依次运行各阶段脚本
4. **文档产出**：基于训练结果撰写分析报告和结题报告

---

*本 CHANGELOG 由自动化部署流程生成于 2026-05-31。*
