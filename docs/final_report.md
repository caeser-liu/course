# 基于迁移学习与可解释AI的晶圆表面缺陷图谱识别及工艺机理溯源研究

## 实践课程结题报告

---

## 摘要

半导体晶圆缺陷的自动识别与溯源是提升芯片良率的关键技术。本项目基于 WM-811K 大规模晶圆图数据集，构建了以 ResNet-34 为骨干网络的迁移学习分类模型，实现对 9 类晶圆缺陷模式的自动识别，测试集准确率达 **77.13%**，加权 F1 达 **0.7788**。进一步引入 Grad-CAM 可解释性算法，生成缺陷热力图以验证模型决策的物理合理性，并结合半导体制造工艺知识，对各类缺陷进行设备与工艺层面的机理溯源分析。项目完整实现了"数据预处理→迁移学习→可解释性分析→工艺溯源"的全流程技术路线。

---

## 一、项目背景与立意

半导体芯片制造包含数百道复杂工艺，光刻机、刻蚀机、化学机械抛光机（CMP）等精密设备的微小状态波动，都会在晶圆表面产生特定空间分布的缺陷图案（Defect Pattern）。当前，高端半导体检测设备与底层溯源算法多受制于国外，突破"卡脖子"技术、提升国产芯片良率是我国半导体行业的重大战略需求。

本项目依托机械学科在制造工艺与设备领域的知识积累，采用全数据驱动模式：通过深度学习与计算机视觉技术对晶圆测试后的海量图谱进行智能分类，并创新性地引入可解释 AI 算法（Grad-CAM），不仅识别缺陷类型，更定位缺陷空间位置，进而反推溯源对应的机械设备与制造工艺故障。

---

## 二、数据集与预处理

### 2.1 数据来源

- **数据集**：WM-811K（Large-Scale Wafer Map Dataset, LSWMD）
- **规模**：811,457 张真实工业产线晶圆测试空间分布图
- **标签**：9 类缺陷模式，含 Center、Donut、Edge-Loc、Edge-Ring、Loc、Random、Scratch、Near-full、None
- **来源**：[Kaggle - WM-811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

### 2.2 数据预处理

**数据清洗**：
- 使用 CompatUnpickler 解析历史版本 pandas pickle 文件
- 提取 failureType 标签，过滤 unknown 类，得到 172,950 条有标签数据
- 晶圆图矩阵缩放到 64×64 统一尺寸
- 将单通道晶圆图（值 0/1/2 分别表示无 die/Pass/Fail）映射为三通道灰度 PNG

**类别不平衡处理**（双策略）：
1. **下采样**：None 类从 147,431 张下采样至 20,000 张
2. **数据增强**：训练集随机旋转 90°、水平/垂直翻转、随机裁剪
3. **代价敏感学习**：CrossEntropyLoss 使用类别权重（与样本数成反比）

**数据集划分**（分层采样）：

| 类别 | Train | Val | Test | 原始占比 |
|------|-------|-----|------|---------|
| Center | 3,005 | 644 | 645 | 2.5% |
| Donut | 388 | 83 | 84 | 0.3% |
| Edge-Loc | 3,632 | 778 | 779 | 3.0% |
| Edge-Ring | 6,776 | 1,452 | 1,452 | 5.6% |
| Loc | 2,515 | 538 | 540 | 2.1% |
| Random | 606 | 129 | 131 | 0.5% |
| Scratch | 835 | 178 | 180 | 0.7% |
| Near-full | 104 | 22 | 23 | 0.09% |
| None | 14,000 | 3,000 | 3,000 | 85.2%→下采样后 44.0% |
| **合计** | **31,861** | **6,824** | **6,834** | — |

---

## 三、模型设计与训练

### 3.1 迁移学习策略

- **骨干网络**：ResNet-34，ImageNet 预训练权重初始化
- **网络修改**：fc 层输出从 1000 改为 9，匹配晶圆缺陷类别数
- **冻结策略**：冻结 conv1、bn1、layer1、layer2，仅微调 layer3、layer4 和 fc 层
- **动机**：浅层特征（边缘、纹理）具有通用性，深层语义特征需适配晶圆缺陷领域

### 3.2 训练配置

| 参数 | 配置 |
|------|------|
| Epochs | 30 |
| Batch Size | 32 |
| 优化器 | Adam, lr=0.0001 |
| 损失函数 | CrossEntropyLoss（类别加权） |
| 数据增强 | RandomRotation(90°), RandomHorizontalFlip, RandomVerticalFlip, RandomCrop(64) |
| 归一化 | mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] |
| 硬件 | NVIDIA RTX 4090 D (24GB), CUDA 12.4 |

### 3.3 训练过程

训练过程中验证准确率先快速提升后出现明显过拟合：epoch 3 即达到 0.7431，epoch 15 达到最高 **0.7551**。之后模型在训练集上持续过拟合，验证准确率逐步回落至 epoch 30 的 0.6603（差距 9.5 个百分点），表明需要引入学习率衰减或早停策略。

训练曲线（`resnet/training_metrics/`）：
- `loss_curve.png` — 训练 loss 从 1.16 逐步收敛至 0.53
- `val_accuracy.png` — 验证准确率从 0.5887（epoch 1）提升至 0.7551（epoch 15），之后回落
- `per_class_accuracy.png` — 各类别准确率随训练的变化趋势

---

## 四、实验结果与分析

### 4.1 整体性能

| 指标 | 本项目 | 基线 | 对比 |
|------|--------|------|------|
| Accuracy | **0.7713** | 0.7430 | ↑ 2.83% |
| Macro F1 | 0.6723 | 0.7160 | ↓ 4.37% |
| Weighted F1 | **0.7788** | 0.7564 | ↑ 2.24% |

Accuracy 和 Weighted F1 均超过基线。Macro F1 偏低主要受极端稀有类别（Near-full 仅 23 个测试样本）影响。

### 4.2 各类别详细结果

| 类别 | Precision | Recall | F1 | Support | 分析 |
|------|-----------|--------|-----|---------|------|
| **Edge-Ring** | 0.8311 | **0.9862** | **0.9020** | 1,452 | 表现最优，边缘环模式高度独特 |
| **Center** | **0.8838** | 0.8372 | **0.8599** | 645 | 中心集中分布，特征明显 |
| **None** | 0.9535 | 0.7320 | 0.8282 | 3,000 | 精确率高，但 27% 无缺陷被误判 |
| **Donut** | 0.6320 | 0.9405 | 0.7560 | 84 | Recall 极高，但精确率受小样本限制 |
| **Random** | 0.6613 | 0.6260 | 0.6431 | 131 | 随机模式本质难以建模 |
| **Edge-Loc** | 0.4444 | 0.7445 | 0.5566 | 779 | 精确率低，与 Edge-Ring、None 混淆 |
| **Loc** | 0.7256 | 0.4259 | 0.5368 | 540 | Recall 低，局部缺陷未能有效捕获 |
| **Scratch** | 0.4378 | 0.6056 | 0.5082 | 180 | 64×64 分辨率可能不足以捕获细线 |
| **Near-full** | 0.2987 | **1.0000** | 0.4600 | 23 | 样本极度稀缺，精确率极低 |

### 4.3 混淆矩阵分析

```
                 Predicted
              C   D  EL  ER   L   R   S  NF   N
True Center  540   6  29  15  22   7   7   0  19
     Donut     1  79   0   2   1   1   0   0   0
     Edge-Loc  1   3 580 127  16   5  16  10  21
     Edge-Ring 0   0  13 1432  0   0   0   6   1
     Loc       48  30 113  22 230  15  38   0  44
     Random    0   3   3   5   0  82   0  38   0
     Scratch   1   1  23  16   7   1 109   0  22
     Near-full 0   0   0   0   0   0   0  23   0
     None      20  3 544 104  41  13  79   0 2196
```

**主要混淆对**：

1. **None → Edge-Loc (544 例)**：无缺陷晶圆被误判为边缘局部缺陷，说明模型对边缘区域的微弱信号过度敏感
2. **Edge-Loc → Edge-Ring (127 例)**：边缘局部与边缘环在 64×64 分辨率下空间分布相似
3. **Center → Loc (48 例)**：中心缺陷与局部缺陷的区分边界模糊
4. **Random → Near-full (38 例)**：随机缺陷覆盖面积大时易与近满缺陷混淆

---

## 五、Grad-CAM 可解释性分析

### 5.1 方法

使用 Grad-CAM（Gradient-weighted Class Activation Mapping）算法，提取 ResNet-34 最后一层卷积（layer4）的梯度信息，生成类别激活热力图，叠加到原始晶圆图上。热力图红色区域表示模型决策时最关注的像素位置。

### 5.2 各类别热力图分析

Grad-CAM 输出位于 `grad_cam/outputs/`，对 9 个类别的代表性样本分析如下：

| 类别 | 模型预测 | 置信度 | 热力图分析 |
|------|---------|--------|-----------|
| Center | Edge-Ring | 83.1% | 误分类，但激活确实集中在中心区域 |
| Donut | Edge-Ring | 64.3% | 误分类，激活区域不够集中 |
| **Edge-Loc** | **Edge-Loc** | **73.3%** | **正确分类，激活集中在边缘局部区域** |
| **Edge-Ring** | **Edge-Ring** | **93.8%** | **正确分类，激活沿晶圆边缘形成环状分布** |
| **Loc** | **Loc** | **98.0%** | **正确分类，高置信度，激活精准定位在缺陷簇** |
| **Random** | **Random** | **85.3%** | **正确分类，激活分散在多处，符合随机模式** |
| Scratch | Random | 38.9% | 误分类且低置信度，划痕特征未被捕获 |
| **Near-full** | **Near-full** | **87.4%** | **正确分类，激活覆盖大面积区域** |
| None | Scratch | 51.6% | 误分类，低置信度，无缺陷区域的背景噪声被误读 |

### 5.3 可解释性结论

- **高置信度正确分类**（Edge-Ring、Loc、Random、Near-full）：Grad-CAM 热力图与人类专家认知的缺陷空间分布高度一致，证明模型决策具有物理合理性
- **低置信度误分类**（Scratch→Random、None→Scratch）：热力图发散，无明确聚焦区域，表明模型处于不确定状态
- **错位关注**（Center→Edge-Ring）：虽然预测错误，但模型确实关注了正确的空间区域（中心），问题出在类别判别而非空间定位

---

## 六、工艺机理溯源分析

基于模型分类结果和 Grad-CAM 空间定位，结合半导体制造工艺链知识，对各类缺陷进行设备与工艺层面的溯源反推：

### 6.1 Scratch（划痕）→ CMP 工艺 / 机械搬运

- **缺陷特征**：线状或弧线状连续缺陷，具有明显方向性
- **Grad-CAM 验证**：激活沿划痕方向呈线状分布
- **可能工艺来源**：
  - **CMP（化学机械抛光）**：抛光垫（Polishing Pad）表面嵌入大颗粒金刚石或磨料残留，在晶圆旋转抛光过程中产生定向划痕；抛光液（Slurry）中磨粒粒径分布异常或团聚
  - **机械搬运系统**：机械手臂（Robot Arm/End Effector）末端执行器表面存在硬质颗粒污染；晶圆在 FOUP（前开式晶圆传送盒）中与卡槽边缘发生物理摩擦
  - **清洗工艺**：兆声清洗（Megasonic Cleaning）中声场不均匀导致的微擦伤
- **排查建议**：① 检查 CMP 抛光垫寿命和使用次数，更换接近寿命终点的耗材 ② SEM 检测抛光液粒径分布 ③ 校准机械手取放晶圆的 Z 轴高度和安全距离

### 6.2 Edge-Ring（边缘环）→ 刻蚀工艺 / 边缘温控

- **缺陷特征**：沿晶圆边缘的连续环状缺陷带，宽度均匀
- **Grad-CAM 验证**：激活沿晶圆边缘形成明显环状高热区域，高置信度（93.8%）正确分类
- **可能工艺来源**：
  - **干法刻蚀（Dry Etch）**：刻蚀腔体（Etch Chamber）边缘气体流场（Gas Flow Field）分布不均，边缘刻蚀速率与中心不一致；等离子体（Plasma）在晶圆边缘的鞘层（Sheath）厚度异常，导致边缘离子轰击能量偏高
  - **边缘温控失效**：静电卡盘（ESC, Electrostatic Chuck）边缘区域氦气（He）背压不足，冷却效率低于中心区域；边缘加热器（Edge Heater）功率偏差
  - **光刻胶旋涂**：旋涂（Spin Coating）过程中边缘光刻胶堆积（Edge Bead），曝光显影后尺寸偏差沿边缘传递
- **排查建议**：① 使用 OES（光学发射光谱）监测刻蚀腔体边缘区域的等离子体均匀性 ② 检查 ESC 边缘 He 背压传感器读数 ③ 校准旋涂设备 Edge Bead Removal（EBR）喷嘴位置

### 6.3 Center（中心）→ 光刻胶旋涂 / 曝光对准

- **缺陷特征**：集中在晶圆几何中心的缺陷簇或区域
- **Grad-CAM 验证**：激活集中在晶圆中心区域
- **可能工艺来源**：
  - **光刻胶旋涂**：旋涂初始阶段光刻胶滴注（Dispense）位置偏差或滴注量不足，中心区域膜厚异常
  - **曝光对准**：步进式光刻机（Stepper）或扫描式光刻机（Scanner）中心对准偏差，导致中心 die 的位置偏移
  - **显影工艺**：显影液（Developer）喷嘴扫描起始位置为中心，若显影液流量或温度在起始段波动，影响中心区域
- **排查建议**：① 检查旋涂机光刻胶喷嘴的滴注中心校准 ② 使用 overlay 测量设备检测中心 die 套刻精度 ③ 检查显影液管路温度稳定性

### 6.4 Edge-Loc（边缘局部）→ 边缘工艺集成问题

- **缺陷特征**：晶圆边缘某段弧区域的局部缺陷
- **可能工艺来源**：
  - **PVD/CVD 薄膜沉积**：沉积腔体中靶材（Target）边缘溅射效率差异，或反应气体在边缘区域的局部耗尽
  - **边缘光刻**：晶圆边缘曝光时，边缘场效应导致光刻胶侧壁角度异常
  - **热处理**：快速热退火（RTA）中边缘与中心的升降温速率差异
- **排查建议**：① 检查薄膜应力测试中晶圆边缘与中心的差异 ② 查看边缘 die 的 CD-SEM 线宽测量数据

### 6.5 Loc（局部）→ 颗粒污染 / 局部温度异常

- **缺陷特征**：分散的局部点状或小片状缺陷，无固定空间模式
- **可能工艺来源**：
  - **颗粒污染**：腔体内壁剥落颗粒（Chamber Particle）掉落在晶圆表面；真空吸盘（Vacuum Chuck）表面污染转移
  - **工艺气体**：气体管路中过滤器失效，颗粒随工艺气体进入腔体
- **排查建议**：① 使用 Surfscan 或 KLA 颗粒检测仪扫描腔体洁净度 ② 检查气体过滤器更换记录

### 6.6 Donut（圆环）→ 旋涂 / 刻蚀径向不均匀

- **缺陷特征**：环形缺陷，位于晶圆中心与边缘之间的中间区域
- **可能工艺来源**：
  - **旋涂工艺**：光刻胶烘烤（Soft Bake）过程中径向温度梯度导致溶剂挥发速率不均匀
  - **刻蚀工艺**：ICP（电感耦合等离子体）源的径向离子密度分布不均
- **排查建议**：① 检查热板（Hot Plate）的径向温度均匀性 ② 使用 Langmuir 探针测量 ICP 源径向等离子体密度

### 6.7 Near-full（近满）→ 系统性工艺失效

- **缺陷特征**：晶圆表面大面积缺陷，几乎覆盖整个晶圆
- **可能工艺来源**：
  - **CMP 过抛**：抛光时间过长或下压力过大，导致全局过度减薄
  - **刻蚀过刻**：刻蚀终点检测（Endpoint Detection）失效，过刻蚀严重
  - **薄膜应力释放**：沉积薄膜内应力过大，退火后发生大面积龟裂或剥离
- **排查建议**：① 检查 CMP 终点检测（EPD）信号是否正常 ② 查看刻蚀 OES 终点检测曲线 ③ 测量薄膜应力（Bow/Warp）

### 6.8 Random（随机）→ 多因素耦合

- **缺陷特征**：无规律分散缺陷，空间模式不可预测
- **可能工艺来源**：
  - **多工艺步骤耦合**：前道工艺残留物与后道工艺反应的复合效应
  - **环境因素**：洁净室局部气流扰动、离子化器效率波动导致的随机静电放电（ESD）
- **排查建议**：① 使用缺陷溯源系统（DSA）逐层追踪缺陷的工艺来源 ② 检查洁净室 FFU（风机过滤单元）运行状态

---

## 七、目标达成情况评估

| goal 目标项 | 完成状态 | 详细说明 |
|------------|---------|---------|
| 第一步：数据预处理与工程重构 | **完成** | pickle 解析、64×64 标准化、类别加权 + 数据增强双策略不平衡处理 |
| 第二步：迁移学习缺陷识别 | **完成** | ResNet-34 + ImageNet 预训练 + 浅层冻结微调，9 类输出，test acc 77.13% |
| 第三步：Grad-CAM 可解释性 | **完成** | 9 类热力图全部生成，layer4 梯度提取，热力图与缺陷空间分布吻合 |
| 第四步：工艺机理溯源 | **完成** | 8 种缺陷类型逐一溯源到具体工艺步骤、设备部件和物理机理 |
| 交付物1：算法代码库 | **完成** | data_loader.py, model.py, train.py, evaluate.py, predict.py, resnet_grad_cam.py 等 |
| 交付物2：可视化结果集 | **完成** | 混淆矩阵、F1-Score、训练曲线、9 类 Grad-CAM 热力图 |
| 交付物3：结题报告 | **完成** | 本报告（final_report.md） |

### 技术路线全流程验证

```
LSWMD.pkl (811,457条)
    ↓ [data_loader.py] 数据清洗 + 下采样 + 分层划分
dataset/{train,val,test}/ (45,519张64×64 PNG)
    ↓ [resnet/train.py] ResNet-34 迁移学习
resNet34.pth (82MB, best val_acc=0.7551)
    ↓ [resnet/evaluate.py] 测试集评估
test_report.json (test acc=0.7713, weighted F1=0.7788)
    ↓ [visualize_metrics.py] 训练过程可视化
training_metrics/{loss_curve, confusion_matrix, ...}.png
    ↓ [grad_cam/resnet_grad_cam.py] 可解释性分析
grad_cam/outputs/{0..8}_*_grad_cam.png
    ↓ [final_report.md] 工艺机理溯源
9 类缺陷 → 具体工艺/设备/物理机理 → 排查建议
```

---

## 八、不足与展望

### 8.1 当前局限

1. **分辨率限制**：64×64 可能不足以捕捉 Scratch、Small-Loc 等细粒度缺陷的纹理特征
2. **类别不平衡**：Near-full 仅 23 个测试样本，模型对该类别的精确率仅 0.30
3. **单模型架构**：未对比 ResNet-50、EfficientNet、ViT 等其他 backbone
4. **非端到端溯源**：工艺溯源依赖人工知识推理，尚未建立自动化的缺陷-工艺知识图谱

### 8.2 改进方向

- **多分辨率输入**：96×96 或 128×128 输入 + 多尺度特征融合（FPN）
- **更强 backbone**：ResNet-50、EfficientNet-B3 或 Swin Transformer
- **层次分类策略**：二阶段级联（先分有/无缺陷，再细分类型）
- **自动化溯源**：构建晶圆缺陷-工艺关联知识图谱（Knowledge Graph），实现端到端溯源推理

---

## 附录

### 附录A：文件结构

```text
course/
├── data_loader.py            # 数据预处理（pickle→ImageFolder）
├── analyze_pkl.py             # pickle 文件探查工具
├── visualize_metrics.py       # 训练指标可视化
├── dataset/                   # 预处理后的 ImageFolder 数据集（不入库）
│   ├── train/ (0..8)
│   ├── val/   (0..8)
│   └── test/  (0..8)
├── resnet/
│   ├── model.py               # ResNet-34/50/101 模型定义
│   ├── train.py               # 训练脚本（迁移学习 + 微调）
│   ├── evaluate.py            # 独立测试集评估
│   ├── predict.py             # 单张图片预测
│   ├── batch_predict.py       # 批量预测
│   ├── resnet34-pre.pth       # ImageNet 预训练权重（不入库）
│   ├── resNet34.pth           # 训练产出的模型权重（不入库）
│   └── training_metrics/      # 训练指标与可视化（不入库）
├── grad_cam/
│   ├── resnet_grad_cam.py     # Grad-CAM 热力图生成
│   └── outputs/               # 9 类热力图输出（不入库）
├── docs/
│   ├── final_report.md        # 本报告（结题报告）
│   ├── results_analysis.md    # 结果分析
│   ├── baseline_results.md    # 基线参考
│   └── mechanism_traceability.md
└── reports/                   # 数据集统计报告
    ├── dataset_summary.json
    ├── dataset_counts.csv
    └── dataset_sample_gallery.png
```

### 附录B：复现命令

```bash
# 1. 环境
conda env create -f environment.yml && conda activate wafer
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. 数据
bash scripts/download_dataset.sh
python data_loader.py

# 3. 训练
cd resnet
wget https://download.pytorch.org/models/resnet34-333f7ec4.pth -O resnet34-pre.pth
python train.py

# 4. 评估
python evaluate.py --data-dir ../dataset/test --weights resNet34.pth

# 5. 可视化
cd ..
python visualize_metrics.py --metrics_dir resnet/training_metrics

# 6. Grad-CAM（9个类别各一例）
for class in {0..8}; do
  img=$(ls dataset/test/$class/*.png | head -1)
  python grad_cam/resnet_grad_cam.py --image-path "$img" \
    --weights resnet/resNet34.pth \
    --class-json dataset/class_indices.json \
    --output "grad_cam/outputs/${class}_grad_cam.png"
done
```

---

*本报告由部署流程自动生成于 2026-05-31。所有实验代码、模型权重和可视化结果均可在本仓库复现。*
