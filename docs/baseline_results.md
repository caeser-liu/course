# 基线结果记录

本页记录当前工作区已有训练结果的核心指标。原始训练产物 `resnet/test_report.json` 和 `resnet/training_metrics/metrics.json` 属于生成文件，默认不提交到 Git。

## 数据划分

当前本地 `dataset/` 中的样本数量如下：

| Split | Total |
| --- | ---: |
| train | 31,861 |
| val | 6,824 |
| test | 6,834 |

类别映射：

| ID | Class |
| ---: | --- |
| 0 | Center |
| 1 | Donut |
| 2 | Edge-Loc |
| 3 | Edge-Ring |
| 4 | Loc |
| 5 | Random |
| 6 | Scratch |
| 7 | Near-full |
| 8 | none |

## 测试集指标

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7430 |
| Macro precision | 0.7184 |
| Macro recall | 0.7556 |
| Macro F1 | 0.7160 |
| Weighted precision | 0.8211 |
| Weighted recall | 0.7430 |
| Weighted F1 | 0.7564 |

## 各类别 F1

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Center | 0.8732 | 0.7364 | 0.7990 |
| Donut | 0.7865 | 0.8333 | 0.8092 |
| Edge-Loc | 0.3655 | 0.7972 | 0.5012 |
| Edge-Ring | 0.8543 | 0.9855 | 0.9153 |
| Loc | 0.7041 | 0.3833 | 0.4964 |
| Random | 0.6543 | 0.8092 | 0.7235 |
| Scratch | 0.5172 | 0.6667 | 0.5825 |
| Near-full | 0.7500 | 0.9130 | 0.8235 |
| none | 0.9602 | 0.6757 | 0.7932 |

## 说明

- 本结果用于课程版基线说明，不代表重新训练后的必然结果。
- 完整复现需重新下载 `LSWMD.pkl`、执行 `data_loader.py`，再运行 `resnet/train.py`。
- 当前本机不强制完成训练；建议在 CUDA GPU 环境中重训并更新本页。
