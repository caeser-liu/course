import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report

from model import resnet34

IMG_SIZE = 64  # 与 data_loader.py 中的设置一致

# 导入指标记录器（可视化用）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualize_metrics import MetricsRecorder


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # ====== 数据路径 ======
    # dataset 目录在 course/ 下，train.py 在 course/resnet/ 下
    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    image_path = os.path.join(data_root, "dataset")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # ====== 数据增强（针对晶圆图特点）=====
    # 训练集：随机旋转90°、水平/垂直翻转、随机裁剪
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(IMG_SIZE + 16),  # 先放大再裁剪
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomRotation(90),      # 晶圆图旋转90°有意义
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # 我们的图像是灰度图复制3通道，像素值 0/128/255
            # 均值和标准差按实际数据估算
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "train"),
        transform=data_transform["train"])
    train_num = len(train_dataset)

    # 类别映射：{'0': 0, '1': 1, ..., '8': 8}
    class_indices = train_dataset.class_to_idx
    print(f"\n类别映射: {class_indices}")

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers'.format(nw))

    # ====== 类别权重（处理不平衡）=====
    # 统计每个类别的样本数
    labels = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(labels, minlength=len(class_indices))
    print(f"\n各类别训练样本数: {class_counts}")

    # 计算权重：权重与样本数成反比
    class_weights = 1.0 / class_counts.astype(float)
    class_weights = class_weights / class_weights.sum() * len(class_indices)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"类别权重: {class_weights}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    validate_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "val"),
        transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    print("\nusing {} images for training, {} images for validation.".format(train_num, val_num))

    # ====== 模型 =====
    NUM_CLASSES = 9
    net = resnet34(num_classes=NUM_CLASSES)

    # 加载预训练权重
    model_weight_path = "./resnet34-pre.pth"
    if os.path.exists(model_weight_path):
        print(f"加载预训练权重: {model_weight_path}")
        # 预训练权重是1000类的，需要排除fc层
        pretrained_dict = torch.load(model_weight_path, map_location='cpu', weights_only=False)
        model_dict = net.state_dict()
        # 过滤掉不匹配的层（fc层）
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        print(f"成功加载 {len(pretrained_dict)}/{len(model_dict)} 层权重")
    else:
        print(f"警告: 未找到预训练权重 {model_weight_path}，将从头训练")

    # 修改 fc 层输出为 9 类
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, NUM_CLASSES)
    net.to(device)

    # ====== 冻结浅层（可选）=====
    # 冻结 layer1 和 layer2，只训练 layer3、layer4 和 fc
    # 如果想全量微调，注释掉下面这段
    FREEZE_EARLY_LAYERS = True
    if FREEZE_EARLY_LAYERS:
        print("\n冻结浅层 (conv1, bn1, layer1, layer2)...")
        for name, param in net.named_parameters():
            if any(name.startswith(prefix) for prefix in ['conv1', 'bn1', 'layer1', 'layer2']):
                param.requires_grad = False

    # ====== 损失函数 + 优化器 =====
    loss_function = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # 只优化需要梯度的参数
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    # 创建指标记录器
    metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    recorder = MetricsRecorder(metrics_dir)

    # ====== 训练循环 =====
    epochs = 30
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        # validate
        net.eval()
        acc = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(val_labels.numpy())

        val_accurate = acc / val_num
        print(f'[epoch {epoch+1}] train_loss: {running_loss / train_steps:.4f}  val_accuracy: {val_accurate:.4f}')

        # 计算每类准确率
        per_class_correct = np.zeros(NUM_CLASSES)
        per_class_total = np.zeros(NUM_CLASSES)
        for pred, true in zip(all_preds, all_labels):
            per_class_correct[true] += (pred == true)
            per_class_total[true] += 1
        per_class_acc = np.where(per_class_total > 0, per_class_correct / per_class_total, 0.0)

        # 记录指标
        recorder.record_epoch(
            epoch=epoch + 1,
            train_loss=running_loss / train_steps,
            val_acc=val_accurate,
            per_class_acc=per_class_acc,
            conf_matrix=confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
        )

        # 每 5 个 epoch 打印详细评估
        if (epoch + 1) % 5 == 0:
            print("\n详细评估:")
            print(classification_report(all_labels, all_preds,
                  target_names=[f"class_{i}" for i in range(NUM_CLASSES)], digits=4))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print(f"  → 保存最佳模型 (val_acc={val_accurate:.4f})")

    print(f'\nFinished Training. Best val accuracy: {best_acc:.4f}')

    # 保存训练指标
    recorder.save()
    print(f"训练指标已保存到 {metrics_dir}/metrics.json")

    # ====== 最终测试集评估 =====
    print("\n" + "=" * 60)
    print("最终测试集评估")
    print("=" * 60)
    test_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "test"),
        transform=data_transform["val"])
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    # 加载最佳模型
    net.load_state_dict(torch.load(save_path, map_location=device))
    net.eval()

    test_correct = 0
    test_preds, test_true = [], []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", file=sys.stdout):
            images, labels = data
            outputs = net(images.to(device))
            preds = torch.max(outputs, dim=1)[1]
            test_correct += torch.eq(preds, labels.to(device)).sum().item()
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(labels.numpy())

    test_acc = test_correct / test_num
    print(f"\n测试集准确率: {test_acc:.4f}")
    print("\n分类报告:")
    print(classification_report(test_true, test_preds,
          target_names=[f"class_{i}" for i in range(NUM_CLASSES)], digits=4))

    # 混淆矩阵
    cm = confusion_matrix(test_true, test_preds)
    print(f"\n混淆矩阵:\n{cm}")

    # 保存测试报告
    report = classification_report(test_true, test_preds,
                   target_names=[f"class_{i}" for i in range(NUM_CLASSES)], digits=4,
                   output_dict=True)
    with open('test_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    print("\n测试报告已保存到 test_report.json")


if __name__ == '__main__':
    main()
