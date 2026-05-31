"""Evaluate a trained wafer defect ResNet model on an ImageFolder split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
from tqdm import tqdm

from model import resnet34


IMG_SIZE = 64
NUM_CLASSES = 9


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained wafer defect model")
    parser.add_argument("--data-dir", default="../dataset/test", help="ImageFolder split path")
    parser.add_argument("--weights", default="resNet34.pth", help="Path to trained weights")
    parser.add_argument("--class-json", default="class_indices.json", help="Class mapping JSON")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-report", default="test_report.json")
    parser.add_argument("--output-confusion", default="confusion_matrix.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    weights_path = Path(args.weights)
    class_json = Path(args.class_json)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data split not found: {data_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}. Evaluation requires a trained model."
        )
    if not class_json.exists():
        raise FileNotFoundError(f"Class mapping not found: {class_json}")

    with class_json.open("r", encoding="utf-8") as file:
        class_indices = json.load(file)
    target_names = [class_indices[str(i)] for i in range(NUM_CLASSES)]

    dataset = datasets.ImageFolder(root=str(data_dir), transform=build_transform())
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet34(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc="Evaluating"):
            logits = model(images.to(device))
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_predictions.tolist())
            labels.extend(batch_labels.numpy().tolist())

    report = classification_report(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=target_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES)))

    with Path(args.output_report).open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)
    with Path(args.output_confusion).open("w", encoding="utf-8") as file:
        json.dump(matrix.tolist(), file, indent=4)

    print(classification_report(labels, predictions, target_names=target_names, digits=4, zero_division=0))
    print(f"accuracy: {report['accuracy']:.4f}")
    print(f"saved report: {args.output_report}")
    print(f"saved confusion matrix: {args.output_confusion}")
    print(f"confusion matrix:\n{np.array(matrix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
