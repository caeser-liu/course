"""Batch inference for wafer defect images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import resnet34


IMG_SIZE = 64
NUM_CLASSES = 9
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def iter_images(path: Path):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        yield path
        return
    for image_path in sorted(path.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a directory of wafer-map images")
    parser.add_argument("--data-path", required=True, help="Image file or directory")
    parser.add_argument("--weights", default="resNet34.pth", help="Path to trained weights")
    parser.add_argument("--class-json", default="class_indices.json", help="Class mapping JSON")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-csv", default="batch_predictions.csv")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    weights_path = Path(args.weights)
    class_json = Path(args.class_json)

    if not data_path.exists():
        raise FileNotFoundError(f"Input path not found: {data_path}")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}. Train first with resnet/train.py "
            "or copy a trained resNet34.pth into this directory."
        )
    if not class_json.exists():
        raise FileNotFoundError(f"Class mapping not found: {class_json}")

    with class_json.open("r", encoding="utf-8") as file:
        class_indices = json.load(file)

    image_paths = list(iter_images(data_path))
    if not image_paths:
        raise RuntimeError(f"No images found under {data_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet34(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    transform = build_transform()

    rows = []
    with torch.no_grad():
        for start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[start : start + args.batch_size]
            batch = []
            for image_path in batch_paths:
                image = Image.open(image_path).convert("RGB")
                batch.append(transform(image))
            batch_tensor = torch.stack(batch, dim=0).to(device)
            probabilities = torch.softmax(model(batch_tensor), dim=1).cpu()
            values, indices = torch.max(probabilities, dim=1)

            for image_path, prob, idx in zip(batch_paths, values, indices):
                class_id = str(int(idx))
                row = {
                    "image": str(image_path),
                    "class_id": class_id,
                    "class_name": class_indices[class_id],
                    "probability": f"{float(prob):.6f}",
                }
                rows.append(row)
                print(f"{row['image']} -> {row['class_name']} ({row['probability']})")

    output_csv = Path(args.output_csv)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "class_id", "class_name", "probability"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
