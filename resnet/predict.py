"""Single-image inference for the wafer defect ResNet model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

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


def load_class_indices(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict one wafer-map image")
    parser.add_argument("--image-path", required=True, help="Path to a PNG/JPG wafer image")
    parser.add_argument("--weights", default="resNet34.pth", help="Path to trained weights")
    parser.add_argument("--class-json", default="class_indices.json", help="Class mapping JSON")
    parser.add_argument("--topk", type=int, default=3, help="Number of classes to print")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    weights_path = Path(args.weights)
    class_json = Path(args.class_json)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}. Train first with resnet/train.py "
            "or copy a trained resNet34.pth into this directory."
        )
    if not class_json.exists():
        raise FileNotFoundError(f"Class mapping not found: {class_json}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_indices = load_class_indices(class_json)

    model = resnet34(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = build_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor).cpu().squeeze(0)
        probabilities = torch.softmax(logits, dim=0)

    topk = min(args.topk, NUM_CLASSES)
    values, indices = torch.topk(probabilities, k=topk)

    print(f"image: {image_path}")
    print(f"device: {device}")
    for rank, (prob, idx) in enumerate(zip(values, indices), start=1):
        class_id = str(int(idx))
        print(f"top{rank}: {class_indices[class_id]} ({class_id})  prob={float(prob):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
