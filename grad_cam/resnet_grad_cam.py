"""Grad-CAM for the project ResNet-34 wafer defect classifier.

This script requires a trained weight file. It does not train the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
RESNET_DIR = ROOT / "resnet"
if str(RESNET_DIR) not in sys.path:
    sys.path.insert(0, str(RESNET_DIR))

from model import resnet34  # noqa: E402


IMG_SIZE = 64
NUM_CLASSES = 9


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, target_category: int | None = None):
        logits = self.model(input_tensor)
        if target_category is None:
            target_category = int(torch.argmax(logits, dim=1).item())

        self.model.zero_grad()
        logits[:, target_category].sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam_map = (weights * self.activations).sum(dim=1, keepdim=True)
        cam_map = torch.relu(cam_map)
        cam_map = torch.nn.functional.interpolate(
            cam_map,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam_map = cam_map.squeeze().cpu().numpy()
        cam_map = cam_map - cam_map.min()
        cam_map = cam_map / (cam_map.max() + 1e-8)
        probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()[0]
        return cam_map, target_category, probabilities

    def close(self):
        for handle in self.handles:
            handle.remove()


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def overlay_cam(image: Image.Image, cam_map: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
    base_np = np.asarray(base).astype(np.float32) / 255.0
    heatmap = cm.get_cmap("jet")(cam_map)[..., :3].astype(np.float32)
    overlay = (1.0 - alpha) * base_np + alpha * heatmap
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap for a wafer image")
    parser.add_argument("--image-path", required=True, help="Input wafer image")
    parser.add_argument("--weights", default="../resnet/resNet34.pth", help="Trained ResNet weight path")
    parser.add_argument("--class-json", default="../resnet/class_indices.json", help="Class mapping JSON")
    parser.add_argument("--target-class", type=int, default=None, help="Class id to explain; default is prediction")
    parser.add_argument("--output", default="outputs/grad_cam.png", help="Output image path")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    weights_path = Path(args.weights)
    class_json = Path(args.class_json)
    output_path = Path(args.output)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}. Grad-CAM requires a trained resNet34.pth."
        )
    if not class_json.exists():
        raise FileNotFoundError(f"Class mapping not found: {class_json}")

    with class_json.open("r", encoding="utf-8") as file:
        class_indices = json.load(file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet34(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    input_tensor = build_transform()(image).unsqueeze(0).to(device)

    grad_cam = GradCAM(model, model.layer4)
    try:
        cam_map, explained_class, probabilities = grad_cam(input_tensor, args.target_class)
    finally:
        grad_cam.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_cam(image, cam_map)
    overlay.save(output_path)

    class_name = class_indices[str(explained_class)]
    print(f"explained class: {class_name} ({explained_class})")
    print(f"probability: {probabilities[explained_class]:.4f}")
    print(f"saved Grad-CAM: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
