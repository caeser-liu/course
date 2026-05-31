"""Create lightweight dataset summary artifacts for the generated ImageFolder data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


CLASS_NAMES = {
    "0": "Center",
    "1": "Donut",
    "2": "Edge-Loc",
    "3": "Edge-Ring",
    "4": "Loc",
    "5": "Random",
    "6": "Scratch",
    "7": "Near-full",
    "8": "none",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def count_images(dataset_dir: Path):
    summary = {}
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        summary[split] = {}
        for class_id, class_name in CLASS_NAMES.items():
            class_dir = split_dir / class_id
            count = 0
            if class_dir.exists():
                count = sum(
                    1
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
            summary[split][class_id] = {"class_name": class_name, "count": count}
    return summary


def write_csv(summary, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "class_id", "class_name", "count"])
        writer.writeheader()
        for split, classes in summary.items():
            for class_id, payload in classes.items():
                writer.writerow(
                    {
                        "split": split,
                        "class_id": class_id,
                        "class_name": payload["class_name"],
                        "count": payload["count"],
                    }
                )


def first_image(class_dir: Path):
    if not class_dir.exists():
        return None
    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    return None


def make_gallery(dataset_dir: Path, output_path: Path, split: str = "test"):
    cell_w, cell_h = 128, 154
    image_size = 96
    padding = 16
    gallery = Image.new("RGB", (cell_w * len(CLASS_NAMES), cell_h), "white")
    draw = ImageDraw.Draw(gallery)

    for column, (class_id, class_name) in enumerate(CLASS_NAMES.items()):
        x = column * cell_w
        image_path = first_image(dataset_dir / split / class_id)
        if image_path is not None:
            image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.Resampling.NEAREST)
            gallery.paste(image, (x + padding, padding))
            label = f"{class_id} {class_name}"
        else:
            label = f"{class_id} {class_name}\nmissing"
        draw.text((x + 8, padding + image_size + 8), label, fill="black")

    gallery.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize generated wafer ImageFolder dataset")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--gallery-split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = count_images(dataset_dir)

    totals = {
        split: sum(payload["count"] for payload in classes.values())
        for split, classes in summary.items()
    }
    payload = {
        "dataset_dir": str(dataset_dir),
        "class_names": CLASS_NAMES,
        "splits": summary,
        "totals": totals,
    }

    summary_path = output_dir / "dataset_summary.json"
    csv_path = output_dir / "dataset_counts.csv"
    gallery_path = output_dir / "dataset_sample_gallery.png"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)
    write_csv(summary, csv_path)
    make_gallery(dataset_dir, gallery_path, split=args.gallery_split)

    print(f"saved: {summary_path}")
    print(f"saved: {csv_path}")
    print(f"saved: {gallery_path}")
    print(f"totals: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
