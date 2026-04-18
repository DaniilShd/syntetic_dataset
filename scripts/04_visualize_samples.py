"""
04_visualize_samples.py - Визуализация примеров из датасета
"""

import cv2
import numpy as np
import random
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from utils import load_json


def visualize_comparison(real_path: str, synthetic_path: str, output_path: str):
    """Сравнение реального и синтетического изображения"""
    
    real_img = cv2.imread(real_path)
    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
    
    syn_img = cv2.imread(synthetic_path)
    syn_img = cv2.cvtColor(syn_img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(real_img)
    axes[0].set_title("Реальное изображение")
    axes[0].axis("off")
    
    axes[1].imshow(syn_img)
    axes[1].set_title("Синтетическое изображение")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Визуализация сохранена: {output_path}")


def visualize_with_defects(dataset_dir: str, num_samples: int, output_dir: str):
    """Визуализация изображений с дефектами"""
    
    images_dir = Path(dataset_dir) / "images"
    masks_dir = Path(dataset_dir) / "masks"
    annotations_path = Path(dataset_dir) / "annotations.json"
    
    if not annotations_path.exists():
        print("❌ Файл аннотаций не найден")
        return
    
    annotations = load_json(annotations_path)
    samples = random.sample(annotations, min(num_samples, len(annotations)))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, ann in enumerate(samples):
        img_path = images_dir / ann["image"]
        mask_path = masks_dir / ann["image"]
        
        if not img_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        
        fig, axes = plt.subplots(1, 2 if mask is not None else 1, figsize=(10, 5))
        
        if mask is None:
            axes = [axes]
        
        axes[0].imshow(img)
        
        # Отрисовка bbox
        for defect in ann["defects"]:
            x, y, w, h = defect["bbox"]
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2)
            axes[0].add_patch(rect)
        
        axes[0].set_title(f"Изображение с дефектами ({len(ann['defects'])})")
        axes[0].axis("off")
        
        if mask is not None:
            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title("Маска")
            axes[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{i:03d}.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"✅ Сохранено {len(samples)} визуализаций в {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Визуализация датасета")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Сравнение
    comp_parser = subparsers.add_parser("compare", help="Сравнение реального и синтетического")
    comp_parser.add_argument("--real", type=str, required=True)
    comp_parser.add_argument("--syn", type=str, required=True)
    comp_parser.add_argument("--output", type=str, default="comparison.png")
    
    # Дефекты
    def_parser = subparsers.add_parser("defects", help="Визуализация с дефектами")
    def_parser.add_argument("--dataset_dir", type=str, required=True)
    def_parser.add_argument("--num_samples", type=int, default=10)
    def_parser.add_argument("--output_dir", type=str, default="./output/viz_defects")
    
    args = parser.parse_args()
    
    if args.command == "compare":
        visualize_comparison(args.real, args.syn, args.output)
    elif args.command == "defects":
        visualize_with_defects(args.dataset_dir, args.num_samples, args.output_dir)


if __name__ == "__main__":
    main()