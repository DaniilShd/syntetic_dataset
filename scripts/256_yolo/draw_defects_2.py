#!/usr/bin/env python3
"""
visualize_bbox.py - Отрисовка YOLO bbox на изображениях
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import random
from tqdm import tqdm


def draw_yolo_bbox(image_path: Path, label_path: Path, output_path: Path = None):
    """Отрисовка YOLO bbox на изображении"""
    
    # Загрузка изображения
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Не удалось загрузить {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    # Загрузка разметки
    if not label_path.exists():
        print(f"⚠️ Нет разметки для {image_path.name}")
        return img
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            width = float(parts[3]) * w
            height = float(parts[4]) * h
            
            # Конвертация в угловые координаты
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Цвет по классу
            color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)][class_id % 4]
            
            # Отрисовка
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"Class {class_id}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Сохранение
    if output_path:
        cv2.imwrite(str(output_path), img)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Визуализация YOLO bbox")
    parser.add_argument("--images", type=str, default="data/dataset_synthetic/defect_patches/images",
                       help="Путь к изображениям")
    parser.add_argument("--labels", type=str, default="data/dataset_synthetic/defect_patches/labels",
                       help="Путь к разметке")
    parser.add_argument("--output", type=str, default="data/dataset_synthetic/defect_patches/visualized",
                       help="Выходная директория")
    parser.add_argument("--samples", type=int, default=30,
                       help="Количество случайных примеров (0 = все)")
    parser.add_argument("--show", action="store_true",
                       help="Показать изображения")
    
    args = parser.parse_args()
    
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Поиск изображений
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    
    if args.samples > 0 and args.samples < len(image_files):
        image_files = random.sample(image_files, args.samples)
    
    print(f"📂 Найдено {len(image_files)} изображений")
    
    for img_path in tqdm(image_files, desc="Отрисовка"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        output_path = output_dir / f"vis_{img_path.name}"
        
        img = draw_yolo_bbox(img_path, label_path, output_path)
        
        if args.show and img is not None:
            cv2.imshow("BBox", img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    
    if args.show:
        cv2.destroyAllWindows()
    
    print(f"✅ Готово! Результаты в {output_dir}")


if __name__ == "__main__":
    main()