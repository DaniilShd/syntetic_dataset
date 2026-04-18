#!/usr/bin/env python3
"""
extract_defects.py - Извлечение дефектов из датасета Severstal
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import json

def rle_to_mask(rle_string, height=256, width=1600):
    """Декодирование RLE в маску"""
    if pd.isna(rle_string) or rle_string == '' or rle_string is None:
        return np.zeros((height, width), dtype=np.uint8)
    
    numbers = list(map(int, str(rle_string).split()))
    starts = np.array(numbers[0::2]) - 1
    lengths = np.array(numbers[1::2])
    
    flat_mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat_mask[start:start + length] = 1
    
    # 🔧 ВАЖНО: Severstal использует column-major order
    mask = flat_mask.reshape(width, height).T
    
    # 🔧 Отладка
    # print(f"RLE decode: sum={np.sum(mask)}, shape={mask.shape}")
    
    return mask


def extract_defect_patches(
    image_path: str,
    mask: np.ndarray,
    output_dir: Path,
    defect_id: int,
    min_area: int = 100,
    padding: int = 5,
    max_width: int = 1024,
    max_height: int = 1024
):
    """Извлечение дефектов с обрезкой вместо сжатия"""
    
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = mask.shape
    
    # 🔧 ВАЖНО: маска должна быть uint8
    mask = mask.astype(np.uint8)
    
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    saved_defects = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # Вырезаем с паддингом
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        # Обрезаем если больше лимита
        if crop_w > max_width:
            excess = crop_w - max_width
            x1 += excess // 2
            x2 = x1 + max_width
            crop_w = max_width
        
        if crop_h > max_height:
            excess = crop_h - max_height
            y1 += excess // 2
            y2 = y1 + max_height
            crop_h = max_height
        
        defect_img = image[y1:y2, x1:x2]
        defect_mask = mask[y1:y2, x1:x2].copy()  # 🔧 .copy() важно!
        
        # 🔧 Бинаризуем маску (0 или 255)
        defect_mask = np.where(defect_mask > 0, 255, 0).astype(np.uint8)
        
        # 🔧 ПРОВЕРКА: не пустая ли маска
        if np.sum(defect_mask) == 0:
            print(f"⚠️ Пустая маска для дефекта {defect_id}_{i}")
            continue
        
        # Сохраняем
        defect_name = f"defect_{defect_id:06d}_{i:03d}"
        img_path = output_dir / f"{defect_name}.png"
        mask_path = output_dir / f"{defect_name}_mask.png"
        
        cv2.imwrite(str(img_path), cv2.cvtColor(defect_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), defect_mask)
        
        # 🔧 Отладка
        # print(f"✅ {defect_name}: размер={defect_img.shape}, пикселей в маске={np.sum(defect_mask > 0)}")
        
        saved_defects.append({
            "name": defect_name,
            "image": str(img_path),
            "mask": str(mask_path),
            "width": defect_img.shape[1],
            "height": defect_img.shape[0],
            "area": int(area)
        })
    
    return saved_defects


def process_severstal_dataset(
    train_csv: str = "/app/data/severstal/train.csv",
    images_dir: str = "/app/data/severstal/train_images",
    output_dir: str = "/app/data/defects",
    min_area: int = 100,
    max_width: int = 1024,
    max_height: int = 1024
):
    """Обработка всего датасета Severstal"""
    
    df = pd.read_csv(train_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Группируем по изображениям
    image_groups = df.groupby('ImageId')
    
    total_defects = 0
    all_metadata = []
    
    for image_id, group in tqdm(image_groups, desc="Обработка изображений"):
        image_path = Path(images_dir) / image_id
        if not image_path.exists():
            print(f"⚠️ Пропуск {image_id}: файл не найден")
            continue
        
        # Объединяем маски всех классов
        combined_mask = np.zeros((256, 1600), dtype=np.uint8)
        has_defect = False
        
        for _, row in group.iterrows():
            if pd.notna(row['EncodedPixels']):
                mask = rle_to_mask(row['EncodedPixels'])
                combined_mask = np.maximum(combined_mask, mask)
                has_defect = True
        
        if not has_defect:
            continue
        
        # Извлекаем отдельные дефекты
        defects = extract_defect_patches(
            image_path=image_path,
            mask=combined_mask,
            output_dir=output_path,
            defect_id=total_defects,
            min_area=min_area,
            max_width=max_width,
            max_height=max_height
        )
        
        for d in defects:
            d["source_image"] = image_id
            all_metadata.append(d)
        
        total_defects += len(defects)
    
    # Сохраняем метаданные
    with open(output_path / "defects_metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n✅ Извлечено {total_defects} дефектов")
    print(f"📁 Сохранено в {output_dir}")
    
    # Статистика по размерам
    widths = [d["width"] for d in all_metadata]
    heights = [d["height"] for d in all_metadata]
    print(f"\n📊 Статистика размеров:")
    print(f"   Ширина: min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.0f}")
    print(f"   Высота: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.0f}")
    
    return all_metadata


def main():
    parser = argparse.ArgumentParser(description="Извлечение дефектов из датасета Severstal")
    parser.add_argument("--train_csv", type=str, default="/app/data/severstal/train.csv")
    parser.add_argument("--images_dir", type=str, default="/app/data/severstal/train_images")
    parser.add_argument("--output_dir", type=str, default="/app/data/defects")
    parser.add_argument("--min_area", type=int, default=100)
    parser.add_argument("--max_width", type=int, default=1024)
    parser.add_argument("--max_height", type=int, default=1024)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 ИЗВЛЕЧЕНИЕ ДЕФЕКТОВ SEVERSTAL")
    print("=" * 60)
    print(f"Train CSV: {args.train_csv}")
    print(f"Images: {args.images_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Max size: {args.max_width}×{args.max_height}")
    
    process_severstal_dataset(
        train_csv=args.train_csv,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        min_area=args.min_area,
        max_width=args.max_width,
        max_height=args.max_height
    )


if __name__ == "__main__":
    main()