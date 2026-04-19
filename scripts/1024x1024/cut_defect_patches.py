#!/usr/bin/env python3
"""
extract_defective_patches.py - Вырезание патчей с дефектами
Патч: 256×1600 (полная высота) → ресайз до 1024×1024
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import json

def rle_to_mask(rle_string, height=256, width=1600):
    """Декодирование RLE для Severstal (column-major order)"""
    if pd.isna(rle_string) or rle_string == '':
        return np.zeros((height, width), dtype=np.uint8)
    
    numbers = list(map(int, rle_string.split()))
    starts = np.array(numbers[0::2]) - 1
    lengths = np.array(numbers[1::2])
    
    flat_mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat_mask[start:start + length] = 1
    
    return flat_mask.reshape(width, height).T


def has_black_background(img, black_threshold=30, max_black_ratio=0.05):
    """Проверяет, есть ли в патче черный фон"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    black_pixels = np.sum(gray < black_threshold)
    total_pixels = img.shape[0] * img.shape[1]
    black_ratio = black_pixels / total_pixels
    return black_ratio > max_black_ratio


def resize_to_square(img, target_size=1024):
    """Ресайз до target_size × target_size с сохранением пропорций"""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    if len(img.shape) == 3:
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def extract_defective_patches(
    train_images_dir="data/severstal/train_images",
    train_csv_path="data/severstal/train.csv",
    output_dir="data/severstal/defective_patches",
    patch_width=256,
    stride=128,
    min_defect_area=100,
    resize_to=1024,
    reject_black=True,
    black_threshold=30,
    max_black_ratio=0.05
):
    """Вырезает патчи с дефектами"""
    
    output_path = Path(output_dir)
    
    # 🔧 СОЗДАЕМ ОТДЕЛЬНЫЕ ПАПКИ ДЛЯ ИЗОБРАЖЕНИЙ И МАСОК
    images_path = output_path / "images"
    masks_path = output_path / "masks"
    images_path.mkdir(parents=True, exist_ok=True)
    masks_path.mkdir(parents=True, exist_ok=True)
    
    # Загружаем разметку
    print("📂 Загрузка разметки...")
    df = pd.read_csv(train_csv_path)
    
    # Создаем маски
    print("🎭 Создание масок...")
    masks = {}
    for img_id, group in tqdm(df.groupby('ImageId')):
        combined = np.zeros((256, 1600), dtype=np.uint8)
        for _, row in group.iterrows():
            combined = np.maximum(combined, rle_to_mask(row['EncodedPixels']))
        masks[img_id] = combined
    
    image_files = list(Path(train_images_dir).glob("*.jpg"))
    print(f"📁 Всего изображений: {len(image_files)}")
    
    if reject_black:
        print(f"🚫 Отбраковка патчей с черным фоном (> {max_black_ratio*100:.0f}% черного)")
    
    defective_patches = []
    total_defects_found = 0
    rejected_black = 0
    
    for img_path in tqdm(image_files, desc="🔍 Поиск дефектов"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = masks.get(img_path.name, np.zeros((256, 1600), dtype=np.uint8))
        
        for x in range(0, 1600 - patch_width + 1, stride):
            patch_img = img_rgb[:, x:x+patch_width]
            patch_mask = mask[:, x:x+patch_width]
            
            if reject_black and has_black_background(patch_img, black_threshold, max_black_ratio):
                rejected_black += 1
                continue
            
            defect_pixels = np.sum(patch_mask)
            
            if defect_pixels >= min_defect_area:
                patch_resized = resize_to_square(patch_img, resize_to)
                mask_resized = resize_to_square(patch_mask, resize_to)
                mask_resized = (mask_resized > 0).astype(np.uint8) * 255
                
                name = f"{img_path.stem}_x{x}_w{patch_width}"
                
                # 🔧 СОХРАНЯЕМ В РАЗНЫЕ ПАПКИ
                img_save_path = images_path / f"{name}.png"
                mask_save_path = masks_path / f"{name}.png"
                
                cv2.imwrite(str(img_save_path), cv2.cvtColor(patch_resized, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(mask_save_path), mask_resized)
                
                defective_patches.append({
                    'image': img_path.name,
                    'x': x,
                    'width': patch_width,
                    'defect_area': int(defect_pixels),
                    'saved_as': name,
                    'image_path': str(img_save_path),
                    'mask_path': str(mask_save_path)
                })
                
                total_defects_found += 1
    
    # Статистика
    print(f"\n{'='*50}")
    print(f"📊 СТАТИСТИКА:")
    print(f"  Найдено патчей с дефектами: {len(defective_patches)}")
    if reject_black:
        print(f"  Отбраковано из-за черного фона: {rejected_black}")
    print(f"  Размер сохраненных патчей: {resize_to}×{resize_to}")
    print(f"  Изображения сохранены в: {images_path}")
    print(f"  Маски сохранены в: {masks_path}")
    
    # Сохраняем метаданные
    if defective_patches:
        pd.DataFrame(defective_patches).to_csv(output_path / "defective_patches.csv", index=False)
        with open(output_path / "annotations.json", "w") as f:
            json.dump(defective_patches, f, indent=2)
    
    return defective_patches


def visualize_defective_patches(patch_dir="data/severstal/defective_patches", num_examples=5):
    """Визуализация патчей с дефектами"""
    import matplotlib.pyplot as plt
    
    images_path = Path(patch_dir) / "images"
    masks_path = Path(patch_dir) / "masks"
    
    patch_files = list(images_path.glob("*.png"))
    
    if not patch_files:
        print("Нет патчей для визуализации")
        return
    
    import random
    random.seed(42)
    selected = random.sample(patch_files, min(num_examples, len(patch_files)))
    
    fig, axes = plt.subplots(2, len(selected), figsize=(3*len(selected), 6))
    if len(selected) == 1:
        axes = axes.reshape(2, 1)
    
    for i, patch_file in enumerate(selected):
        img = cv2.imread(str(patch_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Defect {i+1}", fontsize=10)
        axes[0, i].axis('off')
        
        mask_file = masks_path / patch_file.name
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].set_title(f"Mask {i+1}", fontsize=10)
            axes[1, i].axis('off')
    
    plt.suptitle(f"Патчи с дефектами (1024×1024)", fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(patch_dir) / 'defective_patches_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Примеры сохранены в: {patch_dir}/defective_patches_example.png")


if __name__ == "__main__":
    patches = extract_defective_patches(
        train_images_dir="data/severstal/train_images",
        train_csv_path="data/severstal/train.csv",
        output_dir="data/severstal/defective_patches",
        patch_width=256,
        stride=256,
        min_defect_area=100,
        resize_to=1024,
        reject_black=True,
        black_threshold=30,
        max_black_ratio=0.05
    )
    
    if patches:
        visualize_defective_patches("data/severstal/defective_patches")
        print(f"\n✅ Готово!")
        print(f"✅ Найдено {len(patches)} патчей с дефектами")
        print(f"✅ Сохранены в: data/severstal/defective_patches/")
    else:
        print("❌ Не найдено патчей с дефектами!")