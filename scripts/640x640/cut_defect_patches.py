#!/usr/bin/env python3
"""
extract_defective_patches_yolo.py - Вырезание патчей с дефектами в YOLO формате
Патч: 256×1600 (полная высота) → ресайз до указанного размера
Выход: images/ + labels/ в формате YOLO для LTDETR
🔧 ИСПРАВЛЕНО: поддержка мультиклассовой разметки (ClassId 1,2,3,4)
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
import yaml

# 🆕 Маппинг ClassId → имена дефектов
CLASS_NAMES = {
    1: 'defect_type_1',
    2: 'defect_type_2',
    3: 'defect_type_3',
    4: 'defect_type_4'
}

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


def rle_to_mask_with_class(rle_string, class_id, height=256, width=1600):
    """
    🆕 Декодирование RLE с сохранением класса дефекта
    Возвращает маску, где значение = class_id (1,2,3,4)
    """
    if pd.isna(rle_string) or rle_string == '':
        return np.zeros((height, width), dtype=np.uint8)
    
    numbers = list(map(int, rle_string.split()))
    starts = np.array(numbers[0::2]) - 1
    lengths = np.array(numbers[1::2])
    
    flat_mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat_mask[start:start + length] = class_id  # 🆕 Присваиваем class_id вместо 1
    
    return flat_mask.reshape(width, height).T


def mask_to_yolo_boxes_multiclass(mask, min_area=10):
    """
    🆕 Конвертирует многоклассовую маску в список bbox в формате YOLO
    
    Args:
        mask: маска с class_id (0=фон, 1,2,3,4=дефекты)
        min_area: минимальная площадь объекта в пикселях
    
    Returns:
        list of [class_id, x_center, y_center, width, height]
    """
    h, w = mask.shape
    yolo_boxes = []
    
    # 🆕 Обрабатываем каждый класс отдельно
    for class_id in [1, 2, 3, 4]:
        # Создаем бинарную маску для текущего класса
        class_mask = (mask == class_id).astype(np.uint8)
        
        if np.sum(class_mask) == 0:
            continue
        
        # Находим контуры
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Получаем bounding box в пикселях
            x, y, box_w, box_h = cv2.boundingRect(contour)
            
            # Конвертируем в нормализованные YOLO координаты
            x_center = (x + box_w / 2) / w
            y_center = (y + box_h / 2) / h
            width = box_w / w
            height = box_h / h
            
            # Проверяем, что координаты в диапазоне [0, 1]
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            width = np.clip(width, 1e-6, 1)
            height = np.clip(height, 1e-6, 1)
            
            # 🆕 Сохраняем оригинальный class_id (1-4)
            yolo_boxes.append([class_id, x_center, y_center, width, height])
    
    return yolo_boxes


def has_black_background(img, black_threshold=30, max_black_ratio=0.05):
    """Проверяет, есть ли в патче черный фон"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    black_pixels = np.sum(gray < black_threshold)
    total_pixels = img.shape[0] * img.shape[1]
    black_ratio = black_pixels / total_pixels
    return black_ratio > max_black_ratio


def resize_with_bbox(img, boxes, target_size=1024):
    """
    Ресайз изображения и адаптация bbox в YOLO формате
    
    Args:
        img: исходное изображение (H, W, 3)
        boxes: список bbox [class_id, x_center, y_center, width, height]
        target_size: целевой размер квадрата
    
    Returns:
        resized_img, resized_boxes
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Ресайз изображения
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Создаем квадратный канвас
    if len(img.shape) == 3:
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Адаптируем bbox
    resized_boxes = []
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        
        # Конвертируем обратно в пиксельные координаты
        x_center_px = x_center * w
        y_center_px = y_center * h
        width_px = width * w
        height_px = height * h
        
        # Применяем ресайз
        x_center_px *= scale
        y_center_px *= scale
        width_px *= scale
        height_px *= scale
        
        # Добавляем оффсеты от паддинга
        x_center_px += x_offset
        y_center_px += y_offset
        
        # Конвертируем обратно в нормализованные координаты
        x_center_norm = x_center_px / target_size
        y_center_norm = y_center_px / target_size
        width_norm = width_px / target_size
        height_norm = height_px / target_size
        
        # Клиппинг
        x_center_norm = np.clip(x_center_norm, 0, 1)
        y_center_norm = np.clip(y_center_norm, 0, 1)
        width_norm = np.clip(width_norm, 1e-6, 1)
        height_norm = np.clip(height_norm, 1e-6, 1)
        
        resized_boxes.append([class_id, x_center_norm, y_center_norm, width_norm, height_norm])
    
    return canvas, resized_boxes


def extract_defective_patches_yolo(
    train_images_dir="data/severstal/train_images",
    train_csv_path="data/severstal/train.csv",
    output_dir="data/severstal/yolo_dataset",
    patch_width=256,
    stride=128,
    min_defect_area=100,
    min_box_area=10,
    resize_to=1024,
    reject_black=True,
    black_threshold=30,
    max_black_ratio=0.05
):
    """Вырезает патчи с дефектами и сохраняет в YOLO формате (мультикласс)"""
    
    output_path = Path(output_dir)
    
    # СОЗДАЕМ ПАПКИ ДЛЯ YOLO ФОРМАТА
    train_images_path = output_path / "images" / "train"
    train_labels_path = output_path / "labels" / "train"
    val_images_path = output_path / "images" / "val"
    val_labels_path = output_path / "labels" / "val"
    
    for path in [train_images_path, train_labels_path, val_images_path, val_labels_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    # 🆕 Загружаем разметку с сохранением ClassId
    print("📂 Загрузка разметки с ClassId...")
    df = pd.read_csv(train_csv_path)
    
    # 🆕 Статистика по классам
    class_counts = df['ClassId'].value_counts().to_dict()
    print(f"📊 Распределение классов в исходных данных:")
    for class_id, count in sorted(class_counts.items()):
        print(f"   ClassId {class_id}: {count} RLE")
    
    # 🆕 Создаем маски с сохранением класса
    print("🎭 Создание многоклассовых масок...")
    masks = {}
    for img_id, group in tqdm(df.groupby('ImageId')):
        combined = np.zeros((256, 1600), dtype=np.uint8)
        for _, row in group.iterrows():
            class_id = int(row['ClassId'])
            # 🆕 Добавляем маску с class_id (приоритет: более высокий class_id перезаписывает)
            class_mask = rle_to_mask_with_class(row['EncodedPixels'], class_id)
            # При пересечении масок оставляем максимальный class_id
            combined = np.where(class_mask > 0, class_mask, combined)
        masks[img_id] = combined
    
    image_files = list(Path(train_images_dir).glob("*.jpg"))
    print(f"📁 Всего изображений: {len(image_files)}")
    
    if reject_black:
        print(f"🚫 Отбраковка патчей с черным фоном (> {max_black_ratio*100:.0f}% черного)")
    
    all_patches = []
    total_defects_found = 0
    rejected_black = 0
    
    # 🆕 Счетчики по классам
    class_box_counts = defaultdict(int)
    
    # Для разделения на train/val (80/20)
    np.random.seed(42)
    n_images = len(image_files)
    n_train = int(n_images * 0.8)
    train_indices = set(np.random.choice(n_images, n_train, replace=False))
    
    for idx, img_path in enumerate(tqdm(image_files, desc="🔍 Поиск дефектов")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = masks.get(img_path.name, np.zeros((256, 1600), dtype=np.uint8))
        
        # Определяем, в train или val попадет этот патч
        is_train = idx in train_indices
        img_subdir = train_images_path if is_train else val_images_path
        label_subdir = train_labels_path if is_train else val_labels_path
        
        for x in range(0, 1600 - patch_width + 1, stride):
            patch_img = img_rgb[:, x:x+patch_width]
            patch_mask = mask[:, x:x+patch_width]
            
            if reject_black and has_black_background(patch_img, black_threshold, max_black_ratio):
                rejected_black += 1
                continue
            
            # 🆕 Конвертируем многоклассовую маску в YOLO bboxes
            boxes = mask_to_yolo_boxes_multiclass(patch_mask, min_area=min_box_area)
            
            if len(boxes) == 0:
                continue
            
            # Проверяем общую площадь дефектов (всех классов)
            total_defect_pixels = np.sum(patch_mask > 0)
            if total_defect_pixels < min_defect_area:
                continue
            
            # Ресайз изображения и адаптация bbox
            patch_resized, resized_boxes = resize_with_bbox(patch_img, boxes, resize_to)
            
            # Сохраняем изображение
            patch_name = f"{img_path.stem}_x{x}_w{patch_width}"
            img_save_path = img_subdir / f"{patch_name}.jpg"
            cv2.imwrite(str(img_save_path), cv2.cvtColor(patch_resized, cv2.COLOR_RGB2BGR))
            
            # 🆕 Сохраняем YOLO аннотации с оригинальными class_id
            label_save_path = label_subdir / f"{patch_name}.txt"
            with open(label_save_path, 'w') as f:
                for box in resized_boxes:
                    class_id = box[0]  # 1, 2, 3 или 4
                    # YOLO формат: class_id-1 (так как YOLO использует 0-based индексацию)
                    yolo_class_id = class_id - 1
                    f.write(f"{yolo_class_id} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                    class_box_counts[class_id] += 1
            
            all_patches.append({
                'image': img_path.name,
                'x': x,
                'width': patch_width,
                'defect_area': int(total_defect_pixels),
                'num_boxes': len(resized_boxes),
                'saved_as': patch_name,
                'split': 'train' if is_train else 'val',
                'image_path': str(img_save_path),
                'label_path': str(label_save_path)
            })
            
            total_defects_found += len(resized_boxes)
    
    # 🆕 Создаем dataset.yaml с 4 классами
    dataset_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 4,  # 🆕 4 класса
        'names': ['defect_type_1', 'defect_type_2', 'defect_type_3', 'defect_type_4']
    }
    
    with open(output_path / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    
    # Статистика
    print(f"\n{'='*50}")
    print(f"📊 СТАТИСТИКА:")
    print(f"  Найдено патчей с дефектами: {len(all_patches)}")
    print(f"  Всего bbox (дефектов): {total_defects_found}")
    print(f"  Train патчей: {sum(1 for p in all_patches if p['split']=='train')}")
    print(f"  Val патчей: {sum(1 for p in all_patches if p['split']=='val')}")
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ:")
    for class_id in [1, 2, 3, 4]:
        count = class_box_counts[class_id]
        print(f"  ClassId {class_id} ({CLASS_NAMES[class_id]}): {count} bbox")
    if reject_black:
        print(f"\n  Отбраковано из-за черного фона: {rejected_black}")
    print(f"  Размер сохраненных патчей: {resize_to}×{resize_to}")
    print(f"\n📁 СТРУКТУРА YOLO ДАТАСЕТА:")
    print(f"  {output_path}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/  ({len(list(train_images_path.glob('*.jpg')))} файлов)")
    print(f"  │   └── val/    ({len(list(val_images_path.glob('*.jpg')))} файлов)")
    print(f"  ├── labels/")
    print(f"  │   ├── train/  ({len(list(train_labels_path.glob('*.txt')))} файлов)")
    print(f"  │   └── val/    ({len(list(val_labels_path.glob('*.txt')))} файлов)")
    print(f"  └── dataset.yaml")
    
    # Сохраняем метаданные
    if all_patches:
        pd.DataFrame(all_patches).to_csv(output_path / "patches_metadata.csv", index=False)
        with open(output_path / "annotations.json", "w") as f:
            json.dump(all_patches, f, indent=2)
    
    return all_patches


def visualize_yolo_patches(dataset_dir="data/severstal/yolo_dataset", num_examples=5):
    """Визуализация патчей с YOLO bbox (мультикласс)"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images" / "train"
    
    image_files = list(images_dir.glob("*.jpg"))
    
    if not image_files:
        print("Нет патчей для визуализации")
        return
    
    import random
    random.seed(42)
    selected = random.sample(image_files, min(num_examples, len(image_files)))
    
    # 🆕 Цвета для разных классов
    class_colors = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'orange'
    }
    
    fig, axes = plt.subplots(1, len(selected), figsize=(4*len(selected), 4))
    if len(selected) == 1:
        axes = [axes]
    
    for i, img_file in enumerate(selected):
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        axes[i].imshow(img)
        
        label_file = dataset_dir / "labels" / "train" / f"{img_file.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        yolo_class_id = int(parts[0])  # 0-based (0,1,2,3)
                        class_id = yolo_class_id + 1    # Оригинальный (1,2,3,4)
                        x_c, y_c, box_w, box_h = map(float, parts[1:])
                        
                        x = (x_c - box_w/2) * w
                        y = (y_c - box_h/2) * h
                        width = box_w * w
                        height = box_h * h
                        
                        color = class_colors.get(yolo_class_id, 'purple')
                        rect = patches.Rectangle(
                            (x, y), width, height,
                            linewidth=2, edgecolor=color, facecolor='none'
                        )
                        axes[i].add_patch(rect)
                        axes[i].text(x, y-5, f'{CLASS_NAMES[class_id]}', 
                                   color=color, fontsize=8,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        axes[i].set_title(f"{img_file.name}", fontsize=9)
        axes[i].axis('off')
    
    plt.suptitle(f"YOLO формат: патчи с 4 классами дефектов", fontsize=14)
    plt.tight_layout()
    plt.savefig(dataset_dir / 'yolo_patches_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Визуализация сохранена в: {dataset_dir}/yolo_patches_visualization.png")


if __name__ == "__main__":
    patches = extract_defective_patches_yolo(
        train_images_dir="data/severstal/train_images",
        train_csv_path="data/severstal/train.csv",
        output_dir="data/640x640/defect_patches",
        patch_width=256,
        stride=256,
        min_defect_area=100,
        min_box_area=10,
        resize_to=640,
        reject_black=True,
        black_threshold=30,
        max_black_ratio=0.05
    )
    
    if patches:
        visualize_yolo_patches("data/640x640/defect_patches")
        print(f"\n✅ ГОТОВО ДЛЯ LTDETR + DINOv3 (МУЛЬТИКЛАСС)!")
        print(f"✅ Датасет в YOLO формате: data/640x640/defect_patches")
        print(f"✅ Классов: 4")
        print(f"\n📋 dataset.yaml:")
        print(f"path: ./data/640x640/defect_patches")
        print(f"train: images/train")
        print(f"val: images/val")
        print(f"nc: 4")
        print(f"names: ['defect_type_1', 'defect_type_2', 'defect_type_3', 'defect_type_4']")
    else:
        print("❌ Не найдено патчей с дефектами!")