#!/usr/bin/env python3
"""
extract_defective_patches_yolo_fixed_png.py - СОХРАНЕНИЕ В PNG
Вырезание патчей с дефектами в YOLO формате без потери дефектов при пересечении классов
✅ Все патчи сохраняются в train (без val)
✅ Корректная обработка пересекающихся дефектов разных классов
✅ Сохранение в PNG (без сжатия с потерями)
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
import yaml

# Маппинг ClassId → имена дефектов
CLASS_NAMES = {
    1: 'defect_type_1',
    2: 'defect_type_2',
    3: 'defect_type_3',
    4: 'defect_type_4'
}

def rle_to_binary_mask(rle_string, height=256, width=1600):
    """
    Декодирование RLE в бинарную маску (0/1)
    Column-major order (как в Severstal)
    """
    if pd.isna(rle_string) or rle_string == '' or str(rle_string).strip() == '':
        return np.zeros((height, width), dtype=np.uint8)
    
    numbers = list(map(int, str(rle_string).split()))
    starts = np.array(numbers[0::2]) - 1  # Severstal RLE starts from 1
    lengths = np.array(numbers[1::2])
    
    flat_mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat_mask[start:start + length] = 1
    
    # Severstal использует column-major order (сначала столбцы, потом строки)
    # Поэтому reshape(width, height).T
    return flat_mask.reshape(width, height).T


def create_masks_by_class(group, height=256, width=1600):
    """
    🔧 ИСПРАВЛЕНО: Создает отдельные маски для каждого класса
    Дефекты разных классов НЕ перезаписывают друг друга
    
    Returns:
        dict: {class_id: binary_mask}
    """
    masks_by_class = {}
    
    for _, row in group.iterrows():
        class_id = int(row['ClassId'])
        rle_string = row['EncodedPixels']
        
        if pd.isna(rle_string) or str(rle_string).strip() == '':
            continue
        
        # Получаем бинарную маску для этого RLE
        mask = rle_to_binary_mask(rle_string, height, width)
        
        # 🔧 Ключевое исправление: объединяем маски одного класса через логическое ИЛИ
        if class_id not in masks_by_class:
            masks_by_class[class_id] = mask
        else:
            masks_by_class[class_id] = np.logical_or(masks_by_class[class_id], mask).astype(np.uint8)
    
    return masks_by_class


def masks_to_yolo_boxes(masks_by_class, min_area=10):
    """
    Конвертирует словарь масок {class_id: mask} в YOLO bboxes
    
    Returns:
        list of [class_id, x_center, y_center, width, height]
        class_id: оригинальный 1-4
    """
    if not masks_by_class:
        return []
    
    # Получаем размеры из первой маски
    first_mask = next(iter(masks_by_class.values()))
    h, w = first_mask.shape
    
    yolo_boxes = []
    
    for class_id, mask in masks_by_class.items():
        if np.sum(mask) == 0:
            continue
        
        # Находим контуры для этого класса
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Bounding box в пикселях
            x, y, box_w, box_h = cv2.boundingRect(contour)
            
            # Конвертация в нормализованные YOLO координаты
            x_center = (x + box_w / 2) / w
            y_center = (y + box_h / 2) / h
            width = box_w / w
            height = box_h / h
            
            # Клиппинг для безопасности
            x_center = np.clip(x_center, 0.0, 1.0)
            y_center = np.clip(y_center, 0.0, 1.0)
            width = np.clip(width, 1e-6, 1.0)
            height = np.clip(height, 1e-6, 1.0)
            
            yolo_boxes.append([class_id, x_center, y_center, width, height])
    
    return yolo_boxes


def has_black_background(img, black_threshold=30, max_black_ratio=0.05):
    """Проверяет, есть ли в патче слишком много черного фона"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    black_pixels = np.sum(gray < black_threshold)
    total_pixels = img.shape[0] * img.shape[1]
    black_ratio = black_pixels / total_pixels
    return black_ratio > max_black_ratio


def resize_with_bbox(img, boxes, target_size=256):
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
        
        # Конвертируем в пиксельные координаты
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
        x_center_norm = np.clip(x_center_norm, 0.0, 1.0)
        y_center_norm = np.clip(y_center_norm, 0.0, 1.0)
        width_norm = np.clip(width_norm, 1e-6, 1.0)
        height_norm = np.clip(height_norm, 1e-6, 1.0)
        
        resized_boxes.append([class_id, x_center_norm, y_center_norm, width_norm, height_norm])
    
    return canvas, resized_boxes


def extract_defective_patches_yolo(
    train_images_dir="data/severstal/train_images",
    train_csv_path="data/severstal/train.csv",
    output_dir="data/256_yolo/defect_patches",
    patch_width=256,
    stride=256,
    min_defect_area=100,
    min_box_area=10,
    resize_to=256,
    reject_black=True,
    black_threshold=30,
    max_black_ratio=0.05,
    save_format='png'  # 🆕 Параметр выбора формата ('png' или 'jpg')
):
    """
    🔧 ИСПРАВЛЕННАЯ ВЕРСИЯ: Вырезает патчи с дефектами и сохраняет в YOLO формате
    ✅ Все патчи сохраняются в train (без разделения на val)
    ✅ Дефекты разных классов не перезаписывают друг друга
    ✅ Сохранение в PNG (по умолчанию) для сохранения качества
    """
    
    output_path = Path(output_dir)
    
    # СОЗДАЕМ ПАПКИ ТОЛЬКО ДЛЯ TRAIN
    train_images_path = output_path / "images" / "train"
    train_labels_path = output_path / "labels" / "train"
    
    for path in [train_images_path, train_labels_path]:
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Создана директория: {path}")
    
    # Загружаем разметку
    print("\n📂 Загрузка разметки из CSV...")
    df = pd.read_csv(train_csv_path)
    print(f"   Всего строк в CSV: {len(df):,}")
    print(f"   Уникальных изображений: {df['ImageId'].nunique():,}")
    
    # Статистика по классам в исходных данных
    class_counts_original = df['ClassId'].value_counts().to_dict()
    print(f"\n📊 ИСХОДНОЕ распределение RLE по классам:")
    for class_id in sorted(class_counts_original.keys()):
        count = class_counts_original[class_id]
        name = CLASS_NAMES[class_id]
        print(f"   ClassId {class_id} ({name}): {count:,} RLE")
    
    # 🔧 ИСПРАВЛЕНО: Создаем маски для каждого класса отдельно
    print("\n🎭 Создание масок (каждый класс отдельно, без перезаписи)...")
    masks_by_image = {}
    for img_id, group in tqdm(df.groupby('ImageId')):
        masks_by_image[img_id] = create_masks_by_class(group)
    
    # Проверка изображений
    image_files = list(Path(train_images_dir).glob("*.jpg"))
    print(f"\n📁 Всего изображений: {len(image_files):,}")
    
    if reject_black:
        print(f"🚫 Отбраковка патчей с черным фоном (> {max_black_ratio*100:.0f}% черного)")
    
    print(f"🖼️  Формат сохранения: {save_format.upper()}")
    
    all_patches = []
    total_boxes_found = 0
    rejected_black = 0
    processed_images = 0
    images_with_defects = 0
    
    # 🔧 Счетчики по классам для финальной статистики
    class_box_counts = defaultdict(int)
    
    for img_path in tqdm(image_files, desc="🔍 Поиск дефектов"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_masks = masks_by_image.get(img_path.name, {})
        
        if not img_masks:
            continue  # Нет дефектов на этом изображении
        
        images_with_defects += 1
        processed_images += 1
        
        # Скользящее окно по ширине
        for x in range(0, 1600 - patch_width + 1, stride):
            patch_img = img_rgb[:, x:x+patch_width]
            
            # Отбраковка черного фона
            if reject_black and has_black_background(patch_img, black_threshold, max_black_ratio):
                rejected_black += 1
                continue
            
            # 🔧 Вырезаем маски для каждого класса в этом патче
            patch_masks = {}
            total_defect_pixels = 0
            
            for class_id, full_mask in img_masks.items():
                patch_mask = full_mask[:, x:x+patch_width]
                defect_pixels = np.sum(patch_mask)
                
                if defect_pixels > 0:
                    patch_masks[class_id] = patch_mask
                    total_defect_pixels += defect_pixels
            
            # Проверяем общую площадь дефектов
            if total_defect_pixels < min_defect_area:
                continue
            
            # 🔧 Конвертируем маски в YOLO bboxes
            boxes = masks_to_yolo_boxes(patch_masks, min_area=min_box_area)
            
            if len(boxes) == 0:
                continue
            
            # Ресайз изображения и адаптация bbox
            patch_resized, resized_boxes = resize_with_bbox(patch_img, boxes, resize_to)
            
            # 🆕 Сохраняем изображение в выбранном формате
            patch_name = f"{img_path.stem}_x{x}_w{patch_width}"
            img_save_path = train_images_path / f"{patch_name}.{save_format}"
            
            # Конвертируем RGB обратно в BGR для OpenCV
            img_bgr = cv2.cvtColor(patch_resized, cv2.COLOR_RGB2BGR)
            
            # Сохраняем с оптимальными параметрами
            if save_format.lower() == 'png':
                cv2.imwrite(str(img_save_path), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:  # jpg
                cv2.imwrite(str(img_save_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Сохраняем YOLO аннотации
            label_save_path = train_labels_path / f"{patch_name}.txt"
            with open(label_save_path, 'w') as f:
                for box in resized_boxes:
                    class_id = box[0]  # 1, 2, 3 или 4
                    # YOLO использует 0-based индексацию (class_id - 1)
                    yolo_class_id = class_id - 1
                    f.write(f"{yolo_class_id} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                    class_box_counts[class_id] += 1
            
            all_patches.append({
                'image': img_path.name,
                'x': x,
                'width': patch_width,
                'defect_area': int(total_defect_pixels),
                'num_boxes': len(resized_boxes),
                'classes_present': list(set(b[0] for b in resized_boxes)),
                'saved_as': patch_name,
                'format': save_format,
                'split': 'train',  # Все в train
                'image_path': str(img_save_path),
                'label_path': str(label_save_path)
            })
            
            total_boxes_found += len(resized_boxes)
    
    # Создаем dataset.yaml
    dataset_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/train',  # Временно, потом можно заменить на val
        'nc': 4,
        'names': ['defect_type_1', 'defect_type_2', 'defect_type_3', 'defect_type_4']
    }
    
    with open(output_path / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    
    # ============================================
    # СТАТИСТИКА
    # ============================================
    print(f"\n{'='*60}")
    print(f"📊 ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*60}")
    
    print(f"\n📁 ОБРАБОТАНО ИЗОБРАЖЕНИЙ:")
    print(f"   Всего изображений: {len(image_files):,}")
    print(f"   С дефектами: {images_with_defects:,}")
    print(f"   Без дефектов: {len(image_files) - images_with_defects:,}")
    
    print(f"\n📊 ПАТЧИ:")
    print(f"   Найдено патчей с дефектами: {len(all_patches):,}")
    print(f"   Всего bbox (дефектов): {total_boxes_found:,}")
    print(f"   Среднее bbox на патч: {total_boxes_found/len(all_patches):.2f}" if all_patches else "   Среднее bbox на патч: 0")
    
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ:")
    for class_id in [1, 2, 3, 4]:
        count = class_box_counts[class_id]
        pct = (count / total_boxes_found * 100) if total_boxes_found > 0 else 0
        original_count = class_counts_original.get(class_id, 0)
        retention = (count / original_count * 100) if original_count > 0 else 0
        name = CLASS_NAMES[class_id]
        print(f"   ClassId {class_id} ({name}):")
        print(f"      Найдено bbox: {count:,} ({pct:.1f}%)")
        print(f"      Исходно RLE: {original_count:,}")
        print(f"      Сохранено: {retention:.1f}%")
    
    if reject_black:
        print(f"\n🚫 ОТБРАКОВАНО:")
        print(f"   Из-за черного фона: {rejected_black:,}")
    
    print(f"\n📁 СТРУКТУРА YOLO ДАТАСЕТА:")
    print(f"   {output_path}/")
    print(f"   ├── images/")
    print(f"   │   └── train/  ({len(list(train_images_path.glob(f'*.{save_format}'))):,} файлов)")
    print(f"   ├── labels/")
    print(f"   │   └── train/  ({len(list(train_labels_path.glob('*.txt'))):,} файлов)")
    print(f"   └── dataset.yaml")
    
    # Сохраняем метаданные
    if all_patches:
        pd.DataFrame(all_patches).to_csv(output_path / "patches_metadata.csv", index=False)
        with open(output_path / "annotations.json", "w") as f:
            json.dump(all_patches, f, indent=2)
        print(f"\n✅ Метаданные сохранены:")
        print(f"   - {output_path}/patches_metadata.csv")
        print(f"   - {output_path}/annotations.json")
    
    return all_patches


def visualize_yolo_patches(dataset_dir="data/256_yolo/defect_patches", num_examples=5, save_format='png'):
    """Визуализация патчей с YOLO bbox (мультикласс)"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import random
    
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images" / "train"
    
    # 🆕 Ищем файлы с нужным расширением
    image_files = list(images_dir.glob(f"*.{save_format}"))
    
    if not image_files:
        print(f"❌ Нет патчей в формате .{save_format} для визуализации")
        return
    
    random.seed(42)
    selected = random.sample(image_files, min(num_examples, len(image_files)))
    
    # Цвета для разных классов
    class_colors = {
        0: '#FF6B6B',  # defect_type_1 - красный
        1: '#4ECDC4',  # defect_type_2 - бирюзовый
        2: '#45B7D1',  # defect_type_3 - синий
        3: '#96CEB4'   # defect_type_4 - зеленый
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
                        yolo_class_id = int(parts[0])
                        class_id = yolo_class_id + 1
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
                        axes[i].text(x, y-5, CLASS_NAMES[class_id], 
                                   color=color, fontsize=8, fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        axes[i].set_title(f"{img_file.name}", fontsize=9)
        axes[i].axis('off')
    
    plt.suptitle(f"✅ ИСПРАВЛЕНО: все классы сохранены без перезаписи (PNG)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(dataset_dir / 'yolo_patches_visualization_fixed.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Визуализация сохранена в: {dataset_dir}/yolo_patches_visualization_fixed.png")


if __name__ == "__main__":
    print("="*60)
    print("🔧 ИСПРАВЛЕННЫЙ ЭКСТРАКТОР ПАТЧЕЙ (МУЛЬТИКЛАСС + PNG)")
    print("="*60)
    
    # 🆕 Выбор формата сохранения
    SAVE_FORMAT = 'png'  # 'png' или 'jpg'
    
    patches = extract_defective_patches_yolo(
        train_images_dir="data/severstal/train_images",
        train_csv_path="data/severstal/train.csv",
        output_dir="data/256_yolo/defect_patches",
        patch_width=256,
        stride=256,  # Без перекрытия
        min_defect_area=50,
        min_box_area=10,
        resize_to=256,
        reject_black=True,
        black_threshold=30,
        max_black_ratio=0.95,
        save_format=SAVE_FORMAT  # 🆕 PNG формат
    )
    
    if patches:
        print("\n" + "="*60)
        print("🎉 ВИЗУАЛИЗАЦИЯ")
        print("="*60)
        visualize_yolo_patches("data/256_yolo/defect_patches", num_examples=5, save_format=SAVE_FORMAT)
        
        print("\n" + "="*60)
        print("✅ ГОТОВО! ВСЕ КЛАССЫ СОХРАНЕНЫ КОРРЕКТНО!")
        print("="*60)
        print(f"📍 Датасет: data/256_yolo/defect_patches")
        print(f"📍 Классов: 4")
        print(f"📍 Формат изображений: {SAVE_FORMAT.upper()}")
        print(f"📍 Все патчи в train (без val)")
        print(f"\n📋 dataset.yaml:")
        print(f"   path: ./data/256_yolo/defect_patches")
        print(f"   train: images/train")
        print(f"   val: images/train")
        print(f"   nc: 4")
        print(f"   names: ['defect_type_1', 'defect_type_2', 'defect_type_3', 'defect_type_4']")
    else:
        print("\n❌ НЕ НАЙДЕНО ПАТЧЕЙ С ДЕФЕКТАМИ!")
        print("   Проверьте параметры min_defect_area и min_box_area")