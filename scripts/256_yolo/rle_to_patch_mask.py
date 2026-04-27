#!/usr/bin/env python3
"""
rle_to_patch_mask.py - Корректное декодирование RLE для патчей 256×256
Учитывает сдвиг патча (x0_w256, x256_w256, x512_w256...) из имени файла
и размер оригинального изображения (256×1600)
"""

import numpy as np
import pandas as pd
from pathlib import Path

def parse_patch_offset(filename: str):
    """
    Извлекает сдвиг патча из имени файла.
    Пример: '39bec5e5e_x256_w256.png' → offset_x=256, patch_w=256
            '8a87b9578_x0_w256.png'   → offset_x=0, patch_w=256
    """
    import re
    match = re.search(r'_x(\d+)_w(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 256


def rle_to_patch_mask(rle_string, patch_filename, original_width=1600, original_height=256):
    """
    Декодирует RLE с оригинального изображения (256×1600)
    и извлекает маску для патча 256×256 с учётом сдвига.
    
    Args:
        rle_string: RLE строка из оригинального CSV
        patch_filename: имя файла патча (например '39bec5e5e_x256_w256.png')
        original_width: ширина оригинального изображения (1600)
        original_height: высота оригинального изображения (256)
    
    Returns:
        np.ndarray: маска 256×256 для патча
    """
    if pd.isna(rle_string) or str(rle_string).strip() in ['', 'nan']:
        return np.zeros((256, 256), dtype=np.uint8)
    
    # Парсим сдвиг патча
    offset_x, patch_w = parse_patch_offset(patch_filename)
    
    # Декодируем RLE на полном изображении (256×1600)
    numbers = list(map(int, str(rle_string).split()))
    starts = np.array(numbers[0::2]) - 1
    lengths = np.array(numbers[1::2])
    
    total_pixels = original_width * original_height
    full_mask = np.zeros(total_pixels, dtype=np.uint8)
    
    for start, length in zip(starts, lengths):
        if start < total_pixels:
            end = min(start + length, total_pixels)
            full_mask[start:end] = 1
    
    # Row-major: reshape(original_width, original_height).T
    # Для 256×1600: reshape(1600, 256).T → (256, 1600)
    full_mask_2d = full_mask.reshape(original_width, original_height).T
    
    # Извлекаем патч [:, offset_x:offset_x+256]
    patch_mask = full_mask_2d[:, offset_x:offset_x + 256]
    
    return patch_mask


def rle_to_yolo_bboxes(rle_string, patch_filename, class_id, original_width=1600, original_height=256):
    """
    Конвертирует RLE → YOLO bbox для патча 256×256
    
    Returns:
        list of dict: [{'class': 0, 'x_center': 0.5, 'y_center': 0.3, 'width': 0.1, 'height': 0.05}]
    """
    import cv2
    
    mask = rle_to_patch_mask(rle_string, patch_filename, original_width, original_height)
    
    if mask.sum() == 0:
        return []
    
    # Находим связные компоненты
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        if area < 16 or w < 4 or h < 4:
            continue
        
        bboxes.append({
            'class': class_id,
            'x_center': (x + w/2) / 256,
            'y_center': (y + h/2) / 256,
            'width': w / 256,
            'height': h / 256,
            'component_mask': (labels == i).astype(np.uint8),
            'x': x, 'y': y, 'w': w, 'h': h
        })
    
    return bboxes


# ===== ТЕСТ =====
if __name__ == "__main__":
    # Проверяем на первом изображении с RLE
    df_rle = pd.read_csv("data/256_yolo/balanced_defect_patches_rle/train/train_rle.csv")
    
    # Берём первую непустую запись
    for idx, row in df_rle.iterrows():
        rle = str(row['EncodedPixels'])
        if rle not in ['nan', '', 'None'] and len(rle.split()) > 2:
            break
    
    patch_name = row['ImageId']
    class_id = int(row['ClassId']) - 1
    offset_x, _ = parse_patch_offset(patch_name)
    
    print(f"Патч: {patch_name}")
    print(f"Сдвиг: x={offset_x}")
    print(f"Class: {class_id}")
    print(f"RLE (first 100): {rle[:100]}")
    
    # Декодируем маску
    mask = rle_to_patch_mask(rle, patch_name)
    print(f"Маска: {mask.sum()} пикселей из 65536 ({mask.sum()/65536*100:.1f}%)")
    
    # Конвертируем в YOLO bbox
    bboxes = rle_to_yolo_bboxes(rle, patch_name, class_id)
    print(f"Bbox'ов: {len(bboxes)}")
    for i, bb in enumerate(bboxes):
        print(f"  {i}: class={bb['class']}, x={bb['x_center']:.3f}, y={bb['y_center']:.3f}, "
              f"w={bb['width']:.3f}, h={bb['height']:.3f}")
    
    # Сохраняем маску для визуальной проверки
    import cv2
    cv2.imwrite("scripts/test_rle_mask.png", mask * 255)
    
    # Сравниваем с YOLO-разметкой
    label_path = Path(f"data/256_yolo/balanced_defect_patches_rle/train/labels/{Path(patch_name).stem}.txt")
    if label_path.exists():
        print(f"\nYOLO-разметка из {label_path}:")
        with open(label_path) as f:
            for line in f:
                print(f"  {line.strip()}")
        
        # Визуализация
        img_path = Path(f"data/256_yolo/balanced_defect_patches_rle/train/images/{patch_name}")
        if img_path.exists():
            img = cv2.imread(str(img_path))
            
            # RLE маска — красный
            contours_rle, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours_rle, -1, (0, 0, 255), 2)
            
            # YOLO bbox — зелёный
            img_h, img_w = img.shape[:2]
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, xc, yc, bw, bh = map(float, parts[:5])
                        x1 = int((xc - bw/2) * img_w)
                        y1 = int((yc - bh/2) * img_h)
                        x2 = int((xc + bw/2) * img_w)
                        y2 = int((yc + bh/2) * img_h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            cv2.imwrite("scripts/test_rle_vs_yolo.png", img)
            print(f"\n✅ Сохранено:")
            print(f"  scripts/test_rle_mask.png — RLE маска")
            print(f"  scripts/test_rle_vs_yolo.png — RED=RLE, GREEN=YOLO")