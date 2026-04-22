#!/usr/bin/env python3
"""
compose_defects_adaptive.py - Адаптивная вставка дефектов с цветовой валидацией
Вставляет дефект только если его цветовая гамма совместима с фоном
"""

import sys
import os
sys.path.insert(0, '/app/scripts')
sys.path.insert(0, '/app/src')

import cv2
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

from utils import set_seed, print_system_info, logger


def rle_to_mask(rle_string, height=256, width=1600):
    """Правильное декодирование RLE для Severstal (column-major order)"""
    if pd.isna(rle_string) or rle_string == '':
        return np.zeros((height, width), dtype=np.uint8)
    
    numbers = list(map(int, rle_string.split()))
    starts = np.array(numbers[0::2]) - 1
    lengths = np.array(numbers[1::2])
    
    flat_mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat_mask[start:start + length] = 1
    
    return flat_mask.reshape(width, height).T


def split_mask_into_components(mask: np.ndarray) -> List[np.ndarray]:
    """
    Разбивает бинарную маску на отдельные связные компоненты
    Возвращает список масок (каждая — отдельный дефект)
    """
    # Находим связные компоненты
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    components = []
    for i in range(1, num_labels):  # Пропускаем фон (0)
        component_mask = (labels == i).astype(np.uint8)
        
        # Игнорируем слишком маленькие компоненты (шум)
        if np.sum(component_mask) < 10:
            continue
        
        components.append(component_mask)
    
    return components


def extract_defects_from_mask(image: np.ndarray, mask: np.ndarray, padding: int = 5) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]]:
    """
    Вырезает ВСЕ связные компоненты из маски как отдельные дефекты
    
    Returns:
        Список кортежей: [(roi, roi_mask, bbox), ...]
    """
    components = split_mask_into_components(mask)
    
    defects = []
    for comp_mask in components:
        y_indices, x_indices = np.where(comp_mask > 0)
        
        if len(x_indices) == 0:
            continue
        
        x1 = max(0, np.min(x_indices) - padding)
        y1 = max(0, np.min(y_indices) - padding)
        x2 = min(image.shape[1], np.max(x_indices) + padding)
        y2 = min(image.shape[0], np.max(y_indices) + padding)
        
        roi = image[y1:y2, x1:x2].copy()
        roi_mask = comp_mask[y1:y2, x1:x2].copy()
        
        defects.append((roi, roi_mask, (x1, y1, x2, y2)))
    
    return defects


def create_alpha_mask(mask: np.ndarray, feather_size: int = 5) -> np.ndarray:
    """Создает сильно растушеванную маску без жестких границ"""
    if mask.sum() == 0:
        return mask
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    alpha = np.clip(dist / feather_size, 0, 1)
    return alpha


def calculate_color_histogram(image: np.ndarray, mask: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Вычисляет нормализованную цветовую гистограмму только внутри маски
    Возвращает 2D гистограмму (Hue + Saturation)
    """
    if mask.sum() == 0:
        return None
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Вычисляем 2D гистограмму (Hue + Saturation)
    hist = cv2.calcHist(
        [hsv],           # ← список изображений
        [0, 1],          # каналы Hue и Saturation
        mask,            # маска
        [bins, bins],    # размеры гистограммы
        [0, 180, 0, 256] # диапазоны: Hue 0-180, Saturation 0-256
    )
    
    # Нормализуем
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    
    return hist.flatten()


def calculate_histogram_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Вычисляет схожесть двух гистограмм
    Использует корреляцию (1 = идентичны, 0 = разные, -1 = противоположные)
    """
    if hist1 is None or hist2 is None:
        return 0.0
    
    # Корреляция
    correlation = cv2.compareHist(
        hist1.reshape(-1, 1).astype(np.float32),
        hist2.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    
    return correlation


class SeverstalDefectLibrary:
    """Библиотека дефектов из Severstal датасета"""
    
    def __init__(self, images_dir: Path, csv_path: Path):
        self.images_dir = Path(images_dir)
        self.df = pd.read_csv(csv_path)
        self.defect_groups = self.df.groupby('ImageId')
        
        logger.info(f"📂 Загружено {len(self.df)} дефектов из {len(self.defect_groups)} изображений")
        
        self.cache = {}
    
    def get_random_defect(self) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Возвращает случайный ОТДЕЛЬНЫЙ дефект (один связный компонент)
        """
        image_id = random.choice(list(self.defect_groups.groups.keys()))
        
        if image_id not in self.cache:
            img_path = self.images_dir / image_id
            if not img_path.exists():
                return None
            
            image = cv2.imread(str(img_path))
            if image is None:
                return None
            
            self.cache[image_id] = image
        
        image = self.cache[image_id]
        defects = self.defect_groups.get_group(image_id)
        row = defects.sample(1).iloc[0]
        
        mask = rle_to_mask(row['EncodedPixels'], height=256, width=1600)
        
        if mask.sum() == 0:
            return None
        
        class_id = int(row['ClassId']) - 1
        
        # ✅ Извлекаем ВСЕ компоненты и выбираем случайный
        component_defects = extract_defects_from_mask(image, mask)
        
        if not component_defects:
            return None
        
        # Выбираем случайный компонент
        roi, roi_mask, bbox = random.choice(component_defects)
        
        return roi, roi_mask, class_id


class AdaptiveDefectComposer:
    """Композер с цветовой валидацией перед вставкой"""
    
    def __init__(self, clean_dir: Path, defect_library: SeverstalDefectLibrary, 
                 similarity_threshold: float = 0.5):
        self.clean_images = list(Path(clean_dir).glob("*.png")) + \
                            list(Path(clean_dir).glob("*.jpg"))
        self.defect_library = defect_library
        self.similarity_threshold = similarity_threshold  # Порог схожести гистограмм
        
        logger.info(f"📂 Загружено {len(self.clean_images)} чистых фонов")
        logger.info(f"🎯 Порог цветовой совместимости: {similarity_threshold}")
    
    def match_lighting(self, roi: np.ndarray, roi_mask: np.ndarray, background: np.ndarray, 
                       target_region: Tuple[int, int, int, int]) -> np.ndarray:
        """Мягкая адаптация яркости ROI под фон"""
        x1, y1, x2, y2 = target_region
        bg_region = background[y1:y2, x1:x2]
        
        if bg_region.size == 0 or roi_mask.sum() == 0:
            return roi
        
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        bg_hsv = cv2.cvtColor(bg_region, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        roi_v = roi_hsv[:, :, 2]
        bg_v = bg_hsv[:, :, 2]
        
        roi_v_mean = np.mean(roi_v[roi_mask > 0])
        bg_v_mean = np.mean(bg_v)
        
        if roi_v_mean > 0:
            v_ratio = bg_v_mean / roi_v_mean
            v_ratio = 1.0 + (v_ratio - 1.0) * 0.8  # 80% коррекции
            roi_hsv[:, :, 2] = np.clip(roi_v * v_ratio, 0, 255)
        
        roi_adjusted = cv2.cvtColor(np.clip(roi_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        mask_3d = np.stack([roi_mask] * 3, axis=-1)
        roi_final = roi * (1 - mask_3d) + roi_adjusted * mask_3d
        
        return roi_final.astype(np.uint8)
    
    def is_color_compatible(self, roi: np.ndarray, roi_mask: np.ndarray, 
                            background: np.ndarray, target_region: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = target_region
        bg_region = background[y1:y2, x1:x2]
        
        # 1. Гистограммная схожесть
        defect_hist = calculate_color_histogram(roi, roi_mask)
        bg_mask = np.ones(bg_region.shape[:2], dtype=np.uint8)
        bg_hist = calculate_color_histogram(bg_region, bg_mask)
        
        if defect_hist is None or bg_hist is None:
            return False
        
        similarity = calculate_histogram_similarity(defect_hist, bg_hist)
        
        if similarity < self.similarity_threshold:
            return False
        
        # 2. ✅ ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА ЯРКОСТИ
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(bg_region, cv2.COLOR_BGR2GRAY)
        
        roi_mean = np.mean(roi_gray[roi_mask > 0])
        bg_mean = np.mean(bg_gray)
        
        if roi_mean > 0:
            brightness_ratio = max(roi_mean, bg_mean) / min(roi_mean, bg_mean)
            if brightness_ratio > 2.5:  # Слишком большая разница в яркости
                return False
        
        return True
    
    def compose_single_image(self, bg_path: Path, max_defects: int = 3) -> Tuple[Image.Image, List[Dict]]:
        """Создает одно синтетическое изображение"""
        
        background = cv2.imread(str(bg_path))
        if background is None:
            return None, []
        
        h_bg, w_bg = background.shape[:2]
        
        annotations = []
        inserted_boxes = []
        
        num_defects = random.randint(1, max_defects)
        
        max_attempts = num_defects * 15  # Больше попыток из-за цветовой валидации
        attempts = 0
        inserted = 0
        
        while inserted < num_defects and attempts < max_attempts:
            attempts += 1
            
            result = self.defect_library.get_random_defect()
            if result is None:
                continue
            
            roi, roi_mask, class_id = result
            
            if roi.shape[0] >= h_bg - 20 or roi.shape[1] >= w_bg - 20:
                continue
            
            h_roi, w_roi = roi.shape[:2]
            
            margin = 10
            x1 = random.randint(margin, w_bg - w_roi - margin)
            y1 = random.randint(margin, h_bg - h_roi - margin)
            x2 = x1 + w_roi
            y2 = y1 + h_roi
            
            # Проверка на пересечение
            overlaps = False
            for bx1, by1, bx2, by2 in inserted_boxes:
                if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):
                    overlaps = True
                    break
            
            if overlaps:
                continue
            
            # ✅ ЦВЕТОВАЯ ВАЛИДАЦИЯ — ключевое нововведение!
            if not self.is_color_compatible(roi, roi_mask, background, (x1, y1, x2, y2)):
                continue  # Пропускаем, если цвета не совместимы
            
            # Адаптируем освещение
            roi = self.match_lighting(roi, roi_mask, background, (x1, y1, x2, y2))
            
            # Создаем альфа-канал
            alpha = create_alpha_mask(roi_mask, feather_size=5)
            alpha_3d = np.stack([alpha] * 3, axis=-1)
            
            roi_float = roi.astype(np.float32)
            
            bg_region = background[y1:y2, x1:x2].astype(np.float32)
            bg_blurred = cv2.GaussianBlur(bg_region, (21, 21), 0)
            
            bg_final = bg_region * (1 - alpha_3d) + bg_blurred * alpha_3d
            blended = roi_float * alpha_3d + bg_final * (1 - alpha_3d)
            
            background[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
            
            inserted_boxes.append((x1, y1, x2, y2))
            
            # Создаем YOLO-аннотацию
            y_indices, x_indices = np.where(roi_mask > 0)
            if len(x_indices) > 0:
                rel_x1 = (x1 + np.min(x_indices)) / w_bg
                rel_y1 = (y1 + np.min(y_indices)) / h_bg
                rel_x2 = (x1 + np.max(x_indices)) / w_bg
                rel_y2 = (y1 + np.max(y_indices)) / h_bg
                
                annotations.append({
                    'class': class_id,
                    'x_center': (rel_x1 + rel_x2) / 2,
                    'y_center': (rel_y1 + rel_y2) / 2,
                    'width': rel_x2 - rel_x1,
                    'height': rel_y2 - rel_y1
                })
                
                inserted += 1
        
        if inserted == 0:
            return None, []
        
        result_pil = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        
        return result_pil, annotations
    
    def generate_dataset(self, output_dir: Path, num_images: int):
        """Генерирует датасет"""
        output_images = output_dir / "images"
        output_labels = output_dir / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Генерация {num_images} изображений...")
        
        for i in tqdm(range(num_images), desc="Composing"):
            bg_path = random.choice(self.clean_images)
            img, anns = self.compose_single_image(bg_path, max_defects=random.randint(2, 5))
            
            if img and anns:
                filename = f"syn_{i:06d}"
                img.save(output_images / f"{filename}.png", "PNG")
                
                with open(output_labels / f"{filename}.txt", 'w') as f:
                    for ann in anns:
                        f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                               f"{ann['width']:.6f} {ann['height']:.6f}\n")
        
        logger.info(f"✅ Готово! Результаты в {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, default="data/dataset_synthetic/clean_patches")
    parser.add_argument("--severstal_images", type=str, default="data/severstal/train_images")
    parser.add_argument("--severstal_csv", type=str, default="data/severstal/train.csv")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_adaptive")
    parser.add_argument("--num_images", type=int, default=10000)
    parser.add_argument("--similarity", type=float, default=0.4, 
                       help="Порог схожести гистограмм (0.3-0.6 рекомендуется)")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    print_system_info()
    
    defect_lib = SeverstalDefectLibrary(
        Path(args.severstal_images),
        Path(args.severstal_csv)
    )
    
    composer = AdaptiveDefectComposer(
        Path(args.clean_dir),
        defect_lib,
        similarity_threshold=args.similarity
    )
    
    composer.generate_dataset(Path(args.output_dir), args.num_images)


if __name__ == "__main__":
    main()