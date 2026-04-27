#!/usr/bin/env python3
"""
compose_defects_improved_fast.py - Оптимизированная генерация синтетических дефектов
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
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import json
from dataclasses import dataclass
import time

from utils import set_seed, print_system_info, logger


# ============================================
# 1. RLE И МАСКИ
# ============================================

def rle_to_mask(rle_string, height=256, width=1600):
    """Декодирование RLE для Severstal"""
    if pd.isna(rle_string) or rle_string == '':
        return np.zeros((height, width), dtype=np.uint8)
    
    numbers = list(map(int, rle_string.split()))
    starts = np.array(numbers[0::2]) - 1
    lengths = np.array(numbers[1::2])
    
    flat_mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat_mask[start:start + length] = 1
    
    return flat_mask.reshape(width, height).T


def split_mask_into_components(mask: np.ndarray, min_area: int = 10) -> List[np.ndarray]:
    """Разбивает маску на связные компоненты"""
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    components = []
    for i in range(1, num_labels):
        component_mask = (labels == i).astype(np.uint8)
        if np.sum(component_mask) < min_area:
            continue
        components.append(component_mask)
    
    return components


# ============================================
# 2. БЫСТРЫЙ ЦВЕТОВОЙ АНАЛИЗ
# ============================================

def fast_color_check(roi: np.ndarray, roi_mask: np.ndarray, 
                     bg_region: np.ndarray) -> Tuple[bool, float]:
    """
    Быстрая проверка цветовой совместимости
    Использует средние значения вместо гистограмм для скорости
    """
    if roi_mask.sum() == 0:
        return False, 0.0
    
    # Средние значения в HSV
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    bg_hsv = cv2.cvtColor(bg_region, cv2.COLOR_BGR2HSV)
    
    # Только внутри маски
    roi_pixels = roi_hsv[roi_mask > 0]
    bg_pixels = bg_hsv.reshape(-1, 3)
    
    # Сравниваем средние H, S, V
    roi_mean = np.mean(roi_pixels, axis=0)
    bg_mean = np.mean(bg_pixels, axis=0)
    
    # Нормализованная разница
    hue_diff = abs(roi_mean[0] - bg_mean[0]) / 180.0
    sat_diff = abs(roi_mean[1] - bg_mean[1]) / 256.0
    val_diff = abs(roi_mean[2] - bg_mean[2]) / 256.0
    
    # Взвешенная оценка
    similarity = 1.0 - (hue_diff * 0.5 + sat_diff * 0.3 + val_diff * 0.2)
    
    # Порог: similarity > 0.3 считается совместимым
    is_compatible = similarity > 0.3
    
    return is_compatible, similarity


# ============================================
# 3. БИБЛИОТЕКА ДЕФЕКТОВ (УПРОЩЕННАЯ)
# ============================================

class SeverstalDefectLibrary:
    """Библиотека дефектов без кеширования (быстрая загрузка)"""
    
    def __init__(self, images_dir: Path, csv_path: Path):
        self.images_dir = Path(images_dir)
        self.df = pd.read_csv(csv_path)
        
        # Группируем по классам
        self.class_to_image_ids = defaultdict(set)
        for _, row in self.df.iterrows():
            class_id = int(row['ClassId']) - 1
            self.class_to_image_ids[class_id].add(row['ImageId'])
        
        self.class_counts = {cls: len(ids) for cls, ids in self.class_to_image_ids.items()}
        
        # Кеш изображений
        self.image_cache = {}
        
        logger.info(f"📂 Загружено {len(self.df)} дефектов")
        logger.info(f"📊 Классы: {self.class_counts}")
    
    def _load_image(self, image_id: str) -> np.ndarray:
        """Загружает изображение с кешированием"""
        if image_id in self.image_cache:
            return self.image_cache[image_id]
        
        img_path = self.images_dir / image_id
        image = cv2.imread(str(img_path))
        if image is not None:
            self.image_cache[image_id] = image
        return image
    
    def get_random_defect(self, target_class: Optional[int] = None) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """Возвращает случайный отдельный дефект"""
        # Выбираем изображение
        if target_class is not None and target_class in self.class_to_image_ids:
            image_id = random.choice(list(self.class_to_image_ids[target_class]))
        else:
            # Случайный класс
            all_ids = []
            for ids in self.class_to_image_ids.values():
                all_ids.extend(ids)
            image_id = random.choice(all_ids) if all_ids else None
        
        if image_id is None:
            return None
        
        image = self._load_image(image_id)
        if image is None:
            return None
        
        # Получаем дефекты этого изображения
        image_defects = self.df[self.df['ImageId'] == image_id]
        
        if target_class is not None:
            image_defects = image_defects[image_defects['ClassId'] == target_class + 1]
        
        if len(image_defects) == 0:
            return None
        
        # Выбираем случайный дефект
        row = image_defects.sample(1).iloc[0]
        mask = rle_to_mask(row['EncodedPixels'], height=256, width=1600)
        
        if mask.sum() == 0:
            return None
        
        class_id = int(row['ClassId']) - 1
        
        # Извлекаем компоненты
        components = split_mask_into_components(mask)
        if not components:
            return None
        
        # Выбираем случайный компонент
        comp_mask = random.choice(components)
        
        y_indices, x_indices = np.where(comp_mask > 0)
        padding = 5
        x1 = max(0, np.min(x_indices) - padding)
        y1 = max(0, np.min(y_indices) - padding)
        x2 = min(image.shape[1], np.max(x_indices) + padding)
        y2 = min(image.shape[0], np.max(y_indices) + padding)
        
        roi = image[y1:y2, x1:x2].copy()
        roi_mask = comp_mask[y1:y2, x1:x2].copy()
        
        return roi, roi_mask, class_id


# ============================================
# 4. БАЛАНСИРОВЩИК (УПРОЩЕННЫЙ)
# ============================================

class ClassBalancer:
    """Простой балансировщик классов"""
    
    def __init__(self, target_per_class: Dict[int, int]):
        self.target_per_class = target_per_class
        self.generated = defaultdict(int)
    
    def get_next_class(self) -> Optional[int]:
        """Возвращает класс с наибольшим дефицитом"""
        deficits = []
        for cls, target in self.target_per_class.items():
            deficit = target - self.generated.get(cls, 0)
            if deficit > 0:
                deficits.append((cls, deficit))
        
        if not deficits:
            return None
        
        deficits.sort(key=lambda x: x[1], reverse=True)
        
        # 80% самый дефицитный, 20% случайный
        if random.random() < 0.8:
            return deficits[0][0]
        else:
            return random.choice([c for c, _ in deficits])
    
    def add_generated(self, class_id: int):
        self.generated[class_id] += 1


# ============================================
# 5. БЫСТРЫЙ КОМПОЗЕР
# ============================================

class FastDefectComposer:
    """Оптимизированный композер с быстрой валидацией"""
    
    def __init__(self, clean_dir: Path, defect_library: SeverstalDefectLibrary,
                 class_balancer: Optional[ClassBalancer] = None,
                 use_poisson: bool = True):
        
        self.clean_images = list(Path(clean_dir).glob("*.png")) + \
                            list(Path(clean_dir).glob("*.jpg"))
        self.defect_library = defect_library
        self.class_balancer = class_balancer
        self.use_poisson = use_poisson
        
        logger.info(f"📂 Загружено {len(self.clean_images)} чистых фонов")
        logger.info(f"🎨 Poisson blending: {use_poisson}")
    
    def _create_alpha_mask(self, mask: np.ndarray, feather: int = 3) -> np.ndarray:
        """Быстрое создание альфа-маски"""
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        alpha = np.clip(dist / feather, 0, 1)
        return cv2.GaussianBlur(alpha, (3, 3), 0)
    
    def _match_brightness_fast(self, roi: np.ndarray, roi_mask: np.ndarray, 
                               bg_region: np.ndarray) -> np.ndarray:
        """Быстрая адаптация яркости"""
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(bg_region, cv2.COLOR_BGR2GRAY)
        
        roi_mean = np.mean(roi_gray[roi_mask > 0])
        bg_mean = np.mean(bg_gray)
        
        if roi_mean > 0 and bg_mean > 0:
            ratio = bg_mean / roi_mean
            ratio = 1.0 + (ratio - 1.0) * 0.7  # 70% коррекции
            
            roi_float = roi.astype(np.float32)
            roi_float[roi_mask > 0] *= ratio
            
            return np.clip(roi_float, 0, 255).astype(np.uint8)
        
        return roi
    
    def compose_single_image(self, bg_path: Path, max_defects: int = 3) -> Tuple[Image.Image, List[Dict]]:
        """Быстрое создание одного изображения"""
        background = cv2.imread(str(bg_path))
        if background is None:
            return None, []
        
        h_bg, w_bg = background.shape[:2]
        annotations = []
        inserted_boxes = []
        
        num_defects = random.randint(1, max_defects)
        
        for _ in range(num_defects):
            # Определяем класс
            target_class = None
            if self.class_balancer:
                target_class = self.class_balancer.get_next_class()
            
            # Получаем дефект
            result = self.defect_library.get_random_defect(target_class)
            if result is None:
                continue
            
            roi, roi_mask, class_id = result
            h_roi, w_roi = roi.shape[:2]
            
            # Проверка размера
            if h_roi >= h_bg - 10 or w_roi >= w_bg - 10:
                continue
            
            # Случайная позиция (5 попыток)
            best_pos = None
            best_score = -1
            
            for _ in range(5):
                x1 = random.randint(5, w_bg - w_roi - 5)
                y1 = random.randint(5, h_bg - h_roi - 5)
                x2, y2 = x1 + w_roi, y1 + h_roi
                
                # Проверка пересечений
                overlap = False
                for bx1, by1, bx2, by2 in inserted_boxes:
                    if not (x2 < bx1 - 5 or x1 > bx2 + 5 or y2 < by1 - 5 or y1 > by2 + 5):
                        overlap = True
                        break
                
                if overlap:
                    continue
                
                # Быстрая проверка цвета
                bg_region = background[y1:y2, x1:x2]
                compatible, score = fast_color_check(roi, roi_mask, bg_region)
                
                if compatible and score > best_score:
                    best_score = score
                    best_pos = (x1, y1, x2, y2)
            
            if best_pos is None:
                continue
            
            x1, y1, x2, y2 = best_pos
            
            # Смешивание
            if self.use_poisson and random.random() < 0.4:
                try:
                    mask_uint8 = (roi_mask * 255).astype(np.uint8)
                    center = (x1 + w_roi // 2, y1 + h_roi // 2)
                    
                    clone_method = random.choice([cv2.NORMAL_CLONE, cv2.MIXED_CLONE])
                    blended = cv2.seamlessClone(roi, background, mask_uint8, center, clone_method)
                    background[y1:y2, x1:x2] = blended[y1:y2, x1:x2]
                except:
                    # Fallback
                    self._simple_blend(roi, roi_mask, background, (x1, y1, x2, y2))
            else:
                self._simple_blend(roi, roi_mask, background, (x1, y1, x2, y2))
            
            inserted_boxes.append((x1, y1, x2, y2))
            
            if self.class_balancer:
                self.class_balancer.add_generated(class_id)
            
            # Аннотация
            y_idx, x_idx = np.where(roi_mask > 0)
            if len(x_idx) > 0:
                rel_x1 = (x1 + np.min(x_idx)) / w_bg
                rel_y1 = (y1 + np.min(y_idx)) / h_bg
                rel_x2 = (x1 + np.max(x_idx)) / w_bg
                rel_y2 = (y1 + np.max(y_idx)) / h_bg
                
                annotations.append({
                    'class': class_id,
                    'x_center': (rel_x1 + rel_x2) / 2,
                    'y_center': (rel_y1 + rel_y2) / 2,
                    'width': rel_x2 - rel_x1,
                    'height': rel_y2 - rel_y1
                })
        
        if not annotations:
            return None, []
        
        # Легкая пост-обработка
        if random.random() < 0.3:
            noise = np.random.normal(0, 3, background.shape)
            background = np.clip(background + noise, 0, 255).astype(np.uint8)
        
        result_pil = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        return result_pil, annotations
    
    def _simple_blend(self, roi, roi_mask, background, target_region):
        """Простое альфа-смешивание"""
        x1, y1, x2, y2 = target_region
        bg_region = background[y1:y2, x1:x2]
        
        # Адаптируем яркость
        roi_adj = self._match_brightness_fast(roi, roi_mask, bg_region)
        
        # Альфа-маска
        alpha = self._create_alpha_mask(roi_mask)
        alpha_3d = np.stack([alpha] * 3, axis=-1)
        
        # Смешивание
        blended = roi_adj * alpha_3d + bg_region * (1 - alpha_3d)
        background[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    
    def generate_dataset(self, output_dir: Path, num_images: int):
        """Генерация датасета"""
        output_images = output_dir / "images"
        output_labels = output_dir / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Генерация {num_images} изображений...")
        
        start_time = time.time()
        generated = 0
        
        for i in tqdm(range(num_images), desc="Composing"):
            bg_path = random.choice(self.clean_images)
            img, anns = self.compose_single_image(bg_path, max_defects=random.randint(2, 4))
            
            if img and anns:
                filename = f"syn_{i:06d}"
                img.save(output_images / f"{filename}.png", "PNG")
                
                with open(output_labels / f"{filename}.txt", 'w') as f:
                    for ann in anns:
                        f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                               f"{ann['width']:.6f} {ann['height']:.6f}\n")
                
                generated += 1
                
                # Прогресс каждые 1000
                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - start_time
                    speed = (i + 1) / elapsed
                    eta = (num_images - i - 1) / speed
                    logger.info(f"📊 Прогресс: {i+1}/{num_images}, скорость: {speed:.1f} img/s, ETA: {eta/60:.1f} мин")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Сгенерировано {generated}/{num_images} изображений за {elapsed/60:.1f} мин")
        
        if self.class_balancer:
            logger.info(f"📊 Итоговое распределение: {dict(self.class_balancer.generated)}")
            logger.info(f"🎯 Целевое: {self.class_balancer.target_per_class}")


# ============================================
# 6. MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Быстрая генерация синтетических дефектов")
    
    parser.add_argument("--clean_dir", type=str, default="data/256_yolo/balanced_clean_patches/train")
    parser.add_argument("--severstal_images", type=str, default="data/severstal/train_images")
    parser.add_argument("--severstal_csv", type=str, default="data/severstal/train.csv")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_fast")
    parser.add_argument("--num_images", type=int, default=6000)
    parser.add_argument("--use_poisson", action="store_true", default=True)
    parser.add_argument("--no_poisson", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    # Параметры балансировки
    parser.add_argument("--target_class_0", type=int, default=1500)
    parser.add_argument("--target_class_1", type=int, default=1500)
    parser.add_argument("--target_class_2", type=int, default=1500)
    parser.add_argument("--target_class_3", type=int, default=1500)
    
    args = parser.parse_args()
    
    use_poisson = args.use_poisson and not args.no_poisson
    
    set_seed(args.seed)
    
    # Загружаем библиотеку
    defect_lib = SeverstalDefectLibrary(
        Path(args.severstal_images),
        Path(args.severstal_csv)
    )
    
    # Создаем балансировщик
    target_per_class = {
        0: args.target_class_0,
        1: args.target_class_1,
        2: args.target_class_2,
        3: args.target_class_3
    }
    class_balancer = ClassBalancer(target_per_class)
    
    # Создаем композер
    composer = FastDefectComposer(
        Path(args.clean_dir),
        defect_lib,
        class_balancer=class_balancer,
        use_poisson=use_poisson
    )
    
    # Генерируем
    composer.generate_dataset(Path(args.output_dir), args.num_images)


if __name__ == "__main__":
    main()