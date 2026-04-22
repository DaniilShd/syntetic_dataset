#!/usr/bin/env python3
"""
compose_defects_advanced.py - Точная вставка дефектов по контуру
Использует: RLE маски, полигоны, alpha blending с размытием краев
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
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import json
from scipy import ndimage

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


def mask_to_polygon(mask: np.ndarray, epsilon_factor: float = 0.005) -> List[Tuple[int, int]]:
    """Конвертирует бинарную маску в полигон (контур)"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Берем самый большой контур
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Аппроксимируем для сглаживания
    epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Конвертируем в список точек
    return [(int(pt[0][0]), int(pt[0][1])) for pt in approx]


def extract_defect_by_polygon(image: np.ndarray, mask: np.ndarray, padding: int = 5) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Вырезает дефект по маске с минимальным ограничивающим прямоугольником
    
    Returns:
        roi: вырезанное изображение дефекта
        roi_mask: вырезанная маска
        bbox: (x1, y1, x2, y2) координаты на исходном изображении
    """
    # Находим ненулевые пиксели в маске
    y_indices, x_indices = np.where(mask > 0)
    
    if len(x_indices) == 0:
        return None, None, (0, 0, 0, 0)
    
    # Ограничивающий прямоугольник с паддингом
    x1 = max(0, np.min(x_indices) - padding)
    y1 = max(0, np.min(y_indices) - padding)
    x2 = min(image.shape[1], np.max(x_indices) + padding)
    y2 = min(image.shape[0], np.max(y_indices) + padding)
    
    # Вырезаем ROI и маску
    roi = image[y1:y2, x1:x2].copy()
    roi_mask = mask[y1:y2, x1:x2].copy()
    
    return roi, roi_mask, (x1, y1, x2, y2)


def create_alpha_mask(mask: np.ndarray, feather_size: int = 5) -> np.ndarray:
    """Создает сильно растушеванную маску без жестких границ"""
    if mask.sum() == 0:
        return mask
    # Distance Transform: каждый пиксель маски получает значение расстояния до границы
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Нормализуем и обрезаем, чтобы создать градиентный переход на границе
    alpha = np.clip(dist / feather_size, 0, 1)
    return alpha


class SeverstalDefectLibrary:
    """Библиотека дефектов из Severstal датасета"""
    
    def __init__(self, images_dir: Path, csv_path: Path):
        self.images_dir = Path(images_dir)
        self.df = pd.read_csv(csv_path)
        
        # Группируем по ImageId (один файл может иметь несколько дефектов)
        self.defect_groups = self.df.groupby('ImageId')
        
        logger.info(f"📂 Загружено {len(self.df)} дефектов из {len(self.defect_groups)} изображений")
        
        # Кэш для часто используемых дефектов
        self.cache = {}
    
    def get_random_defect(self) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Возвращает случайный дефект: (roi, mask, class_id)
        """
        # Выбираем случайное изображение
        image_id = random.choice(list(self.defect_groups.groups.keys()))
        
        # Проверяем кэш
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
        
        # Выбираем случайный дефект из этого изображения
        row = defects.sample(1).iloc[0]
        
        # Декодируем маску
        mask = rle_to_mask(row['EncodedPixels'], height=256, width=1600)
        
        if mask.sum() == 0:
            return None
        
        # Класс дефекта (1-4)
        class_id = int(row['ClassId']) - 1  # Severstal классы 1-4 → 0-3
        
        # Вырезаем ROI
        roi, roi_mask, bbox = extract_defect_by_polygon(image, mask)
        
        if roi is None or roi.size == 0:
            return None
        
        return roi, roi_mask, class_id


class AdvancedDefectComposer:
    """Композер с точной вставкой по маске"""
    
    def __init__(self, clean_dir: Path, defect_library: SeverstalDefectLibrary):
        self.clean_images = list(Path(clean_dir).glob("*.png")) + \
                            list(Path(clean_dir).glob("*.jpg"))
        self.defect_library = defect_library
        
        logger.info(f"📂 Загружено {len(self.clean_images)} чистых фонов")
    
    def match_lighting(self, roi: np.ndarray, roi_mask: np.ndarray, background: np.ndarray, 
                   target_region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Мягкая адаптация яркости ROI под фон
        Сохраняет текстуру и цвет дефекта, убирает только резкий перепад
        """
        x1, y1, x2, y2 = target_region
        bg_region = background[y1:y2, x1:x2]
        
        if bg_region.size == 0 or roi_mask.sum() == 0:
            return roi
        
        # Конвертируем в HSV — лучше для сохранения цвета
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        bg_hsv = cv2.cvtColor(bg_region, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Маска для статистики
        mask_1d = roi_mask.flatten() > 0
        
        # Берем ТОЛЬКО яркость (V-канал) для адаптации
        # Hue и Saturation не трогаем — сохраняем цвет дефекта!
        roi_v = roi_hsv[:, :, 2]
        bg_v = bg_hsv[:, :, 2]
        
        # Средняя яркость внутри маски и в области фона
        roi_v_mean = np.mean(roi_v[roi_mask > 0])
        bg_v_mean = np.mean(bg_v)
        
        # Коэффициент коррекции (мягкий — только 40% от разницы)
        if roi_v_mean > 0:
            v_ratio = bg_v_mean / roi_v_mean
            v_ratio = 1.0 + (v_ratio - 1.0) * 0.7  # только 40% коррекции
        
            # Применяем коррекцию яркости
            roi_hsv[:, :, 2] = np.clip(roi_v * v_ratio, 0, 255)

                        # После коррекции яркости добавьте:
            roi_s = roi_hsv[:, :, 1]
            bg_s = bg_hsv[:, :, 1]
            roi_s_mean = np.mean(roi_s[roi_mask > 0])
            bg_s_mean = np.mean(bg_s)

            if roi_s_mean > 0:
                s_ratio = bg_s_mean / roi_s_mean
                s_ratio = 1.0 + (s_ratio - 1.0) * 0.5  # 50% коррекции насыщенности
                roi_hsv[:, :, 1] = np.clip(roi_s * s_ratio, 0, 255)
        
        # Обратно в BGR
        roi_adjusted = cv2.cvtColor(np.clip(roi_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Сохраняем оригинальную текстуру там, где маска = 0
        mask_3d = np.stack([roi_mask] * 3, axis=-1)
        roi_final = roi * (1 - mask_3d) + roi_adjusted * mask_3d
        
        return roi_final.astype(np.uint8)
    
    def compose_single_image(self, bg_path: Path, max_defects: int = 3) -> Tuple[Image.Image, List[Dict]]:
        """Создает одно синтетическое изображение"""
        
        background = cv2.imread(str(bg_path))
        if background is None:
            return None, []
        
        h_bg, w_bg = background.shape[:2]
        
        annotations = []
        inserted_boxes = []  # Список уже вставленных bbox (x1, y1, x2, y2)
        
        num_defects = random.randint(1, max_defects)
        
        max_attempts = num_defects * 10  # Увеличиваем попытки из-за проверки пересечений
        attempts = 0
        inserted = 0
        
        while inserted < num_defects and attempts < max_attempts:
            attempts += 1
            
            # Получаем случайный дефект
            result = self.defect_library.get_random_defect()
            if result is None:
                continue
            
            roi, roi_mask, class_id = result
            
            # Пропускаем слишком большие дефекты
            if roi.shape[0] >= h_bg - 20 or roi.shape[1] >= w_bg - 20:
                continue
            
            h_roi, w_roi = roi.shape[:2]
            
            # Случайная позиция
            margin = 10
            x1 = random.randint(margin, w_bg - w_roi - margin)
            y1 = random.randint(margin, h_bg - h_roi - margin)
            x2 = x1 + w_roi
            y2 = y1 + h_roi
            
            # ✅ Проверка на пересечение с уже вставленными дефектами
            overlaps = False
            for bx1, by1, bx2, by2 in inserted_boxes:
                # Проверяем пересечение двух прямоугольников
                if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):
                    overlaps = True
                    break
            
            if overlaps:
                continue  # Пропускаем эту позицию, пробуем другую
            
            # Адаптируем освещение
            roi = self.match_lighting(roi, roi_mask, background, (x1, y1, x2, y2))
            
            # Создаем альфа-канал с размытием
            alpha = create_alpha_mask(roi_mask, feather_size=5)
            alpha_3d = np.stack([alpha] * 3, axis=-1)
            
            roi_float = roi.astype(np.float32)
            
            # Размываем фон в месте вставки для плавного перехода
            bg_region = background[y1:y2, x1:x2].astype(np.float32)
            bg_blurred = cv2.GaussianBlur(bg_region, (21, 21), 0)
            
            # Смешиваем
            bg_final = bg_region * (1 - alpha_3d) + bg_blurred * alpha_3d
            blended = roi_float * alpha_3d + bg_final * (1 - alpha_3d)
            
            background[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
            
            # Сохраняем bbox вставленного дефекта
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
                
                # Сохраняем YOLO-разметку
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
    parser.add_argument("--output_dir", type=str, default="data/synthetic_severstal_composed")
    parser.add_argument("--num_images", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    print_system_info()
    
    # Создаем библиотеку дефектов
    defect_lib = SeverstalDefectLibrary(
        Path(args.severstal_images),
        Path(args.severstal_csv)
    )
    
    # Создаем композер
    composer = AdvancedDefectComposer(
        Path(args.clean_dir),
        defect_lib
    )
    
    # Генерируем датасет
    composer.generate_dataset(Path(args.output_dir), args.num_images)


if __name__ == "__main__":
    main()