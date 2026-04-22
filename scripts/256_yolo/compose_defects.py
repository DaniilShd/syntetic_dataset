#!/usr/bin/env python3
"""
compose_defects.py - Бесшовная вставка дефектов на чистые фоны
Использует: Poisson Blending + Adaptive Augmentation + Multi-class Support
"""

import sys
import os
sys.path.insert(0, '/app/scripts')
sys.path.insert(0, '/app/src')

import cv2
import random
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import json

# Локальные утилиты
from utils import set_seed, print_system_info, logger

class YOLODatasetHandler:
    """Обработчик YOLO-разметки для дефектов"""
    
    @staticmethod
    def load_annotation(label_path: Path) -> List[Dict]:
        if not label_path.exists():
            return []
        annotations = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    annotations.append({
                        'class': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })
        return annotations

    @staticmethod
    def save_annotation(path: Path, annotations: List[Dict]):
        with open(path, 'w') as f:
            for ann in annotations:
                f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                       f"{ann['width']:.6f} {ann['height']:.6f}\n")

class SeamlessDefectComposer:
    def __init__(self, config):
        self.config = config
        self.clean_images = list(Path(config.clean_dir).glob("*.png")) + \
                            list(Path(config.clean_dir).glob("*.jpg"))
        
        # Кэш для дефектов (чтобы не грузить с диска каждый раз)
        self.defects_cache = self._load_defects_library(config.defects_dir)
        
        logger.info(f"📂 Загружено {len(self.clean_images)} чистых фонов")
        logger.info(f"📂 Загружено {len(self.defects_cache)} дефектов в библиотеку")

    def _load_defects_library(self, defects_dir: Path) -> List[Dict]:
        """Сканирует датасет с дефектами и сохраняет их в кэш"""
        library = []
        images_dir = Path(defects_dir) / "images"
        labels_dir = Path(defects_dir) / "labels"
        
        for img_path in images_dir.glob("*.png"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
                
            annotations = YOLODatasetHandler.load_annotation(label_path)
            if not annotations:
                continue
                
            # Загружаем изображение и храним в оперативной памяти
            img = cv2.imread(str(img_path))
            if img is not None:
                library.append({
                    "image": img,
                    "annotations": annotations,
                    "path": img_path
                })
        return library

    def extract_defect_roi(self, defect_data: Dict, ann_idx: int) -> Tuple[np.ndarray, int, Tuple[float, float]]:
        """Вырезает конкретный дефект из изображения с небольшим контекстом"""
        img = defect_data["image"]
        ann = defect_data["annotations"][ann_idx]
        h, w = img.shape[:2]
        
        # Переводим относительные координаты в абсолютные
        x = int(ann['x_center'] * w)
        y = int(ann['y_center'] * h)
        bw = int(ann['width'] * w)
        bh = int(ann['height'] * h)
        
        # Добавляем контекст (padding) 15% с каждой стороны
        padding_w = int(bw * 0.3)
        padding_h = int(bh * 0.3)
        
        x1 = max(0, x - bw//2 - padding_w)
        y1 = max(0, y - bh//2 - padding_h)
        x2 = min(w, x + bw//2 + padding_w)
        y2 = min(h, y + bh//2 + padding_h)
        
        roi = img[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None, -1, (0, 0)
            
        # Возвращаем класс, ROI и оригинальный центр bbox
        return roi, ann['class'], (x, y)

    def create_poisson_mask(self, roi: np.ndarray) -> np.ndarray:
        """Создает маску для бесшовного клонирования на основе контуров ROI"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Морфологические операции для сглаживания маски
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (5,5), 0)
        
        return mask

    def seamless_insert(self, background: np.ndarray, roi: np.ndarray, 
                       center_x: int, center_y: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Вставка ROI на фон с использованием Poisson Blending.
        Возвращает изображение и координаты вставленного объекта.
        """
        h_bg, w_bg = background.shape[:2]
        h_roi, w_roi = roi.shape[:2]
        
        # Рассчитываем координаты вставки
        x1 = max(0, center_x - w_roi//2)
        y1 = max(0, center_y - h_roi//2)
        x2 = min(w_bg, x1 + w_roi)
        y2 = min(h_bg, y1 + h_roi)
        
        # Корректируем размер ROI если выходит за границы
        roi_cropped = roi[0:y2-y1, 0:x2-x1]
        if roi_cropped.size == 0:
            return background, (0,0,0,0)
            
        try:
            mask = self.create_poisson_mask(roi_cropped)
            
            # Нормальное смешивание (обычное копирование) если Poisson выдает ошибку
            # Определяем центр для смешивания
            center = (x1 + roi_cropped.shape[1]//2, y1 + roi_cropped.shape[0]//2)
            
            # Poisson Blending
            try:
                blended = cv2.seamlessClone(
                    roi_cropped, background, mask, center, cv2.NORMAL_CLONE
                )
            except cv2.error:
                # Fallback: обычное копирование с размытием границ
                blended = background.copy()
                blended[y1:y2, x1:x2] = roi_cropped
                # Легкое размытие по границе для сглаживания
                mask_blur = cv2.GaussianBlur(mask, (7,7), 0) / 255.0
                mask_blur = np.stack([mask_blur]*3, axis=-1)
                blended[y1:y2, x1:x2] = (blended[y1:y2, x1:x2] * mask_blur + 
                                         background[y1:y2, x1:x2] * (1 - mask_blur)).astype(np.uint8)
                
            return blended, (x1, y1, x2, y2)
            
        except Exception as e:
            logger.warning(f"Seamless clone failed: {e}")
            return background, (0,0,0,0)

    def adjust_to_background(self, roi: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Адаптация цвета и контраста ROI под статистику фона"""
        # 1. Приведение средней яркости
        bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        if np.std(roi_gray) > 0:
            roi_mean = np.mean(roi_gray)
            bg_mean = np.mean(bg_gray)
            roi_adjusted = roi.astype(np.float32) * (bg_mean / (roi_mean + 1e-5))
            roi = np.clip(roi_adjusted, 0, 255).astype(np.uint8)
        
        # 2. Добавление легкого шума для маскировки границ
        if random.random() < 0.5:
            noise = np.random.normal(0, 2, roi.shape).astype(np.int16)
            roi = np.clip(roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
        return roi

    def compose_single_image(self, bg_path: Path, num_defects: int = 3) -> Tuple[Image.Image, List[Dict]]:
        """Создает одно синтетическое изображение с несколькими дефектами"""
        
        # Загружаем фон
        background = cv2.imread(str(bg_path))
        if background is None:
            return None, []
            
        h_bg, w_bg = background.shape[:2]
        new_annotations = []
        
        # Выбираем случайные дефекты из библиотеки
        selected_defects = random.sample(self.defects_cache, min(num_defects, len(self.defects_cache)))
        
        for defect_data in selected_defects:
            if not defect_data["annotations"]:
                continue
            ann_idx = random.randint(0, len(defect_data["annotations"]) - 1)
            
            roi, class_id, (src_cx, src_cy) = self.extract_defect_roi(defect_data, ann_idx)
            if roi is None:
                continue
                
            roi = self.adjust_to_background(roi, background)
            
            # Случайное масштабирование (80% - 120%)
            scale = random.uniform(0.8, 1.2)
            new_w = int(roi.shape[1] * scale)
            new_h = int(roi.shape[0] * scale)
            
            margin = 20
            # Проверяем, влезает ли дефект на фон
            if new_w >= w_bg - 2 * margin or new_h >= h_bg - 2 * margin:
                continue
                
            roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            min_x = margin + new_w // 2
            max_x = w_bg - margin - new_w // 2
            min_y = margin + new_h // 2
            max_y = h_bg - margin - new_h // 2
            
            if min_x >= max_x or min_y >= max_y:
                continue
                
            center_x = random.randint(min_x, max_x)
            center_y = random.randint(min_y, max_y)

            # Бесшовная вставка
            background, (x1, y1, x2, y2) = self.seamless_insert(background, roi, center_x, center_y)
            
            if x2 > x1 and y2 > y1:
                # Формируем YOLO-аннотацию для вставленного объекта
                ann = {
                    'class': class_id,
                    'x_center': ((x1 + x2) / 2) / w_bg,
                    'y_center': ((y1 + y2) / 2) / h_bg,
                    'width': (x2 - x1) / w_bg,
                    'height': (y2 - y1) / h_bg
                }
                new_annotations.append(ann)
        
        # Финальные аугментации всего изображения
        result_img = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        
        if self.config.enable_augmentation:
            # Легкое размытие для слияния текстур
            if random.random() < 0.3:
                result_img = result_img.filter(ImageFilter.GaussianBlur(radius=0.8))
            # Яркость/Контраст
            if random.random() < 0.5:
                result_img = ImageEnhance.Brightness(result_img).enhance(random.uniform(0.9, 1.1))
            if random.random() < 0.5:
                result_img = ImageEnhance.Contrast(result_img).enhance(random.uniform(0.9, 1.1))
                
        return result_img, new_annotations

    def generate_dataset(self, output_dir: Path, total_images: int):
        """Генерирует полный датасет"""
        output_images = output_dir / "images"
        output_labels = output_dir / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Старт генерации {total_images} синтетических изображений...")
        
        for i in tqdm(range(total_images), desc="Composing"):
            # Выбираем случайный чистый фон
            bg_path = random.choice(self.clean_images)
            
            # Количество дефектов (1-5)
            num_def = random.randint(1, 5)
            
            img, anns = self.compose_single_image(bg_path, num_def)
            
            if img and anns:
                filename = f"syn_{i:06d}.png"
                img.save(output_images / filename, "PNG")
                YOLODatasetHandler.save_annotation(output_labels / f"syn_{i:06d}.txt", anns)
            else:
                logger.warning(f"Пропуск {i} (не удалось вставить дефекты)")
                
        logger.info(f"✅ Генерация завершена! Результаты в {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Супер-качественный композер дефектов")
    parser.add_argument("--clean_dir", type=str, default="data/dataset_synthetic/clean_patches",
                       help="Директория с ЧИСТЫМИ фонами (патчами)")
    parser.add_argument("--defects_dir", type=str, default="data/256_yolo/balanced_defect_patches/train",
                       help="Директория с РЕАЛЬНЫМИ дефектами (images/ + labels/)")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_composed",
                       help="Куда сохранять результат")
    parser.add_argument("--num_images", type=int, default=5000,
                       help="Сколько всего синтетических изображений создать")
    parser.add_argument("--no_aug", action="store_true",
                       help="Отключить финальные аугментации")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    set_seed(args.seed)
    print_system_info()
    
    # Простой класс-конфиг
    class Config:
        pass
    config = Config()
    config.clean_dir = args.clean_dir
    config.defects_dir = args.defects_dir
    config.enable_augmentation = not args.no_aug
    
    composer = SeamlessDefectComposer(config)
    composer.generate_dataset(Path(args.output_dir), args.num_images)

if __name__ == "__main__":
    main()