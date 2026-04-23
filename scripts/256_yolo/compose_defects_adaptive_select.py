#!/usr/bin/env python3
"""
compose_defects_adaptive.py - Адаптивная вставка дефектов с цветовой валидацией
Вставляет дефект только если его цветовая гамма совместима с фоном
"""

import sys
import os
sys.path.insert(0, '/app/scripts')
sys.path.insert(0, '/app/src')

import json
import cv2
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

from utils import set_seed, print_system_info, logger


class BackgroundQualityChecker:
    """Проверяет качество чистого фона - АДАПТИРОВАН под металлические поверхности"""
    
    def __init__(self, strict_mode: bool = False):
        # АДАПТИРОВАННЫЕ параметры под ваши данные (металл)
        if strict_mode:
            self.max_gradient_magnitude = 40
            self.max_texture_score = 0.6
            self.min_flatness = 0.15      # Слишком высоко для металла
        else:
            # Реалистичные параметры для металла
            self.max_gradient_magnitude = 80    # Увеличили
            self.max_texture_score = 0.85       # Увеличили
            self.min_flatness = 0.001           # КРИТИЧЕСКИ ВАЖНО
            self.min_variance = 0.001
        self.min_region_size = 50

    def check_texture_uniformity(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> float:
        """Проверяет однородность текстуры в области"""
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        texture_score = min(1.0, laplacian_var / 500)
        return texture_score
    
    def check_brightness_gradients(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> float:
        """Проверяет наличие сильных перепадов яркости (градиентов)"""
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        mean_gradient = np.mean(gradient_magnitude)
        return mean_gradient
    
    def check_flatness(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> float:
        """Проверка на 'плоские' области без текстуры"""
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        std_h = np.std(hsv[:, :, 0])
        std_s = np.std(hsv[:, :, 1])
        
        flatness_score = (std_h + std_s) / 256
        return flatness_score
    
    def is_background_suitable(self, image: np.ndarray, 
                               region: Tuple[int, int, int, int],
                               verbose: bool = False) -> Tuple[bool, Dict]:
        """Проверяет с учетом специфики металлических поверхностей"""
        
        x1, y1, x2, y2 = region
        region_width = x2 - x1
        region_height = y2 - y1
        
        if region_width < self.min_region_size or region_height < self.min_region_size:
            return False, {"reason": f"region_too_small_{region_width}x{region_height}"}
        
        texture_score = self.check_texture_uniformity(image, region)
        gradient_mag = self.check_brightness_gradients(image, region)
        flatness = self.check_flatness(image, region)
        
        gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        brightness_variance = np.var(gray)
        normalized_variance = min(1.0, brightness_variance / 5000)
        
        reasons = []
        
        # Мягкие критерии для металла
        if texture_score > self.max_texture_score:
            reasons.append(f"old_texture_{texture_score:.2f}")
        
        if gradient_mag > self.max_gradient_magnitude:
            reasons.append(f"high_gradient_{gradient_mag:.1f}")
        
        # Для металла flatness почти всегда 0 - это НОРМАЛЬНО!
        # Проверяем только экстремальные случаи
        if normalized_variance < self.min_variance:
            reasons.append(f"too_uniform_{normalized_variance:.3f}")
        
        is_suitable = len(reasons) == 0
        
        metrics = {
            "texture_score": float(texture_score),
            "gradient_magnitude": float(gradient_mag),
            "flatness": float(flatness),
            "brightness_variance": float(normalized_variance),
            "reasons": reasons,
            "suitable": is_suitable,
            "warning": len(reasons) > 0,
            "is_metal": flatness < 0.01
        }
        
        return is_suitable, metrics


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
    """Разбивает бинарную маску на отдельные связные компоненты"""
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    components = []
    for i in range(1, num_labels):
        component_mask = (labels == i).astype(np.uint8)
        
        if np.sum(component_mask) < 10:
            continue
        
        components.append(component_mask)
    
    return components


def extract_defects_from_mask(image: np.ndarray, mask: np.ndarray, padding: int = 5) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]]:
    """Вырезает ВСЕ связные компоненты из маски как отдельные дефекты"""
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
    """Создает растушеванную маску"""
    if mask.sum() == 0:
        return mask
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    alpha = np.clip(dist / feather_size, 0, 1)
    return alpha


def calculate_color_histogram(image: np.ndarray, mask: np.ndarray, bins: int = 32) -> np.ndarray:
    """Вычисляет нормализованную цветовую гистограмму только внутри маски"""
    if mask.sum() == 0:
        return None
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        mask,
        [bins, bins],
        [0, 180, 0, 256]
    )
    
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    
    return hist.flatten()


def calculate_histogram_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Вычисляет схожесть двух гистограмм"""
    if hist1 is None or hist2 is None:
        return 0.0
    
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
        """Возвращает случайный ОТДЕЛЬНЫЙ дефект"""
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
        
        component_defects = extract_defects_from_mask(image, mask)
        
        if not component_defects:
            return None
        
        roi, roi_mask, bbox = random.choice(component_defects)
        
        return roi, roi_mask, class_id


class AdaptiveDefectComposer:
    """Композер с цветовой валидацией перед вставкой"""
    
    def __init__(self, clean_dir: Path, defect_library: SeverstalDefectLibrary, 
                 similarity_threshold: float = 0.5,
                 filter_backgrounds: bool = True):
        self.clean_images = list(Path(clean_dir).glob("*.png")) + \
                            list(Path(clean_dir).glob("*.jpg"))
        self.defect_library = defect_library
        self.similarity_threshold = similarity_threshold
        self.filter_backgrounds = filter_backgrounds
        self.strict_filtering = False  # Не используем строгий режим
        
        # Используем НЕстрогий режим для металла
        self.background_checker = BackgroundQualityChecker(strict_mode=False) if filter_backgrounds else None
        
        if filter_backgrounds and len(self.clean_images) > 0:
            self.diagnose_backgrounds()
            self.clean_images = self.filter_suitable_backgrounds()
            logger.info(f"📂 После фильтрации осталось {len(self.clean_images)} фонов")
            
            # Если всё равно 0 - используем все фоны
            if len(self.clean_images) == 0:
                logger.warning("⚠️ Использую все фоны (фильтрация отключена)")
                self.filter_backgrounds = False
                self.clean_images = list(Path(clean_dir).glob("*.png")) + \
                                    list(Path(clean_dir).glob("*.jpg"))

    def diagnose_backgrounds(self, num_samples: int = 100):
        """Диагностика: анализируем выборку фонов"""
        logger.info("🔬 ДИАГНОСТИКА: Анализ чистых фонов...")
        
        stats = {
            "total": 0,
            "texture_scores": [],
            "gradient_mags": [],
            "flatness_scores": [],
            "rejection_reasons": {}
        }
        
        sample_images = self.clean_images[:min(num_samples, len(self.clean_images))]
        
        for img_path in sample_images:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            stats["total"] += 1
            h, w = image.shape[:2]
            
            region_size = min(224, w, h)
            x1 = (w - region_size) // 2
            y1 = (h - region_size) // 2
            x2 = x1 + region_size
            y2 = y1 + region_size
            
            is_suitable, metrics = self.background_checker.is_background_suitable(
                image, (x1, y1, x2, y2)
            )
            
            stats["texture_scores"].append(metrics["texture_score"])
            stats["gradient_mags"].append(metrics["gradient_magnitude"])
            stats["flatness_scores"].append(metrics["flatness"])
            
            for reason in metrics.get("reasons", []):
                reason_key = reason.split("_")[0]
                stats["rejection_reasons"][reason_key] = stats["rejection_reasons"].get(reason_key, 0) + 1
        
        if stats["total"] > 0:
            logger.info(f"📊 Проанализировано {stats['total']} фонов:")
            logger.info(f"   Средняя текстура: {np.mean(stats['texture_scores']):.3f}")
            logger.info(f"   Средний градиент: {np.mean(stats['gradient_mags']):.1f}")
            logger.info(f"   Средняя плоскость: {np.mean(stats['flatness_scores']):.3f}")
            
            if stats["rejection_reasons"]:
                logger.info(f"📋 Причины отбраковки:")
                for reason, count in stats["rejection_reasons"].items():
                    logger.info(f"   - {reason}: {count} ({count/stats['total']*100:.1f}%)")
        
        logger.info(f"💾 Диагностика завершена")

    def filter_suitable_backgrounds(self, min_region_size: int = 224) -> List[Path]:
        """Отбирает подходящие фоны - адаптивный режим"""
        suitable_images = []
        
        logger.info("🔍 Фильтрация чистых фонов (адаптивный режим для металла)...")
        
        for img_path in tqdm(self.clean_images, desc="Filtering"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            if h < min_region_size or w < min_region_size:
                continue
            
            x1 = (w - min_region_size) // 2
            y1 = (h - min_region_size) // 2
            x2 = x1 + min_region_size
            y2 = y1 + min_region_size
            
            is_suitable, metrics = self.background_checker.is_background_suitable(
                image, (x1, y1, x2, y2)
            )
            
            # Для металла пропускаем даже с предупреждениями
            if is_suitable or metrics.get("is_metal", False):
                suitable_images.append(img_path)
        
        logger.info(f"✅ Отобрано {len(suitable_images)} из {len(self.clean_images)} фонов")
        return suitable_images
    
    def find_suitable_region(self, background: np.ndarray, roi_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Ищет подходящую область для вставки дефекта"""
        h_bg, w_bg = background.shape[:2]
        h_roi, w_roi = roi_shape
        
        margin = 20
        max_attempts = 30
        
        for _ in range(max_attempts):
            try:
                x1 = random.randint(margin, w_bg - w_roi - margin)
                y1 = random.randint(margin, h_bg - h_roi - margin)
                x2 = x1 + w_roi
                y2 = y1 + h_roi
                
                return (x1, y1, x2, y2)
            except:
                continue
        
        return None
    
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
            v_ratio = 1.0 + (v_ratio - 1.0) * 0.8
            roi_hsv[:, :, 2] = np.clip(roi_v * v_ratio, 0, 255)
        
        roi_adjusted = cv2.cvtColor(np.clip(roi_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        mask_3d = np.stack([roi_mask] * 3, axis=-1)
        roi_final = roi * (1 - mask_3d) + roi_adjusted * mask_3d
        
        return roi_final.astype(np.uint8)
    
    def is_color_compatible(self, roi: np.ndarray, roi_mask: np.ndarray, 
                            background: np.ndarray, target_region: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = target_region
        bg_region = background[y1:y2, x1:x2]
        
        defect_hist = calculate_color_histogram(roi, roi_mask)
        bg_mask = np.ones(bg_region.shape[:2], dtype=np.uint8)
        bg_hist = calculate_color_histogram(bg_region, bg_mask)
        
        if defect_hist is None or bg_hist is None:
            return False
        
        similarity = calculate_histogram_similarity(defect_hist, bg_hist)
        
        if similarity < self.similarity_threshold:
            return False
        
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(bg_region, cv2.COLOR_BGR2GRAY)
        
        roi_mean = np.mean(roi_gray[roi_mask > 0])
        bg_mean = np.mean(bg_gray)
        
        if roi_mean > 0:
            brightness_ratio = max(roi_mean, bg_mean) / min(roi_mean, bg_mean)
            if brightness_ratio > 2.5:
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
        
        max_attempts = num_defects * 15
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
            try:
                x1 = random.randint(margin, w_bg - w_roi - margin)
                y1 = random.randint(margin, h_bg - h_roi - margin)
                x2 = x1 + w_roi
                y2 = y1 + h_roi
            except:
                continue
            
            # Проверка на пересечение
            overlaps = False
            for bx1, by1, bx2, by2 in inserted_boxes:
                if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):
                    overlaps = True
                    break
            
            if overlaps:
                continue
            
            if not self.is_color_compatible(roi, roi_mask, background, (x1, y1, x2, y2)):
                continue
            
            roi = self.match_lighting(roi, roi_mask, background, (x1, y1, x2, y2))
            
            alpha = create_alpha_mask(roi_mask, feather_size=5)
            alpha_3d = np.stack([alpha] * 3, axis=-1)
            
            roi_float = roi.astype(np.float32)
            
            bg_region = background[y1:y2, x1:x2].astype(np.float32)
            bg_blurred = cv2.GaussianBlur(bg_region, (21, 21), 0)
            
            bg_final = bg_region * (1 - alpha_3d) + bg_blurred * alpha_3d
            blended = roi_float * alpha_3d + bg_final * (1 - alpha_3d)
            
            background[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
            
            inserted_boxes.append((x1, y1, x2, y2))
            
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
        
        generated = 0
        for i in tqdm(range(num_images), desc="Composing"):
            bg_path = random.choice(self.clean_images)
            img, anns = self.compose_single_image(bg_path, max_defects=random.randint(2, 5))
            
            if img and anns:
                filename = f"syn_{generated:06d}"
                img.save(output_images / f"{filename}.png", "PNG")
                
                with open(output_labels / f"{filename}.txt", 'w') as f:
                    for ann in anns:
                        f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                               f"{ann['width']:.6f} {ann['height']:.6f}\n")
                generated += 1
        
        logger.info(f"✅ Готово! Сгенерировано {generated} изображений в {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, default="data/256_yolo/balanced_clean_patches/train")
    parser.add_argument("--severstal_images", type=str, default="data/severstal/train_images")
    parser.add_argument("--severstal_csv", type=str, default="data/severstal/train.csv")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_adaptive_final")
    parser.add_argument("--num_images", type=int, default=10000)
    parser.add_argument("--similarity", type=float, default=0.5, help="Порог схожести гистограмм (0.3-0.6)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter_backgrounds", action="store_true", default=False, help="Включить фильтрацию фонов")
    
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
        similarity_threshold=args.similarity,
        filter_backgrounds=args.filter_backgrounds
    )
    
    if len(composer.clean_images) > 0:
        composer.generate_dataset(Path(args.output_dir), args.num_images)
    else:
        logger.error("❌ Нет изображений для генерации!")


if __name__ == "__main__":
    main()