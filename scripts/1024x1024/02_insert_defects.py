#!/usr/bin/env python3
"""
02_insert_defects.py - Вставка дефектов в синтетические фоны
"""

import cv2
import numpy as np
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from utils import set_seed, load_defects_with_masks, save_json, logger


class DefectInserter:
    """Вставка дефектов с гармонизацией и невидимыми границами"""
    
    def __init__(
        self,
        defects_per_image: Tuple[int, int] = (1, 3),
        scale_range: Tuple[float, float] = (0.7, 1.5),
        blur_kernel: int = 5,
        alpha_feather: int = 15,
        base_scale_factor: float = 4.0,
        harmonize_strength: float = 0.7,
        boundary_noise: float = 0.015
    ):
        self.defects_per_image = defects_per_image
        self.scale_range = scale_range
        self.blur_kernel = blur_kernel
        self.alpha_feather = alpha_feather
        self.base_scale_factor = base_scale_factor
        self.harmonize_strength = harmonize_strength
        self.boundary_noise = boundary_noise
    
    def _match_histograms(
        self,
        source: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Приведение гистограммы source к target в области маски"""
        mask_bool = mask > 127
        
        if np.sum(mask_bool) < 100:
            return source
        
        result = source.copy()
        
        for c in range(3):
            src_channel = source[:, :, c]
            tgt_channel = target[:, :, c]
            
            src_pixels = src_channel[mask_bool]
            tgt_pixels = tgt_channel[mask_bool]
            
            if len(src_pixels) == 0 or len(tgt_pixels) == 0:
                continue
            
            src_mean = np.mean(src_pixels)
            src_std = np.std(src_pixels)
            tgt_mean = np.mean(tgt_pixels)
            tgt_std = np.std(tgt_pixels)
            
            target_mean = tgt_mean * self.harmonize_strength + src_mean * (1 - self.harmonize_strength)
            target_std = tgt_std * self.harmonize_strength + src_std * (1 - self.harmonize_strength)
            
            normalized = (src_channel - src_mean) / (src_std + 1e-6)
            harmonized = normalized * target_std + target_mean
            
            result[:, :, c][mask_bool] = np.clip(harmonized[mask_bool], 0, 255)
        
        return result.astype(np.uint8)
    
    def _add_boundary_texture(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        x: int,
        y: int
    ) -> np.ndarray:
        """
        Добавление микро-текстуры на границе.
        x, y - координаты вставки дефекта
        """
        h, w = mask.shape[:2]
        
        # Создание маски границы (только в области дефекта)
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        mask_eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = (mask_dilated - mask_eroded) > 0
        
        if not np.any(boundary):
            return image
        
        # Генерация шума
        noise = np.random.normal(0, self.boundary_noise * 255, (h, w, 3))
        
        # Вырезаем ROI из изображения
        roi = image[y:y+h, x:x+w].astype(np.float32)
        
        # Применяем шум только к границе
        boundary_3ch = np.stack([boundary] * 3, axis=-1)
        roi[boundary_3ch] = np.clip(
            roi[boundary_3ch] + noise[boundary_3ch],
            0, 255
        )
        
        # Возвращаем ROI обратно
        result = image.copy()
        result[y:y+h, x:x+w] = roi.astype(np.uint8)
        
        return result           
    
    def _create_alpha_mask(self, mask: np.ndarray) -> np.ndarray:
        """Создание альфа-канала с размытыми краями"""
        mask_float = mask.astype(np.float32) / 255.0
        mask_float = cv2.GaussianBlur(mask_float, (self.alpha_feather, self.alpha_feather), 0)
        mask_float = np.power(mask_float, 0.8)
        return np.stack([mask_float] * 3, axis=-1)
    
    def _resize_defect(
        self,
        defect_img: np.ndarray,
        defect_mask: np.ndarray,
        bg_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Масштабирование дефекта"""
        h, w = defect_img.shape[:2]
        bg_h, bg_w = bg_size
        
        total_scale = self.base_scale_factor * random.uniform(*self.scale_range)
        
        new_w = int(w * total_scale)
        new_h = int(h * total_scale)
        
        if new_w > bg_w or new_h > bg_h:
            scale_correction = min(bg_w / new_w, bg_h / new_h) * 0.95
            new_w = int(new_w * scale_correction)
            new_h = int(new_h * scale_correction)
        
        defect_img = cv2.resize(defect_img, (new_w, new_h))
        defect_mask = cv2.resize(defect_mask, (new_w, new_h))
        
        return defect_img, defect_mask
    
    def insert_defect(
        self,
        background: np.ndarray,
        defect_img: np.ndarray,
        defect_mask: np.ndarray
    ) -> Tuple[np.ndarray, List[int]]:
        """Вставка одного дефекта"""
        
        bg_h, bg_w = background.shape[:2]
        defect_mask = (defect_mask > 127).astype(np.uint8) * 255
        
        defect_img, defect_mask = self._resize_defect(defect_img, defect_mask, (bg_h, bg_w))
        h, w = defect_img.shape[:2]
        
        max_x = max(0, bg_w - w)
        max_y = max(0, bg_h - h)
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        
        roi = background[y:y+h, x:x+w].copy()
        defect_harmonized = self._match_histograms(defect_img, roi, defect_mask)
        
        if self.blur_kernel > 0:
            defect_harmonized = cv2.GaussianBlur(
                defect_harmonized,
                (self.blur_kernel, self.blur_kernel),
                0
            )
        
        alpha = self._create_alpha_mask(defect_mask)
        
        result = background.copy().astype(np.float32)
        roi_float = result[y:y+h, x:x+w]
        blended = roi_float * (1 - alpha) + defect_harmonized.astype(np.float32) * alpha
        result[y:y+h, x:x+w] = blended
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 🔧 Передаем координаты
        result = self._add_boundary_texture(result, defect_mask, x, y)
        
        bbox = [x, y, w, h]
        return result, bbox 
    
    def process_image(
        self,
        background_path: Path,
        defect_pool: List[Dict],
        output_dir: Path
    ) -> Optional[Dict]:
        """Обработка одного фона"""
        
        background = cv2.imread(str(background_path))
        if background is None:
            return None
        
        bg_h, bg_w = background.shape[:2]
        num_defects = random.randint(*self.defects_per_image)
        selected_defects = random.sample(defect_pool, min(num_defects, len(defect_pool)))
        
        annotation = {
            "image": background_path.name,
            "width": bg_w,
            "height": bg_h,
            "defects": []
        }
        
        full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        
        for defect_info in selected_defects:
            defect_img = cv2.imread(defect_info["image"])
            defect_mask = cv2.imread(defect_info["mask"], cv2.IMREAD_GRAYSCALE)
            
            if defect_img is None or defect_mask is None:
                continue
            
            background, bbox = self.insert_defect(background, defect_img, defect_mask)
            x, y, w, h = bbox
            
            mask_resized = cv2.resize(defect_mask, (w, h))
            full_mask[y:y+h, x:x+w] = np.maximum(full_mask[y:y+h, x:x+w], mask_resized)
            
            annotation["defects"].append({
                "bbox": bbox,
                "source": defect_info["name"]
            })
        
        img_output = output_dir / "images" / background_path.name
        mask_output = output_dir / "masks" / background_path.name
        img_output.parent.mkdir(parents=True, exist_ok=True)
        mask_output.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(img_output), background)
        cv2.imwrite(str(mask_output), full_mask)
        
        return annotation


def main():
    parser = argparse.ArgumentParser(description="Вставка дефектов в фоны")
    parser.add_argument("--backgrounds_dir", type=str, default="/app/results/synthetic_backgrounds")
    parser.add_argument("--defects_dir", type=str, default="/app/data/defects")
    parser.add_argument("--output_dir", type=str, default="/app/results/final_dataset")
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--defects_min", type=int, default=1)
    parser.add_argument("--defects_max", type=int, default=2)
    parser.add_argument("--scale_min", type=float, default=0.8)
    parser.add_argument("--scale_max", type=float, default=1.3)
    parser.add_argument("--blur", type=int, default=5)
    parser.add_argument("--feather", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_scale", type=float, default=4.0)
    parser.add_argument("--harmonize", type=float, default=0.9)
    parser.add_argument("--boundary_noise", type=float, default=0.02)

    args = parser.parse_args()
    set_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("🔧 ВСТАВКА ДЕФЕКТОВ")
    logger.info("=" * 60)
    logger.info(f"Базовый масштаб: {args.base_scale}x")
    logger.info(f"Вариация масштаба: {args.scale_min}-{args.scale_max}")
    logger.info(f"Гармонизация: {args.harmonize}")
    logger.info(f"Растушевка: {args.feather}px")
    
    bg_files = sorted(Path(args.backgrounds_dir).glob("*.png"))
    logger.info(f"📂 Загружено {len(bg_files)} фонов")
    
    defects = load_defects_with_masks(args.defects_dir)
    logger.info(f"📂 Загружено {len(defects)} дефектов")
    
    if not bg_files or not defects:
        return
    
    output_dir = Path(args.output_dir)
    
    if args.num_images:
        if args.num_images < len(bg_files):
            bg_files = random.sample(bg_files, args.num_images)
        else:
            multiplier = args.num_images // len(bg_files) + 1
            bg_files = (bg_files * multiplier)[:args.num_images]
    
    inserter = DefectInserter(
        defects_per_image=(args.defects_min, args.defects_max),
        scale_range=(args.scale_min, args.scale_max),
        blur_kernel=args.blur,
        alpha_feather=args.feather,
        base_scale_factor=args.base_scale,
        harmonize_strength=args.harmonize,
        boundary_noise=args.boundary_noise
    )
    
    annotations = []
    for bg_file in tqdm(bg_files, desc="Вставка дефектов"):
        ann = inserter.process_image(bg_file, defects, output_dir)
        if ann:
            annotations.append(ann)
    
    save_json(annotations, output_dir / "annotations.json")
    
    total_defects = sum(len(a["defects"]) for a in annotations)
    logger.info(f"\n✅ Обработано {len(annotations)} изображений")
    logger.info(f"✅ Вставлено {total_defects} дефектов")
    logger.info(f"📁 Результаты: {output_dir}")


if __name__ == "__main__":
    main()