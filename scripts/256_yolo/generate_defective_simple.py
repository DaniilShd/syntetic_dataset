#!/usr/bin/env python3
"""
generate_defective_simple.py - ПРОСТОЙ inpainting без галлюцинаций
"""

import sys
import os
sys.path.insert(0, '/app/scripts')
sys.path.insert(0, '/app/src')

import torch
import random
import argparse
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import numpy as np
import json

from diffusers import StableDiffusionXLInpaintPipeline, EulerAncestralDiscreteScheduler

from config import GenerationConfig
from utils import set_seed, print_system_info, logger


class YOLODatasetHandler:
    """Обработчик YOLO-разметки"""
    
    def __init__(self, images_dir: Path, labels_dir: Path):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
    
    def load_annotation(self, image_name: str) -> List[Dict]:
        label_path = self.labels_dir / f"{Path(image_name).stem}.txt"
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
    
    def save_annotation(self, image_name: str, annotations: List[Dict]):
        label_path = self.labels_dir / f"{Path(image_name).stem}.txt"
        with open(label_path, 'w') as f:
            for ann in annotations:
                f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                       f"{ann['width']:.6f} {ann['height']:.6f}\n")
    
    @staticmethod
    def flip_annotation_horizontal(annotations: List[Dict]) -> List[Dict]:
        flipped = []
        for ann in annotations:
            flipped.append({
                'class': ann['class'],
                'x_center': 1.0 - ann['x_center'],
                'y_center': ann['y_center'],
                'width': ann['width'],
                'height': ann['height']
            })
        return flipped


class SimpleInpainter:
    """МАКСИМАЛЬНО ПРОСТОЙ inpainting — только текстура, без изменения формы"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.resize_to = config.resize_to
        
        # ТОЛЬКО ОДИН ПРОМПТ — только текстура
        self.prompt = "realistic metal surface texture, industrial steel, same shape, detailed"
        self.negative_prompt = "different shape, new object, cartoon, color noise, rainbow, abstract"
        
        logger.info(f"🔄 Загрузка ПРОСТОГО inpainting...")
        self._load_pipeline()
        logger.info(f"✅ Готово")
    
    def _load_pipeline(self):
        """Загрузка БЕЗ ControlNet и IP-Adapter"""
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=self.config.cache_dir
        ).to(self.config.device)
        
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing()
    
    def create_defect_mask(
        self, 
        image_size: Tuple[int, int], 
        annotations: List[Dict], 
        dilate_pixels: int = 8
    ) -> Image.Image:
        """Создает МАЛЕНЬКУЮ маску точно вокруг дефекта"""
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        w, h = image_size
        
        for ann in annotations:
            x = ann['x_center'] * w
            y = ann['y_center'] * h
            bw = ann['width'] * w
            bh = ann['height'] * h
            
            # МИНИМАЛЬНОЕ расширение
            padding = dilate_pixels
            x1 = max(0, x - bw/2 - padding)
            y1 = max(0, y - bh/2 - padding)
            x2 = min(w, x + bw/2 + padding)
            y2 = min(h, y + bh/2 + padding)
            
            draw.ellipse([x1, y1, x2, y2], fill=255)
        
        # ЛЕГКОЕ размытие для плавности
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1))
        
        return mask
    
    def inpaint_simple(
        self,
        image: Image.Image,
        annotations: List[Dict],
        strength: float = 0.15,  # ← ОЧЕНЬ МАЛЕНЬКИЙ STRENGTH!
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Dict]:
        """Простой inpainting — только легкое изменение текстуры"""
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Маска только на дефект
        mask = self.create_defect_mask(image.size, annotations, dilate_pixels=6)
        
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        
        result = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=image,
            mask_image=mask,
            strength=strength,  # 0.15 — почти не меняет форму
            guidance_scale=3.0,  # НИЗКИЙ guidance — меньше "творчества"
            num_inference_steps=15,  # МЕНЬШЕ шагов — меньше изменений
            generator=generator
        ).images[0]
        
        meta = {
            "seed": seed,
            "strength": strength,
        }
        
        return result, meta
    
    def apply_augmentations(
        self,
        image: Image.Image,
        annotations: List[Dict]
    ) -> Tuple[Image.Image, List[Dict], Dict]:
        """ТОЛЬКО легкие аугментации"""
        
        aug_info = {"flip": False, "brightness": 1.0, "contrast": 1.0}
        
        if not self.config.enable_augmentation:
            return image, annotations, aug_info
        
        # Горизонтальный флип
        if random.random() < self.config.aug_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            annotations = YOLODatasetHandler.flip_annotation_horizontal(annotations)
            aug_info["flip"] = True
        
        # Яркость — минимальные изменения
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.9, 1.1)
            image = enhancer.enhance(factor)
            aug_info["brightness"] = round(factor, 2)
        
        # Контраст — минимальные изменения
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.9, 1.1)
            image = enhancer.enhance(factor)
            aug_info["contrast"] = round(factor, 2)
        
        return image, annotations, aug_info
    
    def generate_dataset(
        self,
        input_images_dir: Path,
        input_labels_dir: Path,
        output_dir: Path,
        variants_per_image: int,
        total_limit: Optional[int] = None
    ) -> List[Dict]:
        """Генерация датасета"""
        
        output_images = output_dir / "images"
        output_labels = output_dir / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        source_handler = YOLODatasetHandler(input_images_dir, input_labels_dir)
        target_handler = YOLODatasetHandler(output_images, output_labels)
        
        image_paths = list(input_images_dir.glob("*.png")) + \
                     list(input_images_dir.glob("*.jpg")) + \
                     list(input_images_dir.glob("*.jpeg"))
        
        if not image_paths:
            logger.error(f"Нет изображений в {input_images_dir}")
            return []
        
        logger.info(f"📂 Найдено {len(image_paths)} изображений")
        
        total_generated = 0
        all_metadata = []
        
        for img_path in tqdm(image_paths, desc="Inpainting"):
            if total_limit and total_generated >= total_limit:
                break
            
            try:
                ref_image = Image.open(img_path).convert("RGB")
            except:
                continue
            
            annotations = source_handler.load_annotation(img_path.name)
            if not annotations:
                continue
            
            if self.resize_to:
                ref_image = ref_image.resize((self.resize_to, self.resize_to), Image.Resampling.LANCZOS)
            
            for variant in range(variants_per_image):
                if total_limit and total_generated >= total_limit:
                    break
                
                try:
                    # ОЧЕНЬ МЯГКИЙ inpainting
                    strength = random.uniform(0.1, 0.2)
                    syn_image, meta = self.inpaint_simple(ref_image, annotations, strength=strength)
                    
                    # Легкие аугментации
                    current_annotations = [ann.copy() for ann in annotations]
                    syn_image, current_annotations, aug_info = self.apply_augmentations(
                        syn_image, current_annotations
                    )
                    
                    filename = f"simple_{total_generated:06d}_{img_path.stem}_v{variant}.png"
                    syn_image.save(output_images / filename, "PNG", optimize=True)
                    target_handler.save_annotation(filename, current_annotations)
                    
                    meta.update({
                        "filename": filename,
                        "source_image": img_path.name,
                        "variant": variant,
                        "augmentations": aug_info,
                    })
                    all_metadata.append(meta)
                    
                    total_generated += 1
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка: {e}")
                    torch.cuda.empty_cache()
                    continue
            
            if total_generated % 10 == 0:
                torch.cuda.empty_cache()
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        logger.info(f"🎉 Создано {total_generated} изображений")
        return all_metadata


def main():
    parser = argparse.ArgumentParser(description="ПРОСТОЙ inpainting без галлюцинаций")
    
    parser.add_argument("--input_dir", type=str, default="data/256_yolo/balanced_defect_patches/train")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_simple")
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resize_to", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--flip_prob", type=float, default=0.5)
    
    args = parser.parse_args()
    
    print_system_info()
    set_seed(args.seed)
    
    input_path = Path(args.input_dir)
    images_dir = input_path / "images"
    labels_dir = input_path / "labels"
    
    config = GenerationConfig()
    config.resize_to = args.resize_to
    config.enable_augmentation = not args.no_augmentation
    config.aug_flip_prob = args.flip_prob
    
    logger.info("=" * 70)
    logger.info("🎨 ПРОСТОЙ INPAINTING (БЕЗ ГАЛЛЮЦИНАЦИЙ)")
    logger.info("=" * 70)
    logger.info(f"📂 Выход: {args.output_dir}")
    logger.info(f"🔥 Вариантов: {args.variants}")
    logger.info(f"🎲 Strength: 0.1-0.2 (минимальный)")
    logger.info(f"✅ ФОРМА ДЕФЕКТА НЕ МЕНЯЕТСЯ")
    logger.info("=" * 70)
    
    generator = SimpleInpainter(config)
    generator.generate_dataset(
        input_images_dir=images_dir,
        input_labels_dir=labels_dir,
        output_dir=Path(args.output_dir),
        variants_per_image=args.variants,
        total_limit=args.limit
    )
    
    logger.info("✅ Готово!")


if __name__ == "__main__":
    main()