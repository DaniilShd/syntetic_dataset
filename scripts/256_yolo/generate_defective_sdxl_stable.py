#!/usr/bin/env python3
"""
generate_defective_sdxl_stable.py - СТАБИЛЬНЫЙ SDXL (без Turbo)
Работает гарантированно!
"""

import sys
import os
sys.path.insert(0, '/app/scripts')
sys.path.insert(0, '/app/src')

import torch
import random
import argparse
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import numpy as np
import json

from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
from config import GenerationConfig
from utils import set_seed, print_system_info, logger


class YOLODatasetHandler:
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
        return [{
            'class': ann['class'],
            'x_center': 1.0 - ann['x_center'],
            'y_center': ann['y_center'],
            'width': ann['width'],
            'height': ann['height']
        } for ann in annotations]


class SDXLStableGenerator:
    """СТАБИЛЬНЫЙ генератор на обычном SDXL"""
    
    PROMPTS = [
        "industrial metal surface with defect, realistic texture, detailed",
        "steel plate with manufacturing defect, factory lighting, sharp",
        "metal surface imperfection, industrial photography, authentic",
    ]
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.resize_to = config.resize_to
        
        logger.info(f"🔄 Загрузка SDXL (стабильная версия)...")
        self._load_pipeline()
        logger.info(f"✅ SDXL готов")
    
    def _load_pipeline(self):
        """Загрузка пайплайна БЕЗ проблемных оптимизаций"""
        
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=self.config.cache_dir
        ).to(self.config.device)
        
        # Стабильный scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True
        )
        
        # ❌ ОТКЛЮЧАЕМ проблемные оптимизации!
        # if hasattr(self.pipe, 'enable_attention_slicing'):
        #     self.pipe.enable_attention_slicing()
        
        # if hasattr(self.pipe, 'enable_vae_tiling'):
        #     self.pipe.vae.enable_tiling()
        
        # ✅ Вместо этого — принудительная очистка кэша
        torch.cuda.empty_cache()
        
        logger.info("   ✅ Пайплайн загружен (без проблемных оптимизаций)")
    
    def generate_one(
        self,
        reference_image: Image.Image,
        annotations: List[Dict],
        strength: float = 0.35,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Dict]:
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        prompt = random.choice(self.PROMPTS) + ", same shape, same defect type"
        negative = "blurry, distorted, cartoon, painting, text, watermark, different shape"
        
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=reference_image,
            strength=strength,
            guidance_scale=5.0,
            num_inference_steps=15,
            generator=generator
        ).images[0]
        
        return result, {"seed": seed, "strength": strength, "prompt": prompt}
    
    def apply_augmentations(self, image: Image.Image, annotations: List[Dict]) -> Tuple[Image.Image, List[Dict]]:
        if not self.config.enable_augmentation:
            return image, annotations
        
        if random.random() < self.config.aug_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            annotations = YOLODatasetHandler.flip_annotation_horizontal(annotations)
        
        if random.random() < 0.4:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.4:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        
        return image, annotations
    
    def generate_dataset(
        self,
        input_images_dir: Path,
        input_labels_dir: Path,
        output_dir: Path,
        variants_per_image: int,
        total_limit: Optional[int] = None
    ) -> List[Dict]:
        
        output_images = output_dir / "images"
        output_labels = output_dir / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        source_handler = YOLODatasetHandler(input_images_dir, input_labels_dir)
        target_handler = YOLODatasetHandler(output_images, output_labels)
        
        image_paths = list(input_images_dir.glob("*.png")) + list(input_images_dir.glob("*.jpg"))
        
        logger.info(f"📂 Найдено {len(image_paths)} изображений")
        
        total_generated = 0
        all_metadata = []
        
        for img_path in tqdm(image_paths, desc="Генерация"):
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
                    strength = random.uniform(0.3, 0.45)
                    syn_image, meta = self.generate_one(ref_image, annotations, strength=strength)
                    
                    current_annotations = [ann.copy() for ann in annotations]
                    syn_image, current_annotations = self.apply_augmentations(syn_image, current_annotations)
                    
                    filename = f"sdxl_{total_generated:06d}_{img_path.stem}_v{variant}.png"
                    syn_image.save(output_images / filename, "PNG")
                    target_handler.save_annotation(filename, current_annotations)
                    
                    meta.update({"filename": filename, "source": img_path.name})
                    all_metadata.append(meta)
                    
                    total_generated += 1
                    
                    if total_generated % 50 == 0:
                        logger.info(f"📊 Сгенерировано: {total_generated}")
                        
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/256_yolo/balanced_defect_patches/train")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_sdxl_stable")
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resize_to", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--flip_prob", type=float, default=0.5)
    
    args = parser.parse_args()
    
    print_system_info()
    set_seed(args.seed)
    
    config = GenerationConfig()
    config.resize_to = args.resize_to
    config.enable_augmentation = not args.no_augmentation
    config.aug_flip_prob = args.flip_prob
    
    logger.info("=" * 70)
    logger.info("🎨 СТАБИЛЬНЫЙ SDXL (без Turbo)")
    logger.info("=" * 70)
    logger.info(f"Выход: {args.output_dir}")
    logger.info(f"Вариантов: {args.variants}")
    logger.info(f"Скорость: ~1-2 сек/изображение")
    logger.info("=" * 70)
    
    generator = SDXLStableGenerator(config)
    generator.generate_dataset(
        input_images_dir=Path(args.input_dir) / "images",
        input_labels_dir=Path(args.input_dir) / "labels",
        output_dir=Path(args.output_dir),
        variants_per_image=args.variants,
        total_limit=args.limit
    )
    
    logger.info("✅ Готово!")


if __name__ == "__main__":
    main()