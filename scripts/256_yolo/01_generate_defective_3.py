#!/usr/bin/env python3
"""
01_generate_defective_3.py - Генерация синтетических изображений с дефектами
Поддержка SD 2.1 и SDXL для качественной генерации
БЕЗ спектральных артефактов
"""

import sys
import os
sys.path.insert(0, '/app/scripts')
sys.path.insert(0, '/app/src')

import torch
import random
import argparse
from PIL import Image, ImageEnhance
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import numpy as np
import json

from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)

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


class DefectiveGenerator:
    """Генератор с поддержкой SD 2.1 и SDXL"""
    
    # Модели на выбор
    MODELS = {
        "sd15": "runwayml/stable-diffusion-v1-5",
        "sd21": "stabilityai/stable-diffusion-2-1-base",  # ← ИСПРАВЛЕНО
        "sdxl": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "sdxl_base": "stabilityai/stable-diffusion-xl-base-1.0"
    }
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model_id = self.MODELS.get(config.model_version, config.model_id)
        
        # Промпты для industrial текстур
        self.prompts = [
            "industrial steel surface with manufacturing defects, detailed metallic texture, factory photography",
            "metal plate with scratches and imperfections, industrial inspection photo, harsh lighting",
            "defective steel sheet, production line, realistic metal surface, quality control image",
            "worn industrial metal with surface damage, authentic texture, manufacturing environment",
            "steel surface with marks and defects, industrial photography, detailed close-up",
            "metal surface imperfections, factory quality control image, realistic industrial texture"
        ]
        
        # Негативные промпты
        self.negative_prompt = "blurry, low quality, distorted, text, watermark, cartoon, painting, smooth plastic, CGI, 3d render, illustration"
        
        logger.info(f"🔄 Загрузка модели: {self.model_id}")
        
        # Загрузка пайплайна в зависимости от модели
        if "xl" in self.model_id.lower():
            self._load_sdxl()
        else:
            self._load_sd()
        
        logger.info(f"✅ Модель загружена: {self.model_id}")
        logger.info(f"📐 Размер генерации: {config.resize_to}x{config.resize_to}")
    
    def _load_sd(self):
        """Загрузка SD 1.5 или 2.1"""
        try:
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float16,
                cache_dir=self.config.cache_dir
            )
        except:
            vae = None
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=self.config.cache_dir,
            variant="fp16" if "2-1" in self.model_id else None
        ).to(self.config.device)
        
        # Euler scheduler для лучшего качества
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing()
        
        self.is_xl = False
    
    def _load_sdxl(self):
        """Загрузка SDXL"""
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=self.config.cache_dir,
            variant="fp16",
            use_safetensors=True
        ).to(self.config.device)
        
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing()
        
        self.is_xl = True
    
    def generate_one(
        self,
        reference_image: Image.Image,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Dict]:
        """Генерация одного изображения"""
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        strength = random.uniform(self.config.strength_min, self.config.strength_max)
        prompt = random.choice(self.prompts)
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        
        # Подготовка изображения
        if self.is_xl:
            # SDXL ожидает 1024x1024
            target_size = 1024
            if reference_image.size != (target_size, target_size):
                ref_for_gen = reference_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            else:
                ref_for_gen = reference_image
        else:
            target_size = self.config.resize_to
            ref_for_gen = reference_image
        
        # Генерация
        output = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            image=ref_for_gen,
            strength=strength,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            generator=generator
        )
        
        image = output.images[0]
        
        # Возвращаем к исходному размеру если нужно
        if image.size != (self.config.resize_to, self.config.resize_to):
            image = image.resize((self.config.resize_to, self.config.resize_to), Image.Resampling.LANCZOS)
        
        return image, {
            "seed": seed,
            "strength": round(strength, 3),
            "prompt": prompt,
            "model": self.model_id
        }
    
    def apply_augmentations(
        self,
        image: Image.Image,
        annotations: List[Dict]
    ) -> Tuple[Image.Image, List[Dict], Dict]:
        """Аугментации"""
        
        aug_info = {"flip": False, "brightness": 1.0, "contrast": 1.0}
        
        if not self.config.enable_augmentation:
            return image, annotations, aug_info
        
        # Флип
        if random.random() < self.config.aug_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            annotations = YOLODatasetHandler.flip_annotation_horizontal(annotations)
            aug_info["flip"] = True
        
        # Яркость
        if random.random() < self.config.aug_brightness_prob:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(self.config.aug_brightness_min, self.config.aug_brightness_max)
            image = enhancer.enhance(factor)
            aug_info["brightness"] = round(factor, 2)
        
        # Контраст
        if random.random() < self.config.aug_contrast_prob:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(self.config.aug_contrast_min, self.config.aug_contrast_max)
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
        
        logger.info(f"📂 Найдено {len(image_paths)} исходных изображений")
        logger.info(f"🎯 Вариантов на каждое: {variants_per_image}")
        logger.info(f"📊 Ожидаемый выход: ~{len(image_paths) * variants_per_image} изображений")
        
        total_generated = 0
        all_metadata = []
        
        for img_path in tqdm(image_paths, desc="Генерация"):
            if total_limit and total_generated >= total_limit:
                break
            
            try:
                ref_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки {img_path.name}: {e}")
                continue
            
            annotations = source_handler.load_annotation(img_path.name)
            if not annotations:
                logger.warning(f"⚠️ Нет разметки для {img_path.name}")
                continue
            
            if self.config.resize_to:
                ref_image = ref_image.resize(
                    (self.config.resize_to, self.config.resize_to),
                    Image.Resampling.LANCZOS
                )
            
            for variant in range(variants_per_image):
                if total_limit and total_generated >= total_limit:
                    break
                
                try:
                    syn_image, meta = self.generate_one(ref_image)
                    current_annotations = [ann.copy() for ann in annotations]
                    syn_image, current_annotations, aug_info = self.apply_augmentations(
                        syn_image, current_annotations
                    )
                    
                    filename = f"syn_{total_generated:06d}_{img_path.stem}_v{variant}.png"
                    syn_image.save(output_images / filename, "PNG", optimize=True)
                    target_handler.save_annotation(filename, current_annotations)
                    
                    meta.update({
                        "filename": filename,
                        "source_image": img_path.name,
                        "variant": variant,
                        "augmentations": aug_info,
                        "num_objects": len(current_annotations)
                    })
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
        
        # Сохранение метаданных
        with open(output_dir / "generation_metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        logger.info(f"🎉 Генерация завершена! Создано {total_generated} изображений")
        
        if all_metadata:
            avg_strength = np.mean([m['strength'] for m in all_metadata])
            logger.info(f"📊 Средняя strength: {avg_strength:.3f}")
        
        return all_metadata


def main():
    parser = argparse.ArgumentParser(description="Генерация синтетики (SD 2.1 / SDXL)")
    
    parser.add_argument("--input_dir", type=str, default="data/256_yolo/balanced_defect_patches/train")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_sd21")
    
    # 🔥 ВЫБОР МОДЕЛИ
    parser.add_argument("--model", type=str, default="sd21",
                       choices=["sd15", "sd21", "sdxl", "sdxl_base"],
                       help="Модель Stable Diffusion")
    
    # Параметры генерации
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strength_min", type=float, default=0.15)
    parser.add_argument("--strength_max", type=float, default=0.25)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=15)
    
    # Аугментации
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--flip_prob", type=float, default=0.5)
    
    # Технические
    parser.add_argument("--resize_to", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    print_system_info()
    set_seed(args.seed)
    
    input_path = Path(args.input_dir)
    images_dir = input_path / "images"
    labels_dir = input_path / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        logger.error(f"Директории не найдены в {input_path}")
        sys.exit(1)
    
    config = GenerationConfig()
    config.model_version = args.model
    config.strength_min = args.strength_min
    config.strength_max = args.strength_max
    config.guidance_scale = args.guidance_scale
    config.num_inference_steps = args.steps
    config.resize_to = args.resize_to
    
    config.enable_augmentation = not args.no_augmentation
    config.aug_flip_prob = args.flip_prob
    config.aug_brightness_prob = 0.3
    config.aug_brightness_min = 0.7
    config.aug_brightness_max = 1.3
    config.aug_contrast_prob = 0.3
    config.aug_contrast_min = 0.7
    config.aug_contrast_max = 1.3
    
    # Отключаем спектральный контроль
    config.use_spectrum_matching = False
    config.use_high_freq_injection = False
    
    logger.info("=" * 70)
    logger.info(f"🎨 ГЕНЕРАЦИЯ СИНТЕТИКИ ({args.model.upper()})")
    logger.info("=" * 70)
    logger.info(f"📂 Вход: {input_path}")
    logger.info(f"📂 Выход: {args.output_dir}")
    logger.info(f"🤖 Модель: {args.model}")
    logger.info(f"🔥 Вариантов: {args.variants}")
    logger.info(f"🎲 Strength: {config.strength_min:.2f}-{config.strength_max:.2f}")
    logger.info(f"📐 Размер: {args.resize_to}x{args.resize_to}")
    logger.info("=" * 70)
    
    generator = DefectiveGenerator(config)
    generator.generate_dataset(
        input_images_dir=images_dir,
        input_labels_dir=labels_dir,
        output_dir=Path(args.output_dir),
        variants_per_image=args.variants,
        total_limit=args.limit
    )
    
    logger.info("✅ Генерация завершена!")


if __name__ == "__main__":
    main()