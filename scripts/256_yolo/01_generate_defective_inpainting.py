#!/usr/bin/env python3
"""
generate_defective_inpaint.py - Инжекция дефектов с сохранением локации и разметки
Полный скрипт для генерации синтетических дефектов через inpainting
"""

import sys
import os
sys.path.insert(0, '/app/scripts')
sys.path.insert(0, '/app/src')

import torch
import random
import argparse
from PIL import Image, ImageDraw, ImageEnhance
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


class DefectInpainter:
    """Генератор дефектов через inpainting с сохранением локации"""
    
    MODELS = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sd21": "stabilityai/stable-diffusion-2-1-base",
    }
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model_id = self.MODELS.get(config.model_version, config.model_id)
        self.resize_to = config.resize_to
        
        # Промпты для разных типов дефектов
        self.defect_prompts = {
            0: "metal surface with scratch, linear scratch on steel, industrial defect",
            1: "metal plate with dent, concave dent on steel surface, manufacturing damage",
            2: "steel surface with crack, linear crack on metal, material defect",
            3: "metal with corrosion spot, rust on steel surface, oxidation defect",
        }
        
        self.negative_prompt = "blurry, low quality, distorted, text, watermark, cartoon, painting, smooth plastic, different background, changed environment"
        
        logger.info(f"🔄 Загрузка модели inpainting: {self.model_id}")
        self._load_pipeline()
        logger.info(f"✅ Модель загружена")
    
    def _load_pipeline(self):
        """Загрузка пайплайна для inpainting"""
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=self.config.cache_dir
        ).to(self.config.device)
        
        # Оптимизации
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing()
        
        if hasattr(self.pipe, 'enable_vae_tiling'):
            self.pipe.vae.enable_tiling()
    
    def create_defect_mask(
        self, 
        image_size: Tuple[int, int], 
        annotations: List[Dict], 
        dilate_pixels: int = 15,
        mask_type: str = "ellipse"
    ) -> Image.Image:
        """Создает маску вокруг bbox дефектов"""
        mask = Image.new("L", image_size, 0)  # черный = не менять
        draw = ImageDraw.Draw(mask)
        
        w, h = image_size
        
        for ann in annotations:
            # YOLO bbox -> пиксели
            x = ann['x_center'] * w
            y = ann['y_center'] * h
            bw = ann['width'] * w
            bh = ann['height'] * h
            
            # Расширяем область для контекста
            padding = dilate_pixels
            x1 = max(0, x - bw/2 - padding)
            y1 = max(0, y - bh/2 - padding)
            x2 = min(w, x + bw/2 + padding)
            y2 = min(h, y + bh/2 + padding)
            
            # Создаем маску
            if mask_type == "ellipse":
                draw.ellipse([x1, y1, x2, y2], fill=255)
            elif mask_type == "rectangle":
                draw.rectangle([x1, y1, x2, y2], fill=255)
            elif mask_type == "blurred":
                # Градиентная маска (плавный переход)
                draw.ellipse([x1, y1, x2, y2], fill=255)
                mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
        
        return mask
    
    def get_prompt_for_annotations(self, annotations: List[Dict]) -> str:
        """Генерирует промпт на основе классов дефектов"""
        classes = set(ann['class'] for ann in annotations)
        
        if len(classes) == 1:
            class_id = list(classes)[0]
            base_prompt = self.defect_prompts.get(class_id, "metal surface defect")
        else:
            base_prompt = "metal surface with multiple defects, scratches, dents, industrial damage"
        
        # Добавляем модификаторы качества
        quality_mods = [
            "detailed texture",
            "realistic industrial photography",
            "sharp focus",
            "high contrast",
            "authentic metal surface"
        ]
        
        return base_prompt + ", " + ", ".join(random.sample(quality_mods, 3))
    
    def inpaint_defects(
        self, 
        image: Image.Image, 
        annotations: List[Dict],
        strength: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Image.Image, Dict]:
        """Инжектирует новые дефекты в область bbox"""
        
        if strength is None:
            strength = random.uniform(self.config.strength_min, self.config.strength_max)
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Создаем маску
        mask = self.create_defect_mask(
            image.size, 
            annotations, 
            dilate_pixels=12,
            mask_type="blurred"
        )
        
        # Промпт на основе классов
        prompt = self.get_prompt_for_annotations(annotations)
        
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        
        # Inpainting
        result = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            image=image,
            mask_image=mask,
            strength=strength,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            generator=generator
        ).images[0]
        
        meta = {
            "seed": seed,
            "strength": round(strength, 3),
            "prompt": prompt,
            "model": self.model_id
        }
        
        return result, mask, meta
    
    def apply_augmentations(
        self,
        image: Image.Image,
        annotations: List[Dict]
    ) -> Tuple[Image.Image, List[Dict], Dict]:
        """Аугментации с обновлением разметки"""
        
        aug_info = {"flip": False, "brightness": 1.0, "contrast": 1.0}
        
        if not self.config.enable_augmentation:
            return image, annotations, aug_info
        
        # Горизонтальный флип
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
        total_limit: Optional[int] = None,
        save_masks: bool = False
    ) -> List[Dict]:
        """Генерация датасета с inpainting"""
        
        output_images = output_dir / "images"
        output_labels = output_dir / "labels"
        output_masks = output_dir / "masks" if save_masks else None
        
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        if save_masks:
            output_masks.mkdir(parents=True, exist_ok=True)
        
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
        
        for img_path in tqdm(image_paths, desc="Inpainting"):
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
            
            # Ресайз
            if self.resize_to:
                ref_image = ref_image.resize((self.resize_to, self.resize_to), Image.Resampling.LANCZOS)
            
            for variant in range(variants_per_image):
                if total_limit and total_generated >= total_limit:
                    break
                
                try:
                    # Inpainting дефектов
                    syn_image, mask, meta = self.inpaint_defects(ref_image, annotations)
                    
                    # Аугментации (разметка НЕ меняется при inpainting!)
                    current_annotations = [ann.copy() for ann in annotations]
                    syn_image, current_annotations, aug_info = self.apply_augmentations(
                        syn_image, current_annotations
                    )
                    
                    # Сохранение
                    filename = f"inpaint_{total_generated:06d}_{img_path.stem}_v{variant}.png"
                    syn_image.save(output_images / filename, "PNG", optimize=True)
                    target_handler.save_annotation(filename, current_annotations)
                    
                    if save_masks:
                        mask.save(output_masks / f"mask_{filename}", "PNG")
                    
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
                    logger.error(f"❌ Ошибка inpainting: {e}")
                    torch.cuda.empty_cache()
                    continue
            
            if total_generated % 10 == 0:
                torch.cuda.empty_cache()
        
        # Сохранение метаданных
        with open(output_dir / "generation_metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        logger.info(f"🎉 Inpainting завершен! Создано {total_generated} изображений")
        
        if all_metadata:
            avg_strength = np.mean([m['strength'] for m in all_metadata])
            logger.info(f"📊 Средняя strength: {avg_strength:.3f}")
        
        return all_metadata


def main():
    parser = argparse.ArgumentParser(description="Inpainting дефектов с сохранением разметки")
    
    parser.add_argument("--input_dir", type=str, default="data/256_yolo/balanced_defect_patches/train")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_inpaint")
    
    parser.add_argument("--model", type=str, default="sdxl", choices=["sdxl", "sd21"])
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strength_min", type=float, default=0.6)
    parser.add_argument("--strength_max", type=float, default=0.8)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=25)
    
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--flip_prob", type=float, default=0.5)
    parser.add_argument("--save_masks", action="store_true", help="Сохранять маски inpainting")
    
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
    config.aug_brightness_prob = 0.2
    config.aug_brightness_min = 0.8
    config.aug_brightness_max = 1.2
    config.aug_contrast_prob = 0.2
    config.aug_contrast_min = 0.8
    config.aug_contrast_max = 1.2
    
    logger.info("=" * 70)
    logger.info(f"🎨 INPAINTING ДЕФЕКТОВ ({args.model.upper()})")
    logger.info("=" * 70)
    logger.info(f"📂 Вход: {input_path}")
    logger.info(f"📂 Выход: {args.output_dir}")
    logger.info(f"🤖 Модель: {args.model}")
    logger.info(f"🔥 Вариантов: {args.variants}")
    logger.info(f"🎲 Strength: {config.strength_min:.2f}-{config.strength_max:.2f}")
    logger.info(f"📐 Размер: {args.resize_to}x{args.resize_to}")
    logger.info(f"💾 Сохранять маски: {'✅' if args.save_masks else '❌'}")
    logger.info("=" * 70)
    logger.info("⚠️  РАЗМЕТКА СОХРАНЯЕТСЯ НЕИЗМЕННОЙ!")
    logger.info("=" * 70)
    
    generator = DefectInpainter(config)
    generator.generate_dataset(
        input_images_dir=images_dir,
        input_labels_dir=labels_dir,
        output_dir=Path(args.output_dir),
        variants_per_image=args.variants,
        total_limit=args.limit,
        save_masks=args.save_masks
    )
    
    logger.info("✅ Inpainting завершен!")


if __name__ == "__main__":
    main()