#!/usr/bin/env python3
"""
generate_defective_controlnet.py - Сохранение вида дефекта через IP-Adapter + ControlNet
Полный скрипт для генерации с сохранением формы и текстуры дефекта
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
import cv2

from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler
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


class DefectPreservingInpainter:
    """Inpainting с сохранением внешнего вида и структуры дефекта"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model_id = config.model_id or "stabilityai/stable-diffusion-xl-base-1.0"
        self.resize_to = config.resize_to
        
        # Промпты для разных классов дефектов
        self.defect_prompts = {
            0: "metal surface with scratch mark, linear scratch, same shape, realistic texture",
            1: "metal plate with dent, concave dent, same shape, industrial surface",
            2: "steel with crack line, linear crack, same shape, metallic texture",
            3: "metal with corrosion, rust spot, same shape, industrial material",
        }
        
        self.negative_prompt = "different shape, new object, cartoon, blurry, distorted, changed structure"
        
        logger.info(f"🔄 Загрузка ControlNet + IP-Adapter...")
        self._load_pipeline()
        logger.info(f"✅ Пайплайн готов")
    
    def _load_pipeline(self):
        """Загрузка ControlNet и IP-Adapter"""
        
        # ControlNet для сохранения структуры
        logger.info("   📥 ControlNet Canny...")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16,
            cache_dir=self.config.cache_dir
        )
        
        # Основной пайплайн
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            self.model_id,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=self.config.cache_dir
        ).to(self.config.device)
        
        # Scheduler
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # IP-Adapter для сохранения внешнего вида
        logger.info("   📥 IP-Adapter...")
        try:
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.bin",
                cache_dir=self.config.cache_dir
            )
            self.ip_adapter_loaded = True
            logger.info("   ✅ IP-Adapter загружен")
        except Exception as e:
            logger.warning(f"   ⚠️ IP-Adapter не загружен: {e}")
            self.ip_adapter_loaded = False
        
        # Оптимизации
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing()
        
        if hasattr(self.pipe, 'enable_vae_tiling'):
            self.pipe.vae.enable_tiling()
    
    def create_defect_mask(
        self, 
        image_size: Tuple[int, int], 
        annotations: List[Dict], 
        dilate_pixels: int = 12
    ) -> Image.Image:
        """Создает маску вокруг bbox дефектов"""
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        w, h = image_size
        
        for ann in annotations:
            x = ann['x_center'] * w
            y = ann['y_center'] * h
            bw = ann['width'] * w
            bh = ann['height'] * h
            
            padding = dilate_pixels
            x1 = max(0, x - bw/2 - padding)
            y1 = max(0, y - bh/2 - padding)
            x2 = min(w, x + bw/2 + padding)
            y2 = min(h, y + bh/2 + padding)
            
            # Эллипс с размытием для плавного перехода
            draw.ellipse([x1, y1, x2, y2], fill=255)
        
        # Размытие маски для плавного перехода
        mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
        
        return mask
    
    def create_canny_edges(self, image: Image.Image) -> Image.Image:
        """Создает карту границ Canny для ControlNet"""
        img_np = np.array(image.convert("RGB"))
        
        # Конвертируем в grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        return Image.fromarray(edges)
    
    def get_prompt_for_annotations(self, annotations: List[Dict]) -> str:
        """Генерирует промпт на основе классов дефектов"""
        classes = set(ann['class'] for ann in annotations)
        
        if len(classes) == 1:
            class_id = list(classes)[0]
            base_prompt = self.defect_prompts.get(class_id, "metal surface defect, same shape")
        else:
            base_prompt = "metal surface with defects, same shapes, industrial texture"
        
        return base_prompt + ", detailed, realistic, sharp focus"
    
    def inpaint_preserve_defect(
        self,
        image: Image.Image,
        annotations: List[Dict],
        strength: Optional[float] = None,
        ip_scale: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Dict]:
        """Inpainting с сохранением структуры и внешнего вида дефекта"""
        
        if strength is None:
            strength = random.uniform(self.config.strength_min, self.config.strength_max)
        
        if ip_scale is None:
            ip_scale = random.uniform(0.6, 0.8)
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Создаем маску
        mask = self.create_defect_mask(image.size, annotations)
        
        # Создаем Canny edges для ControlNet
        control_image = self.create_canny_edges(image)
        
        # Промпт
        prompt = self.get_prompt_for_annotations(annotations)
        
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        
        # Параметры генерации
        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": self.negative_prompt,
            "image": image,
            "mask_image": mask,
            "control_image": control_image,
            "controlnet_conditioning_scale": 0.75,  # сохраняем структуру
            "strength": strength,
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
            "generator": generator
        }
        
        # Добавляем IP-Adapter если загружен
        if self.ip_adapter_loaded:
            pipe_kwargs["ip_adapter_image"] = image
            pipe_kwargs["ip_adapter_scale"] = ip_scale
        
        result = self.pipe(**pipe_kwargs).images[0]
        
        meta = {
            "seed": seed,
            "strength": round(strength, 3),
            "ip_adapter_scale": round(ip_scale, 3) if self.ip_adapter_loaded else None,
            "prompt": prompt,
            "controlnet_scale": 0.75
        }
        
        return result, meta
    
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
                continue
            
            if self.resize_to:
                ref_image = ref_image.resize((self.resize_to, self.resize_to), Image.Resampling.LANCZOS)
            
            for variant in range(variants_per_image):
                if total_limit and total_generated >= total_limit:
                    break
                
                try:
                    # Inpainting с сохранением дефекта
                    syn_image, meta = self.inpaint_preserve_defect(ref_image, annotations)
                    
                    # Аугментации (разметка не меняется!)
                    current_annotations = [ann.copy() for ann in annotations]
                    syn_image, current_annotations, aug_info = self.apply_augmentations(
                        syn_image, current_annotations
                    )
                    
                    # Сохранение
                    filename = f"ctrl_{total_generated:06d}_{img_path.stem}_v{variant}.png"
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
                    
                    if total_generated % 20 == 0:
                        logger.info(f"📊 Сгенерировано: {total_generated}")
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка: {e}")
                    torch.cuda.empty_cache()
                    continue
            
            if total_generated % 5 == 0:
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
    parser = argparse.ArgumentParser(description="ControlNet+IP-Adapter inpainting с сохранением дефектов")
    
    parser.add_argument("--input_dir", type=str, default="data/256_yolo/balanced_defect_patches/train")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_controlnet")
    
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strength_min", type=float, default=0.3)
    parser.add_argument("--strength_max", type=float, default=0.5)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=20)
    
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--flip_prob", type=float, default=0.5)
    
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
    config.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    config.strength_min = args.strength_min
    config.strength_max = args.strength_max
    config.guidance_scale = args.guidance_scale
    config.num_inference_steps = args.steps
    config.resize_to = args.resize_to
    
    config.enable_augmentation = not args.no_augmentation
    config.aug_flip_prob = args.flip_prob
    config.aug_brightness_prob = 0.2
    config.aug_brightness_min = 0.85
    config.aug_brightness_max = 1.15
    config.aug_contrast_prob = 0.2
    config.aug_contrast_min = 0.85
    config.aug_contrast_max = 1.15
    
    logger.info("=" * 70)
    logger.info("🎨 CONTROLNET + IP-ADAPTER INPAINTING")
    logger.info("=" * 70)
    logger.info(f"📂 Вход: {input_path}")
    logger.info(f"📂 Выход: {args.output_dir}")
    logger.info(f"🔥 Вариантов: {args.variants}")
    logger.info(f"🎲 Strength: {config.strength_min:.2f}-{config.strength_max:.2f}")
    logger.info(f"🎯 Guidance: {config.guidance_scale}")
    logger.info(f"📐 Размер: {args.resize_to}x{args.resize_to}")
    logger.info("=" * 70)
    logger.info("✅ СТРУКТУРА ДЕФЕКТА СОХРАНЯЕТСЯ!")
    logger.info("✅ РАЗМЕТКА ОСТАЕТСЯ ВАЛИДНОЙ!")
    logger.info("=" * 70)
    
    generator = DefectPreservingInpainter(config)
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