#!/usr/bin/env python3
"""
03_generate_defective.py - Генерация синтетических изображений с дефектами
Полный аналог 01_generate_backgrounds.py с множеством настроек
"""

import sys
import os
sys.path.insert(0, '/app/scripts')
sys.path.insert(0, '/app/src')

import torch
import random
import argparse
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, AutoencoderKL

from config import GenerationConfig
from utils import (
    set_seed, load_images_from_dir, save_json, 
    print_system_info, logger
)


class DefectiveGenerator:
    """Генератор изображений с дефектами через img2img"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        
        logger.info("🔄 Загрузка Stable Diffusion...")
        
        # Загрузка улучшенного VAE
        try:
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float16,
                cache_dir=config.cache_dir
            )
            logger.info("✅ Загружен улучшенный VAE")
        except Exception as e:
            logger.warning(f"⚠️ Стандартный VAE: {e}")
            vae = None
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.model_id,
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=config.cache_dir,
            local_files_only=False
        ).to(config.device)
        
        if hasattr(self.pipe.vae, 'enable_tiling'):
            self.pipe.vae.enable_tiling()
            logger.info("✅ VAE tiling включен")
        
        if config.use_ip_adapter:
            logger.info("🔄 Загрузка IP-Adapter...")
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
                cache_dir=config.cache_dir
            )
            self.pipe.set_ip_adapter_scale(0.75)
        
        if config.enable_attention_slicing:
            self.pipe.enable_attention_slicing()
        
        if config.enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("✅ xformers включен")
            except:
                logger.warning("⚠️ xformers не удалось включить")
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        torch.cuda.empty_cache()
        logger.info("✅ Генератор готов")
    
    def generate_one(
        self,
        reference_image: Image.Image,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Dict]:
        """Генерация одного синтетического изображения с дефектом"""
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        ip_scale = random.uniform(
            self.config.ip_adapter_scale_min,
            self.config.ip_adapter_scale_max
        )
        strength = random.uniform(
            self.config.strength_min,
            self.config.strength_max
        )
        
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        
        negative_prompt = (
            "purple, violet, magenta, artifacts, "
            "blurry, low quality, distorted, watermark, text"
        )
        
        if self.config.use_ip_adapter:
            output = self.pipe(
                prompt=self.config.prompt,
                negative_prompt=negative_prompt,
                image=reference_image,
                strength=strength,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                generator=generator,
                ip_adapter_image=reference_image,
                ip_adapter_scale=ip_scale
            )
        else:
            output = self.pipe(
                prompt=self.config.prompt,
                negative_prompt=negative_prompt,
                image=reference_image,
                strength=strength,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                generator=generator
            )
        
        # Извлечение изображения
        if hasattr(output, 'images'):
            image = output.images[0]
        elif isinstance(output, tuple):
            image = output[0]
            if isinstance(image, list):
                image = image[0]
        else:
            image = output
        
        metadata = {
            "seed": seed,
            "strength": round(strength, 3),
            "guidance_scale": self.config.guidance_scale,
            "steps": self.config.num_inference_steps
        }
        
        if self.config.use_ip_adapter:
            metadata["ip_adapter_scale"] = round(ip_scale, 3)
        
        if random.random() < 0.05:
            torch.cuda.empty_cache()
        
        return image, metadata
    
    def augment_image(self, image: Image.Image) -> Image.Image:
        """Аугментация изображения"""
        
        # Случайный горизонтальный флип
        if random.random() < self.config.aug_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Случайный вертикальный флип
        if random.random() < self.config.aug_vflip_prob:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Случайная яркость
        if random.random() < self.config.aug_brightness_prob:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(
                self.config.aug_brightness_min,
                self.config.aug_brightness_max
            )
            image = enhancer.enhance(factor)
        
        # Случайный контраст
        if random.random() < self.config.aug_contrast_prob:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(
                self.config.aug_contrast_min,
                self.config.aug_contrast_max
            )
            image = enhancer.enhance(factor)
        
        return image
    
    def generate_batch(
        self,
        reference_paths: List[Path],
        output_dir: Path,
        variants_per_reference: int,
        total_limit: Optional[int] = None,
        masks_dir: Optional[Path] = None
    ) -> List[Dict]:
        """Массовая генерация изображений с дефектами"""
        
        output_images = output_dir / "images"
        output_masks = output_dir / "masks"
        output_images.mkdir(parents=True, exist_ok=True)
        output_masks.mkdir(parents=True, exist_ok=True)
        
        total_generated = 0
        all_metadata = []
        
        for ref_idx, ref_path in enumerate(reference_paths):
            logger.info(f"📁 Референс {ref_idx+1}/{len(reference_paths)}: {ref_path.name}")
            
            ref_image = Image.open(ref_path).convert("RGB")
            
            # Ресайз если нужно
            if self.config.resize_to and ref_image.size != (self.config.resize_to, self.config.resize_to):
                ref_image = ref_image.resize(
                    (self.config.resize_to, self.config.resize_to),
                    Image.Resampling.LANCZOS
                )
            
            # Загружаем маску если есть
            mask_image = None
            if masks_dir and self.config.copy_masks:
                mask_path = masks_dir / ref_path.name
                if mask_path.exists():
                    mask_image = Image.open(mask_path).convert("L")
                    if self.config.resize_to:
                        mask_image = mask_image.resize(
                            (self.config.resize_to, self.config.resize_to),
                            Image.Resampling.NEAREST
                        )
            
            for variant in range(variants_per_reference):
                if total_limit and total_generated >= total_limit:
                    logger.info(f"✅ Лимит {total_limit} достигнут")
                    return all_metadata
                
                try:
                    syn_image, meta = self.generate_one(ref_image)
                    
                    if syn_image is None:
                        logger.error(f"Пустое изображение для варианта {variant}")
                        continue
                    
                    # Аугментация
                    if self.config.enable_augmentation:
                        syn_image = self.augment_image(syn_image)
                    
                    filename = f"defective_{total_generated:06d}_ref{ref_idx:03d}_v{variant:03d}.png"
                    syn_image.save(output_images / filename, "PNG")
                    
                    # Сохраняем маску
                    if mask_image and self.config.copy_masks:
                        mask_image.save(output_masks / filename, "PNG")
                    
                    meta.update({
                        "filename": filename,
                        "reference_image": str(ref_path.name),
                        "reference_index": ref_idx,
                        "variant": variant,
                        "has_mask": mask_image is not None
                    })
                    all_metadata.append(meta)
                    
                    total_generated += 1
                    
                    if total_generated % 10 == 0:
                        logger.info(f"   📊 Сгенерировано: {total_generated}")
                        
                except Exception as e:
                    logger.error(f"Ошибка при генерации варианта {variant}: {e}")
                    torch.cuda.empty_cache()
                    continue
        
        save_json(all_metadata, output_dir / "metadata.json")
        logger.info(f"🎉 Сгенерировано {total_generated} изображений")
        
        return all_metadata


class DefectiveGenerationConfig(GenerationConfig):
    """Расширенная конфигурация для генерации дефектов"""
    
    def __init__(
        self,
        # Базовые параметры
        ip_adapter_scale_min: float = 0.70,
        ip_adapter_scale_max: float = 0.80,
        strength_min: float = 0.15,
        strength_max: float = 0.25,
        guidance_scale: float = 3.0,
        num_inference_steps: int = 25,
        use_ip_adapter: bool = True,
        # Специфичные параметры
        prompt: str = "steel surface with defects, industrial quality, metal sheet",
        resize_to: int = 1024,
        enable_augmentation: bool = True,
        copy_masks: bool = True,
        # Аугментация
        aug_flip_prob: float = 0.5,
        aug_vflip_prob: float = 0.0,
        aug_brightness_prob: float = 0.3,
        aug_brightness_min: float = 0.9,
        aug_brightness_max: float = 1.1,
        aug_contrast_prob: float = 0.3,
        aug_contrast_min: float = 0.9,
        aug_contrast_max: float = 1.1,
        **kwargs
    ):
        super().__init__(
            ip_adapter_scale_min=ip_adapter_scale_min,
            ip_adapter_scale_max=ip_adapter_scale_max,
            strength_min=strength_min,
            strength_max=strength_max,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            use_ip_adapter=use_ip_adapter,
            **kwargs
        )
        self.prompt = prompt
        self.resize_to = resize_to
        self.enable_augmentation = enable_augmentation
        self.copy_masks = copy_masks
        self.aug_flip_prob = aug_flip_prob
        self.aug_vflip_prob = aug_vflip_prob
        self.aug_brightness_prob = aug_brightness_prob
        self.aug_brightness_min = aug_brightness_min
        self.aug_brightness_max = aug_brightness_max
        self.aug_contrast_prob = aug_contrast_prob
        self.aug_contrast_min = aug_contrast_min
        self.aug_contrast_max = aug_contrast_max


def main():
    parser = argparse.ArgumentParser(description="Генерация синтетических изображений с дефектами")
    
    # Пути
    parser.add_argument("--input_dir", type=str, default="data/severstal/defective_patches/images")
    parser.add_argument("--masks_dir", type=str, default="data/severstal/defective_patches/masks")
    parser.add_argument("--output_dir", type=str, default="/app/results/defective_dataset")
    
    # Параметры генерации
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strength_min", type=float, default=0.30)
    parser.add_argument("--strength_max", type=float, default=0.45)
    parser.add_argument("--ip_scale_min", type=float, default=0.70)
    parser.add_argument("--ip_scale_max", type=float, default=0.80)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--prompt", type=str, default="steel surface with defects, industrial quality")
    
    # Размеры
    parser.add_argument("--resize_to", type=int, default=1024)
    
    # Аугментация
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--no_copy_masks", action="store_true")
    parser.add_argument("--flip_prob", type=float, default=0.8)
    parser.add_argument("--brightness_prob", type=float, default=0.7)
    parser.add_argument("--brightness_min", type=float, default=0.7)
    parser.add_argument("--brightness_max", type=float, default=1.3)
    
    # Прочее
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_ip_adapter", action="store_true")
    
    args = parser.parse_args()
    
    print_system_info()
    set_seed(args.seed)
    
    config = DefectiveGenerationConfig(
        ip_adapter_scale_min=args.ip_scale_min,
        ip_adapter_scale_max=args.ip_scale_max,
        strength_min=args.strength_min,
        strength_max=args.strength_max,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        use_ip_adapter=not args.no_ip_adapter,
        prompt=args.prompt,
        resize_to=args.resize_to,
        enable_augmentation=not args.no_augmentation,
        copy_masks=not args.no_copy_masks,
        aug_flip_prob=args.flip_prob,
        aug_brightness_prob=args.brightness_prob,
        aug_brightness_min=args.brightness_min,
        aug_brightness_max=args.brightness_max,
    )
    
    logger.info("=" * 60)
    logger.info("🎨 ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ИЗОБРАЖЕНИЙ С ДЕФЕКТАМИ")
    logger.info("=" * 60)
    logger.info(f"IP-Adapter: {'включен' if config.use_ip_adapter else 'выключен'}")
    logger.info(f"Strength: {config.strength_min}-{config.strength_max}")
    logger.info(f"Guidance scale: {config.guidance_scale}")
    logger.info(f"Inference steps: {config.num_inference_steps}")
    logger.info(f"Prompt: {config.prompt}")
    logger.info(f"Resize to: {config.resize_to}×{config.resize_to}")
    logger.info(f"Augmentation: {'включена' if config.enable_augmentation else 'выключена'}")
    logger.info(f"Copy masks: {'да' if config.copy_masks else 'нет'}")
    
    # Загрузка референсов
    references = load_images_from_dir(args.input_dir)
    logger.info(f"📂 Найдено {len(references)} референсов с дефектами")
    
    if not references:
        logger.error(f"Нет изображений в {args.input_dir}")
        sys.exit(1)
    
    # Директория с масками
    masks_dir = Path(args.masks_dir) if args.masks_dir and not args.no_copy_masks else None
    if masks_dir and masks_dir.exists():
        logger.info(f"📂 Найдена директория с масками: {masks_dir}")
    
    generator = DefectiveGenerator(config)
    generator.generate_batch(
        reference_paths=references,
        output_dir=Path(args.output_dir),
        variants_per_reference=args.variants,
        total_limit=args.limit,
        masks_dir=masks_dir
    )
    
    logger.info("✅ Генерация завершена")


if __name__ == "__main__":
    main()