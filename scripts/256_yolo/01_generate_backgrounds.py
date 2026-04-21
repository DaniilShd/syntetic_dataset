#!/usr/bin/env python3
"""
01_generate_backgrounds.py - Генерация синтетических фонов в Docker
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

from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, AutoencoderKL

from config import GenerationConfig
from utils import set_seed, load_images_from_dir, save_json, print_system_info, logger


class BackgroundGenerator:
    """Генератор синтетических фонов"""
    
    # Размеры для обработки
    GENERATION_SIZE = (1024, 1024)  # для Stable Diffusion
    OUTPUT_SIZE = (640, 640)        # финальный размер
    
    # Параметры аугментации
    AUG_FLIP_PROB = 0.5
    AUG_BRIGHTNESS_PROB = 0.3
    AUG_BRIGHTNESS_MIN = 0.9
    AUG_BRIGHTNESS_MAX = 1.1
    AUG_CONTRAST_PROB = 0.3
    AUG_CONTRAST_MIN = 0.9
    AUG_CONTRAST_MAX = 1.1
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Настройка пайплайна Stable Diffusion"""
        logger.info("🔄 Загрузка модели...")
        
        # VAE для лучшего качества
        try:
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float16,
                cache_dir=self.config.cache_dir
            )
            logger.info("✅ Улучшенный VAE загружен")
        except:
            logger.warning("⚠️ Стандартный VAE")
            vae = None
        
        # Основной пайплайн
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.config.model_id,
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=self.config.cache_dir
        ).to(self.config.device)
        
        # Оптимизации
        if hasattr(self.pipe.vae, 'enable_tiling'):
            self.pipe.vae.enable_tiling()
        
        if self.config.enable_attention_slicing:
            self.pipe.enable_attention_slicing()
        
        if self.config.enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # IP-Adapter (опционально)
        if self.config.use_ip_adapter:
            self._setup_ip_adapter()
        
        torch.cuda.empty_cache()
        logger.info("✅ Готов к работе")
    
    def _setup_ip_adapter(self):
        """Настройка IP-Adapter"""
        try:
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
                cache_dir=self.config.cache_dir
            )
            self.pipe.set_ip_adapter_scale(0.75)
            logger.info("✅ IP-Adapter загружен")
        except:
            logger.warning("⚠️ IP-Adapter не загружен")
            self.config.use_ip_adapter = False
    
    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Подготовка изображения: ресайз до 1024x1024"""
        return image.resize(self.GENERATION_SIZE, Image.Resampling.LANCZOS)
    
    def _resize_output(self, image: Image.Image) -> Image.Image:
        """Ресайз результата до 640x640"""
        return image.resize(self.OUTPUT_SIZE, Image.Resampling.LANCZOS)
    
    def _augment_image(self, image: Image.Image) -> Image.Image:
        """Аугментация изображения"""
        
        # Горизонтальный флип
        if random.random() < self.AUG_FLIP_PROB:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Яркость
        if random.random() < self.AUG_BRIGHTNESS_PROB:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(self.AUG_BRIGHTNESS_MIN, self.AUG_BRIGHTNESS_MAX)
            image = enhancer.enhance(factor)
        
        # Контраст
        if random.random() < self.AUG_CONTRAST_PROB:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(self.AUG_CONTRAST_MIN, self.AUG_CONTRAST_MAX)
            image = enhancer.enhance(factor)
        
        return image
    
    def generate(self, reference: Image.Image, seed: int = None) -> Tuple[Image.Image, Dict]:
        """Генерация одного фона"""
        
        # Параметры
        seed = seed or random.randint(0, 2**32 - 1)
        strength = random.uniform(self.config.strength_min, self.config.strength_max)
        ip_scale = random.uniform(
            self.config.ip_adapter_scale_min,
            self.config.ip_adapter_scale_max
        )
        
        # Подготовка изображения (256 -> 1024)
        reference_1024 = self._prepare_image(reference)
        
        # Параметры генерации
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        
        pipe_args = {
            "prompt": self.config.prompt,
            "negative_prompt": "purple, violet, magenta, artifacts, noise, blurry",
            "image": reference_1024,
            "strength": strength,
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
            "generator": generator
        }
        
        if self.config.use_ip_adapter:
            pipe_args["ip_adapter_image"] = reference_1024
            pipe_args["ip_adapter_scale"] = ip_scale
        
        # Генерация
        output = self.pipe(**pipe_args)
        
        # Извлечение результата
        if hasattr(output, 'images'):
            result = output.images[0]
        elif isinstance(output, tuple):
            result = output[0][0] if isinstance(output[0], list) else output[0]
        else:
            result = output
        
        # Ресайз до 640x640
        result_640 = self._resize_output(result)
        
        # Аугментация
        result_640 = self._augment_image(result_640)
        
        # Метаданные
        metadata = {
            "seed": seed,
            "strength": round(strength, 3),
            "guidance_scale": self.config.guidance_scale,
            "steps": self.config.num_inference_steps,
            "input_size": self.GENERATION_SIZE,
            "output_size": self.OUTPUT_SIZE,
            "augmented": True
        }
        
        if self.config.use_ip_adapter:
            metadata["ip_adapter_scale"] = round(ip_scale, 3)
        
        # Очистка памяти
        if random.random() < 0.05:
            torch.cuda.empty_cache()
        
        return result_640, metadata
    
    def generate_batch(
        self,
        references: List[Path],
        output_dir: Path,
        variants: int = 1,
        limit: int = None
    ) -> List[Dict]:
        """Пакетная генерация"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated = 0
        metadata_list = []
        
        for ref_idx, ref_path in enumerate(references):
            logger.info(f"📁 [{ref_idx+1}/{len(references)}] {ref_path.name}")
            
            # Загрузка референса (ожидаем 256x256)
            ref_image = Image.open(ref_path).convert("RGB")
            
            for variant in range(variants):
                if limit and generated >= limit:
                    logger.info(f"✅ Лимит {limit} достигнут")
                    return metadata_list
                
                try:
                    # Генерация
                    result, meta = self.generate(ref_image)
                    
                    # Сохранение
                    filename = f"bg_{generated:06d}_ref{ref_idx:03d}_v{variant:03d}.png"
                    result.save(output_dir / filename, "PNG")
                    
                    meta.update({
                        "filename": filename,
                        "reference": ref_path.name,
                        "variant": variant
                    })
                    metadata_list.append(meta)
                    
                    generated += 1
                    
                    if generated % 10 == 0:
                        logger.info(f"   📊 Сгенерировано: {generated}")
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка: {e}")
                    torch.cuda.empty_cache()
                    continue
        
        # Сохранение метаданных
        save_json(metadata_list, output_dir / "metadata.json")
        logger.info(f"🎉 Завершено! Всего: {generated}")
        
        return metadata_list


def main():
    parser = argparse.ArgumentParser(description="Генерация синтетических фонов")
    
    # Основные параметры
    parser.add_argument("--input_dir", default="data/256_yolo/balanced_clean_patches/train", help="Директория с изображениями 256x256")
    parser.add_argument("--output_dir", default="data/dataset_synthetic/clean_patches", help="Куда сохранять результат 640x640")
    parser.add_argument("--variants", type=int, default=5, help="Вариантов на одно изображение")
    parser.add_argument("--limit", type=int, default=None, help="Общий лимит генерации")
    
    # Параметры генерации
    parser.add_argument("--strength_min", type=float, default=0.35)
    parser.add_argument("--strength_max", type=float, default=0.45)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_ip_adapter", action="store_true")
    
    args = parser.parse_args()
    
    # Инициализация
    print_system_info()
    set_seed(args.seed)
    
    config = GenerationConfig(
        strength_min=args.strength_min,
        strength_max=args.strength_max,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        use_ip_adapter=not args.no_ip_adapter
    )
    
    logger.info("=" * 60)
    logger.info("🎨 ГЕНЕРАЦИЯ ФОНОВ (256 -> 1024 -> 640 + АУГМЕНТАЦИЯ)")
    logger.info("=" * 60)
    logger.info(f"Вход: {args.input_dir} (256x256)")
    logger.info(f"Выход: {args.output_dir} (640x640)")
    logger.info(f"Промежуточный размер: 1024x1024")
    logger.info(f"IP-Adapter: {'вкл' if config.use_ip_adapter else 'выкл'}")
    logger.info(f"Аугментация: флип(0.5), яркость(0.3, 0.9-1.1), контраст(0.3, 0.9-1.1)")
    
    # Загрузка изображений
    references = load_images_from_dir(args.input_dir)
    if not references:
        logger.error(f"❌ Нет изображений в {args.input_dir}")
        sys.exit(1)
    
    logger.info(f"📦 Загружено изображений: {len(references)}")
    
    # Генерация
    generator = BackgroundGenerator(config)
    generator.generate_batch(
        references=references,
        output_dir=Path(args.output_dir),
        variants=args.variants,
        limit=args.limit
    )
    
    logger.info("✅ Готово!")


if __name__ == "__main__":
    main()