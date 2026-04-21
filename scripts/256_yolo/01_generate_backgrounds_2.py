#!/usr/bin/env python3
"""
01_generate_backgrounds.py - Генерация синтетических фонов в Docker
Генерация сразу в 256x256 со спектральным контролем
"""

import sys
import os
sys.path.insert(0, '/app/scripts')
sys.path.insert(0, '/app/src')

import torch
import cv2
import random
import argparse
from PIL import Image, ImageEnhance
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import numpy as np

from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, AutoencoderKL

from config import GenerationConfig
from utils import set_seed, load_images_from_dir, save_json, print_system_info, logger


# ===== ФУНКЦИИ СПЕКТРАЛЬНОГО КОНТРОЛЯ =====
def match_spectrum(source: Image.Image, target: Image.Image) -> Image.Image:
    """Приведение спектра к референсу"""
    src = np.array(source).astype(np.float32)
    tgt = np.array(target).astype(np.float32)
    
    src_f = np.fft.fft2(src, axes=(0, 1))
    tgt_f = np.fft.fft2(tgt, axes=(0, 1))
    
    src_phase = np.angle(src_f)
    tgt_amp = np.abs(tgt_f)
    
    result_f = tgt_amp * np.exp(1j * src_phase)
    result = np.fft.ifft2(result_f, axes=(0, 1)).real
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def inject_high_freq(source: Image.Image, target: Image.Image, alpha: float = 0.3) -> Image.Image:
    """Инжекция высоких частот из референса"""
    src = np.array(source).astype(np.float32)
    tgt = np.array(target).astype(np.float32)
    
    blur = cv2.GaussianBlur(tgt, (0, 0), sigmaX=3)
    high = tgt - blur
    
    result = src + alpha * high
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)


class BackgroundGenerator:
    """Генератор синтетических фонов со спектральным контролем"""
    
    TARGET_SIZE = (256, 256)
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.use_spectrum_matching = getattr(config, 'use_spectrum_matching', True)
        self.use_high_freq_injection = getattr(config, 'use_high_freq_injection', True)
        self.high_freq_alpha = getattr(config, 'high_freq_alpha', 0.3)
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Настройка пайплайна"""
        logger.info("🔄 Загрузка Stable Diffusion...")
        
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
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.config.model_id,
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=self.config.cache_dir
        ).to(self.config.device)
        
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
        
        if self.config.use_ip_adapter:
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
        
        torch.cuda.empty_cache()
        logger.info("🚀 Генератор готов")
        logger.info(f"📊 Spectrum matching: {'✅ ВКЛ' if self.use_spectrum_matching else '❌ ВЫКЛ'}")
        logger.info(f"🔍 High-freq injection: {'✅ ВКЛ' if self.use_high_freq_injection else '❌ ВЫКЛ'}")
    
    def _augment_image(self, image: Image.Image) -> Image.Image:
        """Аугментация"""
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.9, 1.1))
        
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.9, 1.1))
        
        return image
    
    def generate(self, reference: Image.Image, seed: int = None) -> Tuple[Image.Image, Dict]:
        """Генерация одного фона"""
        
        seed = seed or random.randint(0, 2**32 - 1)
        strength = random.uniform(self.config.strength_min, self.config.strength_max)
        ip_scale = random.uniform(0.6, 0.85)
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        
        pipe_args = {
            "prompt": self.config.prompt,
            "negative_prompt": "blurry, low quality, distorted, text, watermark, cartoon, painting",
            "image": reference,
            "strength": strength,
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
            "generator": generator
        }
        
        if self.config.use_ip_adapter:
            pipe_args["ip_adapter_image"] = reference
            pipe_args["ip_adapter_scale"] = ip_scale
        
        output = self.pipe(**pipe_args)
        
        if hasattr(output, 'images'):
            result = output.images[0]
        elif isinstance(output, tuple):
            result = output[0][0] if isinstance(output[0], list) else output[0]
        else:
            result = output
        
        # ===== СПЕКТРАЛЬНЫЙ КОНТРОЛЬ =====
        if self.use_spectrum_matching:
            result = match_spectrum(result, reference)
        
        if self.use_high_freq_injection:
            result = inject_high_freq(result, reference, alpha=self.high_freq_alpha)
        
        result = self._augment_image(result)
        
        metadata = {
            "seed": seed,
            "strength": round(strength, 3),
            "spectrum_matched": self.use_spectrum_matching,
            "high_freq_injected": self.use_high_freq_injection
        }
        
        if random.random() < 0.05:
            torch.cuda.empty_cache()
        
        return result, metadata
    
    def generate_batch(self, references: List[Path], output_dir: Path, variants: int = 1, limit: int = None) -> List[Dict]:
        """Пакетная генерация"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated = 0
        metadata_list = []
        
        for ref_idx, ref_path in enumerate(references):
            logger.info(f"📁 [{ref_idx+1}/{len(references)}] {ref_path.name}")
            
            ref_image = Image.open(ref_path).convert("RGB")
            if ref_image.size != self.TARGET_SIZE:
                ref_image = ref_image.resize(self.TARGET_SIZE, Image.Resampling.LANCZOS)
            
            for variant in range(variants):
                if limit and generated >= limit:
                    return metadata_list
                
                try:
                    result, meta = self.generate(ref_image)
                    filename = f"bg_{generated:06d}_ref{ref_idx:03d}_v{variant:03d}.png"
                    result.save(output_dir / filename, "PNG")
                    
                    meta.update({"filename": filename, "reference": ref_path.name, "variant": variant})
                    metadata_list.append(meta)
                    generated += 1
                    
                    if generated % 10 == 0:
                        logger.info(f"📊 Сгенерировано: {generated}")
                except Exception as e:
                    logger.error(f"❌ Ошибка: {e}")
                    torch.cuda.empty_cache()
        
        save_json(metadata_list, output_dir / "metadata.json")
        logger.info(f"🎉 Завершено! Всего: {generated}")
        return metadata_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/256_yolo/balanced_clean_patches/train")
    parser.add_argument("--output_dir", default="data/dataset_synthetic/clean_patches")
    parser.add_argument("--variants", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strength_min", type=float, default=0.15)
    parser.add_argument("--strength_max", type=float, default=0.25)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_ip_adapter", action="store_true")
    parser.add_argument("--no_spectrum_matching", action="store_true")
    parser.add_argument("--no_high_freq", action="store_true")
    parser.add_argument("--high_freq_alpha", type=float, default=0.3)
    parser.add_argument("--size", type=int, default=256)
    
    args = parser.parse_args()
    
    print_system_info()
    set_seed(args.seed)
    
    BackgroundGenerator.TARGET_SIZE = (args.size, args.size)
    
    config = GenerationConfig()
    config.strength_min = args.strength_min
    config.strength_max = args.strength_max
    config.guidance_scale = args.guidance_scale
    config.num_inference_steps = args.steps
    config.use_ip_adapter = not args.no_ip_adapter
    config.use_spectrum_matching = not args.no_spectrum_matching
    config.use_high_freq_injection = not args.no_high_freq
    config.high_freq_alpha = args.high_freq_alpha
    
    logger.info("=" * 60)
    logger.info("🎨 ГЕНЕРАЦИЯ ФОНОВ СО СПЕКТРАЛЬНЫМ КОНТРОЛЕМ")
    logger.info("=" * 60)
    
    references = load_images_from_dir(args.input_dir)
    if not references:
        logger.error(f"❌ Нет изображений в {args.input_dir}")
        sys.exit(1)
    
    generator = BackgroundGenerator(config)
    generator.generate_batch(references, Path(args.output_dir), args.variants, args.limit)
    
    logger.info("✅ Готово!")


if __name__ == "__main__":
    main()