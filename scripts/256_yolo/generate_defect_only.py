#!/usr/bin/env python3
"""
generate_defect_only.py - Генерация синтетики ТОЛЬКО в областях дефектов
Все дефекты на изображении изменяются ОДНОВРЕМЕННО
RLE трансформируется с учётом сдвига патча
"""

import sys
import os
import re
import torch
import cv2
import random
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, AutoencoderKL

sys.path.insert(0, '/app/scripts')
sys.path.insert(0, '/app/src')
from config import GenerationConfig
from utils import set_seed, print_system_info, logger


# ================= RLE С УЧЁТОМ СДВИГА ПАТЧА =================

def parse_patch_offset(filename: str) -> Tuple[int, int]:
    match = re.search(r'_x(\d+)_w(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 256


def rle_to_patch_mask(rle_string: str, patch_filename: str,
                      original_width: int = 1600, original_height: int = 256) -> np.ndarray:
    if pd.isna(rle_string) or str(rle_string).strip() in ['', 'nan']:
        return np.zeros((256, 256), dtype=np.uint8)
    
    offset_x, _ = parse_patch_offset(patch_filename)
    
    numbers = list(map(int, str(rle_string).split()))
    starts = np.array(numbers[0::2]) - 1
    lengths = np.array(numbers[1::2])
    
    full_mask = np.zeros(original_width * original_height, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        if start < len(full_mask):
            full_mask[start:min(start + length, len(full_mask))] = 1
    
    full_mask_2d = full_mask.reshape(original_width, original_height).T
    return full_mask_2d[:, offset_x:offset_x + 256]


def rle_to_defect_bboxes(rle_string: str, patch_filename: str, class_id: int) -> List[Dict]:
    mask = rle_to_patch_mask(rle_string, patch_filename)
    if mask.sum() == 0:
        return []
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 16 or w < 4 or h < 4:
            continue
        bboxes.append({
            'class': class_id,
            'x_center': (x + w/2) / 256,
            'y_center': (y + h/2) / 256,
            'width': w / 256,
            'height': h / 256,
            'component_mask': (labels == i).astype(np.uint8),
            'x': x, 'y': y, 'w': w, 'h': h
        })
    return bboxes


# ================= СПЕКТРАЛЬНЫЙ КОНТРОЛЬ =================

def match_spectrum(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src_f = np.fft.fft2(source.astype(np.float32), axes=(0, 1))
    tgt_f = np.fft.fft2(target.astype(np.float32), axes=(0, 1))
    result = np.fft.ifft2(np.abs(tgt_f) * np.exp(1j * np.angle(src_f)), axes=(0, 1)).real
    return np.clip(result, 0, 255).astype(np.uint8)


def inject_high_freq(source: np.ndarray, target: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    blur = cv2.GaussianBlur(target.astype(np.float32), (0, 0), sigmaX=3)
    result = source.astype(np.float32) + alpha * (target.astype(np.float32) - blur)
    return np.clip(result, 0, 255).astype(np.uint8)


# ================= ГЛАВНЫЙ КЛАСС =================

class DefectOnlyGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        
        logger.info("🔄 Загрузка Stable Diffusion...")
        try:
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16, cache_dir=config.cache_dir)
        except:
            vae = None
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.model_id, vae=vae, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False,
            cache_dir=config.cache_dir
        ).to(config.device)
        
        if hasattr(self.pipe.vae, 'enable_tiling'):
            self.pipe.vae.enable_tiling()
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        torch.cuda.empty_cache()
        logger.info("🚀 Готов (defect-only, все дефекты одновременно)")
    
    def generate_defect_region(self, crop: Image.Image) -> np.ndarray:
        w, h = crop.size
        sd_w = max(64, ((w + 7) // 8) * 8)
        sd_h = max(64, ((h + 7) // 8) * 8)
        
        crop_sd = crop.resize((sd_w, sd_h), Image.Resampling.LANCZOS)
        
        strength = random.uniform(self.config.strength_min, self.config.strength_max)
        generator = torch.Generator(device=self.config.device).manual_seed(
            random.randint(0, 2**32 - 1))
        
        output = self.pipe(
            prompt=self.config.prompt,
            negative_prompt=self.config.negative_prompt,
            image=crop_sd, strength=strength,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            generator=generator
        )
        
        generated = output.images[0]
        if sd_w != w or sd_h != h:
            generated = generated.resize((w, h), Image.Resampling.LANCZOS)
        
        generated_np = np.array(generated).astype(np.float32)
        crop_np = np.array(crop).astype(np.float32)
        
        if self.config.use_spectrum_matching:
            generated_np = match_spectrum(generated_np, crop_np)
        if self.config.use_high_freq_injection:
            generated_np = inject_high_freq(generated_np, crop_np, self.config.high_freq_alpha)
        
        return generated_np
    
    def process_image_all_defects(
        self, img_path: Path, all_bboxes: List[Dict],
        output_dir: Path, variant: int, total_idx: int
    ) -> Optional[Dict]:
        """Изменяет ВСЕ дефекты на одном изображении ОДНОВРЕМЕННО"""
        try:
            original = Image.open(img_path).convert("RGB")
            original_np = np.array(original).astype(np.float32)
            result_np = original_np.copy()
            yolo_annotations = []
            
            for bbox_info in all_bboxes:
                x, y, w, h = bbox_info['x'], bbox_info['y'], bbox_info['w'], bbox_info['h']
                comp_mask = bbox_info['component_mask']
                
                pad = int(max(w, h) * 0.3)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(256, x + w + pad), min(256, y + h + pad)
                
                if x2 - x1 < 16 or y2 - y1 < 16:
                    continue
                
                # Вырезаем из ОРИГИНАЛА
                crop = original.crop((x1, y1, x2, y2))
                crop_w, crop_h = x2 - x1, y2 - y1
                
                generated_np = self.generate_defect_region(crop)
                if generated_np.shape[:2] != (crop_h, crop_w):
                    generated_np = cv2.resize(generated_np, (crop_w, crop_h))
                
                crop_mask = comp_mask[y1:y2, x1:x2].astype(np.float32)
                ksize = min(7, min(crop_w, crop_h) // 4 * 2 + 1)
                feathered = cv2.GaussianBlur(crop_mask, (ksize, ksize), ksize // 2)
                feathered_3ch = np.stack([feathered] * 3, axis=-1)
                
                target = result_np[y1:y2, x1:x2]
                if generated_np.shape != target.shape:
                    continue
                if feathered_3ch.shape != target.shape:
                    feathered_3ch = cv2.resize(feathered_3ch, (target.shape[1], target.shape[0]))
                    if len(feathered_3ch.shape) == 2:
                        feathered_3ch = np.stack([feathered_3ch] * 3, axis=-1)
                
                result_np[y1:y2, x1:x2] = (
                    generated_np * feathered_3ch + target * (1 - feathered_3ch))
                
                yolo_annotations.append({
                    'class': bbox_info['class'],
                    'x_center': bbox_info['x_center'],
                    'y_center': bbox_info['y_center'],
                    'width': bbox_info['width'],
                    'height': bbox_info['height']
                })
            
            if not yolo_annotations:
                return None
            
            stem = img_path.stem
            filename = f"syn_{total_idx:06d}_{stem}_v{variant}.png"
            
            Image.fromarray(np.clip(result_np, 0, 255).astype(np.uint8)).save(
                output_dir / "images" / filename, "PNG", optimize=True)
            
            with open(output_dir / "labels" / f"{Path(filename).stem}.txt", 'w') as f:
                for ann in yolo_annotations:
                    f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                           f"{ann['width']:.6f} {ann['height']:.6f}\n")
            
            return {"filename": filename, "num_defects": len(yolo_annotations)}
        
        except Exception as e:
            logger.error(f"❌ {img_path.name}: {e}")
            return None
    
    def generate_dataset(
        self, input_dir: Path, rle_csv: Path, output_dir: Path,
        variants: int, limit: Optional[int] = None
    ):
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        rle_df = pd.read_csv(rle_csv)
        groups = rle_df.groupby('ImageId')
        logger.info(f"📂 Патчей: {len(groups)} (с несколькими дефектами: {(groups.size() > 1).sum()})")
        
        all_images = {img.name: img for img in input_dir.glob("*")
                     if img.suffix.lower() in ['.png', '.jpg', '.jpeg']}
        
        total = 0
        for image_id, group in tqdm(groups, desc="Генерация"):
            if limit and total >= limit:
                break
            if image_id not in all_images:
                continue
            
            img_path = all_images[image_id]
            
            # Собираем ВСЕ дефекты
            all_bboxes = []
            for _, row in group.iterrows():
                rle = str(row['EncodedPixels'])
                if rle == 'nan' or rle.strip() == '':
                    continue
                all_bboxes.extend(
                    rle_to_defect_bboxes(rle, image_id, int(row['ClassId']) - 1))
            
            if not all_bboxes:
                continue
            
            # Генерируем варианты (все дефекты одновременно)
            for v in range(variants):
                if limit and total >= limit:
                    break
                self.process_image_all_defects(img_path, all_bboxes, output_dir, v, total)
                total += 1
            
            if total % 20 == 0:
                torch.cuda.empty_cache()
        
        logger.info(f"✅ Сгенерировано: {total} изображений")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--rle_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strength_min", type=float, default=0.08)
    parser.add_argument("--strength_max", type=float, default=0.15)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--high_freq_alpha", type=float, default=0.40)
    parser.add_argument("--prompt", type=str,
                       default="metal surface defect, scratch, industrial steel texture")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = GenerationConfig()
    config.strength_min = args.strength_min
    config.strength_max = args.strength_max
    config.guidance_scale = args.guidance_scale
    config.num_inference_steps = args.steps
    config.prompt = args.prompt
    config.negative_prompt = "smooth, plastic, wood, rust"
    config.use_spectrum_matching = True
    config.use_high_freq_injection = True
    config.high_freq_alpha = args.high_freq_alpha
    
    generator = DefectOnlyGenerator(config)
    generator.generate_dataset(
        Path(args.input_dir), Path(args.rle_csv),
        Path(args.output_dir), args.variants, args.limit)
    
    logger.info("✅ Готово!")


if __name__ == "__main__":
    main()