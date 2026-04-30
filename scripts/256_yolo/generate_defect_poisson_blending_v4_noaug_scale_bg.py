#!/usr/bin/env python3
"""
generate_defect_poisson_blending.py
Poisson Blending + SD для дефекта и фона + Масштабирование + Color Correction
БЕЗ аугментации фона
ДОБАВЛЕНО: Мягкая SD-генерация фона для сохранения текстуры
"""

import sys
import re
import torch
import cv2
import random
import argparse
import logging
import traceback
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PoissonBlendConfig:
    gradient_mixing: float = 0.0
    # Параметры SD для дефекта
    sd_strength_min: float = 0.25
    sd_strength_max: float = 0.45
    sd_steps: int = 20
    sd_guidance_scale: float = 1.8
    # Параметры SD для фона (значительно мягче)
    bg_sd_strength_min: float = 0.05
    bg_sd_strength_max: float = 0.12
    bg_sd_steps: int = 15
    bg_sd_guidance_scale: float = 1.5
    
    random_seed: int = 42
    use_spectrum_matching: bool = True
    use_high_freq_injection: bool = True
    high_freq_alpha: float = 0.35
    prompt: str = "metal surface defect, scratch, industrial steel texture, metallic sheen, brushed metal"
    negative_prompt: str = "smooth, plastic, wood, rust, glass, organic, painted, colorful, rainbow, vibrant"
    # Промпт для фона - только текстура, без дефектов
    bg_prompt: str = "industrial steel surface, brushed metal texture, metallic sheen, uniform metal"
    bg_negative_prompt: str = "defect, scratch, crack, dent, damage, rust, colorful, rainbow, vibrant"
    
    defect_consistency: float = 0.8
    scale_factors: List[float] = None
    color_correction_strength: float = 0.85
    bg_color_correction_strength: float = 0.6  # Мягче для фона

    def __post_init__(self):
        if self.scale_factors is None:
            self.scale_factors = [1.0, 1.05, 1.1, 1.15]


def color_transfer_lab(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Перенос цвета из source в target в пространстве Lab.
    """
    source_lab = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    
    result_lab = target_lab.copy()
    
    for channel in range(3):
        src_mean, src_std = source_lab[:, :, channel].mean(), source_lab[:, :, channel].std()
        tgt_mean, tgt_std = target_lab[:, :, channel].mean(), target_lab[:, :, channel].std()
        
        if tgt_std > 0:
            result_lab[:, :, channel] = ((target_lab[:, :, channel] - tgt_mean) * (src_std / tgt_std)) + src_mean
    
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
    
    return result_rgb.astype(np.float32)


def adaptive_color_correction(sd_result: np.ndarray, original_bg: np.ndarray,
                             defect_mask: np.ndarray, strength: float = 0.85) -> np.ndarray:
    """
    Адаптивная цветокоррекция SD-результата.
    
    Стратегия:
    1. Для фона (вне маски) - полный color transfer от оригинала
    2. Для дефекта (внутри маски) - частичный color transfer
    3. На границе - плавное смешивание
    
    Это предотвращает появление цветных пятен от SD.
    """
    # Ensure consistent float32 type
    sd_result = sd_result.astype(np.float32)
    original_bg = original_bg.astype(np.float32)
    defect_mask = defect_mask.astype(np.float32)
    
    h, w = defect_mask.shape[:2]
    
    # Расширяем маску для создания зоны перехода
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_outer = cv2.dilate(defect_mask, kernel)
    mask_inner = cv2.erode(defect_mask, kernel)
    
    # Создаём веса для разных зон
    transition_zone = mask_outer - mask_inner
    transition_zone = np.clip(transition_zone, 0, 1)
    
    # Размываем для плавности
    transition_zone = cv2.GaussianBlur(transition_zone, (15, 15), 7)
    
    # Цветокоррекция всего изображения
    corrected_full = color_transfer_lab(original_bg, sd_result).astype(np.float32)
    
    # Частичная коррекция для дефекта (сохраняем 30% оригинальных цветов SD)
    corrected_defect = cv2.addWeighted(sd_result, 1.0 - strength * 0.3, corrected_full, strength * 0.3, 0)
    
    # Веса для смешивания
    bg_weight = mask_outer * strength
    defect_weight = mask_inner * (1.0 - strength * 0.3)
    
    # Собираем результат
    result = sd_result.copy()
    
    # Применяем коррекцию к фону (расширяем маски до 3 каналов)
    bg_weight_3ch = np.stack([bg_weight] * 3, axis=-1).astype(np.float32)
    defect_weight_3ch = np.stack([defect_weight] * 3, axis=-1).astype(np.float32)
    
    result = corrected_full * bg_weight_3ch + result * (1 - bg_weight_3ch)
    result = corrected_defect * defect_weight_3ch + result * (1 - defect_weight_3ch)
    
    return np.clip(result, 0, 255)


def create_blend_mask(component_mask: np.ndarray, crop_h: int, crop_w: int,
                     feather_inner: int = 3, feather_outer: int = 5) -> np.ndarray:
    mask_float = component_mask.astype(np.float32)
    
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_outer*2+1, feather_outer*2+1))
    mask_outer = cv2.dilate(mask_float, kernel_outer)
    
    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_inner*2+1, feather_inner*2+1))
    mask_inner = cv2.erode(mask_float, kernel_inner)
    
    mask_inner_blurred = cv2.GaussianBlur(mask_inner, (feather_inner*2+1, feather_inner*2+1), feather_inner)
    mask_outer_blurred = cv2.GaussianBlur(mask_outer, (feather_outer*2+1, feather_outer*2+1), feather_outer)
    
    blend_mask = np.maximum(mask_inner_blurred, mask_outer_blurred * 0.7)
    blend_mask = np.clip(blend_mask, 0, 1)
    
    return blend_mask


def apply_multiscale_blend(defect: np.ndarray, background: np.ndarray,
                          component_mask: np.ndarray) -> np.ndarray:
    h, w = component_mask.shape
    mask_float = component_mask.astype(np.float32)
    
    kernel_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    core_mask = cv2.erode(mask_float, kernel_core, iterations=2)
    core_mask = cv2.GaussianBlur(core_mask, (5, 5), 2)
    
    main_mask = cv2.GaussianBlur(mask_float, (7, 7), 3)
    
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    outer_mask = cv2.dilate(mask_float, kernel_outer)
    outer_mask = cv2.GaussianBlur(outer_mask, (15, 15), 7)
    
    core_mask_3ch = np.stack([core_mask] * 3, axis=-1)
    main_mask_3ch = np.stack([main_mask] * 3, axis=-1)
    outer_mask_3ch = np.stack([outer_mask] * 3, axis=-1)
    
    result_core = defect * core_mask_3ch + background * (1 - core_mask_3ch)
    
    blend_mid = defect * 0.85 + background * 0.15
    mid_weight = main_mask_3ch - core_mask_3ch
    mid_weight = np.clip(mid_weight, 0, 1)
    
    blend_outer = defect * 0.4 + background * 0.6
    outer_weight = outer_mask_3ch - main_mask_3ch
    outer_weight = np.clip(outer_weight, 0, 1)
    
    result = background.copy()
    result = result * (1 - outer_mask_3ch) + blend_outer * outer_mask_3ch
    result = result * (1 - mid_weight) + blend_mid * mid_weight
    result = result * (1 - core_mask_3ch) + result_core * core_mask_3ch
    
    transition_zone = outer_mask_3ch - main_mask_3ch
    transition_zone = np.clip(transition_zone, 0, 1)
    
    if transition_zone.max() > 0:
        bg_blur = cv2.GaussianBlur(background, (0, 0), sigmaX=5)
        bg_details = background.astype(np.float32) - bg_blur.astype(np.float32)
        result = result + bg_details * transition_zone * 0.3
    
    return np.clip(result, 0, 255).astype(np.uint8)


def scale_defect_and_mask(background_crop: np.ndarray, crop_comp_mask: np.ndarray,
                         scale_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    if scale_factor == 1.0:
        return background_crop, crop_comp_mask
    
    h, w = background_crop.shape[:2]
    
    moments = cv2.moments(crop_comp_mask)
    if moments['m00'] == 0:
        return background_crop, crop_comp_mask
    
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    scaled_bg = cv2.resize(background_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    scaled_mask = cv2.resize(crop_comp_mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled_mask = (scaled_mask > 0.5).astype(np.uint8)
    
    new_cx = int(cx * scale_factor)
    new_cy = int(cy * scale_factor)
    
    x_offset = cx - new_cx
    y_offset = cy - new_cy
    
    output_bg = np.zeros_like(background_crop)
    output_mask = np.zeros_like(crop_comp_mask)
    
    src_x1 = max(0, -x_offset)
    src_y1 = max(0, -y_offset)
    src_x2 = min(new_w, w - x_offset)
    src_y2 = min(new_h, h - y_offset)
    
    dst_x1 = max(0, x_offset)
    dst_y1 = max(0, y_offset)
    dst_x2 = min(w, x_offset + new_w)
    dst_y2 = min(h, y_offset + new_h)
    
    copy_w = min(src_x2 - src_x1, dst_x2 - dst_x1)
    copy_h = min(src_y2 - src_y1, dst_y2 - dst_y1)
    
    if copy_w > 0 and copy_h > 0:
        output_bg[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
            scaled_bg[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w]
        output_mask[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
            scaled_mask[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w]
    
    empty_mask = (output_bg.sum(axis=2) == 0) if len(output_bg.shape) == 3 else (output_bg == 0)
    if len(output_bg.shape) == 3:
        empty_mask_3ch = np.stack([empty_mask] * 3, axis=-1)
        output_bg[empty_mask_3ch] = background_crop[empty_mask_3ch]
    
    return output_bg, output_mask


# ================= RLE =================

def parse_patch_offset(filename: str) -> Tuple[int, int]:
    match = re.search(r'_x(\d+)_w(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 256


def rle_to_patch_mask(rle_string: Optional[str], patch_filename: str,
                      original_width: int = 1600, original_height: int = 256) -> np.ndarray:
    if rle_string is None or pd.isna(rle_string):
        return np.zeros((256, 256), dtype=np.uint8)
    
    rle_str = str(rle_string).strip()
    if rle_str == '' or rle_str.lower() == 'nan':
        return np.zeros((256, 256), dtype=np.uint8)
    
    try:
        offset_x, patch_width = parse_patch_offset(patch_filename)
        numbers = list(map(int, rle_str.split()))
        
        if len(numbers) % 2 != 0:
            logger.warning(f"Нечётное количество чисел в RLE для {patch_filename}")
            return np.zeros((256, 256), dtype=np.uint8)
        
        starts = np.array(numbers[0::2]) - 1
        lengths = np.array(numbers[1::2])
        total_pixels = original_width * original_height
        
        full_mask = np.zeros(total_pixels, dtype=np.uint8)
        for start, length in zip(starts, lengths):
            if start < 0:
                continue
            end = min(start + length, total_pixels)
            if start < total_pixels:
                full_mask[start:end] = 1
        
        full_mask_2d = full_mask.reshape(original_width, original_height).T
        
        if offset_x + patch_width > original_width:
            patch_width = original_width - offset_x
        
        patch_mask = full_mask_2d[:, offset_x:offset_x + patch_width]
        
        if patch_mask.shape != (256, 256):
            patch_mask = cv2.resize(patch_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        return patch_mask.astype(np.uint8)
    
    except Exception as e:
        logger.error(f"Ошибка декодирования RLE для {patch_filename}: {e}")
        return np.zeros((256, 256), dtype=np.uint8)


def rle_to_defect_bboxes(rle_string: Optional[str], patch_filename: str,
                         class_id: int) -> List[Dict]:
    mask = rle_to_patch_mask(rle_string, patch_filename)
    
    if mask.sum() == 0:
        return []
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bboxes = []
    kernel = np.ones((3, 3), np.uint8)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        if area < 16 or w < 4 or h < 4:
            continue
        
        component_mask = (labels == i).astype(np.uint8)
        dilated_once = cv2.dilate(component_mask, kernel, iterations=1)
        boundary_mask = dilated_once - component_mask
        
        bboxes.append({
            'class': class_id,
            'x_center': (x + w/2) / 256,
            'y_center': (y + h/2) / 256,
            'width': w / 256,
            'height': h / 256,
            'component_mask': component_mask,
            'boundary_mask': boundary_mask,
            'eroded_mask': cv2.erode(component_mask, kernel, iterations=1),
            'x': x, 'y': y, 'w': w, 'h': h,
            'centroid': (int(centroids[i, 0]), int(centroids[i, 1]))
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


# ================= SD ГЕНЕРАЦИЯ =================

class SDDefectGenerator:
    def __init__(self, config: PoissonBlendConfig, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA недоступна, используется CPU")
        
        self.device = device
        self.config = config
        
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, AutoencoderKL
        
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        logger.info("Загрузка Stable Diffusion...")
        
        # Загружаем улучшенный VAE
        try:
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            logger.info("✅ Загружен улучшенный VAE (MSE)")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить MSE VAE: {e}")
            vae = None
            logger.info("ℹ️ Используется стандартный VAE")
        
        # Основной пайплайн с VAE
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            vae=vae,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # НЕ используем .to(device) при offload
        # Вместо этого перемещаем на устройство вручную только если не CUDA
        if device == "cpu":
            self.pipe = self.pipe.to(device)
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Полностью отключаем tiling
        if hasattr(self.pipe.vae, 'disable_tiling'):
            self.pipe.vae.disable_tiling()
            logger.info("ℹ️ Tiling VAE отключен")
        
        # Для CUDA используем ТОЛЬКО attention slicing, БЕЗ offload
        if device == "cuda":
            self.pipe = self.pipe.to(device)
            try:
                self.pipe.enable_attention_slicing(slice_size="auto")
                logger.info("✅ Attention slicing включен (auto)")
            except Exception as e:
                logger.warning(f"Не удалось включить attention slicing: {e}")
                try:
                    self.pipe.enable_attention_slicing(1)
                    logger.info("✅ Attention slicing включен (slice=1)")
                except Exception as e2:
                    logger.warning(f"Не удалось включить attention slicing вообще: {e2}")
        
        # Предварительно прогреваем VAE на тестовом тензоре
        self._warmup_vae()
        
        logger.info("SD готов")
    
    def _warmup_vae(self):
        """Прогрев VAE для предотвращения ошибок с пустыми тензорами"""
        try:
            logger.info("Прогрев VAE...")
            # Создаём тестовый латентный тензор правильного размера
            test_latent = torch.randn(1, 4, 64, 64, 
                                     device=self.device if self.device == "cuda" else "cpu",
                                     dtype=torch.float16 if self.device == "cuda" else torch.float32)
            
            # Прогоняем через VAE
            with torch.no_grad():
                if hasattr(self.pipe.vae, 'decode'):
                    _ = self.pipe.vae.decode(test_latent).sample
            
            # Очищаем кеш
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ VAE прогрет успешно")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось прогреть VAE: {e}")
    
    @torch.no_grad()
    def generate_defect(self, defect_crop: Image.Image, seed: int = None) -> np.ndarray:
        """Генерация дефекта с высокой strength для разнообразия"""
        w, h = defect_crop.size
        
        # Ресайзим до размера, кратного 64, минимум 512
        sd_w = max(((w + 63) // 64) * 64, 512)
        sd_h = max(((h + 63) // 64) * 64, 512)
        
        crop_sd = defect_crop.resize((sd_w, sd_h), Image.Resampling.LANCZOS)
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device=self.device if torch.cuda.is_available() else "cpu").manual_seed(seed)
        strength = random.uniform(self.config.sd_strength_min, self.config.sd_strength_max)
        
        if random.random() > self.config.defect_consistency:
            return np.array(defect_crop).astype(np.float32)
        
        try:
            output = self.pipe(
                prompt=self.config.prompt,
                negative_prompt=self.config.negative_prompt,
                image=crop_sd,
                strength=strength,
                guidance_scale=self.config.sd_guidance_scale,
                num_inference_steps=self.config.sd_steps,
                generator=generator,
                height=sd_h,
                width=sd_w
            )
            
            generated = output.images[0]
            
            if generated.size != (w, h):
                generated = generated.resize((w, h), Image.Resampling.LANCZOS)
            
            generated_np = np.array(generated).astype(np.float32)
            crop_np = np.array(defect_crop).astype(np.float32)
            
            if self.config.color_correction_strength > 0:
                generated_np = adaptive_color_correction(
                    generated_np, crop_np,
                    np.ones((h, w), dtype=np.uint8),
                    self.config.color_correction_strength
                )
            
            if self.config.use_spectrum_matching:
                generated_np = match_spectrum(generated_np, crop_np)
            
            if self.config.use_high_freq_injection:
                generated_np = inject_high_freq(generated_np, crop_np, self.config.high_freq_alpha)
            
            return generated_np
        
        except Exception as e:
            logger.error(f"SD defect generation error: {e}")
            traceback.print_exc()
            return np.array(defect_crop).astype(np.float32)

    @torch.no_grad()
    def generate_background(self, bg_crop: Image.Image, seed: int = None) -> np.ndarray:
        """Мягкая генерация фона для сохранения текстуры.
        При ошибке возвращает исходный фон без изменений."""
        w, h = bg_crop.size
        
        # Ресайзим до размера, кратного 64, минимум 512
        sd_w = max(((w + 63) // 64) * 64, 512)
        sd_h = max(((h + 63) // 64) * 64, 512)
        
        crop_sd = bg_crop.resize((sd_w, sd_h), Image.Resampling.LANCZOS)
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device=self.device if torch.cuda.is_available() else "cpu").manual_seed(seed)
        strength = random.uniform(self.config.bg_sd_strength_min, self.config.bg_sd_strength_max)
        
        try:
            output = self.pipe(
                prompt=self.config.bg_prompt,
                negative_prompt=self.config.bg_negative_prompt,
                image=crop_sd,
                strength=strength,
                guidance_scale=self.config.bg_sd_guidance_scale,
                num_inference_steps=self.config.bg_sd_steps,
                generator=generator,
                height=sd_h,
                width=sd_w
            )
            
            generated = output.images[0]
            
            if generated.size != (w, h):
                generated = generated.resize((w, h), Image.Resampling.LANCZOS)
            
            generated_np = np.array(generated).astype(np.float32)
            crop_np = np.array(bg_crop).astype(np.float32)
            
            if self.config.bg_color_correction_strength > 0:
                generated_np = color_transfer_lab(crop_np, generated_np)
            
            if self.config.use_spectrum_matching:
                generated_np = match_spectrum(generated_np, crop_np).astype(np.float32)
            
            if self.config.use_high_freq_injection:
                generated_np = inject_high_freq(generated_np, crop_np, self.config.high_freq_alpha).astype(np.float32)
            
            result = cv2.addWeighted(crop_np, 0.5, generated_np, 0.5, 0)
            
            return result
        
        except Exception as e:
            logger.error(f"SD background generation error: {e}")
            logger.warning("Пропускаю SD-генерацию фона, используется оригинальный фон")
            # Возвращаем исходный фон без изменений
            return np.array(bg_crop).astype(np.float32)
            



# ================= ГЛАВНЫЙ ГЕНЕРАТОР =================

class PoissonDefectGenerator:
    def __init__(self, config: PoissonBlendConfig = None):
        self.config = config or PoissonBlendConfig()
        
        logger.info("Инициализация PoissonDefectGenerator...")
        logger.info(f"Масштабирование: {self.config.scale_factors}")
        logger.info(f"SD defect strength: {self.config.sd_strength_min}-{self.config.sd_strength_max}")
        logger.info(f"SD background strength: {self.config.bg_sd_strength_min}-{self.config.bg_sd_strength_max}")
        logger.info(f"Color correction: {self.config.color_correction_strength}")
        
        try:
            self.sd_generator = SDDefectGenerator(self.config)
            logger.info("SD генератор успешно загружен")
        except Exception as e:
            logger.error(f"Ошибка загрузки SD генератора: {e}")
            self.sd_generator = None
            raise
        
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)
    
    def process_image_all_defects(self, img_path: Path, all_bboxes: List[Dict],
                                  output_dir: Path, variant: int, total_idx: int) -> Optional[Dict]:
        if self.sd_generator is None:
            logger.error("SD генератор не инициализирован!")
            return None
        
        try:
            original = cv2.imread(str(img_path))
            if original is None:
                logger.warning(f"Не удалось загрузить {img_path}")
                return None
            
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            img_h, img_w = original.shape[:2]
            
            # ===== МЯГКАЯ SD-ГЕНЕРАЦИЯ ВСЕГО ФОНА (с fallback) =====
            original_pil = Image.fromarray(original)  # <-- вынесли из try-except
            bg_seed = self.config.random_seed + total_idx * 1000 + variant * 100
            
            # generate_background теперь сам обрабатывает ошибки
            background = self.sd_generator.generate_background(original_pil, seed=bg_seed)
            background = np.clip(background, 0, 255).astype(np.uint8)
            # =========================================================
            
            result = background.copy().astype(np.float32)
            yolo_annotations = []
            
            for bbox_idx, bbox_info in enumerate(all_bboxes):
                x, y, w, h = bbox_info['x'], bbox_info['y'], bbox_info['w'], bbox_info['h']
                comp_mask = bbox_info['component_mask']
                boundary_mask = bbox_info.get('boundary_mask', comp_mask)
                
                pad = int(max(w, h) * 0.3)
                pad = max(pad, 4)
                
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(img_w, x + w + pad)
                y2 = min(img_h, y + h + pad)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop_w = x2 - x1
                crop_h = y2 - y1
                
                if crop_w < 16 or crop_h < 16:
                    continue
                
                crop_comp_mask = comp_mask[y1:y2, x1:x2].copy()
                
                if crop_comp_mask.sum() == 0:
                    continue
                
                # Берём кроп из СГЕНЕРИРОВАННОГО фона
                background_crop = background[y1:y2, x1:x2].copy()
                
                # Масштабирование
                scale_factor = random.choice(self.config.scale_factors)
                scaled_bg_crop, scaled_comp_mask = scale_defect_and_mask(
                    background_crop, crop_comp_mask, scale_factor
                )
                
                # SD генерация дефекта (более агрессивная)
                background_crop_pil = Image.fromarray(
                    np.clip(scaled_bg_crop, 0, 255).astype(np.uint8)
                )
                
                seed = self.config.random_seed + total_idx * 100 + variant * 10 + bbox_idx
                
                try:
                    generated_np = self.sd_generator.generate_defect(background_crop_pil, seed=seed)
                except Exception as e:
                    logger.error(f"Ошибка SD генерации дефекта: {e}")
                    continue
                
                if generated_np.shape[:2] != (crop_h, crop_w):
                    generated_np = cv2.resize(
                        generated_np, (crop_w, crop_h), 
                        interpolation=cv2.INTER_LANCZOS4
                    )
                
                # Цветокоррекция дефекта
                if self.config.color_correction_strength > 0:
                    generated_np = adaptive_color_correction(
                        generated_np,
                        background_crop.astype(np.float32),
                        scaled_comp_mask,
                        self.config.color_correction_strength
                    )
                
                # Смешиваем SD-дефект с SD-фоном
                blended = apply_multiscale_blend(
                    generated_np,
                    background_crop.astype(np.float32),
                    scaled_comp_mask
                )
                
                if blended.shape[:2] != (crop_h, crop_w):
                    blended = cv2.resize(blended, (crop_w, crop_h))
                
                result[y1:y2, x1:x2] = blended.astype(np.float32)
                
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
            
            result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
            
            # Лёгкая постобработка
            blur = cv2.GaussianBlur(result_uint8, (0, 0), sigmaX=1.0)
            result_uint8 = cv2.addWeighted(result_uint8, 1.1, blur, -0.1, 0)
            result_uint8 = np.clip(result_uint8, 0, 255).astype(np.uint8)
            
            result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(output_dir / "images" / filename),
                result_bgr,
                [cv2.IMWRITE_PNG_COMPRESSION, 3]
            )
            
            with open(output_dir / "labels" / f"{Path(filename).stem}.txt", 'w') as f:
                for ann in yolo_annotations:
                    f.write(
                        f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                        f"{ann['width']:.6f} {ann['height']:.6f}\n"
                    )
            
            return {"filename": filename, "num_defects": len(yolo_annotations)}
        
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
            traceback.print_exc()
            return None
    
    def generate_dataset(self, input_dir: Path, rle_csv: Path,
                        output_dir: Path, variants: int = 3,
                        limit: Optional[int] = None) -> int:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        rle_df = pd.read_csv(rle_csv)
        groups = rle_df.groupby('ImageId')
        
        logger.info(f"Найдено патчей: {len(groups)}")
        logger.info(f"  С несколькими дефектами: {(groups.size() > 1).sum()}")
        
        all_images = {}
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img in input_dir.glob(ext):
                all_images[img.name] = img
                all_images[img.stem] = img
        
        stats = {'total': 0, 'errors': 0, 'defects': 0}
        
        for image_id, group in tqdm(groups, desc="SD bg + SD defect + Scale"):
            if limit and stats['total'] >= limit:
                break
            
            img_path = all_images.get(image_id)
            if img_path is None:
                logger.warning(f"Изображение не найдено: {image_id}")
                continue
            
            all_bboxes = []
            for _, row in group.iterrows():
                rle = row.get('EncodedPixels')
                if pd.isna(rle) or str(rle).strip().lower() in ['', 'nan']:
                    continue
                all_bboxes.extend(
                    rle_to_defect_bboxes(str(rle), image_id, int(row['ClassId']) - 1)
                )
            
            if not all_bboxes:
                continue
            
            for v in range(variants):
                if limit and stats['total'] >= limit:
                    break
                
                result = self.process_image_all_defects(
                    img_path, all_bboxes, output_dir, v, stats['total']
                )
                
                if result:
                    stats['total'] += 1
                    stats['defects'] += result['num_defects']
                else:
                    stats['errors'] += 1
                
                if stats['total'] % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        logger.info(f"Итого: {stats['total']} изображений, "
                   f"{stats['defects']} дефектов, {stats['errors']} ошибок")
        return stats['total']


# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser(
        description="SD фон + SD дефект + Масштабирование + Color Correction"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--rle_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sd_strength_min", type=float, default=0.15)
    parser.add_argument("--sd_strength_max", type=float, default=0.25)
    parser.add_argument("--bg_strength_min", type=float, default=0.05)
    parser.add_argument("--bg_strength_max", type=float, default=0.12)
    parser.add_argument("--sd_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_spectrum", action="store_true")
    parser.add_argument("--no_high_freq", action="store_true")
    parser.add_argument("--color_correction", type=float, default=0.85,
                       help="Сила цветокоррекции дефекта")
    parser.add_argument("--bg_color_correction", type=float, default=0.6,
                       help="Сила цветокоррекции фона")
    parser.add_argument("--prompt", type=str,
                       default="metal surface defect, scratch, industrial steel texture, metallic sheen, brushed metal")
    parser.add_argument("--negative_prompt", type=str,
                       default="smooth, plastic, wood, rust, glass, organic, painted, colorful, rainbow, vibrant")
    parser.add_argument("--bg_prompt", type=str,
                       default="industrial steel surface, brushed metal texture, metallic sheen, uniform metal")
    parser.add_argument("--bg_negative_prompt", type=str,
                       default="defect, scratch, crack, dent, damage, rust, colorful, rainbow, vibrant")
    
    args = parser.parse_args()
    
    config = PoissonBlendConfig(
        random_seed=args.seed,
        sd_steps=args.sd_steps,
        sd_guidance_scale=args.guidance_scale,
        sd_strength_min=args.sd_strength_min,
        sd_strength_max=args.sd_strength_max,
        bg_sd_strength_min=args.bg_strength_min,
        bg_sd_strength_max=args.bg_strength_max,
        use_spectrum_matching=not args.no_spectrum,
        use_high_freq_injection=not args.no_high_freq,
        color_correction_strength=args.color_correction,
        bg_color_correction_strength=args.bg_color_correction,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        bg_prompt=args.bg_prompt,
        bg_negative_prompt=args.bg_negative_prompt
    )
    
    generator = PoissonDefectGenerator(config)
    total = generator.generate_dataset(
        Path(args.input_dir),
        Path(args.rle_csv),
        Path(args.output_dir),
        args.variants,
        args.limit
    )
    
    logger.info(f"Готово! Сгенерировано {total} изображений")


if __name__ == "__main__":
    main()