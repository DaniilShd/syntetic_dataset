#!/usr/bin/env python3
"""
generate_defect_poisson_blending_v5_orig_bg.py
Poisson Blending + Случайный чистый фон + Аугментация фона ПЕРЕД вставкой дефекта
Основано на: Pérez et al., "Poisson Image Editing", SIGGRAPH 2003
ИСПРАВЛЕНИЯ: подбор фона по яркости, сглаживание маски от рамок
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
import albumentations as A
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
    sd_strength_min: float = 0.08
    sd_strength_max: float = 0.15
    sd_steps: int = 25
    sd_guidance_scale: float = 2.0
    random_seed: int = 42
    augment_background: bool = True
    use_spectrum_matching: bool = True
    use_high_freq_injection: bool = True
    high_freq_alpha: float = 0.40
    prompt: str = "metal surface defect, scratch, industrial steel texture"
    negative_prompt: str = "smooth, plastic, wood, rust"
    clean_dir: Optional[Path] = None
    brightness_threshold: float = 25.0  # ★ Порог разницы яркости


# ========== ВСЕ ОРИГИНАЛЬНЫЕ ФУНКЦИИ БЕЗ ИЗМЕНЕНИЙ ==========

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
    return np.clip(blend_mask, 0, 1)




def adapt_defect_brightness(defect: np.ndarray, background: np.ndarray, 
                            mask: np.ndarray) -> np.ndarray:
    """
    ★ Адаптирует ТОЛЬКО КРАЯ дефекта под фон.
    Центр остаётся оригинальным.
    """
    mask_float = mask.astype(np.float32)
    
    if mask_float.sum() < 10:
        return defect
    
    mask_3ch = np.stack([mask_float] * 3, axis=-1)
    
    # Средний цвет фона ПОД дефектом
    bg_under_mask = background.astype(np.float32) * mask_3ch
    bg_mean = np.sum(bg_under_mask, axis=(0, 1)) / max(mask_float.sum(), 1)
    
    # Средний цвет дефекта
    defect_mean = np.sum(defect.astype(np.float32) * mask_3ch, axis=(0, 1)) / max(mask_float.sum(), 1)
    
    bg_brightness = np.mean(bg_mean)
    defect_brightness = np.mean(defect_mean)
    
    threshold = 5.0
    
    if abs(bg_brightness - defect_brightness) > threshold:
        # Коррекция яркости
        brightness_ratio = bg_brightness / max(defect_brightness, 1.0)
        brightness_ratio = np.clip(brightness_ratio, 0.4, 2.5)
        
        defect_adapted = defect.astype(np.float32) * brightness_ratio
        
        # Поканальная коррекция цвета
        channel_ratios = bg_mean / np.maximum(defect_mean, 1.0)
        channel_ratios = np.clip(channel_ratios, 0.5, 2.0)
        defect_adapted = defect_adapted * channel_ratios
        
        # ★ ТОЛЬКО КРАЯ: distance transform от границы
        dist = cv2.distanceTransform((1 - mask_float).astype(np.uint8), cv2.DIST_L2, 5)
        max_dist = dist.max()
        if max_dist > 0:
            dist = dist / max_dist
        
        # ★ Зона адаптации: только первые 30% от границы
        # Ближе к границе — больше адаптации, дальше — оригинал
        edge_zone = np.clip(1 - dist / 0.3, 0, 1)  # 1 на границе, 0 в центре
        edge_zone = cv2.GaussianBlur(edge_zone, (9, 9), 4)  # Плавный переход
        
        edge_zone_3ch = np.stack([edge_zone] * 3, axis=-1)
        
        # Смешиваем: края адаптированы, центр — оригинал
        defect_result = defect.astype(np.float32) * (1 - edge_zone_3ch) + defect_adapted * edge_zone_3ch
        
        # ★ Если дефект светлый на тёмном — затемняем края сильнее
        if defect_brightness > bg_brightness:
            # Дополнительное затемнение краёв через смешивание с фоном
            boundary_blend = edge_zone_3ch * 0.6  # 60% фон на границе
            defect_result = defect_result * (1 - boundary_blend) + background.astype(np.float32) * boundary_blend
        
        return np.clip(defect_result, 0, 255)
    
    return defect


def apply_multiscale_blend(defect: np.ndarray, background: np.ndarray,
                          component_mask: np.ndarray) -> np.ndarray:
    """
    Многослойное смешивание с предварительной адаптацией яркости дефекта
    """
    h, w = component_mask.shape
    mask_float = component_mask.astype(np.float32)
    
    # ★ АДАПТИРУЕМ ЯРКОСТЬ ДЕФЕКТА ПОД ФОН
    defect_adapted = adapt_defect_brightness(defect, background, mask_float)
    
    # ★ Размываем маску
    mask_float = cv2.GaussianBlur(mask_float, (21, 21), 10)
    
    # Ядро дефекта
    kernel_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    core_mask = cv2.erode(mask_float, kernel_core, iterations=3)
    core_mask = cv2.GaussianBlur(core_mask, (11, 11), 5)
    
    # Основная маска
    main_mask = cv2.GaussianBlur(mask_float, (15, 15), 7)
    
    # Расширенная маска для перехода
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    outer_mask = cv2.dilate(mask_float, kernel_outer)
    outer_mask = cv2.GaussianBlur(outer_mask, (31, 31), 15)
    
    core_mask_3ch = np.stack([core_mask] * 3, axis=-1)
    main_mask_3ch = np.stack([main_mask] * 3, axis=-1)
    outer_mask_3ch = np.stack([outer_mask] * 3, axis=-1)
    
    # ★ Используем адаптированный дефект
    result_core = defect_adapted * (0.9 * core_mask_3ch) + background * (1 - 0.9 * core_mask_3ch)
    
    blend_mid = defect_adapted * 0.6 + background * 0.4
    mid_weight = main_mask_3ch - core_mask_3ch
    mid_weight = np.clip(mid_weight, 0, 1)
    
    blend_outer = defect_adapted * 0.25 + background * 0.75
    outer_weight = outer_mask_3ch - main_mask_3ch
    outer_weight = np.clip(outer_weight, 0, 1)
    
    result = background.copy()
    result = result * (1 - outer_mask_3ch) + blend_outer * outer_mask_3ch
    result = result * (1 - mid_weight) + blend_mid * mid_weight
    result = result * (1 - core_mask_3ch) + result_core * core_mask_3ch
    
    # Добавляем микро-текстуру фона
    transition_zone = outer_mask_3ch - main_mask_3ch
    transition_zone = np.clip(transition_zone, 0, 1)
    
    if transition_zone.max() > 0:
        bg_blur = cv2.GaussianBlur(background, (0, 0), sigmaX=10)
        bg_details = background.astype(np.float32) - bg_blur.astype(np.float32)
        result = result + bg_details * transition_zone * 0.4
    
    return np.clip(result, 0, 255).astype(np.uint8)


def get_background_augmentation() -> A.Compose:
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.4),
    ])


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
        eroded = cv2.erode(component_mask, kernel, iterations=1)
        bboxes.append({
            'class': class_id,
            'x_center': (x + w/2) / 256,
            'y_center': (y + h/2) / 256,
            'width': w / 256,
            'height': h / 256,
            'component_mask': component_mask,
            'boundary_mask': boundary_mask,
            'eroded_mask': eroded,
            'x': x, 'y': y, 'w': w, 'h': h,
            'centroid': (int(centroids[i, 0]), int(centroids[i, 1]))
        })
    return bboxes


def match_spectrum(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src_f = np.fft.fft2(source.astype(np.float32), axes=(0, 1))
    tgt_f = np.fft.fft2(target.astype(np.float32), axes=(0, 1))
    result = np.fft.ifft2(np.abs(tgt_f) * np.exp(1j * np.angle(src_f)), axes=(0, 1)).real
    return np.clip(result, 0, 255).astype(np.uint8)


def inject_high_freq(source: np.ndarray, target: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    blur = cv2.GaussianBlur(target.astype(np.float32), (0, 0), sigmaX=3)
    result = source.astype(np.float32) + alpha * (target.astype(np.float32) - blur)
    return np.clip(result, 0, 255).astype(np.uint8)


def compute_guidance_field(defect_patch: np.ndarray, background_patch: np.ndarray,
                          mask: np.ndarray, mix: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    grad_defect_y, grad_defect_x = np.gradient(defect_patch.astype(np.float64))
    grad_bg_y, grad_bg_x = np.gradient(background_patch.astype(np.float64))
    mag_defect = np.sqrt(grad_defect_y**2 + grad_defect_x**2)
    mag_bg = np.sqrt(grad_bg_y**2 + grad_bg_x**2)
    use_defect = mag_defect > mag_bg
    vy = np.where(use_defect, grad_defect_y, grad_bg_y)
    vx = np.where(use_defect, grad_defect_x, grad_bg_x)
    if mix > 0:
        vy = (1 - mix) * vy + mix * grad_bg_y
        vx = (1 - mix) * vx + mix * grad_bg_x
    return vy, vx


def solve_poisson_sparse(background: np.ndarray, guidance_vy: np.ndarray,
                         guidance_vx: np.ndarray, mask: np.ndarray,
                         boundary_mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    div_v = np.gradient(guidance_vy, axis=0) + np.gradient(guidance_vx, axis=1)
    solve_mask = (mask > 0) & (boundary_mask == 0)
    solve_indices = np.where(solve_mask.ravel() > 0)[0]
    n = len(solve_indices)
    if n == 0:
        return background.copy()
    flat_to_matrix = {flat_idx: i for i, flat_idx in enumerate(solve_indices)}
    A = lil_matrix((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    div_flat = div_v.ravel()
    bg_flat = background.ravel()
    for i, flat_idx in enumerate(solve_indices):
        y, x = divmod(flat_idx, w)
        A[i, i] = -4
        b[i] = div_flat[flat_idx]
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                neighbor_flat = ny * w + nx
                if solve_mask.ravel()[neighbor_flat]:
                    A[i, flat_to_matrix[neighbor_flat]] = 1
                else:
                    b[i] -= bg_flat[neighbor_flat]
    try:
        x = spsolve(csr_matrix(A), b)
    except Exception as e:
        logger.warning(f"Sparse solve failed: {e}")
        return background.copy()
    result = background.copy().astype(np.float64)
    result_flat = result.ravel()
    result_flat[solve_indices] = np.clip(x, 0, 255)
    boundary_indices = np.where((boundary_mask > 0).ravel())[0]
    result_flat[boundary_indices] = bg_flat[boundary_indices]
    return result.reshape(h, w)


def poisson_blend_single_channel(background: np.ndarray, defect: np.ndarray,
                                 mask: np.ndarray, boundary_mask: np.ndarray,
                                 mix: float = 0.0) -> np.ndarray:
    mask_area = mask.sum()
    boundary_area = boundary_mask.sum()
    if mask_area == 0:
        return background
    if boundary_area >= mask_area * 0.5:
        vy, vx = compute_guidance_field(defect, background, mask, mix=0.3)
    else:
        vy, vx = compute_guidance_field(defect, background, mask, mix)
    result = solve_poisson_sparse(background, vy, vx, mask, boundary_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    mask_eroded_float = mask_eroded.astype(np.float32)
    center_zone = cv2.GaussianBlur(mask_eroded_float, (3, 3), 1)
    result = result * (1 - center_zone * 0.3) + defect * (center_zone * 0.3)
    return np.clip(result, 0, 255).astype(np.uint8)


def poisson_blend_color(background: np.ndarray, defect: np.ndarray,
                       mask: np.ndarray, boundary_mask: np.ndarray,
                       config: PoissonBlendConfig) -> np.ndarray:
    result = np.zeros_like(background)
    for channel in range(3):
        result[:, :, channel] = poisson_blend_single_channel(
            background[:, :, channel].astype(np.float64),
            defect[:, :, channel].astype(np.float64),
            mask, boundary_mask, config.gradient_mixing)
    return result


def apply_adaptive_blend(defect: np.ndarray, background: np.ndarray,
                         component_mask: np.ndarray) -> np.ndarray:
    """
    ★ Адаптивное смешивание: подстраивает цвет дефекта под локальный фон
    в каждой точке периметра
    """
    h, w = component_mask.shape
    mask_float = component_mask.astype(np.float32)
    
    # 1. Находим границу дефекта
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask_float, kernel, iterations=1)
    boundary = mask_dilated - mask_float
    
    # 2. Для каждой точки на границе вычисляем локальный цвет фона
    # Размываем фон — это даст средний цвет в окрестности каждой точки
    bg_blurred = cv2.GaussianBlur(background.astype(np.float32), (21, 21), 10)
    
    # 3. Вычисляем разницу между дефектом и фоном на границе
    boundary_3ch = np.stack([boundary] * 3, axis=-1)
    
    # Локальная коррекция цвета: на границе дефект должен плавно переходить в фон
    correction = (bg_blurred - defect.astype(np.float32)) * boundary_3ch
    
    # Размываем коррекцию чтобы она плавно затухала внутрь дефекта
    correction_blurred = cv2.GaussianBlur(correction, (31, 31), 15)
    
    # 4. Расстояние от границы (для плавного затухания коррекции)
    dist = cv2.distanceTransform((1 - mask_float).astype(np.uint8), cv2.DIST_L2, 5)
    dist = np.clip(dist / max(dist.max(), 1), 0, 1)  # Нормализация 0..1
    dist_3ch = np.stack([dist] * 3, axis=-1)
    
    # 5. Адаптируем дефект: чем ближе к границе, тем больше коррекция
    adapted_defect = defect.astype(np.float32) + correction_blurred * (1 - dist_3ch)
    
    # 6. Многослойное смешивание как раньше, но с адаптированным дефектом
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
    
    # Ядро: адаптированный дефект
    result_core = adapted_defect * core_mask_3ch + background * (1 - core_mask_3ch)
    
    # Средний слой: 70% адаптированный дефект + 30% фон
    blend_mid = adapted_defect * 0.7 + background * 0.3
    mid_weight = main_mask_3ch - core_mask_3ch
    mid_weight = np.clip(mid_weight, 0, 1)
    
    # Внешний слой: 30% адаптированный дефект + 70% фон (плавный переход)
    blend_outer = adapted_defect * 0.3 + background * 0.7
    outer_weight = outer_mask_3ch - main_mask_3ch
    outer_weight = np.clip(outer_weight, 0, 1)
    
    result = background.copy()
    result = result * (1 - outer_mask_3ch) + blend_outer * outer_mask_3ch
    result = result * (1 - mid_weight) + blend_mid * mid_weight
    result = result * (1 - core_mask_3ch) + result_core * core_mask_3ch
    
    # Добавляем микро-текстуру фона в переходной зоне
    transition_zone = outer_mask_3ch - main_mask_3ch
    transition_zone = np.clip(transition_zone, 0, 1)
    
    if transition_zone.max() > 0:
        bg_details = background.astype(np.float32) - bg_blurred
        result = result + bg_details * transition_zone * 0.15
    
    return np.clip(result, 0, 255).astype(np.uint8)

class SDDefectGenerator:
    def __init__(self, config: PoissonBlendConfig, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA недоступна, используется CPU")
        self.device = device
        self.config = config
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info("🔄 Загрузка Stable Diffusion...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        if device == "cuda":
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
            if hasattr(self.pipe, 'enable_vae_slicing'):
                self.pipe.enable_vae_slicing()
            elif hasattr(self.pipe, 'enable_vae_tiling'):
                self.pipe.enable_vae_tiling()
            try:
                self.pipe.enable_model_cpu_offload()
            except:
                pass
        logger.info("🚀 SD готов")
    
    @torch.no_grad()
    def generate(self, defect_crop: Image.Image, seed: int = None) -> np.ndarray:
        w, h = defect_crop.size
        sd_w = max(64, ((w + 7) // 8) * 8)
        sd_h = max(64, ((h + 7) // 8) * 8)
        crop_sd = defect_crop.resize((sd_w, sd_h), Image.Resampling.LANCZOS)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        strength = random.uniform(self.config.sd_strength_min, self.config.sd_strength_max)
        try:
            output = self.pipe(
                prompt=self.config.prompt,
                negative_prompt=self.config.negative_prompt,
                image=crop_sd,
                strength=strength,
                guidance_scale=self.config.sd_guidance_scale,
                num_inference_steps=self.config.sd_steps,
                generator=generator
            )
            generated = output.images[0]
            if sd_w != w or sd_h != h:
                generated = generated.resize((w, h), Image.Resampling.LANCZOS)
            generated_np = np.array(generated).astype(np.float32)
            crop_np = np.array(defect_crop).astype(np.float32)
            if self.config.use_spectrum_matching:
                generated_np = match_spectrum(generated_np, crop_np)
            if self.config.use_high_freq_injection:
                generated_np = inject_high_freq(generated_np, crop_np, self.config.high_freq_alpha)
            return generated_np
        except Exception as e:
            logger.error(f"SD generation error: {e}")
            return np.array(defect_crop).astype(np.float32)


# ★ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ
def get_image_brightness(image: np.ndarray) -> float:
    """Вычисляет среднюю яркость изображения"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return gray.mean()


def get_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    ★ Сравнивает гистограммы HSV. Возвращает 0 (полное совпадение) до 1 (макс. различие).
    Учитывает тон (H), насыщенность (S) и яркость (V).
    """
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    
    score = 0.0
    # H: тон (самый важный) — вес 0.5
    hist1 = cv2.calcHist([hsv1], [0], None, [30], [0, 180])
    hist2 = cv2.calcHist([hsv2], [0], None, [30], [0, 180])
    score += 0.5 * cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    
    # S: насыщенность — вес 0.2
    hist1 = cv2.calcHist([hsv1], [1], None, [32], [0, 256])
    hist2 = cv2.calcHist([hsv2], [1], None, [32], [0, 256])
    score += 0.2 * cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    
    # V: яркость — вес 0.3
    hist1 = cv2.calcHist([hsv1], [2], None, [32], [0, 256])
    hist2 = cv2.calcHist([hsv2], [2], None, [32], [0, 256])
    score += 0.3 * cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    
    return score


class PoissonDefectGenerator:
    def __init__(self, config: PoissonBlendConfig = None):
        self.config = config or PoissonBlendConfig()
        logger.info("Инициализация PoissonDefectGenerator...")
        try:
            self.sd_generator = SDDefectGenerator(self.config)
            logger.info("✓ SD генератор успешно загружен")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки SD генератора: {e}")
            self.sd_generator = None
            raise
        self.bg_augmenter = get_background_augmentation() if self.config.augment_background else None
        self.clean_backgrounds = []
        if self.config.clean_dir and self.config.clean_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                for p in self.config.clean_dir.glob(ext):
                    img = cv2.imread(str(p))
                    if img is not None:
                        self.clean_backgrounds.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            logger.info(f"✓ Загружено {len(self.clean_backgrounds)} чистых фонов")
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)
    
    def get_matching_background(self, target_img: np.ndarray, max_attempts: int = 100) -> np.ndarray:
        """★ Усиленный подбор фона с похожей цветовой гистограммой"""
        if not self.clean_backgrounds:
            raise ValueError("Нет чистых фонов!")
        
        best_bg = None
        best_score = float('inf')
        
        for _ in range(max_attempts):
            bg = random.choice(self.clean_backgrounds).copy()
            score = get_histogram_similarity(target_img, bg)
            
            if score < best_score:
                best_score = score
                best_bg = bg
            
            # ★ УЖЕСТОЧЕНО: < 0.15 почти идеальное совпадение
            if score < 0.15:
                break
        
        if best_bg is None:
            best_bg = random.choice(self.clean_backgrounds).copy()
        
        # ★ Логируем если плохо подобрали
        if best_score > 0.3:
            logger.debug(f"⚠ Плохой подбор фона, dissimilarity: {best_score:.3f}")
        
        if best_bg.shape[:2] != (256, 256):
            best_bg = cv2.resize(best_bg, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        
        return best_bg
        
    def augment_background(self, image: np.ndarray) -> np.ndarray:
        if self.bg_augmenter is None:
            return image
        augmented = self.bg_augmenter(image=image)
        return augmented['image']
    
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
            
            # ★ Подбор фона по яркости
            try:
                background = self.get_matching_background(original)
            except ValueError:
                logger.error("Нет чистых фонов!")
                return None
            
            background = self.augment_background(background)
            result = background.copy().astype(np.float32)
            yolo_annotations = []
            for bbox_idx, bbox_info in enumerate(all_bboxes):
                x, y, w, h = bbox_info['x'], bbox_info['y'], bbox_info['w'], bbox_info['h']
                comp_mask = bbox_info['component_mask']
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
                
                # ★ Сглаживание маски для устранения прямоугольных рамок
                crop_comp_mask = cv2.GaussianBlur(crop_comp_mask.astype(np.float32), (5, 5), 2)
                crop_comp_mask = (crop_comp_mask > 0.3).astype(np.uint8)
                if crop_comp_mask.sum() == 0:
                    continue
                
                defect_crop = original[y1:y2, x1:x2].copy()
                defect_crop_pil = Image.fromarray(defect_crop)
                seed = self.config.random_seed + total_idx * 100 + variant * 10 + bbox_idx
                try:
                    generated_np = self.sd_generator.generate(defect_crop_pil, seed=seed)
                except Exception as e:
                    logger.error(f"Ошибка SD генерации: {e}")
                    continue
                if generated_np.shape[:2] != (crop_h, crop_w):
                    generated_np = cv2.resize(generated_np, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
                background_crop = background[y1:y2, x1:x2].copy()
                blended = apply_adaptive_blend(generated_np, background_crop.astype(np.float32), crop_comp_mask)
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
            blur = cv2.GaussianBlur(result_uint8, (0, 0), sigmaX=1.5)
            result_uint8 = cv2.addWeighted(result_uint8, 1.3, blur, -0.3, 0)
            result_uint8 = np.clip(result_uint8, 0, 255).astype(np.uint8)
            result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / "images" / filename), result_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            with open(output_dir / "labels" / f"{Path(filename).stem}.txt", 'w') as f:
                for ann in yolo_annotations:
                    f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
            return {"filename": filename, "num_defects": len(yolo_annotations)}
        except Exception as e:
            logger.error(f"❌ Error processing {img_path.name}: {e}")
            traceback.print_exc()
            return None
    
    def generate_dataset(self, input_dir: Path, rle_csv: Path,
                    output_dir: Path, variants: int = 3,
                    limit: Optional[int] = None) -> int:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
        rle_df = pd.read_csv(rle_csv)
        groups = rle_df.groupby('ImageId')
        logger.info(f"📂 Найдено патчей: {len(groups)}")
        all_images = {}
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img in input_dir.glob(ext):
                all_images[img.name] = img
                all_images[img.stem] = img
        stats = {'total': 0, 'errors': 0, 'defects': 0}
        for image_id, group in tqdm(groups, desc="Poisson + Clean BG"):
            if limit and stats['total'] >= limit:
                break
            img_path = all_images.get(image_id)
            if img_path is None:
                continue
            
            # ★ Собираем ВСЕ компоненты дефектов для этого изображения
            all_bboxes = []
            for _, row in group.iterrows():
                rle = row.get('EncodedPixels')
                if pd.isna(rle) or str(rle).strip().lower() in ['', 'nan']:
                    continue
                all_bboxes.extend(rle_to_defect_bboxes(str(rle), image_id, int(row['ClassId']) - 1))
            
            if not all_bboxes:
                continue
            
            # ★ Генерируем варианты, но для каждого берём только ОДИН случайный дефект
            for v in range(variants):
                if limit and stats['total'] >= limit:
                    break
                
                # ★ Выбираем случайный ОДИН дефект
                selected_bbox = [random.choice(all_bboxes)]
                
                result = self.process_image_all_defects(img_path, selected_bbox, output_dir, v, stats['total'])
                if result:
                    stats['total'] += 1
                    stats['defects'] += result['num_defects']
                else:
                    stats['errors'] += 1
                if stats['total'] % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        logger.info(f"✅ Итого: {stats['total']} изображений, {stats['defects']} дефектов, {stats['errors']} ошибок")
        return stats['total']


def main():
    parser = argparse.ArgumentParser(description="Poisson Blending + Brightness Match")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--rle_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--clean_dir", type=str, default="data/256_yolo/balanced_clean_patches/train")
    parser.add_argument("--brightness_threshold", type=float, default=25.0)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sd_strength", type=float, default=None)
    parser.add_argument("--sd_strength_min", type=float, default=0.08)
    parser.add_argument("--sd_strength_max", type=float, default=0.15)
    parser.add_argument("--sd_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--no_spectrum", action="store_true")
    parser.add_argument("--no_high_freq", action="store_true")
    parser.add_argument("--prompt", type=str, default="metal surface defect, scratch, industrial steel texture")
    parser.add_argument("--negative_prompt", type=str, default="smooth, plastic, wood, rust")
    args = parser.parse_args()
    config = PoissonBlendConfig(
        random_seed=args.seed, sd_steps=args.sd_steps,
        sd_guidance_scale=args.guidance_scale,
        augment_background=not args.no_augment,
        use_spectrum_matching=not args.no_spectrum,
        use_high_freq_injection=not args.no_high_freq,
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        clean_dir=Path(args.clean_dir),
        brightness_threshold=args.brightness_threshold
    )
    if args.sd_strength:
        config.sd_strength_min = config.sd_strength_max = args.sd_strength
    else:
        config.sd_strength_min = args.sd_strength_min
        config.sd_strength_max = args.sd_strength_max
    generator = PoissonDefectGenerator(config)
    total = generator.generate_dataset(Path(args.input_dir), Path(args.rle_csv),
                                       Path(args.output_dir), args.variants, args.limit)
    logger.info(f"✅ Готово! Сгенерировано {total} изображений")


if __name__ == "__main__":
    main()