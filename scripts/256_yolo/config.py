"""
config.py - Конфигурации для Docker-окружения
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
import cv2

# Базовые пути в контейнере
BASE_DATA_DIR = "/app/data"
BASE_RESULTS_DIR = "/app/results"
BASE_MODELS_DIR = "/app/models"
BASE_CONFIGS_DIR = "/app/configs"


@dataclass
class GenerationConfig:
    """Конфигурация для генерации синтетических фонов"""
    
    # Модель
    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda"
    
    # Промпты
    # prompt: str = "steel surface, metal sheet, industrial material"
    prompt: str = (
        "industrial steel surface, metallic texture, "
        "manufacturing grade metal, uniform illumination, "
        "high quality industrial photography, sharp details"
    )
    # negative_prompt="blurry, low quality, distorted, text, watermark, cartoon, painting"
    negative_prompt: str = (
        "blurry, low quality, distorted, noise, grain, "
        "text, watermark, logo, signature, "
        "rust, corrosion, defect, crack, hole, scratch, "
        "person, human, face, hand, "
        "object, tool, equipment, background, "
        "cartoon, painting, drawing, 3d render, "
        "overexposed, underexposed, dark, shadow"
    )
        
    # Ключевые параметры для Severstal
    ip_adapter_scale_min: float = 0.70
    ip_adapter_scale_max: float = 0.80
    ip_adapter_scale_default: float = 0.75  # ← ДОБАВЛЕНО (базовое значение)
    strength_min: float = 0.15
    strength_max: float = 0.22
    guidance_scale: float = 2.0
    num_inference_steps: int = 25
    
    # Размеры
    resolution: int = 1024
    resize_to: Optional[int] = None  # Для ресайза в defective генераторе
    
    # ===== СПЕКТРАЛЬНЫЙ КОНТРОЛЬ (НОВОЕ) =====
    use_spectrum_matching: bool = True      # ← ДОБАВЛЕНО (FFT matching)
    use_high_freq_injection: bool = True    # ← ДОБАВЛЕНО (инжекция высоких частот)
    high_freq_alpha: float = 0.3            # ← ДОБАВЛЕНО (сила инжекции)
    
    # Аугментация
    enable_augmentation: bool = True
    aug_flip_prob: float = 0.5
    aug_brightness_prob: float = 0.3
    aug_brightness_min: float = 0.8
    aug_brightness_max: float = 1.2
    aug_contrast_prob: float = 0.3          # ← ДОБАВЛЕНО
    aug_contrast_min: float = 0.8           # ← ДОБАВЛЕНО
    aug_contrast_max: float = 1.2           # ← ДОБАВЛЕНО
    
    # Оптимизации
    enable_attention_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_xformers: bool = True  # Для RTX 5070 Ti
    
    # Режимы
    use_ip_adapter: bool = True
    use_random_seed: bool = True
    
    # Кеширование
    cache_dir: str = "/app/cache/huggingface"


@dataclass
class InsertionConfig:
    """Конфигурация для вставки дефектов"""
    
    method: str = "poisson"
    defects_per_image_min: int = 1
    defects_per_image_max: int = 3
    max_placement_attempts: int = 50
    blend_mode: int = cv2.NORMAL_CLONE
    gaussian_blur_kernel: int = 7
    auto_scale_defect: bool = True
    max_scale_ratio: float = 0.8


@dataclass
class ValidationConfig:
    """Конфигурация для валидации качества"""
    
    compute_fid: bool = True
    compute_kid: bool = True
    fid_batch_size: int = 50
    device: str = "cuda"
    fid_threshold: float = 15.0
    kid_threshold: float = 0.01


@dataclass
class PipelineConfig:
    """Основная конфигурация пайплайна для Docker"""
    
    # Директории (монтируемые)
    clean_textures_dir: str = os.path.join(BASE_DATA_DIR, "clean_textures")
    defects_dir: str = os.path.join(BASE_DATA_DIR, "defects")
    output_dir: str = BASE_RESULTS_DIR
    models_dir: str = BASE_MODELS_DIR
    
    # Поддиректории (автоматически)
    backgrounds_dir: str = os.path.join(BASE_RESULTS_DIR, "synthetic_backgrounds")
    final_dataset_dir: str = os.path.join(BASE_RESULTS_DIR, "final_dataset")
    validation_dir: str = os.path.join(BASE_RESULTS_DIR, "validation")
    viz_dir: str = os.path.join(BASE_RESULTS_DIR, "visualizations")
    
    # Параметры генерации
    variants_per_reference: int = 50
    total_backgrounds_limit: Optional[int] = None
    total_final_images: int = 10000
    
    # Подконфигурации
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    insertion: InsertionConfig = field(default_factory=InsertionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Случайные seed
    random_seed: int = 42
    
    def __post_init__(self):
        """Валидация путей"""
        import os
        os.makedirs(self.backgrounds_dir, exist_ok=True)
        os.makedirs(self.final_dataset_dir, exist_ok=True)
        os.makedirs(self.validation_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)