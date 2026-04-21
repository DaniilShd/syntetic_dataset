#!/usr/bin/env python3
"""
03_generate_defective.py - Генерация синтетических изображений с дефектами
Поддержка YOLO-разметки с автоматической трансформацией bbox
Включает спектральный контроль для сохранения текстуры
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
from utils import set_seed, print_system_info, logger


def match_spectrum(source: Image.Image, target: Image.Image) -> Image.Image:
    """
    Приведение спектра сгенерированного изображения к спектру референса
    Сохраняет частотное распределение оригинала при новой структуре (фазе)
    """
    src = np.array(source).astype(np.float32)
    tgt = np.array(target).astype(np.float32)

    # FFT по пространственным осям
    src_f = np.fft.fft2(src, axes=(0, 1))
    tgt_f = np.fft.fft2(tgt, axes=(0, 1))

    # Извлекаем амплитуду и фазу
    src_phase = np.angle(src_f)  # Фаза от сгенерированного (новая структура)
    tgt_amp = np.abs(tgt_f)      # Амплитуда от референса (частотный состав)

    # Комбинируем: амплитуда референса + фаза генерации
    result_f = tgt_amp * np.exp(1j * src_phase)
    result = np.fft.ifft2(result_f, axes=(0, 1)).real

    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def inject_high_freq(source: Image.Image, target: Image.Image, alpha: float = 0.3) -> Image.Image:
    """
    Инжекция высокочастотной составляющей из референса
    Добавляет микротекстуру и детализацию
    """
    src = np.array(source).astype(np.float32)
    tgt = np.array(target).astype(np.float32)

    # High-pass фильтр через гауссово размытие
    blur = cv2.GaussianBlur(tgt, (0, 0), sigmaX=3)
    high = tgt - blur  # Высокочастотная компонента

    result = src + alpha * high
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


class YOLODatasetHandler:
    """Обработчик YOLO-разметки"""
    
    def __init__(self, images_dir: Path, labels_dir: Path):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        
    
    def load_annotation(self, image_name: str) -> List[Dict]:
        """Загрузка YOLO-аннотации из txt файла"""
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
        """Сохранение YOLO-аннотации в txt файл"""
        label_path = self.labels_dir / f"{Path(image_name).stem}.txt"
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                       f"{ann['width']:.6f} {ann['height']:.6f}\n")
    
    @staticmethod
    def flip_annotation_horizontal(annotations: List[Dict]) -> List[Dict]:
        """Горизонтальное отражение bbox (меняется только x_center)"""
        flipped = []
        for ann in annotations:
            flipped.append({
                'class': ann['class'],
                'x_center': 1.0 - ann['x_center'],  # Отражение по горизонтали
                'y_center': ann['y_center'],        # Не меняется
                'width': ann['width'],              # Размеры те же
                'height': ann['height']
            })
        return flipped


class DefectiveGenerator:
    """Генератор изображений с дефектами через img2img со спектральным контролем"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.use_spectrum_matching = config.use_spectrum_matching
        self.use_high_freq_injection = config.use_high_freq_injection
        self.high_freq_alpha = config.high_freq_alpha
        
        logger.info("🔄 Загрузка Stable Diffusion...")
        
        # VAE с обработкой ошибок
        try:
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float16,
                cache_dir=config.cache_dir
            )
            logger.info("✅ Загружен улучшенный VAE (MSE)")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить MSE VAE: {e}")
            vae = None
            logger.info("ℹ️ Используется стандартный VAE")
        
        # Основной пайплайн
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.model_id,
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=config.cache_dir
        ).to(config.device)
        
        # Оптимизации
        if hasattr(self.pipe.vae, 'enable_tiling'):
            self.pipe.vae.enable_tiling()
            logger.info("✅ Tiling VAE включен")
        
        if config.use_ip_adapter:
            logger.info("🔄 Загрузка IP-Adapter...")
            try:
                self.pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name="ip-adapter_sd15.bin",
                    cache_dir=config.cache_dir
                )
                self.pipe.set_ip_adapter_scale(config.ip_adapter_scale_default)
                logger.info(f"✅ IP-Adapter загружен (scale={config.ip_adapter_scale_default})")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось загрузить IP-Adapter: {e}")
                config.use_ip_adapter = False
        
        if config.enable_attention_slicing:
            self.pipe.enable_attention_slicing()
            logger.info("✅ Attention slicing включен")
        
        if config.enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("✅ xFormers включен")
            except Exception as e:
                logger.warning(f"⚠️ xFormers не доступен: {e}")
                pass
        
        # DDIM scheduler для лучшего контроля
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        logger.info("✅ Используется DDIM scheduler")
        
        torch.cuda.empty_cache()
        logger.info("🚀 Генератор готов")
        
        # Логирование настроек
        logger.info("=" * 50)
        logger.info("Настройки генерации:")
        logger.info(f"  Strength: {config.strength_min:.2f}-{config.strength_max:.2f}")
        logger.info(f"  Guidance scale: {config.guidance_scale}")
        logger.info(f"  Steps: {config.num_inference_steps}")
        logger.info(f"  Spectrum matching: {self.use_spectrum_matching}")
        logger.info(f"  High-freq injection: {self.use_high_freq_injection}")
        if self.use_high_freq_injection:
            logger.info(f"    Alpha: {self.high_freq_alpha}")
        logger.info("=" * 50)
    
    def generate_one(
        self,
        reference_image: Image.Image,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Dict]:
        """Генерация одного синтетического изображения с дефектом"""
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        # Случайные параметры
        ip_scale = random.uniform(self.config.ip_adapter_scale_min, self.config.ip_adapter_scale_max)
        strength = random.uniform(self.config.strength_min, self.config.strength_max)
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        
        # Генерация
        if self.config.use_ip_adapter:
            output = self.pipe(
                prompt=self.config.prompt,
                negative_prompt="blurry, low quality, distorted, text, watermark, cartoon, painting",
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
                negative_prompt="blurry, low quality, distorted, text, watermark, cartoon, painting",
                image=reference_image,
                strength=strength,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                generator=generator
            )
        
        # Извлечение изображения
        image = output.images[0] if hasattr(output, 'images') else output[0]
        if isinstance(image, list):
            image = image[0]
        
        # ===== СПЕКТРАЛЬНЫЙ КОНТРОЛЬ =====
        if self.use_spectrum_matching:
            image = match_spectrum(image, reference_image)
        
        # ===== ИНЖЕКЦИЯ ВЫСОКИХ ЧАСТОТ =====
        if self.use_high_freq_injection:
            image = inject_high_freq(
                image,
                reference_image,
                alpha=self.high_freq_alpha
            )
        
        return image, {
            "seed": seed,
            "strength": round(strength, 3),
            "ip_adapter_scale": round(ip_scale, 3) if self.config.use_ip_adapter else None,
            "spectrum_matched": self.use_spectrum_matching,
            "high_freq_injected": self.use_high_freq_injection
        }
    
    def apply_augmentations(
        self,
        image: Image.Image,
        annotations: List[Dict]
    ) -> Tuple[Image.Image, List[Dict], bool]:
        """Применение аугментаций с обновлением разметки"""
        
        if not self.config.enable_augmentation:
            return image, annotations, False
        
        flipped_h = False
        
        # Горизонтальный флип
        if random.random() < self.config.aug_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            annotations = YOLODatasetHandler.flip_annotation_horizontal(annotations)
            flipped_h = True
        
        # Яркость
        if random.random() < self.config.aug_brightness_prob:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(self.config.aug_brightness_min, self.config.aug_brightness_max)
            image = enhancer.enhance(factor)
        
        # Контраст
        if random.random() < self.config.aug_contrast_prob:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(self.config.aug_contrast_min, self.config.aug_contrast_max)
            image = enhancer.enhance(factor)
        
        return image, annotations, flipped_h
    
    def generate_dataset(
        self,
        input_images_dir: Path,
        input_labels_dir: Path,
        output_dir: Path,
        variants_per_image: int,
        total_limit: Optional[int] = None
    ) -> List[Dict]:
        """Генерация датасета с сохранением YOLO-разметки"""
        
        # Создание выходных директорий
        output_images = output_dir / "images"
        output_labels = output_dir / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        # Инициализация обработчиков
        source_handler = YOLODatasetHandler(input_images_dir, input_labels_dir)
        target_handler = YOLODatasetHandler(output_images, output_labels)
        
        # Поиск изображений
        image_paths = list(input_images_dir.glob("*.png")) + \
                     list(input_images_dir.glob("*.jpg")) + \
                     list(input_images_dir.glob("*.jpeg"))
        
        if not image_paths:
            logger.error(f"Нет изображений в {input_images_dir}")
            return []
        
        logger.info(f"📂 Найдено {len(image_paths)} изображений")
        
        total_generated = 0
        all_metadata = []
        
        for img_path in tqdm(image_paths, desc="Обработка изображений"):
            if total_limit and total_generated >= total_limit:
                break
            
            # Загрузка исходных данных
            try:
                ref_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки {img_path.name}: {e}")
                continue
            
            annotations = source_handler.load_annotation(img_path.name)
            
            if not annotations:
                logger.warning(f"⚠️ Нет разметки для {img_path.name}, пропускаем")
                continue
            
            # Ресайз если нужно
            if self.config.resize_to and ref_image.size != (self.config.resize_to, self.config.resize_to):
                # Сохраняем пропорции для аннотаций
                old_w, old_h = ref_image.size
                ref_image = ref_image.resize(
                    (self.config.resize_to, self.config.resize_to),
                    Image.Resampling.LANCZOS
                )
                # Аннотации уже нормализованы, не требуют изменения
            
            # Генерация вариантов
            for variant in range(variants_per_image):
                if total_limit and total_generated >= total_limit:
                    break
                
                try:
                    # Генерация синтетического изображения
                    syn_image, meta = self.generate_one(ref_image)
                    
                    # Аугментация с обновлением разметки
                    current_annotations = [ann.copy() for ann in annotations]
                    syn_image, current_annotations, was_flipped = self.apply_augmentations(
                        syn_image, current_annotations
                    )
                    
                    # Сохранение
                    filename = f"syn_{total_generated:06d}_{img_path.stem}_v{variant}.png"
                    syn_image.save(output_images / filename, "PNG", optimize=True)
                    target_handler.save_annotation(filename, current_annotations)
                    
                    # Метаданные
                    meta.update({
                        "filename": filename,
                        "source_image": img_path.name,
                        "variant": variant,
                        "flipped_horizontal": was_flipped,
                        "num_objects": len(current_annotations)
                    })
                    all_metadata.append(meta)
                    
                    total_generated += 1
                    
                    if total_generated % 10 == 0:
                        logger.info(f"📊 Сгенерировано: {total_generated}")
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка генерации варианта {variant} для {img_path.name}: {e}")
                    torch.cuda.empty_cache()
                    continue
            
            # Очистка памяти
            if total_generated % 5 == 0:
                torch.cuda.empty_cache()
        
        # Сохранение метаданных
        import json
        with open(output_dir / "generation_metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        logger.info(f"🎉 Генерация завершена. Создано {total_generated} изображений с разметкой")
        logger.info(f"📁 Результаты сохранены в {output_dir}")
        
        # Статистика
        if all_metadata:
            avg_strength = np.mean([m['strength'] for m in all_metadata])
            logger.info(f"📊 Средняя сила преобразования: {avg_strength:.3f}")
        
        return all_metadata


def main():
    parser = argparse.ArgumentParser(description="Генерация синтетических изображений с YOLO-разметкой и спектральным контролем")
    
    # Основные пути
    parser.add_argument("--input_dir", type=str, default="data/256_yolo/balanced_defect_patches/train",
                       help="Директория с изображениями (содержит images/ и labels/)")
    parser.add_argument("--output_dir", type=str, default="data/dataset_synthetic/defect_patches",
                       help="Выходная директория")
    
    # Параметры генерации (ОПТИМИЗИРОВАНЫ для сохранения текстуры)
    parser.add_argument("--variants", type=int, default=5,
                       help="Количество синтетических вариантов на одно изображение")
    parser.add_argument("--limit", type=int, default=None,
                       help="Общий лимит генерируемых изображений")
    parser.add_argument("--strength_min", type=float, default=0.15,
                       help="Минимальная сила преобразования (меньше = больше сохранения структуры)")
    parser.add_argument("--strength_max", type=float, default=0.25,
                       help="Максимальная сила преобразования (рекомендуется 0.15-0.25)")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                       help="Сила следования промпту (меньше = меньше галлюцинаций)")
    parser.add_argument("--steps", type=int, default=20,
                       help="Количество шагов денойзинга (меньше = меньше сглаживания)")
    parser.add_argument("--prompt", type=str, 
                       default="industrial steel surface with manufacturing texture, detailed metallic structure",
                       help="Промпт для генерации")
    
    # Спектральный контроль (НОВЫЕ ПАРАМЕТРЫ)
    parser.add_argument("--no_spectrum_matching", action="store_true",
                       help="Отключить спектральное согласование (FFT matching)")
    parser.add_argument("--no_high_freq", action="store_true",
                       help="Отключить инжекцию высоких частот")
    parser.add_argument("--high_freq_alpha", type=float, default=0.3,
                       help="Сила инжекции высоких частот (0.2-0.4 рекомендуется)")
    
    # Аугментация
    parser.add_argument("--no_augmentation", action="store_true",
                       help="Отключить аугментацию")
    parser.add_argument("--flip_prob", type=float, default=0.5,
                       help="Вероятность горизонтального отражения")
    parser.add_argument("--brightness_prob", type=float, default=0.2,
                       help="Вероятность изменения яркости")
    parser.add_argument("--brightness_range", type=float, nargs=2, default=[0.8, 1.2],
                       help="Диапазон изменения яркости")
    
    # Технические параметры
    parser.add_argument("--resize_to", type=int, default=256,
                       help="Размер изображения (квадрат)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Сид для воспроизводимости")
    parser.add_argument("--no_ip_adapter", action="store_true",
                       help="Отключить IP-Adapter")
    
    args = parser.parse_args()
    
    # Настройка
    print_system_info()
    set_seed(args.seed)
    
    # Определение путей
    input_path = Path(args.input_dir)
    images_dir = input_path / "images"
    labels_dir = input_path / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        logger.error(f"Директории images/ и labels/ не найдены в {input_path}")
        sys.exit(1)
    
    # Конфигурация с новыми параметрами
    config = GenerationConfig()
    
    # Основные параметры генерации (оптимизированы)
    config.strength_min = args.strength_min
    config.strength_max = args.strength_max
    config.guidance_scale = args.guidance_scale
    config.num_inference_steps = args.steps
    config.use_ip_adapter = not args.no_ip_adapter
    config.prompt = args.prompt
    config.resize_to = args.resize_to
    
    # Спектральный контроль (НОВОЕ)
    config.use_spectrum_matching = not args.no_spectrum_matching
    config.use_high_freq_injection = not args.no_high_freq
    config.high_freq_alpha = args.high_freq_alpha
    
    # Аугментация
    config.enable_augmentation = not args.no_augmentation
    config.aug_flip_prob = args.flip_prob
    config.aug_brightness_prob = args.brightness_prob
    config.aug_brightness_min = args.brightness_range[0]
    config.aug_brightness_max = args.brightness_range[1]
    
    # Дополнительные параметры
    config.ip_adapter_scale_default = 0.75
    config.ip_adapter_scale_min = 0.6
    config.ip_adapter_scale_max = 0.85
    config.aug_contrast_prob = 0.2
    config.aug_contrast_min = 0.8
    config.aug_contrast_max = 1.2
    
    logger.info("=" * 70)
    logger.info("🎨 ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДЕФЕКТОВ СО СПЕКТРАЛЬНЫМ КОНТРОЛЕМ")
    logger.info("=" * 70)
    logger.info(f"📂 Вход: {input_path}")
    logger.info(f"📂 Выход: {args.output_dir}")
    logger.info(f"🔢 Вариантов на изображение: {args.variants}")
    logger.info(f"⚙️  Strength: {config.strength_min:.2f}-{config.strength_max:.2f} (оптимизировано)")
    logger.info(f"🎯 Guidance scale: {config.guidance_scale} (снижено)")
    logger.info(f"🔄 Steps: {config.num_inference_steps} (оптимально)")
    logger.info(f"📊 Spectrum matching: {'✅ ВКЛ' if config.use_spectrum_matching else '❌ ВЫКЛ'}")
    logger.info(f"🔍 High-freq injection: {'✅ ВКЛ' if config.use_high_freq_injection else '❌ ВЫКЛ'}")
    if config.use_high_freq_injection:
        logger.info(f"   Alpha: {config.high_freq_alpha}")
    logger.info(f"🖼️  IP-Adapter: {'✅ ВКЛ' if config.use_ip_adapter else '❌ ВЫКЛ'}")
    logger.info("=" * 70)
    
    # Запуск генерации
    generator = DefectiveGenerator(config)
    generator.generate_dataset(
        input_images_dir=images_dir,
        input_labels_dir=labels_dir,
        output_dir=Path(args.output_dir),
        variants_per_image=args.variants,
        total_limit=args.limit
    )
    
    logger.info("✅ Процесс завершен успешно")


if __name__ == "__main__":
    main()