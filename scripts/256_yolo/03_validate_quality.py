#!/usr/bin/env python3
"""
03_validate_quality.py - Валидация качества синтетических данных
Вычисляет FID, KID и проверяет systematic bias
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import ValidationConfig
from utils import save_json, logger


class SystematicBiasChecker:
    """Проверка систематических смещений между реальными и синтетическими данными"""
    
    def __init__(self, real_dir: str, fake_dir: str):
        self.real_dir = Path(real_dir)
        self.fake_dir = Path(fake_dir)
        
    def load_images(self, directory: Path, max_images: int = 500) -> List[np.ndarray]:
        """Загрузка изображений для анализа"""
        images = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for ext in extensions:
            for img_path in directory.glob(ext):
                if len(images) >= max_images:
                    break
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(np.array(img))
                except:
                    continue
                    
        return images[:max_images]
    
    def compute_histogram_stats(self, images: List[np.ndarray]) -> Dict:
        """Вычисление статистик гистограмм"""
        all_means = []
        all_stds = []
        all_skewness = []
        
        for img in images:
            # Поканальные статистики
            for c in range(3):
                channel = img[:, :, c].flatten()
                all_means.append(np.mean(channel))
                all_stds.append(np.std(channel))
                all_skewness.append(self._skewness(channel))
        
        return {
            "mean": {"value": np.mean(all_means), "std": np.std(all_means)},
            "std": {"value": np.mean(all_stds), "std": np.std(all_stds)},
            "skewness": {"value": np.mean(all_skewness), "std": np.std(all_skewness)}
        }
    
    def _skewness(self, data: np.ndarray) -> float:
        """Вычисление асимметрии распределения"""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.sum(((data - mean) / std) ** 3) / n
    
    def compute_color_balance(self, images: List[np.ndarray]) -> Dict:
        """Анализ цветового баланса"""
        r_means, g_means, b_means = [], [], []
        
        for img in images:
            r_means.append(np.mean(img[:, :, 0]))
            g_means.append(np.mean(img[:, :, 1]))
            b_means.append(np.mean(img[:, :, 2]))
        
        return {
            "R_mean": np.mean(r_means),
            "G_mean": np.mean(g_means),
            "B_mean": np.mean(b_means),
            "R/G_ratio": np.mean(r_means) / np.mean(g_means) if np.mean(g_means) > 0 else 1,
            "B/G_ratio": np.mean(b_means) / np.mean(g_means) if np.mean(g_means) > 0 else 1
        }
    
    def compute_frequency_spectrum(self, images: List[np.ndarray]) -> Dict:
        """Анализ частотных характеристик через FFT"""
        all_spectral_energy = []
        
        for img in tqdm(images[:100], desc="FFT анализ"):  # FFT дорогой, ограничиваем
            gray = np.mean(img, axis=2)
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Энергия в разных частотных диапазонах
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            
            # Низкие частоты (центр)
            low_freq = magnitude[center_h-10:center_h+10, center_w-10:center_w+10]
            low_energy = np.sum(low_freq) / np.sum(magnitude)
            
            # Высокие частоты (края)
            high_freq = np.concatenate([
                magnitude[:20, :].flatten(),
                magnitude[-20:, :].flatten(),
                magnitude[:, :20].flatten(),
                magnitude[:, -20:].flatten()
            ])
            high_energy = np.sum(np.abs(high_freq)) / np.sum(magnitude)
            
            all_spectral_energy.append({
                "low_freq_ratio": low_energy,
                "high_freq_ratio": high_energy
            })
        
        return {
            "low_freq_ratio": np.mean([e["low_freq_ratio"] for e in all_spectral_energy]),
            "high_freq_ratio": np.mean([e["high_freq_ratio"] for e in all_spectral_energy])
        }
    
    def check_bias(self) -> Dict:
        """Полная проверка systematic bias"""
        logger.info("Загрузка реальных изображений...")
        real_images = self.load_images(self.real_dir)
        logger.info(f"Загружено {len(real_images)} реальных изображений")
        
        logger.info("Загрузка синтетических изображений...")
        fake_images = self.load_images(self.fake_dir)
        logger.info(f"Загружено {len(fake_images)} синтетических изображений")
        
        if not real_images or not fake_images:
            return {"error": "Нет изображений для анализа"}
        
        # Статистики гистограмм
        logger.info("Анализ гистограмм...")
        real_hist = self.compute_histogram_stats(real_images)
        fake_hist = self.compute_histogram_stats(fake_images)
        
        # Цветовой баланс
        logger.info("Анализ цветового баланса...")
        real_color = self.compute_color_balance(real_images)
        fake_color = self.compute_color_balance(fake_images)
        
        # Частотный спектр
        logger.info("Анализ частотного спектра...")
        real_spectral = self.compute_frequency_spectrum(real_images)
        fake_spectral = self.compute_frequency_spectrum(fake_images)
        
        # Вычисление отклонений и проверка порогов
        bias_report = {
            "histogram": {
                "mean_diff": abs(real_hist["mean"]["value"] - fake_hist["mean"]["value"]),
                "mean_diff_percent": round(abs(real_hist["mean"]["value"] - fake_hist["mean"]["value"]) / real_hist["mean"]["value"] * 100, 2),
                "std_diff": abs(real_hist["std"]["value"] - fake_hist["std"]["value"]),
                "std_diff_percent": round(abs(real_hist["std"]["value"] - fake_hist["std"]["value"]) / real_hist["std"]["value"] * 100, 2) if real_hist["std"]["value"] > 0 else 0,
                "pass": False  # Будет установлено ниже
            },
            "color_balance": {
                "R_diff": abs(real_color["R_mean"] - fake_color["R_mean"]),
                "G_diff": abs(real_color["G_mean"] - fake_color["G_mean"]),
                "B_diff": abs(real_color["B_mean"] - fake_color["B_mean"]),
                "RG_ratio_diff": abs(real_color["R/G_ratio"] - fake_color["R/G_ratio"]),
                "BG_ratio_diff": abs(real_color["B/G_ratio"] - fake_color["B/G_ratio"]),
                "pass": False
            },
            "frequency": {
                "low_freq_diff": abs(real_spectral["low_freq_ratio"] - fake_spectral["low_freq_ratio"]),
                "high_freq_diff": abs(real_spectral["high_freq_ratio"] - fake_spectral["high_freq_ratio"]),
                "pass": False
            }
        }
        
        # Пороги для systematic bias
        HIST_MEAN_THRESHOLD = 5.0  # 5% разницы в среднем
        HIST_STD_THRESHOLD = 10.0  # 10% разницы в std
        COLOR_THRESHOLD = 10.0     # 10% разницы в цветовых каналах
        RATIO_THRESHOLD = 0.05     # 5% разницы в отношениях
        FREQ_THRESHOLD = 0.1       # 10% разницы в частотных компонентах
        
        bias_report["histogram"]["pass"] = (
            bias_report["histogram"]["mean_diff_percent"] < HIST_MEAN_THRESHOLD and
            bias_report["histogram"]["std_diff_percent"] < HIST_STD_THRESHOLD
        )
        
        bias_report["color_balance"]["pass"] = (
            bias_report["color_balance"]["R_diff"] / real_color["R_mean"] * 100 < COLOR_THRESHOLD and
            bias_report["color_balance"]["G_diff"] / real_color["G_mean"] * 100 < COLOR_THRESHOLD and
            bias_report["color_balance"]["B_diff"] / real_color["B_mean"] * 100 < COLOR_THRESHOLD and
            bias_report["color_balance"]["RG_ratio_diff"] < RATIO_THRESHOLD and
            bias_report["color_balance"]["BG_ratio_diff"] < RATIO_THRESHOLD
        )
        
        bias_report["frequency"]["pass"] = (
            bias_report["frequency"]["low_freq_diff"] < FREQ_THRESHOLD and
            bias_report["frequency"]["high_freq_diff"] < FREQ_THRESHOLD
        )
        
        bias_report["overall_pass"] = (
            bias_report["histogram"]["pass"] and
            bias_report["color_balance"]["pass"] and
            bias_report["frequency"]["pass"]
        )
        
        # Добавляем сырые данные для отчетности
        bias_report["raw"] = {
            "real_histogram": real_hist,
            "fake_histogram": fake_hist,
            "real_color": real_color,
            "fake_color": fake_color,
            "real_spectral": real_spectral,
            "fake_spectral": fake_spectral
        }
        
        return bias_report


def compute_fid_kid(real_dir: str, fake_dir: str, config: ValidationConfig) -> Dict:
    """Вычисление FID и KID метрик"""
    metrics = {}
    
    if config.compute_fid:
        try:
            from cleanfid import fid
            fid_score = fid.compute_fid(
                real_dir, fake_dir,
                device=config.device,
                batch_size=config.fid_batch_size
            )
            metrics["fid"] = round(fid_score, 2)
            metrics["fid_pass"] = fid_score < config.fid_threshold
            logger.info(f"📊 FID: {fid_score:.2f} {'✅' if metrics['fid_pass'] else '❌'}")
        except ImportError:
            logger.error("⚠️ cleanfid не установлен")
            metrics["fid"] = None
    
    if config.compute_kid:
        try:
            from cleanfid import kid
            kid_score = kid.compute_kid(real_dir, fake_dir, device=config.device)
            metrics["kid"] = round(kid_score, 4)
            metrics["kid_pass"] = kid_score < config.kid_threshold
            logger.info(f"📊 KID: {kid_score:.4f} {'✅' if metrics['kid_pass'] else '❌'}")
        except ImportError:
            logger.error("⚠️ cleanfid не установлен")
            metrics["kid"] = None
    
    return metrics


def plot_histogram_comparison(real_dir: str, fake_dir: str, output_path: str):
    """Визуализация сравнения гистограмм"""
    checker = SystematicBiasChecker(real_dir, fake_dir)
    real_images = checker.load_images(Path(real_dir), max_images=100)
    fake_images = checker.load_images(Path(fake_dir), max_images=100)
    
    if not real_images or not fake_images:
        return
    
    # Сбор всех пикселей
    real_pixels = np.concatenate([img.flatten() for img in real_images])
    fake_pixels = np.concatenate([img.flatten() for img in fake_images])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Гистограмма яркости
    axes[0, 0].hist(real_pixels, bins=50, alpha=0.7, label='Real', color='blue')
    axes[0, 0].hist(fake_pixels, bins=50, alpha=0.7, label='Synthetic', color='red')
    axes[0, 0].set_title('Pixel Intensity Distribution')
    axes[0, 0].set_xlabel('Intensity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Поканальные средние
    real_means = []
    fake_means = []
    for c in range(3):
        real_means.append(np.mean([np.mean(img[:, :, c]) for img in real_images]))
        fake_means.append(np.mean([np.mean(img[:, :, c]) for img in fake_images]))
    
    x = np.arange(3)
    width = 0.35
    axes[0, 1].bar(x - width/2, real_means, width, label='Real', color='blue')
    axes[0, 1].bar(x + width/2, fake_means, width, label='Synthetic', color='red')
    axes[0, 1].set_title('Channel Means')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(['R', 'G', 'B'])
    axes[0, 1].legend()
    
    # QQ-plot для проверки распределения
    from scipy import stats
    real_sample = np.random.choice(real_pixels, min(10000, len(real_pixels)))
    fake_sample = np.random.choice(fake_pixels, min(10000, len(fake_pixels)))
    
    axes[1, 0].scatter(np.sort(real_sample), np.sort(fake_sample), alpha=0.5, s=1)
    axes[1, 0].plot([0, 255], [0, 255], 'r--', label='y=x (ideal)')
    axes[1, 0].set_title('Q-Q Plot: Real vs Synthetic')
    axes[1, 0].set_xlabel('Real Quantiles')
    axes[1, 0].set_ylabel('Synthetic Quantiles')
    axes[1, 0].legend()
    
    # Распределение стандартных отклонений
    real_stds = [np.std(img) for img in real_images]
    fake_stds = [np.std(img) for img in fake_images]
    axes[1, 1].boxplot([real_stds, fake_stds], labels=['Real', 'Synthetic'])
    axes[1, 1].set_title('Standard Deviation Distribution')
    axes[1, 1].set_ylabel('Std Dev')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📈 Визуализация сохранена: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Полная валидация качества генерации")
    parser.add_argument("--real_dir", type=str, default="./data/256_yolo/defect_patches/images/train", help="Реальные текстуры")
    parser.add_argument("--fake_dir", type=str, default="./results/defective_dataset/images",  help="Синтетические фоны")
    parser.add_argument("--output", type=str, default="./results/validation")
    parser.add_argument("--skip_bias", action="store_true", help="Пропустить проверку bias")
    
    args = parser.parse_args()
    
    config = ValidationConfig()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("📊 ПОЛНАЯ ВАЛИДАЦИЯ КАЧЕСТВА")
    logger.info("=" * 60)
    
    # 1. FID и KID
    logger.info("\n🔍 ЭТАП 1: Вычисление FID и KID")
    logger.info("-" * 40)
    metrics = compute_fid_kid(args.real_dir, args.fake_dir, config)
    
    # 2. Systematic bias
    bias_report = {}
    if not args.skip_bias:
        logger.info("\n🔍 ЭТАП 2: Проверка systematic bias")
        logger.info("-" * 40)
        checker = SystematicBiasChecker(args.real_dir, args.fake_dir)
        bias_report = checker.check_bias()
        
        if "error" not in bias_report:
            logger.info(f"Гистограммы: {'✅' if bias_report['histogram']['pass'] else '❌'} "
                       f"(Δmean={bias_report['histogram']['mean_diff_percent']:.1f}%)")
            logger.info(f"Цветовой баланс: {'✅' if bias_report['color_balance']['pass'] else '❌'}")
            logger.info(f"Частотный спектр: {'✅' if bias_report['frequency']['pass'] else '❌'}")
            logger.info(f"ОБЩИЙ ВЕРДИКТ: {'✅ PRODUCTION READY' if bias_report['overall_pass'] else '❌ ТРЕБУЕТСЯ ДОРАБОТКА'}")
        
        # Визуализация
        plot_histogram_comparison(
            args.real_dir, 
            args.fake_dir, 
            output_dir / "histogram_comparison.png"
        )
    
    # 3. Сохранение полного отчета
    full_report = {
        "metrics": metrics,
        "systematic_bias": bias_report,
        "thresholds": {
            "fid": config.fid_threshold,
            "kid": config.kid_threshold
        }
    }
    
    report_path = output_dir / "full_validation_report.json"
    save_json(full_report, report_path)
    
    # 4. Итоговое резюме
    logger.info("\n" + "=" * 60)
    logger.info("📋 ИТОГОВОЕ РЕЗЮМЕ")
    logger.info("=" * 60)
    
    all_passed = True
    if metrics.get("fid_pass") is not None:
        logger.info(f"FID: {metrics['fid']:.2f} {'✅' if metrics['fid_pass'] else '❌'} (порог: {config.fid_threshold})")
        all_passed = all_passed and metrics['fid_pass']
    
    if metrics.get("kid_pass") is not None:
        logger.info(f"KID: {metrics['kid']:.4f} {'✅' if metrics['kid_pass'] else '❌'} (порог: {config.kid_threshold})")
        all_passed = all_passed and metrics['kid_pass']
    
    if bias_report and "overall_pass" in bias_report:
        logger.info(f"Systematic bias: {'✅' if bias_report['overall_pass'] else '❌'}")
        all_passed = all_passed and bias_report['overall_pass']
    
    logger.info("-" * 40)
    if all_passed:
        logger.info("✅ ДАТАСЕТ ГОТОВ ДЛЯ ДООБУЧЕНИЯ DINOv2")
    else:
        logger.info("⚠️ РЕКОМЕНДУЕТСЯ УЛУЧШИТЬ КАЧЕСТВО ГЕНЕРАЦИИ")
    
    logger.info(f"📁 Полный отчет: {report_path}")


if __name__ == "__main__":
    main()