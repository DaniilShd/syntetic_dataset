#!/usr/bin/env python3
"""
03_validate_quality.py - Валидация качества синтетических данных (без ресайза)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import ValidationConfig
from utils import logger


class SystematicBiasChecker:
    """Проверка систематических смещений (изображения уже 256x256)"""
    
    def __init__(self, real_dir: str, fake_dir: str, config: Optional[ValidationConfig] = None):
        self.real_dir = Path(real_dir)
        self.fake_dir = Path(fake_dir)
        self.config = config or ValidationConfig()
        
    def load_images(self, directory: Path, max_images: int = 500) -> List[np.ndarray]:
        """Загрузка изображений БЕЗ ресайза"""
        images = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for ext in extensions:
            for img_path in directory.glob(ext):
                if len(images) >= max_images:
                    break
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(np.array(img))
                except Exception as e:
                    logger.debug(f"Ошибка загрузки {img_path}: {e}")
                    continue
                    
        return images[:max_images]
    
    def check_bias(self) -> Dict:
        """Проверка systematic bias (без ресайза)"""
        
        logger.info("Загрузка реальных изображений...")
        real_images = self.load_images(self.real_dir)
        logger.info(f"Загружено {len(real_images)} реальных изображений")
        
        logger.info("Загрузка синтетических изображений...")
        fake_images = self.load_images(self.fake_dir)
        logger.info(f"Загружено {len(fake_images)} синтетических изображений")
        
        if not real_images or not fake_images:
            return {"error": "Нет изображений для анализа"}
        
        real_shapes = set(img.shape[:2] for img in real_images)
        fake_shapes = set(img.shape[:2] for img in fake_images)
        logger.info(f"✅ Размеры реальных: {real_shapes}")
        logger.info(f"✅ Размеры синтетических: {fake_shapes}")
        
        logger.info("Анализ гистограмм...")
        real_hist = self.compute_histogram_stats(real_images)
        fake_hist = self.compute_histogram_stats(fake_images)
        
        logger.info("Анализ цветового баланса...")
        real_color = self.compute_color_balance(real_images)
        fake_color = self.compute_color_balance(fake_images)
        
        logger.info("Анализ частотного спектра...")
        real_spectral = self.compute_frequency_spectrum(real_images)
        fake_spectral = self.compute_frequency_spectrum(fake_images)
        
        bias_report = {
            "histogram": {
                "mean_diff": abs(real_hist["mean"]["value"] - fake_hist["mean"]["value"]),
                "mean_diff_percent": round(abs(real_hist["mean"]["value"] - fake_hist["mean"]["value"]) / real_hist["mean"]["value"] * 100, 2) if real_hist["mean"]["value"] > 0 else 0,
                "std_diff": abs(real_hist["std"]["value"] - fake_hist["std"]["value"]),
                "std_diff_percent": round(abs(real_hist["std"]["value"] - fake_hist["std"]["value"]) / real_hist["std"]["value"] * 100, 2) if real_hist["std"]["value"] > 0 else 0,
                "pass": False
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
        
        HIST_MEAN = getattr(self.config, 'hist_mean_threshold', 5.0)
        HIST_STD = getattr(self.config, 'hist_std_threshold', 10.0)
        COLOR = getattr(self.config, 'color_threshold', 10.0)
        RATIO = getattr(self.config, 'ratio_threshold', 0.05)
        FREQ = getattr(self.config, 'freq_threshold', 0.1)
        
        bias_report["histogram"]["pass"] = (
            bias_report["histogram"]["mean_diff_percent"] < HIST_MEAN and
            bias_report["histogram"]["std_diff_percent"] < HIST_STD
        )
        
        r_diff = (bias_report["color_balance"]["R_diff"] / real_color["R_mean"] * 100) if real_color["R_mean"] > 0 else 0
        g_diff = (bias_report["color_balance"]["G_diff"] / real_color["G_mean"] * 100) if real_color["G_mean"] > 0 else 0
        b_diff = (bias_report["color_balance"]["B_diff"] / real_color["B_mean"] * 100) if real_color["B_mean"] > 0 else 0
        
        bias_report["color_balance"]["pass"] = (
            r_diff < COLOR and g_diff < COLOR and b_diff < COLOR and
            bias_report["color_balance"]["RG_ratio_diff"] < RATIO and
            bias_report["color_balance"]["BG_ratio_diff"] < RATIO
        )
        
        bias_report["frequency"]["pass"] = (
            bias_report["frequency"]["low_freq_diff"] < FREQ and
            bias_report["frequency"]["high_freq_diff"] < FREQ
        )
        
        bias_report["overall_pass"] = (
            bias_report["histogram"]["pass"] and
            bias_report["color_balance"]["pass"] and
            bias_report["frequency"]["pass"]
        )
        
        return bias_report
    
    def compute_histogram_stats(self, images: List[np.ndarray]) -> Dict:
        all_means, all_stds, all_skewness = [], [], []
        for img in tqdm(images, desc="Гистограммы", leave=False):
            for c in range(3):
                channel = img[:, :, c].flatten()
                all_means.append(np.mean(channel))
                all_stds.append(np.std(channel))
                all_skewness.append(self._skewness(channel))
        return {
            "mean": {"value": float(np.mean(all_means)), "std": float(np.std(all_means))},
            "std": {"value": float(np.mean(all_stds)), "std": float(np.std(all_stds))},
            "skewness": {"value": float(np.mean(all_skewness)), "std": float(np.std(all_skewness))}
        }
    
    def _skewness(self, data: np.ndarray) -> float:
        n = len(data)
        if n < 3: return 0.0
        mean, std = np.mean(data), np.std(data)
        if std == 0: return 0.0
        return float(np.sum(((data - mean) / std) ** 3) / n)
    
    def compute_color_balance(self, images: List[np.ndarray]) -> Dict:
        r, g, b = [], [], []
        for img in tqdm(images, desc="Цвета", leave=False):
            r.append(np.mean(img[:, :, 0]))
            g.append(np.mean(img[:, :, 1]))
            b.append(np.mean(img[:, :, 2]))
        r_mean, g_mean, b_mean = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
        return {
            "R_mean": r_mean, "G_mean": g_mean, "B_mean": b_mean,
            "R/G_ratio": r_mean / g_mean if g_mean > 0 else 1.0,
            "B/G_ratio": b_mean / g_mean if g_mean > 0 else 1.0
        }
    
    def compute_frequency_spectrum(self, images: List[np.ndarray]) -> Dict:
        all_spectral = []
        sample_size = min(100, len(images))
        for img in tqdm(images[:sample_size], desc="FFT", leave=False):
            gray = np.mean(img, axis=2)
            fft = np.fft.fft2(gray)
            magnitude = np.abs(np.fft.fftshift(fft))
            h, w = magnitude.shape
            ch, cw = h // 2, w // 2
            r = min(10, h//4, w//4)
            low = magnitude[ch-r:ch+r, cw-r:cw+r]
            total = np.sum(magnitude)
            low_ratio = np.sum(low) / total if total > 0 else 0
            bw = min(20, h//4, w//4)
            high = np.concatenate([magnitude[:bw, :].flatten(), magnitude[-bw:, :].flatten(),
                                   magnitude[:, :bw].flatten(), magnitude[:, -bw:].flatten()])
            high_ratio = np.sum(np.abs(high)) / total if total > 0 else 0
            all_spectral.append({"low": float(low_ratio), "high": float(high_ratio)})
        return {
            "low_freq_ratio": float(np.mean([e["low"] for e in all_spectral])),
            "high_freq_ratio": float(np.mean([e["high"] for e in all_spectral]))
        }


def compute_fid_kid(real_dir: str, fake_dir: str, config: ValidationConfig) -> Dict:
    """FID/KID без ресайза (изображения уже одного размера)"""
    metrics = {}
    try:
        from cleanfid import fid
        fid_score = fid.compute_fid(real_dir, fake_dir, device=config.device, batch_size=config.fid_batch_size)
        metrics["fid"] = round(float(fid_score), 2)
        metrics["fid_pass"] = fid_score < config.fid_threshold
        logger.info(f"📊 FID: {fid_score:.2f} {'✅' if metrics['fid_pass'] else '❌'}")
    except Exception as e:
        logger.error(f"❌ FID error: {e}")
        metrics["fid"] = None
    return metrics


def plot_histogram_comparison(real_dir: str, fake_dir: str, output_path: Path):
    checker = SystematicBiasChecker(real_dir, fake_dir)
    real_images = checker.load_images(Path(real_dir), max_images=100)
    fake_images = checker.load_images(Path(fake_dir), max_images=100)
    if not real_images or not fake_images: return
    
    real_pixels = np.concatenate([img.flatten() for img in real_images[:50]])
    fake_pixels = np.concatenate([img.flatten() for img in fake_images[:50]])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(real_pixels, bins=50, alpha=0.7, label='Real', color='blue', density=True)
    axes[0, 0].hist(fake_pixels, bins=50, alpha=0.7, label='Synthetic', color='red', density=True)
    axes[0, 0].set_title('Pixel Intensity'), axes[0, 0].legend(), axes[0, 0].grid(alpha=0.3)
    
    real_means = [np.mean([np.mean(img[:, :, c]) for img in real_images]) for c in range(3)]
    fake_means = [np.mean([np.mean(img[:, :, c]) for img in fake_images]) for c in range(3)]
    x = np.arange(3)
    axes[0, 1].bar(x - 0.175, real_means, 0.35, label='Real', color='blue', alpha=0.7)
    axes[0, 1].bar(x + 0.175, fake_means, 0.35, label='Synthetic', color='red', alpha=0.7)
    axes[0, 1].set_xticks(x), axes[0, 1].set_xticklabels(['R', 'G', 'B']), axes[0, 1].legend(), axes[0, 1].grid(alpha=0.3, axis='y')
    
    real_stds = [np.std(img) for img in real_images]
    fake_stds = [np.std(img) for img in fake_images]
    bp = axes[1, 1].boxplot([real_stds, fake_stds], labels=['Real', 'Synthetic'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue'), bp['boxes'][1].set_facecolor('lightcoral')
    axes[1, 1].set_title('Std Dev'), axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout(), plt.savefig(output_path, dpi=150, bbox_inches='tight'), plt.close()
    logger.info(f"📈 Визуализация: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", default="./data/256_yolo/balanced_defect_patches/train/images")
    parser.add_argument("--fake_dir", default="./data/dataset_synthetic/defect_patches/images")
    parser.add_argument("--output", default="./results/validation")
    parser.add_argument("--skip_bias", action="store_true")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--fid_threshold", type=float, default=15.0)
    args = parser.parse_args()
    
    config = ValidationConfig()
    config.fid_threshold = args.fid_threshold
    config.compute_fid = not args.skip_fid
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📊 ВАЛИДАЦИЯ (без ресайза): {args.real_dir} vs {args.fake_dir}")
    
    metrics = compute_fid_kid(args.real_dir, args.fake_dir, config) if config.compute_fid else {}
    
    bias_report_clean = {}
    if not args.skip_bias:
        checker = SystematicBiasChecker(args.real_dir, args.fake_dir, config)
        bias_report = checker.check_bias()
        if "error" not in bias_report:
            logger.info(f"Гистограммы: {'✅' if bias_report['histogram']['pass'] else '❌'} (Δmean={bias_report['histogram']['mean_diff_percent']:.1f}%)")
            logger.info(f"Цвет: {'✅' if bias_report['color_balance']['pass'] else '❌'}")
            logger.info(f"Спектр: {'✅' if bias_report['frequency']['pass'] else '❌'}")
            logger.info(f"ВЕРДИКТ: {'✅ READY' if bias_report['overall_pass'] else '❌ FIX'}")
            bias_report_clean = {k: v for k, v in bias_report.items() if k != "raw"}
        plot_histogram_comparison(args.real_dir, args.fake_dir, output_dir / "histogram_comparison.png")
    
    with open(output_dir / "report.json", 'w') as f:
        json.dump({"metrics": metrics, "bias": bias_report_clean}, f, indent=2)
    
    logger.info(f"✅ Отчет: {output_dir}/report.json")


if __name__ == "__main__":
    main()