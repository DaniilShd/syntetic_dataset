#!/usr/bin/env python3
"""
03_validate_quality.py - Валидация качества синтетических данных (без ресайза)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import ValidationConfig
from utils import logger


def convert_to_serializable(obj):
    """Конвертация numpy типов в JSON-сериализуемые"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class SystematicBiasChecker:
    """Проверка систематических смещений (изображения уже 256x256)"""
    
    def __init__(self, real_dir: str, fake_dir: str, config: Optional[ValidationConfig] = None):
        self.real_dir = Path(real_dir)
        self.fake_dir = Path(fake_dir)
        self.config = config or ValidationConfig()
        
    def load_images(self, directory: Path, max_images: int = 500) -> List[np.ndarray]:
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_path in directory.glob(ext):
                if len(images) >= max_images:
                    break
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(np.array(img))
                except:
                    continue
        return images[:max_images]
    
    def check_bias(self) -> Dict:
        logger.info("Загрузка изображений...")
        real = self.load_images(self.real_dir)
        fake = self.load_images(self.fake_dir)
        logger.info(f"Реальных: {len(real)}, Синтетических: {len(fake)}")
        
        if not real or not fake:
            return {"error": "Нет изображений"}
        
        real_hist = self.compute_histogram_stats(real)
        fake_hist = self.compute_histogram_stats(fake)
        real_color = self.compute_color_balance(real)
        fake_color = self.compute_color_balance(fake)
        real_spec = self.compute_frequency_spectrum(real)
        fake_spec = self.compute_frequency_spectrum(fake)
        
        # Конвертация в float для безопасной сериализации
        report = {
            "histogram": {
                "mean_diff": float(abs(real_hist["mean"]["value"] - fake_hist["mean"]["value"])),
                "mean_diff_percent": float(round(abs(real_hist["mean"]["value"] - fake_hist["mean"]["value"]) / real_hist["mean"]["value"] * 100, 2) if real_hist["mean"]["value"] > 0 else 0),
                "std_diff": float(abs(real_hist["std"]["value"] - fake_hist["std"]["value"])),
                "std_diff_percent": float(round(abs(real_hist["std"]["value"] - fake_hist["std"]["value"]) / real_hist["std"]["value"] * 100, 2) if real_hist["std"]["value"] > 0 else 0),
                "pass": bool((abs(real_hist["mean"]["value"] - fake_hist["mean"]["value"]) / real_hist["mean"]["value"] * 100 < 5.0) if real_hist["mean"]["value"] > 0 else False)
            },
            "color_balance": {
                "R_diff": float(abs(real_color["R_mean"] - fake_color["R_mean"])),
                "G_diff": float(abs(real_color["G_mean"] - fake_color["G_mean"])),
                "B_diff": float(abs(real_color["B_mean"] - fake_color["B_mean"])),
                "pass": bool(abs(real_color["R_mean"] - fake_color["R_mean"]) / real_color["R_mean"] * 100 < 10.0 if real_color["R_mean"] > 0 else False)
            },
            "frequency": {
                "low_freq_diff": float(abs(real_spec["low_freq_ratio"] - fake_spec["low_freq_ratio"])),
                "high_freq_diff": float(abs(real_spec["high_freq_ratio"] - fake_spec["high_freq_ratio"])),
                "pass": bool(abs(real_spec["low_freq_ratio"] - fake_spec["low_freq_ratio"]) < 0.1)
            }
        }
        report["overall_pass"] = bool(report["histogram"]["pass"] and report["color_balance"]["pass"] and report["frequency"]["pass"])
        return report
    
    def compute_histogram_stats(self, images: List[np.ndarray]) -> Dict:
        means, stds = [], []
        for img in images:
            for c in range(3):
                ch = img[:, :, c].flatten()
                means.append(float(np.mean(ch)))
                stds.append(float(np.std(ch)))
        return {"mean": {"value": float(np.mean(means))}, "std": {"value": float(np.mean(stds))}}
    
    def compute_color_balance(self, images: List[np.ndarray]) -> Dict:
        r, g, b = [], [], []
        for img in images:
            r.append(float(np.mean(img[:, :, 0])))
            g.append(float(np.mean(img[:, :, 1])))
            b.append(float(np.mean(img[:, :, 2])))
        return {"R_mean": float(np.mean(r)), "G_mean": float(np.mean(g)), "B_mean": float(np.mean(b))}
    
    def compute_frequency_spectrum(self, images: List[np.ndarray]) -> Dict:
        lows, highs = [], []
        for img in images[:min(100, len(images))]:
            gray = np.mean(img, axis=2)
            mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
            h, w = mag.shape
            ch, cw = h // 2, w // 2
            r = min(10, h//4, w//4)
            total = np.sum(mag)
            low = float(np.sum(mag[ch-r:ch+r, cw-r:cw+r]) / total if total > 0 else 0)
            bw = min(20, h//4, w//4)
            high = float((np.sum(np.abs(mag[:bw, :])) + np.sum(np.abs(mag[-bw:, :])) + 
                         np.sum(np.abs(mag[:, :bw])) + np.sum(np.abs(mag[:, -bw:]))) / total if total > 0 else 0)
            lows.append(low)
            highs.append(high)
        return {"low_freq_ratio": float(np.mean(lows)), "high_freq_ratio": float(np.mean(highs))}


def compute_fid_kid(real_dir: str, fake_dir: str, config: ValidationConfig) -> Dict:
    try:
        from cleanfid import fid
        score = fid.compute_fid(real_dir, fake_dir, device=config.device, batch_size=config.fid_batch_size)
        return {"fid": float(score), "fid_pass": bool(score < config.fid_threshold)}
    except:
        return {"fid": None, "fid_pass": None}


def plot_histogram_comparison(real_dir: str, fake_dir: str, output_path: Path):
    checker = SystematicBiasChecker(real_dir, fake_dir)
    real = checker.load_images(Path(real_dir), 100)
    fake = checker.load_images(Path(fake_dir), 100)
    if not real or not fake:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    real_pix = np.concatenate([img.flatten() for img in real[:50]])
    fake_pix = np.concatenate([img.flatten() for img in fake[:50]])
    axes[0].hist(real_pix, bins=50, alpha=0.7, label='Real', color='blue', density=True)
    axes[0].hist(fake_pix, bins=50, alpha=0.7, label='Synthetic', color='red', density=True)
    axes[0].set_title('Pixel Intensity'), axes[0].legend()
    
    real_stds = [np.std(img) for img in real]
    fake_stds = [np.std(img) for img in fake]
    axes[1].boxplot([real_stds, fake_stds], labels=['Real', 'Synthetic'])
    axes[1].set_title('Std Dev')
    
    plt.tight_layout(), plt.savefig(output_path, dpi=150), plt.close()
    logger.info(f"📈 {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", default="./data/256_yolo/balanced_clean_patches/train")
    parser.add_argument("--fake_dir", default="./data/dataset_synthetic/clean_patches")
    parser.add_argument("--output", default="./results/validation")
    parser.add_argument("--skip_bias", action="store_true")
    parser.add_argument("--skip_fid", action="store_true")
    args = parser.parse_args()
    
    config = ValidationConfig()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {} if args.skip_fid else compute_fid_kid(args.real_dir, args.fake_dir, config)
    bias = {}
    if not args.skip_bias:
        bias = SystematicBiasChecker(args.real_dir, args.fake_dir, config).check_bias()
        if "error" not in bias:
            logger.info(f"Гистограммы: {'✅' if bias['histogram']['pass'] else '❌'}")
            logger.info(f"Цвет: {'✅' if bias['color_balance']['pass'] else '❌'}")
            logger.info(f"ВЕРДИКТ: {'✅' if bias['overall_pass'] else '❌'}")
        plot_histogram_comparison(args.real_dir, args.fake_dir, output_dir / "histogram.png")
    
    # Безопасное сохранение JSON
    report = {"metrics": metrics, "bias": bias}
    with open(output_dir / "report.json", 'w') as f:
        json.dump(report, f, indent=2, default=convert_to_serializable)
    
    logger.info(f"✅ {output_dir}/report.json")


if __name__ == "__main__":
    main()