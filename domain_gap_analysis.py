#!/usr/bin/env python3
"""
domain_gap_analysis.py - Исправленная версия
Анализ domain gap с корректными метриками и читаемым выводом
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime
import argparse
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')


class Config:
    model_name = "facebook/dinov2-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    num_samples = 500
    random_seed = 42
    nn_test_samples = 200  # Количество семплов для 1-NN теста


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], resize_to: int = 256):
        self.image_paths = image_paths
        self.resize_to = resize_to
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.resize_to and img.size != (self.resize_to, self.resize_to):
            img = img.resize((self.resize_to, self.resize_to), Image.Resampling.LANCZOS)
        
        inputs = processor(images=img, return_tensors="pt")
        return inputs.pixel_values.squeeze(0), path.name


class DomainGapAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": Config.model_name,
                "device": Config.device,
                "num_samples": Config.num_samples,
                "seed": Config.random_seed
            }
        }
        
    def extract_features(self, image_paths: List[Path], dataset_name: str) -> Tuple[np.ndarray, List[str]]:
        print(f"\n📊 Извлечение признаков из {dataset_name} ({len(image_paths)} изображений)...")
        
        dataset = ImageDataset(image_paths)
        loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)
        
        features_list = []
        filenames = []
        
        with torch.no_grad():
            for batch, names in tqdm(loader, desc=f"  {dataset_name}"):
                batch = batch.to(Config.device)
                output = backbone(batch)
                features = output.pooler_output.cpu().numpy()
                
                features_list.append(features)
                filenames.extend(names)
        
        features = np.concatenate(features_list, axis=0)
        return features, filenames
    
    def compute_statistics(self, features: np.ndarray) -> Dict:
        stats = {
            "mean": float(np.mean(features)),
            "std": float(np.std(features)),
            "min": float(np.min(features)),
            "max": float(np.max(features)),
            "median": float(np.median(features)),
            "variance": float(np.var(features)),
            "l2_norm_mean": float(np.mean(np.linalg.norm(features, axis=1))),
            "l2_norm_std": float(np.std(np.linalg.norm(features, axis=1)))
        }
        return stats
    
    def compute_per_channel_emd(self, features1: np.ndarray, features2: np.ndarray) -> Dict:
        emd_values = []
        for i in range(features1.shape[1]):
            emd = wasserstein_distance(features1[:, i], features2[:, i])
            emd_values.append(emd)
        
        emd_values = np.array(emd_values)
        return {
            "mean_emd": float(np.mean(emd_values)),
            "std_emd": float(np.std(emd_values)),
            "max_emd": float(np.max(emd_values)),
            "min_emd": float(np.min(emd_values)),
            "median_emd": float(np.median(emd_values)),
            "top10_unstable_channels": np.argsort(emd_values)[-10:].tolist(),
            "percentile_95_emd": float(np.percentile(emd_values, 95))
        }
    
    def compute_similarity_metrics(self, features1: np.ndarray, features2: np.ndarray) -> Dict:
        """
        ИСПРАВЛЕННАЯ версия метрик схожести
        """
        n_test = min(Config.nn_test_samples, len(features1), len(features2))
        
        # Центроиды
        centroid1 = np.mean(features1, axis=0)
        centroid2 = np.mean(features2, axis=0)
        centroid_distance = np.linalg.norm(centroid1 - centroid2)
        
        # Cosine similarity между центроидами
        cosine_sim = np.dot(centroid1, centroid2) / (np.linalg.norm(centroid1) * np.linalg.norm(centroid2))
        
        # ===== ИСПРАВЛЕННЫЙ 1-NN ТЕСТ =====
        # Берём n_test семплов из каждого распределения
        indices1 = np.random.choice(len(features1), n_test, replace=False)
        indices2 = np.random.choice(len(features2), n_test, replace=False)
        
        feats1 = features1[indices1]
        feats2 = features2[indices2]
        
        # Объединяем и создаём метки: 0 = domain A, 1 = domain B
        combined = np.concatenate([feats1, feats2])
        labels = np.array([0] * n_test + [1] * n_test)
        
        # Для каждого семпла ищем ближайшего соседа ВО ВСЁМ объединённом сете
        # ИСКЛЮЧАЯ сам этот семпл
        correct_same_domain = 0
        total = 0
        
        for i in range(2 * n_test):
            # Ищем расстояния до всех других семплов
            query = combined[i:i+1]
            distances = cdist(query, combined, metric='cosine')[0]
            
            # Исключаем self-distance (0)
            distances[i] = np.inf
            
            # Находим ближайшего соседа
            nn_idx = np.argmin(distances)
            
            # Проверяем, совпадает ли домен
            if labels[i] == labels[nn_idx]:
                correct_same_domain += 1
            total += 1
        
        nn_accuracy = correct_same_domain / total if total > 0 else 0.5
        
        # ===== ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ =====
        
        # Nearest neighbor distances
        distances_a_to_b = cdist(feats1, feats2, metric='cosine')
        distances_a_to_a = cdist(feats1, feats1, metric='cosine')
        distances_b_to_b = cdist(feats2, feats2, metric='cosine')
        
        # Убираем self-distances
        np.fill_diagonal(distances_a_to_a, np.inf)
        np.fill_diagonal(distances_b_to_b, np.inf)
        
        # Среднее расстояние до ближайшего соседа ВНУТРИ домена
        mean_intra_a = np.mean(np.min(distances_a_to_a, axis=1))
        mean_intra_b = np.mean(np.min(distances_b_to_b, axis=1))
        
        # Среднее расстояние до ближайшего соседа МЕЖДУ доменами
        mean_inter_a_to_b = np.mean(np.min(distances_a_to_b, axis=1))
        mean_inter_b_to_a = np.mean(np.min(distances_a_to_b, axis=0))
        
        # Domain Gap Ratio: насколько меж-доменное расстояние больше внутри-доменного
        intra_avg = (mean_intra_a + mean_intra_b) / 2
        inter_avg = (mean_inter_a_to_b + mean_inter_b_to_a) / 2
        gap_ratio = inter_avg / (intra_avg + 1e-8)
        
        # Domain Overlap Score (0 = полностью разделены, 1 = полностью перекрываются)
        # Если 1-NN даёт 50% точность → домены неразличимы → overlap = 1.0
        overlap_score = 2.0 * (1.0 - nn_accuracy)  # Трансформация: 50%→1.0, 100%→0.0
        overlap_score = np.clip(overlap_score, 0.0, 1.0)
        
        return {
            "centroid_distance": float(centroid_distance),
            "centroid_cosine_similarity": float(cosine_sim),
            "mean_intra_domain_distance_A": float(mean_intra_a),
            "mean_intra_domain_distance_B": float(mean_intra_b),
            "mean_inter_domain_distance": float(inter_avg),
            "domain_gap_ratio": float(gap_ratio),
            "1nn_domain_accuracy": float(nn_accuracy),
            "1nn_expected_for_identical": 0.5,
            "domain_overlap_score": float(overlap_score)
        }
    
    def visualize_pca(self, features_list: List[Tuple[np.ndarray, str, np.ndarray]], save_path: Path):
        """
        Визуализация PCA с эллипсами распределений
        features_list: [(features, name, labels)]
        """
        print("\n📊 Создание PCA визуализации...")
        
        all_features = []
        all_labels = []
        all_names = []
        
        for features, name in features_list:
            n = min(len(features), 500)
            indices = np.random.choice(len(features), n, replace=False)
            
            all_features.append(features[indices])
            all_names.extend([name] * n)
        
        all_features = np.concatenate(all_features, axis=0)
        
        # Стандартизация
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(all_features)
        
        # PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        explained_var = pca.explained_variance_ratio_ * 100
        
        # Цвета
        colors = {'original': '#2196F3', 'synthetic': '#FF9800', 'augmented': '#4CAF50'}
        
        # Построение
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # 1. Scatter plot с эллипсами
        ax1 = axes[0]
        
        for name in set(all_names):
            mask = np.array([n == name for n in all_names])
            data = features_2d[mask]
            color = colors.get(name, '#999999')
            
            # Scatter
            ax1.scatter(data[:, 0], data[:, 1], c=color, label=name, 
                       alpha=0.4, s=8, edgecolors='none')
            
            # Эллипс (2 стандартных отклонения)
            if len(data) > 2:
                try:
                    from matplotlib.patches import Ellipse
                    cov = np.cov(data.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    
                    # Угол эллипса
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                    
                    # Размеры (2 std)
                    width, height = 2 * np.sqrt(eigenvalues) * 2
                    
                    ellipse = Ellipse(
                        xy=np.mean(data, axis=0),
                        width=width, height=height,
                        angle=angle, facecolor='none',
                        edgecolor=color, linewidth=2, linestyle='--', alpha=0.8
                    )
                    ax1.add_patch(ellipse)
                except:
                    pass
        
        ax1.set_xlabel(f'PC1 ({explained_var[0]:.1f}% variance)')
        ax1.set_ylabel(f'PC2 ({explained_var[1]:.1f}% variance)')
        ax1.set_title('Распределение признаков DINOv2 (PCA)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Косой график (расстояния)
        ax2 = axes[1]
        
        # Вычисляем расстояния между доменами
        names_list = list(set(all_names))
        if len(names_list) >= 2:
            # Выбираем по 100 семплов
            data_a = features_2d[np.array([n == names_list[0] for n in all_names])][:100]
            data_b = features_2d[np.array([n == names_list[1] for n in all_names])][:100]
            
            min_len = min(len(data_a), len(data_b))
            data_a = data_a[:min_len]
            data_b = data_b[:min_len]
            
            # Меж-доменные расстояния
            diff = data_a - data_b
            distances = np.linalg.norm(diff, axis=1)
            
            # Гистограмма расстояний
            ax2.hist(distances, bins=30, alpha=0.7, color='#673AB7', edgecolor='black')
            ax2.axvline(x=np.mean(distances), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(distances):.3f}')
            ax2.axvline(x=np.median(distances), color='g', linestyle=':', 
                       label=f'Median: {np.median(distances):.3f}')
            
            ax2.set_xlabel('Pairwise distance in PCA space')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Distances between {names_list[0]} and {names_list[1]}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'pca_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ PCA визуализация сохранена: {save_path / 'pca_visualization.png'}")
        
        return {
            "explained_variance_ratio": explained_var.tolist(),
            "total_explained_variance": float(np.sum(explained_var))
        }
    
    def generate_human_readable_report(self):
        """Генерирует читаемый отчёт с корректной интерпретацией"""
        report = []
        report.append("=" * 80)
        report.append("DOMAIN GAP ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {self.results['timestamp']}")
        report.append(f"Model: {Config.model_name}")
        report.append("")
        
        original_stats = self.results.get('original_statistics', {})
        synthetic_stats = self.results.get('synthetic_statistics', {})
        similarity = self.results.get('similarity_metrics', {})
        emd_data = self.results.get('emd_metrics', {})
        
        report.append("-" * 80)
        report.append("1. СТАТИСТИКИ РАСПРЕДЕЛЕНИЯ ПРИЗНАКОВ")
        report.append("-" * 80)
        report.append(f"{'Метрика':<30} {'Оригинал':<15} {'Синтетика':<15} {'Разница %':<15}")
        report.append("-" * 80)
        
        for key in ['mean', 'std', 'variance', 'l2_norm_mean']:
            orig_val = original_stats.get(key, 0)
            syn_val = synthetic_stats.get(key, 0)
            diff_pct = abs(syn_val - orig_val) / (abs(orig_val) + 1e-8) * 100
            report.append(f"{key:<30} {orig_val:<15.6f} {syn_val:<15.6f} {diff_pct:<15.2f}%")
        
        report.append("")
        report.append("-" * 80)
        report.append("2. МЕТРИКИ СХОДСТВА ДОМЕНОВ")
        report.append("-" * 80)
        
        overlap = similarity.get('domain_overlap_score', 0)
        nn_acc = similarity.get('1nn_domain_accuracy', 1.0)
        gap_ratio = similarity.get('domain_gap_ratio', 999)
        
        report.append(f"  {'Cosine similarity центроид:':<40} {similarity.get('centroid_cosine_similarity', 0):.4f}")
        report.append(f"  {'1-NN точность разделения:':<40} {nn_acc:.4f} (0.50 = домены неразличимы)")
        report.append(f"  {'Domain Overlap Score:':<40} {overlap:.4f} (1.00 = полное перекрытие)")
        report.append(f"  {'Domain Gap Ratio:':<40} {gap_ratio:.4f} (1.00 = одинаковые)")
        report.append(f"  {'Внутри-доменное расстояние (A):':<40} {similarity.get('mean_intra_domain_distance_A', 0):.6f}")
        report.append(f"  {'Внутри-доменное расстояние (B):':<40} {similarity.get('mean_intra_domain_distance_B', 0):.6f}")
        report.append(f"  {'Меж-доменное расстояние:':<40} {similarity.get('mean_inter_domain_distance', 0):.6f}")
        
        report.append("")
        report.append("-" * 80)
        report.append("3. РАСХОЖДЕНИЕ РАСПРЕДЕЛЕНИЙ (EMD)")
        report.append("-" * 80)
        report.append(f"  {'Средний EMD:':<30} {emd_data.get('mean_emd', 0):.6f}")
        report.append(f"  {'Медианный EMD:':<30} {emd_data.get('median_emd', 0):.6f}")
        report.append(f"  {'95-й перцентиль EMD:':<30} {emd_data.get('percentile_95_emd', 0):.6f}")
        report.append(f"  {'Максимальный EMD:':<30} {emd_data.get('max_emd', 0):.6f}")
        
        report.append("")
        report.append("-" * 80)
        report.append("4. ИНТЕРПРЕТАЦИЯ КАЧЕСТВА СИНТЕТИКИ")
        report.append("-" * 80)
        
        # КОРРЕКТНАЯ интерпретация
        if overlap > 0.90 and nn_acc < 0.55:
            quality = "✅ ОТЛИЧНО"
            details = [
                "Синтетика практически неотличима от оригинала",
                "Распределения признаков сильно перекрываются",
                "1-NN классификатор не может разделить домены (< 55% точность)",
                "Рекомендация: Синтетику можно использовать без ограничений"
            ]
        elif overlap > 0.75 and nn_acc < 0.65:
            quality = "🟡 ХОРОШО"
            details = [
                "Синтетика близка к оригиналу, но есть небольшой сдвиг",
                "Распределения в основном перекрываются",
                "1-NN классификатор показывает умеренную разделимость (55-65%)",
                "Рекомендация: Использовать синтетику с весом 0.3-0.5"
            ]
        elif overlap > 0.50 and nn_acc < 0.75:
            quality = "🟠 УДОВЛЕТВОРИТЕЛЬНО"
            details = [
                "Заметный domain gap между синтетикой и оригиналом",
                "Распределения частично перекрываются",
                "1-NN классификатор хорошо разделяет домены (65-75%)",
                "Рекомендация: Уменьшить strength до 0.05-0.10, увеличить high_freq_alpha"
            ]
        elif overlap > 0.25:
            quality = "🔴 ПЛОХО"
            details = [
                "Сильный domain gap",
                "Распределения значительно различаются",
                "1-NN классификатор легко разделяет домены (> 75%)",
                "Рекомендация: Кардинально уменьшить strength до 0.02-0.05"
            ]
        else:
            quality = "⛔ КРИТИЧНО"
            details = [
                "Синтетика полностью не соответствует оригиналу",
                "Распределения практически не перекрываются",
                "1-NN классификатор с 100% точностью определяет источник",
                "Рекомендация: Пересмотреть параметры генерации полностью"
            ]
        
        report.append(f"\n  Качество синтетики: {quality}")
        report.append(f"  Domain Overlap Score: {overlap:.3f}/1.000")
        report.append(f"  1-NN accuracy (чем ближе к 0.5, тем лучше): {nn_acc:.3f}")
        report.append("")
        for detail in details:
            report.append(f"  • {detail}")
        
        report.append("")
        report.append("=" * 80)
        report.append("КАК ЧИТАТЬ МЕТРИКИ:")
        report.append("  Domain Overlap Score: 1.0 = идеально, 0.0 = полностью разные домены")
        report.append("  1-NN accuracy: 0.50 = домены неразличимы (хорошо)")
        report.append("                 0.75+ = домены легко разделяются (плохо)")
        report.append("  Cosine similarity: >0.95 = отлично, <0.85 = плохо")
        report.append("  Domain Gap Ratio: ~1.0 = одинаковые домены, >2.0 = сильный gap")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        report_path = self.output_dir / "domain_gap_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        return report_text
    
    def analyze(self, original_dir: Path, synthetic_dir: Path):
        print("=" * 80)
        print("🔬 DOMAIN GAP ANALYSIS")
        print("=" * 80)
        
        original_images = sorted(list((original_dir / "images").glob("*.png")) + 
                                list((original_dir / "images").glob("*.jpg")))
        synthetic_images = sorted(list((synthetic_dir / "images").glob("*.png")) + 
                                 list((synthetic_dir / "images").glob("*.jpg")))
        
        print(f"\n📂 Original images: {len(original_images)}")
        print(f"📂 Synthetic images: {len(synthetic_images)}")
        
        if len(original_images) > Config.num_samples:
            indices = np.random.choice(len(original_images), Config.num_samples, replace=False)
            original_images = [original_images[i] for i in indices]
        
        if len(synthetic_images) > Config.num_samples:
            indices = np.random.choice(len(synthetic_images), Config.num_samples, replace=False)
            synthetic_images = [synthetic_images[i] for i in indices]
        
        print(f"📊 Анализируем по {min(Config.num_samples, len(original_images))} изображений")
        
        # Извлечение признаков
        original_features, _ = self.extract_features(original_images, "original")
        synthetic_features, _ = self.extract_features(synthetic_images, "synthetic")
        
        # Статистики
        self.results['original_statistics'] = self.compute_statistics(original_features)
        self.results['synthetic_statistics'] = self.compute_statistics(synthetic_features)
        
        # Метрики схожести (ИСПРАВЛЕННЫЕ)
        self.results['similarity_metrics'] = self.compute_similarity_metrics(
            original_features, synthetic_features
        )
        
        # EMD
        self.results['emd_metrics'] = self.compute_per_channel_emd(
            original_features, synthetic_features
        )
        
        # Визуализации
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        pca_stats = self.visualize_pca(
            [(original_features, 'original'), (synthetic_features, 'synthetic')],
            viz_dir
        )
        self.results['pca_statistics'] = pca_stats
        
        # Сохраняем JSON
        json_path = self.output_dir / "domain_gap_results.json"
        with open(json_path, 'w') as f:
            def convert(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(self.results, f, indent=2, default=convert)
        
        print(f"\n✅ JSON результаты сохранены: {json_path}")
        
        # Генерируем читаемый отчёт
        self.generate_human_readable_report()
        
        print(f"\n✅ Все результаты сохранены в: {self.output_dir}")
        print("=" * 80)


processor = None
backbone = None


def main():
    parser = argparse.ArgumentParser(description="Анализ domain gap (исправленная версия)")
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--synthetic_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="result_domain_gap")
    parser.add_argument("--num_samples", type=int, default=1000)
    
    args = parser.parse_args()
    
    original_path = Path(args.original_dir)
    synthetic_path = Path(args.synthetic_dir)
    
    if not (original_path / "images").exists():
        print(f"❌ Директория не найдена: {original_path / 'images'}")
        sys.exit(1)
    
    if not (synthetic_path / "images").exists():
        print(f"❌ Директория не найдена: {synthetic_path / 'images'}")
        sys.exit(1)
    
    Config.num_samples = args.num_samples
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Результаты: {output_dir}")
    
    global processor, backbone
    print("\n🔄 Загрузка DINOv2...")
    processor = AutoImageProcessor.from_pretrained(Config.model_name)
    backbone = AutoModel.from_pretrained(Config.model_name).to(Config.device)
    backbone.eval()
    print("✅ Модель загружена")
    
    analyzer = DomainGapAnalyzer(output_dir)
    analyzer.analyze(original_path, synthetic_path)
    
    print("\n✅ Анализ завершен!")
    print(f"📁 Результаты в: {output_dir}")


if __name__ == "__main__":
    main()