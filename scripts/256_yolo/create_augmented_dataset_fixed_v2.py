#!/usr/bin/env python3
"""
augment_defect_synthetic.py
Аугментация синтетических данных (defect-only/Poisson blending)
с учётом статичного фона и изменённых областей дефектов.
Оптимизировано для DINO на металлических поверхностях.
"""

import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import shutil
import random
from tqdm import tqdm
import yaml
import argparse
from typing import List, Tuple, Optional


class SyntheticDataAugmenter:
    """
    Аугментация синтетических данных с учётом:
    - Фон статичный (оригинальный лист металла)
    - Области дефектов изменены SD/Poisson blending
    - Локальные искажения для разнообразия текстур
    """
    
    def __init__(self, original_path: Path, output_path: Path, 
                 target_train_size: Optional[int] = None):
        self.original_path = Path(original_path)
        self.output_path = Path(output_path)
        
        # Загружаем конфиг
        self.original_yaml = self.original_path / "data.yaml"
        if self.original_yaml.exists():
            with open(self.original_yaml, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                'nc': 4,
                'names': ['defect_1', 'defect_2', 'defect_3', 'defect_4']
            }
        
        # Собираем train изображения
        train_img_dir = self.original_path / "images"
        self.original_train_images = sorted(
            list(train_img_dir.glob("*.jpg")) +
            list(train_img_dir.glob("*.png")) +
            list(train_img_dir.glob("*.jpeg"))
        )
        self.num_original = len(self.original_train_images)
        
        if self.num_original == 0:
            raise ValueError(f"Нет изображений в {train_img_dir}")
        
        # Целевой размер
        self.target_train_size = target_train_size or self.num_original
        self.num_to_generate = max(0, self.target_train_size - self.num_original)
        
        print(f"\n{'='*60}")
        print(f"📊 СТАТИСТИКА ДАННЫХ")
        print(f"{'='*60}")
        print(f"  Исходных изображений : {self.num_original}")
        print(f"  Целевой размер train  : {self.target_train_size}")
        print(f"  Нужно сгенерировать   : {self.num_to_generate}")
        print(f"  Классов дефектов      : {self.config.get('nc', 4)}")
        print(f"  Имена классов         : {self.config.get('names', [])}")
        print(f"{'='*60}\n")
    
    def get_dino_augmentation(self) -> A.Compose:
        """
        Реалистичные аугментации для синтетических данных.
        БЕЗ геометрических искажений, НО с заметными вариациями освещения и текстуры.
        """
        return A.Compose([
            # === ОСВЕЩЕНИЕ (заметные вариации, как на реальном заводе) ===
            A.RandomBrightnessContrast(
                brightness_limit=0.15,            # ±15% яркости (разное время суток/освещение)
                contrast_limit=0.15,              # ±15% контраста
                p=0.5                             # Применять к 50% изображений
            ),
            
            # === ЦВЕТОВЫЕ ВАРИАЦИИ (разные камеры/настройки) ===
            A.HueSaturationValue(
                hue_shift_limit=5,                # ±5° оттенка (минимально)
                sat_shift_limit=10,               # ±10% насыщенности
                val_shift_limit=10,               # ±10% яркости
                p=0.4
            ),
            
            # === ТОНОВАЯ КРИВАЯ (естественные вариации сенсора) ===
            A.RandomToneCurve(
                scale=0.15,                       # Заметное, но реалистичное
                p=0.3
            ),
            
            # === ГАММА-КОРРЕКЦИЯ (разные настройки камеры) ===
            A.RandomGamma(
                gamma_limit=(90, 110),            # ±10% гаммы
                p=0.3
            ),
            
            # === ЛЁГКИЙ ШУМ (вариации качества сенсора) ===
            A.GaussNoise(
                std_range=(0.005, 0.015),         # Очень лёгкий шум (0.5-1.5% std)
                p=0.2
            ),
            
            # === ЛОКАЛЬНЫЕ КОНТРАСТНЫЕ ВАРИАЦИИ (CLAHE) ===
            A.CLAHE(
                clip_limit=2.0,                   # Лёгкое улучшение контраста
                tile_grid_size=(8, 8),
                p=0.2
            ),
            
            # БЕЗ Affine (без поворотов, сдвигов, масштаба - геометрия фиксирована)
            # БЕЗ HorizontalFlip (дефекты не симметричны)
            # БЕЗ ElasticTransform (не ломаем структуру дефектов)
            # БЕЗ CoarseDropout (не удаляем части дефектов)
            # БЕЗ MotionBlur (камера статична)
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.85,                   # Высокая видимость дефектов
            min_area=25
        ))
    
    def read_yolo_labels(self, label_path: Path) -> Tuple[List[List[float]], List[int]]:
        """Читает YOLO-аннотации с валидацией."""
        bboxes = []
        class_labels = []
        
        if not label_path.exists() or label_path.stat().st_size == 0:
            return bboxes, class_labels
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    cls = int(float(parts[0]))
                    xc = np.clip(float(parts[1]), 0.0, 1.0)
                    yc = np.clip(float(parts[2]), 0.0, 1.0)
                    w = np.clip(float(parts[3]), 0.0, 1.0)
                    h = np.clip(float(parts[4]), 0.0, 1.0)
                    
                    # Валидация размера
                    if w > 0.005 and h > 0.005 and w <= 1.0 and h <= 1.0:
                        bboxes.append([xc, yc, w, h])
                        class_labels.append(cls)
        
        except Exception as e:
            print(f"  ⚠️ Ошибка чтения {label_path}: {e}")
        
        return bboxes, class_labels
    
    def write_yolo_labels(self, label_path: Path, bboxes: List[List[float]], 
                         class_labels: List[int]) -> None:
        """Записывает YOLO-аннотации."""
        label_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(label_path, 'w') as f:
            for bbox, cls in zip(bboxes, class_labels):
                xc = np.clip(bbox[0], 0.0, 1.0)
                yc = np.clip(bbox[1], 0.0, 1.0)
                w = np.clip(bbox[2], 0.0, 1.0)
                h = np.clip(bbox[3], 0.0, 1.0)
                
                if w > 0.003 and h > 0.003:  # Минимальный размер
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    
    def augment_single(self, image: np.ndarray, bboxes: List[List[float]], 
                      class_labels: List[int], transform: A.Compose
                      ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Аугментирует одно изображение с обработкой ошибок.
        Возвращает изображение + валидные bboxes + лейблы.
        """
        if not bboxes:
            # Если нет дефектов — всё равно аугментируем фон
            try:
                augmented = transform(image=image, bboxes=[], class_labels=[])
                return augmented['image'], [], []
            except Exception:
                return image, [], []
        
        try:
            augmented = transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            # Фильтруем невалидные bbox'ы
            valid_bboxes = []
            valid_labels = []
            
            for bbox, label in zip(augmented['bboxes'], augmented['class_labels']):
                x, y, w, h = bbox
                
                # Проверка границ
                if (0.0 < x < 1.0 and 0.0 < y < 1.0 and 
                    0.0 < w <= 1.0 and 0.0 < h <= 1.0 and 
                    w > 0.005 and h > 0.005):
                    valid_bboxes.append([x, y, w, h])
                    valid_labels.append(label)
            
            return augmented['image'], valid_bboxes, valid_labels
        
        except Exception as e:
            # Fallback: возвращаем оригинал
            return image, bboxes, class_labels
    
    def create_augmented_dataset(self) -> None:
        """
        Создаёт аугментированный датасет с сохранением структуры.
        """
        print(f"\n{'='*60}")
        print(f"🎨 СОЗДАНИЕ АУГМЕНТИРОВАННОГО ДАТАСЕТА")
        print(f"{'='*60}\n")
        
        transform = self.get_dino_augmentation()
        
        # Создаём структуру папок
        for split in ['train', 'val', 'test']:
            (self.output_path / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_path / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # === 1. Копируем val и test без изменений ===
        print("📁 Копирование val/test...")
        for split in ['val', 'test']:
            for sub in ['images', 'labels']:
                src_dir = self.original_path / split / sub
                dst_dir = self.output_path / split / sub
                if src_dir.exists():
                    files = list(src_dir.glob("*"))
                    for f in files:
                        shutil.copy2(f, dst_dir / f.name)
                    if files:
                        print(f"  {split}/{sub}: {len(files)} файлов")
        
        # === 2. Копируем оригиналы train ===
        print(f"\n📁 Копирование оригинальных train ({self.num_original} шт)...")
        dst_img_dir = self.output_path / "train" / "images"
        dst_lbl_dir = self.output_path / "train" / "labels"

        for img_path in tqdm(self.original_train_images, desc="  Копирование оригиналов"):
            # Изображение
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            
            # Лейбл (из плоской структуры: original_path/labels/file.txt)
            lbl_path = self.original_path / "labels" / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)
            else:
                (dst_lbl_dir / f"{img_path.stem}.txt").touch()
        
        # === 3. Генерируем аугментации ===
        if self.num_to_generate <= 0:
            print(f"\n✅ Аугментации не нужны (цель ≤ исходных)")
            self._save_data_yaml()
            self._print_stats()
            return
        
        print(f"\n🎨 Генерация {self.num_to_generate} аугментированных изображений...")
        print(f"   Аугментации: Affine + Brightness + Contrast + ToneCurve + Noise")
        print(f"   Реалистичные для металлургии (без Flip/Rotate/Blur)\n")
        
        generated = 0
        attempts = 0
        max_attempts = self.num_to_generate * 4  # Запас попыток
        efficiency_list = []
        
        pbar = tqdm(total=self.num_to_generate, desc="  Генерация")
        
        while generated < self.num_to_generate and attempts < max_attempts:
            attempts += 1
            
            # Случайный выбор исходного изображения
            src_img = random.choice(self.original_train_images)
            src_lbl = self.original_path / "labels" / f"{src_img.stem}.txt"

            # Загрузка изображения
            image = cv2.imread(str(src_img))
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Чтение аннотаций
            bboxes, class_labels = self.read_yolo_labels(src_lbl)
            
            # Аугментация
            aug_image, aug_bboxes, aug_labels = self.augment_single(
                image_rgb, bboxes, class_labels, transform
            )
            
            # Проверка: если были дефекты, но после аугментации пропали — пропускаем
            if bboxes and not aug_bboxes:
                continue
            
            # Сохранение
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            
            # Имя файла с информацией об аугментации
            new_name = f"{src_img.stem}_aug{generated:06d}"
            
            cv2.imwrite(
                str(dst_img_dir / f"{new_name}.jpg"),
                aug_image_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            
            self.write_yolo_labels(
                dst_lbl_dir / f"{new_name}.txt",
                aug_bboxes,
                aug_labels
            )
            
            generated += 1
            pbar.update(1)
            
            # Метрики эффективности
            if generated % 100 == 0:
                eff = generated / attempts * 100
                efficiency_list.append(eff)
                pbar.set_postfix({
                    'attempts': attempts,
                    'efficiency': f"{eff:.1f}%"
                })
        
        pbar.close()
        
        # Итоговая статистика генерации
        avg_eff = np.mean(efficiency_list) if efficiency_list else 0
        print(f"\n📊 Статистика генерации:")
        print(f"   Сгенерировано : {generated}")
        print(f"   Всего попыток : {attempts}")
        print(f"   Эффективность: {avg_eff:.1f}%")
        
        # Сохраняем конфиг и выводим статистику
        self._save_data_yaml()
        self._print_stats()
    
    def _save_data_yaml(self) -> None:
        """Сохраняет data.yaml для YOLO."""
        data_config = {
            'path': str(self.output_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': self.config.get('nc', 4),
            'names': self.config.get('names', ['defect_1', 'defect_2', 'defect_3', 'defect_4'])
        }
        
        yaml_path = self.output_path / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n📄 Конфиг сохранён: {yaml_path}")
    
    def _print_stats(self) -> None:
        """Выводит итоговую статистику датасета."""
        print(f"\n{'='*60}")
        print(f"📊 ИТОГОВАЯ СТАТИСТИКА ДАТАСЕТА")
        print(f"{'='*60}")
        
        total_images = 0
        total_labels = 0
        
        for split in ['train', 'val', 'test']:
            img_dir = self.output_path / split / "images"
            lbl_dir = self.output_path / split / "labels"
            
            if img_dir.exists():
                images = list(img_dir.glob("*"))
                labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
                
                # Считаем непустые лейблы
                non_empty_labels = sum(1 for l in labels if l.stat().st_size > 0)
                
                print(f"  {split:6s}: {len(images):5d} изображений, {non_empty_labels:5d} с дефектами")
                
                total_images += len(images)
                total_labels += non_empty_labels
        
        print(f"{'='*60}")
        print(f"  Всего : {total_images} изображений, {total_labels} размеченных")
        print(f"{'='*60}\n")
        
        # Рекомендации для обучения
        print(f"💡 Рекомендации для DINO:")
        print(f"   1. Использовать --img 640 (или --img 256)")
        print(f"   2. Локальные аугментации уже в данных")
        print(f"   3. При обучении добавить: mosaic=0.3, mixup=0.1")
        print(f"   4. Learning rate: 0.001 (меньше из-за синтетики)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='🎨 Аугментация синтетических данных (defect-only) для DINO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Базовая аугментация
  python augment_defect_synthetic.py \\
    --original_path /data/synthetic_poisson \\
    --output_path /data/synthetic_augmented

  # Увеличить датасет до 5000
  python augment_defect_synthetic.py \\
    --original_path /data/synthetic_poisson \\
    --output_path /data/synthetic_augmented \\
    --target_size 5000
        """
    )
    
    parser.add_argument(
        '--original_path', type=str, required=True,
        help='Путь к исходному синтетическому датасету (с data.yaml)'
    )
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Путь для сохранения аугментированного датасета'
    )
    parser.add_argument(
        '--target_size', type=int, default=None,
        help='Целевой размер train (если нужно увеличить)'
    )
    
    args = parser.parse_args()
    
    # Создаём и запускаем
    augmenter = SyntheticDataAugmenter(
        original_path=Path(args.original_path),
        output_path=Path(args.output_path),
        target_train_size=args.target_size
    )
    
    augmenter.create_augmented_dataset()
    
    print("✅ Готово!")


if __name__ == "__main__":
    main()