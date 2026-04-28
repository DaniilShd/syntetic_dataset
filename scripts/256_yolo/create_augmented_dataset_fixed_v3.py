#!/usr/bin/env python3
"""
generate_augmented_only.py - Генерация ТОЛЬКО аугментированных изображений
Берёт исходные изображения и создаёт N аугментированных копий
"""

import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import random
from tqdm import tqdm
import yaml
import argparse
from typing import List, Tuple


class AugmentedGenerator:
    def __init__(self, input_path: Path, output_path: Path, num_generate: int):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.num_generate = num_generate
        
        # Ищем изображения
        self.images_dir = self.input_path / "images"
        self.labels_dir = self.input_path / "labels"
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Директория с изображениями не найдена: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Директория с лейблами не найдена: {self.labels_dir}")
        
        # Получаем список исходных изображений
        self.source_images = sorted(
            list(self.images_dir.glob("*.jpg")) + 
            list(self.images_dir.glob("*.png"))
        )
        
        if not self.source_images:
            raise ValueError(f"Нет изображений в {self.images_dir}")
        
        print(f"\n📊 Параметры генерации:")
        print(f"   Исходных изображений: {len(self.source_images)}")
        print(f"   Сгенерировать: {num_generate}")
        print(f"   Всего будет: {len(self.source_images) + num_generate}")
    
    def get_augmentation(self) -> A.Compose:
        """Аугментации для defect-only синтетики"""
        return A.Compose([
            # Горизонтальный флип (прокат может идти в любую сторону)
            A.HorizontalFlip(p=0.5),
            
            # Яркость и контраст (разное освещение)
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.6
            ),
            
            # Тон и насыщенность (оттенки металла)
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.3
            ),
            
            # Очень лёгкое аффинное
            A.Affine(
                scale=(0.98, 1.02),
                translate_percent=(-0.02, 0.02),
                rotate=(-1, 1),
                p=0.2
            ),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.7,
            min_area=25
        ))
    
    def read_yolo_labels(self, label_path: Path) -> Tuple[List, List]:
        """Чтение YOLO лейблов с клиппингом"""
        bboxes, class_labels = [], []
        if label_path.exists() and label_path.stat().st_size > 0:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(float(parts[0]))
                        xc = np.clip(float(parts[1]), 0.002, 0.998)
                        yc = np.clip(float(parts[2]), 0.002, 0.998)
                        w = np.clip(float(parts[3]), 0.002, 0.998)
                        h = np.clip(float(parts[4]), 0.002, 0.998)
                        
                        if w > 0.001 and h > 0.001:
                            bboxes.append([xc, yc, w, h])
                            class_labels.append(cls)
        return bboxes, class_labels
    
    def write_yolo_labels(self, label_path: Path, bboxes: List, class_labels: List):
        """Запись YOLO лейблов"""
        with open(label_path, 'w') as f:
            for bbox, cls in zip(bboxes, class_labels):
                xc = np.clip(bbox[0], 0.0, 1.0)
                yc = np.clip(bbox[1], 0.0, 1.0)
                w = np.clip(bbox[2], 0.0, 1.0)
                h = np.clip(bbox[3], 0.0, 1.0)
                if w > 0.001 and h > 0.001:
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    
    def augment_image_and_labels(self, image, bboxes, class_labels, transform):
        """Применение аугментации"""
        if not bboxes:
            try:
                augmented = transform(image=image, bboxes=[], class_labels=[])
                return augmented['image'], [], []
            except:
                return image, [], []
        
        # Клиппим bbox'ы
        safe_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x = np.clip(x, 0.002, 0.998)
            y = np.clip(y, 0.002, 0.998)
            w = np.clip(w, 0.002, 0.998)
            h = np.clip(h, 0.002, 0.998)
            safe_bboxes.append([x, y, w, h])
        
        try:
            augmented = transform(image=image, bboxes=safe_bboxes, class_labels=class_labels)
            valid_bboxes, valid_labels = [], []
            
            for bbox, label in zip(augmented['bboxes'], augmented['class_labels']):
                x, y, w, h = bbox
                x = np.clip(x, 0.0, 1.0)
                y = np.clip(y, 0.0, 1.0)
                w = np.clip(w, 0.001, 1.0)
                h = np.clip(h, 0.001, 1.0)
                
                if w > 0.002 and h > 0.002:
                    valid_bboxes.append([x, y, w, h])
                    valid_labels.append(label)
            
            return augmented['image'], valid_bboxes, valid_labels
        except:
            return image, bboxes, class_labels
    
    def generate(self):
        print(f"\n{'='*60}")
        print(f"🎨 ГЕНЕРАЦИЯ АУГМЕНТИРОВАННЫХ ИЗОБРАЖЕНИЙ")
        print(f"   Только аугментации, без оригиналов")
        print(f"{'='*60}")
        
        transform = self.get_augmentation()
        
        # Создаём выходные директории
        output_images_dir = self.output_path / "images"
        output_labels_dir = self.output_path / "labels"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎨 Генерация {self.num_generate} аугментированных изображений...")
        
        generated = 0
        attempts = 0
        max_attempts = self.num_generate * 10
        
        pbar = tqdm(total=self.num_generate, desc="  Генерация")
        
        while generated < self.num_generate and attempts < max_attempts:
            attempts += 1
            
            # Случайно выбираем исходное изображение
            src_img = random.choice(self.source_images)
            src_lbl = self.labels_dir / f"{src_img.stem}.txt"
            
            # Загружаем изображение
            image = cv2.imread(str(src_img))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Читаем лейблы
            bboxes, class_labels = self.read_yolo_labels(src_lbl)
            
            # Применяем аугментацию
            aug_image, aug_bboxes, aug_labels = self.augment_image_and_labels(
                image, bboxes, class_labels, transform)
            
            # Пропускаем если потеряли все bbox'ы
            if not aug_bboxes and bboxes:
                continue
            
            # Сохраняем результат
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            new_name = f"aug_{generated:06d}_{src_img.stem}"
            
            cv2.imwrite(str(output_images_dir / f"{new_name}.jpg"), aug_image_bgr,
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.write_yolo_labels(output_labels_dir / f"{new_name}.txt", aug_bboxes, aug_labels)
            
            generated += 1
            pbar.update(1)
            
            if generated % 500 == 0:
                efficiency = (generated / attempts * 100) if attempts > 0 else 0
                pbar.set_postfix({
                    "attempts": attempts,
                    "eff": f"{efficiency:.1f}%"
                })
        
        pbar.close()
        
        # Создаём data.yaml
        data_config = {
            'path': str(self.output_path.absolute()),
            'train': 'images',
            'val': 'images',
            'nc': 4,
            'names': ['defect1', 'defect2', 'defect3', 'defect4']
        }
        
        with open(self.output_path / "data.yaml", 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n{'='*60}")
        print(f"📊 Результаты генерации:")
        print(f"   Сгенерировано: {generated} изображений")
        print(f"   Попыток: {attempts}")
        if attempts > 0:
            print(f"   Эффективность: {generated/attempts*100:.1f}%")
        print(f"\n✅ Готово: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description='Генерация аугментированных изображений')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Путь к исходным данным (с images/ и labels/)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Путь для сохранения аугментированных изображений')
    parser.add_argument('--num_generate', type=int, required=True,
                       help='Количество аугментированных изображений для генерации')
    
    args = parser.parse_args()
    
    try:
        generator = AugmentedGenerator(
            input_path=Path(args.input_path),
            output_path=Path(args.output_path),
            num_generate=args.num_generate
        )
        generator.generate()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()