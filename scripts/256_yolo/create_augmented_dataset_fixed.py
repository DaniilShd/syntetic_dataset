#!/usr/bin/env python3
"""
augment_defect_only.py - Лёгкие аугментации поверх defect-only синтетики
Горизонтальный флип + яркость + шум + оригинальные аугментации
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
from typing import List, Tuple


class AugmentedDatasetCreator:
    def __init__(self, original_path: Path, output_path: Path, 
                 target_train_size: int = None):
        self.original_path = Path(original_path)
        self.output_path = Path(output_path)
        
        # Ищем data.yaml или создаём
        self.original_yaml = self.original_path / "data.yaml"
        if self.original_yaml.exists():
            with open(self.original_yaml, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {'nc': 4, 'names': ['defect1', 'defect2', 'defect3', 'defect4']}
        
        # Собираем train изображения
        self.original_train_images = sorted(
            list((self.original_path / "train" / "images").glob("*.jpg")) +
            list((self.original_path / "train" / "images").glob("*.png"))
        )
        self.num_original = len(self.original_train_images)
        
        # Целевой размер
        self.target_train_size = target_train_size or self.num_original
        self.num_to_generate = max(0, self.target_train_size - self.num_original)
        
        print(f"\n📊 Статистика:")
        print(f"   Исходных изображений: {self.num_original}")
        print(f"   Целевой размер: {self.target_train_size}")
        print(f"   Сгенерировать: {self.num_to_generate}")
    
    def get_defect_only_augmentation(self) -> A.Compose:
        """
        Аугментации для defect-only синтетики:
        - ✅ Горизонтальный флип (прокат может идти в любую сторону)
        - ✅ Яркость/контраст (разное освещение)
        - ✅ Лёгкий шум (камера)
        - ✅ Hue/Saturation (оттенки металла)
        - ❌ Без поворотов (геометрия дефекта важна)
        """
        return A.Compose([
            # Горизонтальный флип
            A.HorizontalFlip(p=0.5),
            
            # Яркость и контраст
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.6
            ),
            
            # Тон и насыщенность
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.3
            ),
            
            # Очень лёгкое аффинное
            A.Affine(
                scale=(0.97, 1.03),
                translate_percent=(-0.03, 0.03),
                rotate=(-2, 2),
                p=0.3
            ),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.7,
            min_area=25
            ))
    
    def read_yolo_labels(self, label_path: Path) -> Tuple[List, List]:
        bboxes, class_labels = [], []
        if label_path.exists() and label_path.stat().st_size > 0:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(float(parts[0]))
                        xc = np.clip(float(parts[1]), 0, 1)
                        yc = np.clip(float(parts[2]), 0, 1)
                        w = np.clip(float(parts[3]), 0, 1)
                        h = np.clip(float(parts[4]), 0, 1)
                        if w > 0.005 and h > 0.005:
                            bboxes.append([xc, yc, w, h])
                            class_labels.append(cls)
        return bboxes, class_labels
    
    def write_yolo_labels(self, label_path: Path, bboxes: List, class_labels: List):
        with open(label_path, 'w') as f:
            for bbox, cls in zip(bboxes, class_labels):
                xc = np.clip(bbox[0], 0, 1)
                yc = np.clip(bbox[1], 0, 1)
                w = np.clip(bbox[2], 0, 1)
                h = np.clip(bbox[3], 0, 1)
                if w > 0 and h > 0:
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    
    def augment_image_and_labels(self, image, bboxes, class_labels, transform):
        if not bboxes:
            try:
                augmented = transform(image=image, bboxes=[], class_labels=[])
                return augmented['image'], [], []
            except:
                return image, [], []
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            valid_bboxes, valid_labels = [], []
            for bbox, label in zip(augmented['bboxes'], augmented['class_labels']):
                x, y, w, h = bbox
                if 0 < x < 1 and 0 < y < 1 and 0 < w <= 1 and 0 < h <= 1 and w > 0.005:
                    valid_bboxes.append(bbox)
                    valid_labels.append(label)
            return augmented['image'], valid_bboxes, valid_labels
        except:
            return image, bboxes, class_labels
    
    def create_dataset(self):
        print(f"\n{'='*60}")
        print(f"🎨 АУГМЕНТАЦИЯ DEFECT-ONLY СИНТЕТИКИ")
        print(f"   Flip + Brightness + Noise + Hue")
        print(f"{'='*60}")
        
        transform = self.get_defect_only_augmentation()
        
        # Создаём структуру
        for split in ['train', 'val', 'test']:
            (self.output_path / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_path / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Копируем val и test
        print("\n📁 Копирование val и test...")
        for split in ['val', 'test']:
            for sub in ['images', 'labels']:
                src = self.original_path / split / sub
                dst = self.output_path / split / sub
                if src.exists():
                    for f in src.glob("*"):
                        shutil.copy2(f, dst / f.name)
        
        # Копируем оригиналы train
        print(f"\n📁 Копирование исходных изображений ({self.num_original} шт)...")
        dst_img = self.output_path / "train" / "images"
        dst_lbl = self.output_path / "train" / "labels"
        
        for img_path in tqdm(self.original_train_images, desc="  Копирование"):
            shutil.copy2(img_path, dst_img / img_path.name)
            lbl_path = self.original_path / "train" / "labels" / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl / lbl_path.name)
            else:
                (dst_lbl / f"{img_path.stem}.txt").touch()
        
        # Генерируем аугментации
        if self.num_to_generate > 0:
            print(f"\n🎨 Генерация {self.num_to_generate} аугментированных изображений...")
            
            generated = 0
            attempts = 0
            max_attempts = self.num_to_generate * 5
            
            pbar = tqdm(total=self.num_to_generate, desc="  Генерация")
            
            while generated < self.num_to_generate and attempts < max_attempts:
                attempts += 1
                
                src_img = random.choice(self.original_train_images)
                src_lbl = self.original_path / "train" / "labels" / f"{src_img.stem}.txt"
                
                image = cv2.imread(str(src_img))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                bboxes, class_labels = self.read_yolo_labels(src_lbl)
                aug_image, aug_bboxes, aug_labels = self.augment_image_and_labels(
                    image, bboxes, class_labels, transform)
                
                if not aug_bboxes and bboxes:
                    continue
                
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                new_name = f"{src_img.stem}_aug{generated:06d}"
                
                cv2.imwrite(str(dst_img / f"{new_name}.jpg"), aug_image_bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.write_yolo_labels(dst_lbl / f"{new_name}.txt", aug_bboxes, aug_labels)
                
                generated += 1
                pbar.update(1)
                
                if generated % 500 == 0:
                    pbar.set_postfix({"attempts": attempts,
                                     "efficiency": f"{generated/attempts*100:.1f}%"})
            
            pbar.close()
            print(f"  Сгенерировано: {generated}, попыток: {attempts}")
        
        # data.yaml
        data_config = {
            'path': str(self.output_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': self.config.get('nc', 4),
            'names': self.config.get('names', ['defect1', 'defect2', 'defect3', 'defect4'])
        }
        with open(self.output_path / "data.yaml", 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        # Статистика
        for split in ['train', 'val', 'test']:
            n = len(list((self.output_path / split / "images").glob("*")))
            print(f"  {split}: {n} images")
        
        print(f"\n✅ Готово: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description='Аугментация defect-only синтетики')
    parser.add_argument('--original_path', type=str, required=True,
                       help='Путь к исходному датасету')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Путь для сохранения')
    parser.add_argument('--target_size', type=int, default=None,
                       help='Целевой размер train')
    args = parser.parse_args()
    
    creator = AugmentedDatasetCreator(
        original_path=Path(args.original_path),
        output_path=Path(args.output_path),
        target_train_size=args.target_size
    )
    creator.create_dataset()


if __name__ == "__main__":
    main()