import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

def rle_decode(mask_rle, shape=(256, 1600)):
    """
    Правильная декодировка RLE для формата Severstal.
    RLE использует column-major order (по столбцам).
    
    :param mask_rle: RLE строка (например, '1 3 10 5')
    :param shape: Tuple (height, width) изображения
    :return: NumPy array (H, W) с значениями 0/1
    """
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)
    
    # Парсим RLE
    s = mask_rle.split()
    starts = np.asarray(s[0:][::2], dtype=int)
    lengths = np.asarray(s[1:][::2], dtype=int)
    
    # Конвертируем в zero-based индексы
    starts -= 1
    
    # Создаем плоскую маску
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    # Заполняем маску
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    
    # Reshape и transpose (важно! RLE использует column-major)
    return mask.reshape(shape[::-1]).T

def visualize_defects_correct(
    train_images_dir="data/severstal/train_images",
    train_csv_path="data/severstal/train.csv",
    output_dir="data/severstal/defects_visualization_correct",
    num_examples=20,
    draw_bbox=True,
    contour_color=(0, 255, 0),
    contour_thickness=2
):
    """
    Правильная визуализация дефектов с корректным RLE декодированием.
    """
    
    # Создаем выходные папки
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Читаем CSV
    print(f"Чтение разметки из {train_csv_path}...")
    df = pd.read_csv(train_csv_path)
    
    print(f"Колонки: {list(df.columns)}")
    print(f"Всего строк: {len(df)}")
    
    # Группируем по изображениям
    print("Группировка дефектов...")
    image_defects = {}
    
    for image_id, group in tqdm(df.groupby('ImageId'), desc="Обработка"):
        defects = []
        for _, row in group.iterrows():
            # Правильное декодирование RLE
            mask = rle_decode(row['EncodedPixels'], shape=(256, 1600))
            class_id = row['ClassId']
            defects.append({
                'mask': mask,
                'class_id': class_id
            })
        image_defects[image_id] = defects
    
    print(f"Найдено изображений с дефектами: {len(image_defects)}")
    
    # Выбираем изображения для визуализации
    image_ids = list(image_defects.keys())
    
    # Берем случайные или первые N
    import random
    random.seed(42)
    selected_ids = random.sample(image_ids, min(num_examples, len(image_ids)))
    
    # Создаем подпапки
    contours_dir = Path(output_dir) / "contours"
    masks_dir = Path(output_dir) / "masks"
    overlay_dir = Path(output_dir) / "overlay"
    bbox_dir = Path(output_dir) / "bbox"
    
    contours_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)
    bbox_dir.mkdir(exist_ok=True)
    
    # Визуализация
    print("\nВизуализация дефектов...")
    stats = []
    
    for image_id in tqdm(selected_ids, desc="Обработка"):
        # Загружаем изображение
        img_path = Path(train_images_dir) / image_id
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"  Ошибка: не удалось загрузить {image_id}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        defects = image_defects[image_id]
        
        # Создаем копии для разных типов визуализации
        img_contours = img_rgb.copy()
        img_bbox = img_rgb.copy()
        img_overlay = img_rgb.copy()
        
        combined_mask = np.zeros((256, 1600), dtype=np.uint8)
        
        for defect in defects:
            mask = defect['mask']
            class_id = defect['class_id']
            combined_mask = np.maximum(combined_mask, mask)
            
            # Находим контуры
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Рисуем контур
                cv2.drawContours(img_contours, [contour], -1, contour_color, contour_thickness)
                
                # Рисуем bounding box
                if draw_bbox:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(img_bbox, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Подпись класса
                    label = f"Class {class_id}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img_bbox, (x, y - label_h - 5), (x + label_w + 5, y), (0, 0, 0), -1)
                    cv2.putText(img_bbox, label, (x + 2, y - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Для overlay - полупрозрачная заливка
                color = (0, 255, 0) if class_id == 1 else (255, 0, 0) if class_id == 2 else (0, 0, 255)
                mask_colored = np.zeros_like(img_overlay)
                mask_colored[mask == 1] = color
                img_overlay = cv2.addWeighted(img_overlay, 0.7, mask_colored, 0.3, 0)
                
                # Центр масс для подписи на контурах
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Подпись на контурах
                    cv2.putText(img_contours, str(class_id), (cX, cY),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Сохраняем маску
        mask_path = masks_dir / f"{image_id.replace('.jpg', '')}_mask.png"
        cv2.imwrite(str(mask_path), combined_mask * 255)
        
        # Сохраняем контуры
        contours_path = contours_dir / f"{image_id.replace('.jpg', '')}_contours.jpg"
        cv2.imwrite(str(contours_path), cv2.cvtColor(img_contours, cv2.COLOR_RGB2BGR))
        
        # Сохраняем bounding boxes
        bbox_path = bbox_dir / f"{image_id.replace('.jpg', '')}_bbox.jpg"
        cv2.imwrite(str(bbox_path), cv2.cvtColor(img_bbox, cv2.COLOR_RGB2BGR))
        
        # Сохраняем overlay
        overlay_path = overlay_dir / f"{image_id.replace('.jpg', '')}_overlay.jpg"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))
        
        # Статистика
        stats.append({
            'image_id': image_id,
            'num_defects': len(defects),
            'classes': [d['class_id'] for d in defects],
            'mask_path': str(mask_path),
            'contours_path': str(contours_path),
            'bbox_path': str(bbox_path),
            'overlay_path': str(overlay_path)
        })
    
    # Сохраняем статистику
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(Path(output_dir) / "visualization_stats.csv", index=False)
    
    # Выводим статистику
    print("\n" + "="*60)
    print("ИТОГОВАЯ СТАТИСТИКА:")
    print("="*60)
    print(f"  Обработано изображений: {len(stats)}")
    print(f"  Всего дефектов: {stats_df['num_defects'].sum()}")
    print(f"  Среднее дефектов на изображение: {stats_df['num_defects'].mean():.1f}")
    
    # Статистика по классам
    all_classes = []
    for classes in stats_df['classes']:
        all_classes.extend(classes)
    
    class_counts = pd.Series(all_classes).value_counts().sort_index()
    print(f"\n  Распределение по классам:")
    for class_id, count in class_counts.items():
        print(f"    Class {class_id}: {count} дефектов")
    
    return stats_df

def create_side_by_side_comparison(
    train_images_dir="data/severstal/train_images",
    train_csv_path="data/severstal/train.csv",
    output_dir="data/severstal/defects_comparison",
    num_examples=10
):
    """
    Создает сравнение: оригинал vs маска vs контуры для нескольких изображений.
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Читаем CSV и декодируем
    df = pd.read_csv(train_csv_path)
    
    # Выбираем случайные изображения
    import random
    random.seed(42)
    image_ids = random.sample(df['ImageId'].unique().tolist(), min(num_examples, len(df['ImageId'].unique())))
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5*num_examples))
    
    for idx, image_id in enumerate(image_ids):
        # Загружаем изображение
        img_path = Path(train_images_dir) / image_id
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Получаем все дефекты для этого изображения
        defects_data = df[df['ImageId'] == image_id]
        combined_mask = np.zeros((256, 1600), dtype=np.uint8)
        
        for _, row in defects_data.iterrows():
            mask = rle_decode(row['EncodedPixels'], shape=(256, 1600))
            combined_mask = np.maximum(combined_mask, mask)
        
        # Оригинал
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f"Оригинал: {image_id}", fontsize=10)
        axes[idx, 0].axis('off')
        
        # Маска
        axes[idx, 1].imshow(combined_mask, cmap='gray', aspect='auto')
        axes[idx, 1].set_title(f"Маска дефектов ({len(defects_data)} дефектов)", fontsize=10)
        axes[idx, 1].axis('off')
        
        # Наложение
        overlay = img.copy()
        mask_colored = np.zeros_like(img)
        mask_colored[combined_mask == 1] = [0, 255, 0]  # Зеленый
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title("Наложение маски", fontsize=10)
        axes[idx, 2].axis('off')
    
    plt.suptitle("Визуализация дефектов (правильное RLE декодирование)", fontsize=16)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Сравнение сохранено в: {Path(output_dir) / 'comparison.png'}")

def test_rle_decoding(train_csv_path="data/severstal/train.csv", num_tests=5):
    """
    Тестирует правильность декодирования RLE на нескольких примерах.
    """
    df = pd.read_csv(train_csv_path)
    
    print("="*60)
    print("ТЕСТИРОВАНИЕ RLE ДЕКОДИРОВАНИЯ")
    print("="*60)
    
    # Берем несколько случайных дефектов
    test_samples = df.sample(min(num_tests, len(df)))
    
    for idx, row in test_samples.iterrows():
        image_id = row['ImageId']
        class_id = row['ClassId']
        rle = row['EncodedPixels']
        
        # Декодируем
        mask = rle_decode(rle, shape=(256, 1600))
        
        # Проверяем
        print(f"\nИзображение: {image_id}, Class {class_id}")
        print(f"  RLE: {rle[:100]}..." if len(rle) > 100 else f"  RLE: {rle}")
        print(f"  Маска: {mask.sum()} пикселей ({(mask.sum()/(256*1600))*100:.2f}% изображения)")
        print(f"  Форма маски: {mask.shape}")
        print(f"  Уникальные значения: {np.unique(mask)}")
        
        # Проверяем, что дефект не на всю ширину
        row_sum = mask.sum(axis=1)  # Сумма по столбцам для каждой строки
        max_row_defects = row_sum.max()
        print(f"  Максимум дефектов в строке: {max_row_defects} пикселей")
        
        if max_row_defects == 1600:
            print("  ⚠️ ВНИМАНИЕ: Дефект на всю ширину! Возможно проблема с декодированием.")
        else:
            print("  ✅ Дефект выглядит корректно")

if __name__ == "__main__":
    # Пути
    TRAIN_IMAGES_DIR = "data/severstal/train_images"
    TRAIN_CSV_PATH = "data/severstal/train.csv"
    OUTPUT_DIR = "data/severstal/defects_visualization_correct"
    
    # Сначала тестируем декодирование
    test_rle_decoding(TRAIN_CSV_PATH, num_tests=5)
    
    # Создаем сравнение оригинал vs маска
    create_side_by_side_comparison(
        train_images_dir=TRAIN_IMAGES_DIR,
        train_csv_path=TRAIN_CSV_PATH,
        output_dir=OUTPUT_DIR,
        num_examples=5
    )
    
    # Визуализируем дефекты
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАЦИЯ ДЕФЕКТОВ (ПРАВИЛЬНОЕ RLE)")
    print("="*60)
    
    stats = visualize_defects_correct(
        train_images_dir=TRAIN_IMAGES_DIR,
        train_csv_path=TRAIN_CSV_PATH,
        output_dir=OUTPUT_DIR,
        num_examples=20000,  # Показываем 20 примеров
        draw_bbox=True,
        contour_color=(0, 255, 0),
        contour_thickness=2
    )
    
    print(f"\n✅ Все результаты сохранены в: {OUTPUT_DIR}")
    print("  - contours/  - изображения с контурами дефектов")
    print("  - masks/     - бинарные маски дефектов")
    print("  - overlay/   - полупрозрачное наложение")
    print("  - bbox/      - bounding boxes с подписями классов")