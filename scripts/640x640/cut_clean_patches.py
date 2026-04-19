import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

def rle_to_mask(rle_string, height=256, width=1600):
    """Правильное декодирование RLE для Severstal (column-major order)"""
    if pd.isna(rle_string) or rle_string == '':
        return np.zeros((height, width), dtype=np.uint8)
    
    numbers = list(map(int, rle_string.split()))
    starts = np.array(numbers[0::2]) - 1
    lengths = np.array(numbers[1::2])
    
    flat_mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat_mask[start:start + length] = 1
    
    return flat_mask.reshape(width, height).T

def has_black_pixels(patch, black_threshold=10):
    """Проверяет, есть ли в патче черные пиксели"""
    is_black = np.all(patch < 30, axis=2)
    black_count = np.sum(is_black)
    return black_count > black_threshold

def resize_patch(patch, target_size=448):
    """Ресайзит патч до target_size x target_size"""
    return cv2.resize(patch, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

def find_clean_patches(
    train_images_dir="data/severstal/train_images",
    train_csv_path="data/severstal/train.csv",
    output_dir="data/severstal/clean_patches",
    patch_size=256,
    stride=256,
    min_clean_ratio=1.0,
    reject_black_patches=True,
    black_threshold=10,
    resize_to=448  # Новый параметр для ресайза
):
    """Находит и сохраняет чистые патчи (без дефектов и без черного цвета) с ресайзом"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Загружаем разметку
    print("Загрузка разметки...")
    df = pd.read_csv(train_csv_path)
    
    # Создаем маски для всех изображений
    print("Создание масок...")
    masks = {}
    for img_id, group in tqdm(df.groupby('ImageId')):
        combined = np.zeros((256, 1600), dtype=np.uint8)
        for _, row in group.iterrows():
            combined = np.maximum(combined, rle_to_mask(row['EncodedPixels']))
        masks[img_id] = combined
    
    # Все изображения в папке
    image_files = list(Path(train_images_dir).glob("*.jpg"))
    print(f"Всего изображений: {len(image_files)}")
    
    # Позиции патчей
    x_positions = range(0, 1600 - patch_size + 1, stride)
    y_positions = range(0, 256 - patch_size + 1, stride)
    total_patches_per_image = len(x_positions) * len(y_positions)
    print(f"Патчей на изображение: {total_patches_per_image}")
    print(f"Ресайз патчей: {patch_size}x{patch_size} -> {resize_to}x{resize_to}")
    
    if reject_black_patches:
        print(f"Отбраковка патчей с черным цветом (> {black_threshold} черных пикселей)")
    
    # Поиск чистых патчей
    clean_patches = []
    rejected_black = 0
    rejected_defects = 0
    
    for img_path in tqdm(image_files, desc="Поиск чистых патчей"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = masks.get(img_path.name, np.zeros((256, 1600), dtype=np.uint8))
        
        for y in y_positions:
            for x in x_positions:
                patch_img = img_rgb[y:y+patch_size, x:x+patch_size]
                patch_mask = mask[y:y+patch_size, x:x+patch_size]
                
                # Проверка на дефекты
                defect_pixels = np.sum(patch_mask)
                clean_ratio = 1.0 - (defect_pixels / (patch_size * patch_size))
                
                if clean_ratio < min_clean_ratio:
                    rejected_defects += 1
                    continue
                
                # Проверка на черный цвет
                if reject_black_patches and has_black_pixels(patch_img, black_threshold):
                    rejected_black += 1
                    continue
                
                # Ресайз патча
                if resize_to and resize_to != patch_size:
                    patch_img = resize_patch(patch_img, resize_to)
                
                # Сохраняем патч
                name = f"{img_path.stem}_x{x}_y{y}_{patch_size}to{resize_to}.png"
                cv2.imwrite(str(Path(output_dir) / name), 
                           cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))
                clean_patches.append({
                    'image': img_path.name,
                    'x': x, 'y': y,
                    'original_size': patch_size,
                    'resized_size': resize_to,
                    'clean_ratio': clean_ratio
                })
    
    # Статистика
    print(f"\n{'='*50}")
    print(f"СТАТИСТИКА:")
    print(f"  Найдено чистых патчей: {len(clean_patches)}")
    print(f"  Отбраковано из-за дефектов: {rejected_defects}")
    if reject_black_patches:
        print(f"  Отбраковано из-за черного цвета: {rejected_black}")
    print(f"  Полностью чистых (без дефектов): {sum(1 for p in clean_patches if p['clean_ratio'] == 1.0)}")
    print(f"  Размер сохраненных патчей: {resize_to}x{resize_to}")
    
    if clean_patches:
        pd.DataFrame(clean_patches).to_csv(Path(output_dir) / "patches_info.csv", index=False)
        print(f"  Сохранено в: {output_dir}")
    else:
        print(f"  Нет подходящих патчей!")
    
    return clean_patches

def visualize_resized_patches(patch_dir="data/severstal/clean_patches", num_examples=5):
    """Визуализирует ресайзнутые патчи для проверки"""
    import matplotlib.pyplot as plt
    
    patch_path = Path(patch_dir)
    patch_files = list(patch_path.glob("*.png"))
    
    if not patch_files:
        print("Нет патчей для визуализации")
        return
    
    # Берем случайные патчи
    import random
    random.seed(42)
    selected = random.sample(patch_files, min(num_examples, len(patch_files)))
    
    fig, axes = plt.subplots(1, len(selected), figsize=(3*len(selected), 3))
    if len(selected) == 1:
        axes = [axes]
    
    for i, patch_file in enumerate(selected):
        img = cv2.imread(str(patch_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(f"{img.shape[0]}x{img.shape[1]}", fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle(f"Ресайзнутые патчи (448x448)", fontsize=14)
    plt.tight_layout()
    plt.savefig('resized_patches_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Примеры сохранены в 'resized_patches_example.png'")

if __name__ == "__main__":
    # Поиск чистых патчей с ресайзом до 1024х1024
    patches = find_clean_patches(
        train_images_dir="data/severstal/train_images",
        train_csv_path="data/severstal/train.csv",
        output_dir="data/640x640/clean_patches",
        patch_size=256,          # Исходный размер патча
        stride=256,              # Без перекрытия
        min_clean_ratio=1.0,     # Только без дефектов
        reject_black_patches=True,
        black_threshold=10,
        resize_to=1024       
    )
    
    # Визуализируем результаты
    if patches:
        visualize_resized_patches("data/severstal/clean_patches")
        
        print(f"\n✅ Готово!")
        print(f"✅ Найдено {len(patches)} чистых патчей размером 1024x1024")
        print(f"✅ Сохранены в: data/severstal/clean_patches")