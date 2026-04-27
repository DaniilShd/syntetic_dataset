import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# Укажите КОНКРЕТНЫЙ путь для сохранения
output_dir = Path("./debug_rle_check")
output_dir.mkdir(exist_ok=True)

# Загрузите CSV
df = pd.read_csv("data/256_yolo/balanced_defect_patches_rle/train/train_rle.csv")

# Первое изображение с НЕПУСТЫМ RLE
for idx, row in df.iterrows():
    rle = str(row['EncodedPixels'])
    if rle not in ['nan', '', 'None']:
        break

image_id = row['ImageId']
rle = row['EncodedPixels']

print(f"Image: {image_id}")
print(f"RLE pixels: {len(rle.split())}")

# Декодируем
numbers = list(map(int, rle.split()))
starts = np.array(numbers[0::2]) - 1
lengths = np.array(numbers[1::2])

flat = np.zeros(256*256, dtype=np.uint8)
for s, l in zip(starts, lengths):
    if s < len(flat):
        flat[s:min(s+l, len(flat))] = 1

# Оба варианта
mask_row = flat.reshape(256, 256)       # row-major
mask_col = flat.reshape(256, 256).T     # column-major (Severstal)

print(f"Row-major pixels: {mask_row.sum()}")
print(f"Col-major pixels: {mask_col.sum()}")

# Ищем изображение
import glob
candidates = glob.glob(f"data/**/{image_id}", recursive=True)

if candidates:
    img_path = candidates[0]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    
    # Row-major маска (красная)
    overlay_row = img.copy()
    overlay_row[mask_row > 0] = [0, 0, 255]
    cv2.imwrite(str(output_dir / "1_row_major_red.png"), overlay_row)
    
    # Column-major маска (зелёная)
    overlay_col = img.copy()
    overlay_col[mask_col > 0] = [0, 255, 0]
    cv2.imwrite(str(output_dir / "2_col_major_green.png"), overlay_col)
    
    # Оригинал
    cv2.imwrite(str(output_dir / "0_original.png"), img)
    
    # Совмещённая
    overlay_both = img.copy()
    overlay_both[mask_row > 0] = [0, 0, 255]
    overlay_both[mask_col > 0] = [0, 255, 0]
    cv2.imwrite(str(output_dir / "3_both.png"), overlay_both)
    
    print(f"\n✅ Сохранено в: {output_dir.absolute()}")
    print(f"  0_original.png — оригинал")
    print(f"  1_row_major_red.png — row-major (КРАСНЫЙ)")
    print(f"  2_col_major_green.png — column-major (ЗЕЛЁНЫЙ)")
    print(f"  3_both.png — обе маски вместе")
    print(f"\nКакая маска попала на дефекты?")
else:
    print(f"❌ Изображение {image_id} не найдено")
    print("Ищу в: data/256_yolo/balanced_defect_patches_rle/train/images/")
    print("Или укажите путь вручную")