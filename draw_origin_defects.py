#!/usr/bin/env python3
"""
visualize_rle_bbox.py - Отрисовка RLE bbox из train.csv на оригинальных изображениях
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

CLASS_NAMES = {1: 'defect_1', 2: 'defect_2', 3: 'defect_3', 4: 'defect_4'}
COLORS = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), 4: (255, 255, 0)}

def rle_to_mask(rle, h=256, w=1600):
    if pd.isna(rle) or str(rle).strip() == '':
        return np.zeros((h, w), dtype=np.uint8)
    nums = list(map(int, str(rle).split()))
    starts, lengths = np.array(nums[0::2]) - 1, np.array(nums[1::2])
    flat = np.zeros(h * w, dtype=np.uint8)
    for s, l in zip(starts, lengths):
        flat[s:s+l] = 1
    return flat.reshape(w, h).T

def mask_to_bbox(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/severstal/train_images")
    parser.add_argument("--csv", default="data/severstal/train.csv")
    parser.add_argument("--output", default="data/severstal/visualized")
    parser.add_argument("--samples", type=int, default=15000)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.csv)
    image_ids = df['ImageId'].unique()[:args.samples]
    
    for img_id in tqdm(image_ids):
        img = cv2.imread(str(Path(args.images) / img_id))
        if img is None:
            continue
            
        img_data = df[df['ImageId'] == img_id]
        for _, row in img_data.iterrows():
            mask = rle_to_mask(row['EncodedPixels'])
            for x, y, w, h in mask_to_bbox(mask):
                color = COLORS[row['ClassId']]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, CLASS_NAMES[row['ClassId']], (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(str(output_dir / img_id), img)
    
    print(f"✅ Визуализировано {len(image_ids)} изображений в {output_dir}")

if __name__ == "__main__":
    main()