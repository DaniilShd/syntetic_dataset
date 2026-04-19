import cv2
from pathlib import Path

def draw_boxes(input_dir, output_dir):
    """Отрисовка bbox на изображениях"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Цвета для классов (BGR)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    class_names = ['defect_1', 'defect_2', 'defect_3', 'defect_4']
    
    for img_file in (input_path / "images").glob("*.png"):
        # Загружаем изображение
        img = cv2.imread(str(img_file))
        h, w = img.shape[:2]
        
        # Ищем соответствующий label файл
        label_file = input_path / "labels" / f"{img_file.stem}.txt"
        if not label_file.exists():
            continue
            
        # Рисуем боксы
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x, y, bw, bh = map(float, parts[1:5])
                    
                    # YOLO -> пиксели
                    x1 = int((x - bw/2) * w)
                    y1 = int((y - bh/2) * h)
                    x2 = int((x + bw/2) * w)
                    y2 = int((y + bh/2) * h)
                    
                    color = colors[cls_id % len(colors)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, class_names[cls_id], (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(str(output_path / img_file.name), img)
    
    print(f"✅ Сохранено в {output_path}")

# Использование
draw_boxes("data/256_yolo/balanced_defect_patches", "data/256_yolo/visualized")