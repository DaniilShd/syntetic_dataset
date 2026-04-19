"""
utils.py - Вспомогательные функции для Docker-окружения
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import cv2
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Фиксация случайных seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Для детерминированности
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.info(f"Random seed set to {seed}")


def check_directories(config) -> Dict[str, Path]:
    """Проверка существования входных директорий"""
    dirs = {
        "clean": Path(config.clean_textures_dir),
        "defects": Path(config.defects_dir),
        "output": Path(config.output_dir),
        "backgrounds": Path(config.backgrounds_dir),
        "final": Path(config.final_dataset_dir),
    }
    
    for name, path in dirs.items():
        if name in ["clean", "defects"]:
            if not path.exists():
                raise FileNotFoundError(f"Директория {name} не найдена: {path}")
            logger.info(f"✅ {name}: {path} (найдено)")
        else:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 {name}: {path} (создано)")
    
    return dirs


def load_images_from_dir(
    directory: str,
    extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]
) -> List[Path]:
    """Загрузка всех изображений из директории"""
    dir_path = Path(directory)
    images = []
    
    for ext in extensions:
        images.extend(dir_path.glob(f"*{ext}"))
        images.extend(dir_path.glob(f"*{ext.upper()}"))
    
    logger.info(f"Загружено {len(images)} изображений из {directory}")
    return sorted(images)


def load_defects_with_masks(defects_dir: str) -> List[Dict]:
    """Загрузка дефектов с соответствующими масками"""
    defects = []
    dir_path = Path(defects_dir)
    
    for img_path in dir_path.iterdir():
        if not img_path.is_file():
            continue
        
        ext = img_path.suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        
        mask_candidates = [
            dir_path / f"{img_path.stem}_mask{ext}",
            dir_path / f"{img_path.stem}_mask.png",
            dir_path / f"{img_path.stem}_mask.jpg",
            dir_path / f"mask_{img_path.stem}{ext}",
            dir_path / f"mask_{img_path.name}"
        ]
        
        mask_path = None
        for candidate in mask_candidates:
            if candidate.exists():
                mask_path = candidate
                break
        
        if mask_path:
            defects.append({
                "name": img_path.stem,
                "image": str(img_path),
                "mask": str(mask_path)
            })
    
    logger.info(f"Загружено {len(defects)} дефектов с масками")
    return defects


def save_json(data: Dict, path: str, indent: int = 2):
    """Сохранение данных в JSON"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.info(f"JSON сохранен: {path}")


def load_json(path: str) -> Dict:
    """Загрузка данных из JSON"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_gpu_info() -> str:
    """Получение информации о GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"{gpu_name} ({memory:.1f} GB)"
        return "CUDA not available"
    except:
        return "Unknown"


def print_system_info():
    """Вывод информации о системе"""
    logger.info("=" * 50)
    logger.info("SYSTEM INFO")
    logger.info("=" * 50)
    logger.info(f"GPU: {get_gpu_info()}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"HF Cache: {os.environ.get('HF_HOME', 'not set')}")
    logger.info("=" * 50)