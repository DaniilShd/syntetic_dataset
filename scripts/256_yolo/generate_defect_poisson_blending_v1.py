#!/usr/bin/env python3
"""
generate_defect_poisson_blending.py
Использует генерацию дефектов из defect_only.py + Poisson blending для бесшовной вставки.
RLE интерпретируется с учётом сдвига патча (как в defect_only.py).
"""

import sys, re, torch, cv2, random, argparse, logging
import pandas as pd, numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, AutoencoderKL

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Конфигурация для генерации дефектов."""
    sd_strength_min: float = 0.08
    sd_strength_max: float = 0.15
    sd_steps: int = 30
    sd_guidance: float = 2.0
    high_freq_alpha: float = 0.40

# ================= RLE С УЧЁТОМ СДВИГА ПАТЧА (из defect_only.py) =================

def parse_patch_offset(filename: str) -> Tuple[int, int]:
    """Извлекает смещение и ширину патча из имени файла."""
    match = re.search(r'_x(\d+)_w(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 256


def rle_to_patch_mask(rle_string: str, patch_filename: str,
                      original_width: int = 1600, original_height: int = 256) -> np.ndarray:
    """
    Преобразует RLE-строку в маску для патча.
    Работает как в defect_only.py - RLE от полного изображения 1600×256,
    маска вырезается с учётом offset_x из имени файла.
    """
    if pd.isna(rle_string) or str(rle_string).strip() in ['', 'nan']:
        return np.zeros((256, 256), dtype=np.uint8)
    
    offset_x, _ = parse_patch_offset(patch_filename)
    
    numbers = list(map(int, str(rle_string).split()))
    starts = np.array(numbers[0::2]) - 1
    lengths = np.array(numbers[1::2])
    
    # Создаём маску для полного изображения 1600×256
    full_mask = np.zeros(original_width * original_height, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        if start < len(full_mask):
            full_mask[start:min(start + length, len(full_mask))] = 1
    
    # .reshape(1600, 256).T -> (256, 1600) - высота × ширина
    full_mask_2d = full_mask.reshape(original_width, original_height).T
    
    # Вырезаем патч: все строки (256), колонки от offset_x до offset_x + 256
    return full_mask_2d[:, offset_x:offset_x + 256]


def rle_to_bboxes(rle: str, fname: str, cls: int) -> List[Dict]:
    """Преобразует RLE в список bounding boxes с дополнительной информацией."""
    mask = rle_to_patch_mask(rle, fname)
    if mask.sum() == 0:
        return []
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 16 or w < 4 or h < 4:
            continue
        
        cm = (labels == i).astype(np.uint8)
        # Расширяем границу для блендинга
        boundary = cv2.dilate(cm, np.ones((3,3), np.uint8), iterations=3) - cm
        
        bboxes.append({
            'class': cls,
            'x_center': (x + w/2) / 256,
            'y_center': (y + h/2) / 256,
            'width': w / 256,
            'height': h / 256,
            'component_mask': cm,
            'boundary_mask': boundary,
            'x': x, 'y': y, 'w': w, 'h': h
        })
    return bboxes


# ================= Спектральный контроль (из defect_only.py) =================

def match_spectrum(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Замена амплитудного спектра source на спектр target."""
    src_f = np.fft.fft2(source.astype(np.float32), axes=(0, 1))
    tgt_f = np.fft.fft2(target.astype(np.float32), axes=(0, 1))
    result = np.fft.ifft2(np.abs(tgt_f) * np.exp(1j * np.angle(src_f)), axes=(0, 1)).real
    return np.clip(result, 0, 255).astype(np.uint8)


def inject_high_freq(source: np.ndarray, target: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Перенос высокочастотной составляющей с target на source."""
    blur = cv2.GaussianBlur(target.astype(np.float32), (0, 0), sigmaX=3)
    result = source.astype(np.float32) + alpha * (target.astype(np.float32) - blur)
    return np.clip(result, 0, 255).astype(np.uint8)


# ================= POISSON BLENDING =================

def poisson_blend_color(bg: np.ndarray, defect: np.ndarray, mask: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Применяет Poisson blending поканально."""
    result = np.zeros_like(bg)
    for c in range(3):
        result[:,:,c] = _blend_channel(bg[:,:,c].astype(np.float64), defect[:,:,c].astype(np.float64), mask, boundary)
    return result


def _blend_channel(bg: np.ndarray, defect: np.ndarray, mask: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Решение уравнения Пуассона для одного канала."""
    if mask.sum() == 0:
        return bg
        
    inner = (mask > 0) & (boundary == 0)
    idx = np.where(inner.ravel())[0]
    if len(idx) == 0:
        return bg
    
    h, w = mask.shape
    gy, gx = np.gradient(defect)
    div_v = np.gradient(gy, axis=0) + np.gradient(gx, axis=1)
    
    m = {v: i for i, v in enumerate(idx)}
    A = lil_matrix((len(idx), len(idx)))
    b = np.zeros(len(idx))
    
    for i, flat in enumerate(idx):
        y, x = divmod(flat, w)
        A[i,i] = -4
        b[i] = div_v.ravel()[flat]
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w:
                nf = ny*w + nx
                if inner.ravel()[nf]:
                    A[i, m[nf]] = 1
                else:
                    b[i] -= bg.ravel()[nf]
    try:
        x = spsolve(csr_matrix(A), b)
    except Exception as e:
        logger.warning(f"Не удалось решить уравнение Пуассона: {e}. Возвращен исходный фон.")
        return bg
    
    res = bg.copy()
    res.ravel()[idx] = np.clip(x, 0, 255)
    res.ravel()[(boundary > 0).ravel()] = bg.ravel()[(boundary > 0).ravel()]
    return np.clip(res, 0, 255).astype(np.uint8)


# ================= SD Генерация (из defect_only.py) =================

class SDGenerator:
    """Класс для генерации изображений с помощью Stable Diffusion."""
    def __init__(self):
        try:
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
        except Exception:
            logger.warning("Не удалось загрузить sd-vae-ft-mse. Используется стандартный VAE.")
            vae = None
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to("cuda")
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        if hasattr(self.pipe.vae, 'enable_tiling'):
            self.pipe.vae.enable_tiling()
        self.pipe.enable_attention_slicing()
    
    @torch.no_grad()
    def generate_defect_region(self, crop: Image.Image) -> np.ndarray:
        """Генерирует дефект в заданной области (как в defect_only.py)."""
        w, h = crop.size
        sd_w = max(64, ((w + 7) // 8) * 8)
        sd_h = max(64, ((h + 7) // 8) * 8)
        
        crop_sd = crop.resize((sd_w, sd_h), Image.Resampling.LANCZOS)
        
        strength = random.uniform(0.08, 0.15)
        generator = torch.Generator(device="cuda").manual_seed(
            random.randint(0, 2**32 - 1))
        
        output = self.pipe(
            prompt="metal surface defect, scratch, industrial steel texture, manufacturing flaw",
            negative_prompt="smooth, plastic, wood, rust, colorful",
            image=crop_sd,
            strength=strength,
            guidance_scale=2.0,
            num_inference_steps=25,
            generator=generator
        )
        
        generated = output.images[0]
        if sd_w != w or sd_h != h:
            generated = generated.resize((w, h), Image.Resampling.LANCZOS)
        
        generated_np = np.array(generated).astype(np.float32)
        crop_np = np.array(crop).astype(np.float32)
        
        generated_np = match_spectrum(generated_np, crop_np)
        generated_np = inject_high_freq(generated_np, crop_np, alpha=0.4)
        
        return generated_np


# ================= Главный генератор =================

class Generator:
    """Основной класс для обработки изображений и вставки дефектов."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sd = None
    
    def process(self, img_path: Path, bboxes: List[Dict], out_dir: Path, variant: int, idx: int) -> Optional[str]:
        """Обрабатывает одно изображение, вставляя сгенерированные дефекты."""
        try:
            orig = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            result = orig.copy()
            
            if self.sd is None:
                self.sd = SDGenerator()
            
            yolo_annotations = []
            
            for bb in bboxes:
                x, y, w, h = bb['x'], bb['y'], bb['w'], bb['h']
                
                # Добавляем отступы вокруг дефекта
                pad = int(max(w, h) * 0.3)
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(256, x + w + pad), min(256, y + h + pad)
                
                # Пропускаем слишком маленькие или граничные области
                if x2 - x1 < 16 or y2 - y1 < 16:
                    continue
                
                # Вырезаем область из оригинала
                crop = orig[y1:y2, x1:x2]
                hc, wc = crop.shape[:2]
                
                # Генерация дефекта
                generated_np = self.sd.generate_defect_region(Image.fromarray(crop))
                if generated_np.shape[:2] != (hc, wc):
                    generated_np = cv2.resize(generated_np, (wc, hc))
                
                # Подготовка масок
                cm = bb['component_mask'][y1:y2, x1:x2]
                bm = bb['boundary_mask'][y1:y2, x1:x2]
                
                # Проверка и коррекция размеров масок
                if cm.shape[:2] != (hc, wc):
                    cm = cv2.resize(cm, (wc, hc), interpolation=cv2.INTER_NEAREST)
                if bm.shape[:2] != (hc, wc):
                    bm = cv2.resize(bm, (wc, hc), interpolation=cv2.INTER_NEAREST)
                
                # Poisson blending для бесшовной вставки
                blended = poisson_blend_color(crop, generated_np, (cm > 0.5).astype(np.uint8), (bm > 0.5).astype(np.uint8))
                result[y1:y2, x1:x2] = blended
                
                yolo_annotations.append({
                    'class': bb['class'],
                    'x_center': bb['x_center'],
                    'y_center': bb['y_center'],
                    'width': bb['width'],
                    'height': bb['height']
                })
            
            if not yolo_annotations:
                return None
            
            # Сохранение результата
            stem = img_path.stem
            fn = f"syn_{idx:06d}_{stem}_v{variant}.png"
            
            Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)).save(
                out_dir / "images" / fn, "PNG", optimize=True)
            
            # Сохранение меток в YOLO формате
            with open(out_dir / "labels" / f"{Path(fn).stem}.txt", 'w') as f:
                for ann in yolo_annotations:
                    f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} "
                           f"{ann['width']:.6f} {ann['height']:.6f}\n")
            
            return fn
            
        except Exception as e:
            logger.error(f"❌ {img_path.name}: {e}")
            return None
    
    def run(self, input_dir: Path, rle_csv: Path, output_dir: Path, variants: int = 3, limit: Optional[int] = None) -> None:
        """Запускает генерацию для всего датасета."""
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        rle_df = pd.read_csv(rle_csv)
        groups = rle_df.groupby('ImageId')
        logger.info(f"📂 Патчей: {len(groups)}")
        
        all_images = {img.name: img for img in input_dir.glob("*")
                     if img.suffix.lower() in ['.png', '.jpg', '.jpeg']}
        
        total = 0
        for image_id, group in tqdm(groups, desc="Обработка изображений"):
            if limit and total >= limit:
                break
            if image_id not in all_images:
                logger.warning(f"Изображение {image_id} не найдено в {input_dir}. Пропуск.")
                continue
            
            # Собираем ВСЕ дефекты
            all_bboxes = []
            for _, row in group.iterrows():
                rle = str(row['EncodedPixels'])
                if rle == 'nan' or rle.strip() == '':
                    continue
                all_bboxes.extend(
                    rle_to_bboxes(rle, image_id, int(row['ClassId']) - 1))
            
            if not all_bboxes:
                continue
            
            # Генерируем варианты
            for v in range(variants):
                if limit and total >= limit:
                    break
                self.process(all_images[image_id], all_bboxes, output_dir, v, total)
                total += 1
            
            if total % 20 == 0:
                torch.cuda.empty_cache()
        
        logger.info(f"✅ Обработано и сохранено {total} изображений")


def main():
    parser = argparse.ArgumentParser(description="Генерация дефектов с Poisson blending")
    parser.add_argument("--input_dir", required=True, help="Путь к директории с исходными изображениями")
    parser.add_argument("--rle_csv", required=True, help="Путь к CSV файлу с RLE разметкой")
    parser.add_argument("--output_dir", required=True, help="Путь к директории для сохранения результатов")
    parser.add_argument("--variants", type=int, default=3, help="Количество вариаций для каждого изображения")
    parser.add_argument("--limit", type=int, default=None, help="Ограничение на количество обрабатываемых изображений")
    args = parser.parse_args()
    
    Generator(Config()).run(
        Path(args.input_dir),
        Path(args.rle_csv),
        Path(args.output_dir),
        args.variants,
        args.limit
    )


if __name__ == "__main__":
    main()