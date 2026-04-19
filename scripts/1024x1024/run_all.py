#!/usr/bin/env python3
"""
run_all.py - Запуск полного пайплайна в Docker
"""

import sys
import os
sys.path.insert(0, '/app/scripts')

import subprocess
import argparse
from pathlib import Path

from config import PipelineConfig
from utils import set_seed, print_system_info, logger, check_directories


def run_command(cmd: str, description: str):
    """Запуск команды с выводом статуса"""
    logger.info("=" * 60)
    logger.info(f"🚀 {description}")
    logger.info("=" * 60)
    
    result = subprocess.run(cmd, shell=True, executable='/bin/bash')
    
    if result.returncode != 0:
        logger.error(f"Ошибка на этапе: {description}")
        sys.exit(1)
    
    logger.info(f"✅ {description} завершен")


def main():
    parser = argparse.ArgumentParser(description="Запуск пайплайна в Docker")
    parser.add_argument("--clean_dir", type=str, default="/app/data/clean_textures")
    parser.add_argument("--defects_dir", type=str, default="/app/data/defects")
    parser.add_argument("--output_dir", type=str, default="/app/results")
    parser.add_argument("--variants", type=int, default=50)
    parser.add_argument("--final_images", type=int, default=10000)
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    print_system_info()
    set_seed(args.seed)
    
    config = PipelineConfig(
        clean_textures_dir=args.clean_dir,
        defects_dir=args.defects_dir,
        output_dir=args.output_dir,
        variants_per_reference=args.variants,
        total_final_images=args.final_images
    )
    
    # Проверка директорий
    dirs = check_directories(config)
    
    logger.info(f"Вариантов на референс: {config.variants_per_reference}")
    logger.info(f"Финальных изображений: {config.total_final_images}")
    
    # Этап 1: Генерация фонов
    cmd1 = (
        f"python /app/scripts/01_generate_backgrounds.py "
        f"--clean_dir {config.clean_textures_dir} "
        f"--output_dir {config.backgrounds_dir} "
        f"--variants {config.variants_per_reference} "
        f"--seed {args.seed}"
    )
    run_command(cmd1, "ЭТАП 1: Генерация синтетических фонов")
    
    # Этап 2: Валидация (опционально)
    if not args.skip_validation:
        cmd2 = (
            f"python /app/scripts/03_validate_quality.py "
            f"--real_dir {config.clean_textures_dir} "
            f"--fake_dir {config.backgrounds_dir} "
            f"--output {config.validation_dir}/metrics.json"
        )
        run_command(cmd2, "ЭТАП 2: Валидация качества")
    
    # Этап 3: Вставка дефектов
    cmd3 = (
        f"python /app/scripts/02_insert_defects.py "
        f"--backgrounds_dir {config.backgrounds_dir} "
        f"--defects_dir {config.defects_dir} "
        f"--output_dir {config.final_dataset_dir} "
        f"--num_images {config.total_final_images} "
        f"--seed {args.seed}"
    )
    run_command(cmd3, "ЭТАП 3: Вставка дефектов")
    
    # Этап 4: Визуализация
    cmd4 = (
        f"python /app/scripts/04_visualize_samples.py defects "
        f"--dataset_dir {config.final_dataset_dir} "
        f"--num_samples 20 "
        f"--output_dir {config.viz_dir}"
    )
    run_command(cmd4, "ЭТАП 4: Визуализация")
    
    logger.info("=" * 60)
    logger.info("🎉 ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
    logger.info("=" * 60)
    logger.info(f"📁 Результаты в: {config.output_dir}")


if __name__ == "__main__":
    main()