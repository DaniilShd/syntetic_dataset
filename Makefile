# Makefile для управления Docker-пайплайном

.PHONY: build up down shell generate insert validate viz all clean

# Сборка образа
build:
	docker compose build

# Запуск контейнера
up:
	docker compose up -d

# Остановка контейнера
down:
	docker compose down

# Интерактивная оболочка
shell:
	docker exec -it synthetic_generator_v2 /bin/bash

# Генерация фонов
generate:
	docker exec synthetic_generator python /app/scripts/01_generate_backgrounds.py \
		--clean_dir /app/data/clean_textures \
		--output_dir /app/results/synthetic_backgrounds \
		--variants 50

# Генерация с параметрами
generate-custom:
	docker exec synthetic_generator python /app/scripts/01_generate_backgrounds.py \
		--clean_dir /app/data/clean_textures \
		--output_dir /app/results/synthetic_backgrounds \
		--variants $(VARIANTS) \
		--strength_min $(STRENGTH_MIN) \
		--strength_max $(STRENGTH_MAX) \
		--ip_scale_min $(IP_MIN) \
		--ip_scale_max $(IP_MAX)

# Вставка дефектов
insert:
	docker exec synthetic_generator python /app/scripts/02_insert_defects.py \
		--backgrounds_dir /app/results/synthetic_backgrounds \
		--defects_dir /app/data/defects \
		--output_dir /app/results/final_dataset \
		--num_images $(NUM)

# Валидация
validate:
	docker exec synthetic_generator python /app/scripts/03_validate_quality.py \
		--real_dir /app/data/clean_textures \
		--fake_dir /app/results/synthetic_backgrounds \
		--output /app/results/validation/metrics.json

# Визуализация
viz:
	docker exec synthetic_generator python /app/scripts/04_visualize_samples.py defects \
		--dataset_dir /app/results/final_dataset \
		--num_samples 20 \
		--output_dir /app/results/visualizations

# Полный пайплайн
all:
	docker exec synthetic_generator python /app/scripts/run_all.py \
		--clean_dir /app/data/clean_textures \
		--defects_dir /app/data/defects \
		--output_dir /app/results \
		--variants 50 \
		--final_images 10000

# Очистка результатов
clean-results:
	docker exec synthetic_generator rm -rf /app/results/*

# Очистка кеша
clean-cache:
	docker exec synthetic_generator rm -rf /app/cache/*

# Просмотр логов
logs:
	docker logs -f synthetic_generator

# Проверка GPU
gpu:
	docker exec synthetic_generator python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# Проверка памяти GPU
memory:
	docker exec synthetic_generator nvidia-smi

# Полная пересборка
rebuild: down build up
	@echo "Ожидание инициализации..."
	sleep 5
	docker exec synthetic_generator python -c "import torch; print('Ready')"