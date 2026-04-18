FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/scripts:/app/src

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential git ffmpeg libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --no-cache-dir -U pip setuptools wheel

# 🔧 МИНИМАЛЬНЫЕ ИЗМЕНЕНИЯ: правильные версии для IP-Adapter
RUN pip install --no-cache-dir \
    huggingface_hub==0.20.3 \
    tokenizers==0.15.2

RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# 🔧 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: diffusers 0.25.0 и transformers 4.36.0
RUN pip install --no-cache-dir \
    diffusers==0.25.0 \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    peft==0.10.0 \
    safetensors==0.4.5 \
    pillow==10.4.0 \
    opencv-python-headless==4.10.0.84 \
    tqdm==4.66.5 \
    scipy==1.14.0 \
    matplotlib==3.8.0 \
    pyyaml==6.0.1 \ 
    pandas

RUN pip install --no-cache-dir xformers --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir clean-fid==0.1.35

RUN mkdir -p /app/{scripts,src,configs,data,models,results,cache}

WORKDIR /app

ENV HF_HOME=/app/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch \
    CUDA_VISIBLE_DEVICES=0

RUN python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

CMD ["tail", "-f", "/dev/null"]