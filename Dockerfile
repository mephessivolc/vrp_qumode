# Permite trocar a imagem base via docker-compose (padrão: versão CPU)
ARG BASE_IMAGE=tensorflow/tensorflow:2.15.0
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV MPLCONFIGDIR=/tmp/matplotlib
ENV NUMBA_CACHE_DIR=/tmp
ENV TMPDIR=/tmp

RUN mkdir -p /tmp/matplotlib && chmod -R 777 /tmp /tmp/matplotlib

RUN useradd -m -s /bin/bash -u 1025 clovis \
    && apt-get update && apt-get install -y sudo \
    && echo "clovis ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

# Instalação limpa sem salvar cache do pip no disco
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt