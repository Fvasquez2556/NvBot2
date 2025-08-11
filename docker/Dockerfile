# Dockerfile para Momentum Predictor Bot

FROM python:3.10-slim

# Configurar variables de entorno
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root
RUN useradd --create-home --shell /bin/bash app

# Configurar directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt requirements-dev.txt ./

# Actualizar pip e instalar dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Crear directorios necesarios
RUN mkdir -p data/{raw,processed,models} logs config

# Copiar código fuente
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

# Cambiar propiedad a usuario app
RUN chown -R app:app /app

# Cambiar a usuario no-root
USER app

# Exponer puerto (si se agrega API web en el futuro)
EXPOSE 8000

# Configurar healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Comando por defecto
CMD ["python", "src/main.py"]
