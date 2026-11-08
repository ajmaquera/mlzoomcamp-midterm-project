# Imagen base ligera de Python
FROM python:3.12-slim

# Copiar binarios de uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Variables de entorno para ejecucion en contenedor
ENV PATH="/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar definicion de dependencias e instalar
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copiar modelo y aplicacion
COPY model.bin predict.py ./

# Exponer puerto de la API
EXPOSE 9696

# Comando de inicio del servidor
ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
