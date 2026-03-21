# ================================
# Stage: Production
# Base: python:3.11-slim (nhỏ hơn full image ~400MB)
# ================================
FROM python:3.11-slim

# Tránh interactive prompts từ apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# -----------------------------------------------------------
# Cài system dependencies cần thiết cho PIL / OpenCV / torch
# libgl1          → OpenGL (cần cho cv2 / open_clip)
# libglib2.0-0    → GLib (cần cho PIL trên Linux)
# libgomp1        → OpenMP (cần cho PyTorch parallel ops)
# -----------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------
# Cài Python dependencies
# --no-cache-dir  → KHÔNG lưu pip cache → tiết kiệm RAM/disk
# Cài trước requirements để tận dụng Docker layer cache
# -----------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------
# Copy source code (sau cùng để layer cache hiệu quả nhất)
# .dockerignore sẽ loại bỏ .env, data/models/, *.zip, .venv
# -----------------------------------------------------------
COPY . .

# Railway sẽ tự set biến $PORT (thường là 8080)
# Expose chỉ mang tính tài liệu, Railway tự map port
EXPOSE 8080

# -----------------------------------------------------------
# CMD dùng sh -c để expand $PORT từ environment Railway
# - host 0.0.0.0 bắt mọi network interface trong container
# - workers 1 vì RAM Railway free tier hạn chế
# - timeout-keep-alive 75s cho Railway health check
# -----------------------------------------------------------
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 75"]
