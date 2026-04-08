FROM python:3.11-slim

# System deps that rasterio/GDAL need
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Help rasterio find GDAL
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Install Python deps first (layer cache — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Render sets PORT dynamically; default to 8080 for local docker run
ENV PORT=8080
EXPOSE 8080

# --timeout 120 because GEE calls can be slow
# --workers 1 because the ML model is loaded in memory (not thread-safe with multiple workers)
CMD gunicorn server:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
