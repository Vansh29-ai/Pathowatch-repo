FROM python:3.11-slim

# Prevent Python from writing pyc files & enable logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (GDAL + build tools + runtime libs)
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables (important for rasterio)
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Set working directory
WORKDIR /app

# Upgrade pip (important for compiled libs)
RUN pip install --upgrade pip

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create non-root user (security best practice)
RUN useradd -m appuser
USER appuser

# Port (Render / Docker compatible)
ENV PORT=8080
EXPOSE 8080

# Gunicorn settings tuned for ML workload
CMD ["gunicorn", "server:app", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "180", \
     "--workers", "1", \
     "--threads", "2"]