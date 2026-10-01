# ========================================================
# Bomeli Motorcycle Financing - ML Engine Production Docker
# Compatible with Hugging Face Spaces (Docker Space) & Render
# ========================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000

# Install minimal OS utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files, data, and trained model artifacts
COPY . /app

# Ensure directories exist
RUN mkdir -p /app/data /app/models /app/logs

# Expose default port (Render standard port 10000 or dynamic $PORT)
EXPOSE 10000

# Launch FastAPI web application
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
