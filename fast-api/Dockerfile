# Use slim Python image
FROM python:3.12-slim

# Environment tweaks
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (for numpy / xgboost etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default port inside container (Render will map its own $PORT)
ENV PORT=8000
EXPOSE 8000

# Start uvicorn using $PORT provided by Render
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
