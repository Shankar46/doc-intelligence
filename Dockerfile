FROM python:3.11-slim

# Install system dependencies (Tesseract OCR for scanned PDF/image support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Copy backend & frontend source code
COPY backend ./backend
COPY frontend ./frontend

WORKDIR /app/backend

# Create uploads and db directory
RUN mkdir -p data/uploads

# Expose default port
ENV PORT=8000
EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
