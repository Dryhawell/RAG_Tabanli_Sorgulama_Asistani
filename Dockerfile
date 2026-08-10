FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları (PyMuPDF + Tesseract OCR TR/EN)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-tur \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY rag ./rag
COPY README.md .

RUN mkdir -p data indexes metadata

EXPOSE 8501

ENV RAG_DATA_DIR=data \
    RAG_INDEX_PATH=indexes/faiss.index \
    RAG_DOCSTORE_PATH=metadata/docstore.json \
    OLLAMA_HOST=http://host.docker.internal:11434

CMD ["streamlit", "run", "app/ui.py", "--server.address=0.0.0.0", "--server.port=8501"]
