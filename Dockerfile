# Data Pulse - TensorForge 2.0 Phase 2
#   docker build -t datapulse .
#   docker run -p 8000:8000 -e API_KEY=<key> datapulse
# Everything the service needs (code + model weights) is copied into the image,
# so the container makes no network calls at runtime.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/runtime

# uid 1000 is what Hugging Face Spaces runs containers as; it also avoids running as root anywhere else
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --chown=appuser requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=appuser app ./app
COPY --chown=appuser models ./models
RUN mkdir -p /app/runtime && chown appuser /app/runtime

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

# One worker on purpose: job state is shared through SQLite, and one process keeps memory low on 2 vCPU / 4 GB.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
