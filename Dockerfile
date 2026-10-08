FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY worker.py .
COPY workflows/ workflows/

RUN useradd --uid 1001 --create-home worker && chown -R 1001 /app
USER 1001

# The platform runs the image as-is; this must start the worker.
CMD ["python", "worker.py"]
