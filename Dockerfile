FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV STORAGE_BACKEND=gcs \
    LOAD_BACKEND=bigquery \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "src.pipelines.main"]
