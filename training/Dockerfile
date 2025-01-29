
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
COPY src/ ./src/
COPY data/ ./data/
COPY training/train.py ./train.py

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p models

CMD ["python", "train.py"]