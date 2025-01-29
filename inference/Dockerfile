FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/inference_data.csv data/
COPY models/model.pth models/
COPY inference/inference.py .

RUN mkdir -p results

CMD ["python", "inference.py"]