# Iris Flower Classification Project

This project implements a deep learning classifier for the Iris flower dataset
using PyTorch. The project is containerized using Docker with separate
containers for training and inference.

## Project Structure

```
iris-classification/
├── data/
│   ├── iris_data.csv
│   ├── train_data.csv
│   └── inference_data.csv
├── training/
│   ├── Dockerfile
│   └── train.py
├── inference/
│   ├── Dockerfile
│   └── inference.py
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   └── test_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Running the Project

### 1. Training

```bash
cd training
docker build -t iris-training .
docker run -v $(pwd)/models:/app/models iris-training
```

### 2. Inference

```bash
cd inference
docker build -t iris-inference .
docker run -v $(pwd)/results:/app/results iris-inference
```

## Requirements

- Python 3.8+
- Docker
- See requirements.txt for Python dependencies