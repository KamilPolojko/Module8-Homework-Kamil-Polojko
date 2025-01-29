import logging
import os
import sys
from datetime import datetime

import pandas as pd
import torch

# Add parent directory to Python path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import IrisClassifier, ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_inference_data(file_path):
    """Load and preprocess inference data"""
    try:
        data = pd.read_csv(file_path)
        features = torch.FloatTensor(data.iloc[:, :-1].values)
        true_labels = data.iloc[:, -1].values
        return features, true_labels
    except Exception as e:
        logger.error(f"Error loading inference data: {str(e)}")
        raise


def main():
    try:
        logger.info("Starting inference process")

        model_path = '/app/models/model.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model file not found! Please ensure the model is trained first.")

        inference_data_path = '/app/data/inference_data.csv'
        features, true_labels = load_inference_data(inference_data_path)
        logger.info(f"Loaded inference data with {len(features)} samples")

        model = IrisClassifier()
        trainer = ModelTrainer(model)
        trainer.load_model(model_path)

        model.eval()
        with torch.no_grad():
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)

        accuracy = (predicted == torch.LongTensor(
            true_labels)).sum().item() / len(true_labels) * 100
        logger.info(f"Inference accuracy: {accuracy:.2f}%")

        results = pd.DataFrame({
            'true_label': true_labels,
            'predicted_label': predicted.numpy()
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f'/app/results/inference_results_{timestamp}.csv'
        results.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise


if __name__ == "__main__":
    main()
