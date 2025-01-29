import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import IrisClassifier, ModelTrainer
from src.data_processing import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data_loaders(train_data, test_data, batch_size=32):
    """Convert pandas DataFrames to PyTorch DataLoaders"""
    X_train = torch.FloatTensor(train_data.iloc[:, :-1].values)
    y_train = torch.LongTensor(train_data.iloc[:, -1].values)

    X_test = torch.FloatTensor(test_data.iloc[:, :-1].values)
    y_test = torch.LongTensor(test_data.iloc[:, -1].values)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def main():
    try:
        logger.info("Starting training process")

        processor = DataProcessor()
        train_data, test_data, _ = processor.process_and_split_data()
        logger.info(
            f"Training data size: {len(train_data)}, Test data size: {len(test_data)}")

        train_loader, test_loader = prepare_data_loaders(train_data, test_data)

        model = IrisClassifier()
        trainer = ModelTrainer(model)

        logger.info("Starting model training")
        trainer.train(train_loader, test_loader, epochs=100)

        final_accuracy = trainer.evaluate(test_loader)
        logger.info(f"Final test accuracy: {final_accuracy:.2f}%")

        model_save_path = '/app/models/model.pth'
        trainer.save_model(model_save_path)
        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
