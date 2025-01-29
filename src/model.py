import logging
import time
from datetime import datetime

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IrisClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=10, num_classes=3):
        super(IrisClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3)
        )
        self.layer3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class ModelTrainer:
    def __init__(self, model,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        logger.info(f"Using device: {device}")

    def train(self, train_loader, test_loader, epochs=100,
              learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        start_time = time.time()
        logger.info(f"Starting training at {datetime.now()}")

        for epoch in range(epochs):
            self.model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                accuracy = self.evaluate(test_loader)
                logger.info(f'Epoch [{epoch + 1}/{epochs}], '
                            f'Loss: {loss.item():.4f}, '
                            f'Test Accuracy: {accuracy:.2f}%')

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def save_model(self, path):
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
