import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

    def load_iris_data(self):
        """Load the Iris dataset and perform initial preprocessing."""
        try:
            from sklearn.datasets import load_iris
            iris = load_iris()
            data = pd.DataFrame(
                data=iris.data,
                columns=['sepal_length', 'sepal_width', 'petal_length',
                         'petal_width']
            )
            data['species'] = iris.target
            logger.info(f"Loaded Iris dataset with {len(data)} samples")
            return data
        except Exception as e:
            logger.error(f"Error loading Iris dataset: {str(e)}")
            raise

    def process_and_split_data(self, test_size=0.2, inference_size=0.2):
        """Process the data and split into train, test, and inference sets."""
        try:
            data = self.load_iris_data()

            train_valid_data, inference_data = train_test_split(
                data,
                test_size=inference_size,
                random_state=self.random_state
            )

            train_data, test_data = train_test_split(
                train_valid_data,
                test_size=test_size,
                random_state=self.random_state
            )

            os.makedirs('data', exist_ok=True)
            train_data.to_csv('data/train_data.csv', index=False)
            test_data.to_csv('data/test_data.csv', index=False)
            inference_data.to_csv('data/inference_data.csv', index=False)

            logger.info(f"Split data into: train({len(train_data)}), "
                        f"test({len(test_data)}), inference({len(inference_data)})")

            return train_data, test_data, inference_data

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise


if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_and_split_data()
