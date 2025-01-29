import os

import pandas as pd
import pytest

from src.data_processing import DataProcessor


def test_data_processor_initialization():
    """Test if DataProcessor initializes correctly"""
    processor = DataProcessor(random_state=42)
    assert processor.random_state == 42
    assert hasattr(processor, 'label_encoder')


def test_load_iris_data():
    """Test if iris data is loaded correctly"""
    processor = DataProcessor()
    data = processor.load_iris_data()

    assert isinstance(data, pd.DataFrame)

    expected_columns = ['sepal_length', 'sepal_width', 'petal_length',
                        'petal_width', 'species']
    assert all(col in data.columns for col in expected_columns)

    assert len(data['species'].unique()) == 3

    assert len(data) == 150


def test_process_and_split_data():
    """Test if data is split correctly"""
    processor = DataProcessor(random_state=42)
    train_data, test_data, inference_data = processor.process_and_split_data(
        test_size=0.2,
        inference_size=0.2
    )

    total_samples = len(train_data) + len(test_data) + len(inference_data)
    assert abs(
        len(inference_data) / total_samples - 0.2) < 0.01

    remaining_samples = len(train_data) + len(test_data)
    assert abs(
        len(test_data) / remaining_samples - 0.2) < 0.01

    assert os.path.exists('data/train_data.csv')
    assert os.path.exists('data/test_data.csv')
    assert os.path.exists('data/inference_data.csv')

    os.remove('data/train_data.csv')
    os.remove('data/test_data.csv')
    os.remove('data/inference_data.csv')


def test_data_consistency():
    """Test if the processed data maintains the correct structure"""
    processor = DataProcessor()
    train_data, test_data, inference_data = processor.process_and_split_data()

    expected_columns = ['sepal_length', 'sepal_width', 'petal_length',
                        'petal_width', 'species']
    for dataset in [train_data, test_data, inference_data]:
        assert all(col in dataset.columns for col in expected_columns)

        for col in dataset.columns[:-1]:
            assert dataset[col].dtype in ['float64', 'int64']

        assert not dataset.isnull().any().any()


def test_random_state_reproducibility():
    """Test if using the same random_state produces the same splits"""
    processor1 = DataProcessor(random_state=42)
    train1, test1, inf1 = processor1.process_and_split_data()

    processor2 = DataProcessor(random_state=42)
    train2, test2, inf2 = processor2.process_and_split_data()

    assert train1.equals(train2)
    assert test1.equals(test2)
    assert inf1.equals(inf2)


def test_invalid_split_sizes():
    """Test if processor raises error for invalid split sizes"""
    processor = DataProcessor()

    with pytest.raises(ValueError):
        processor.process_and_split_data(test_size=0.9, inference_size=0.2)

    with pytest.raises(ValueError):
        processor.process_and_split_data(test_size=-0.1, inference_size=0.2)

    with pytest.raises(ValueError):
        processor.process_and_split_data(test_size=0.2, inference_size=1.1)


if __name__ == '__main__':
    pytest.main([__file__])
