import os

import pytest
import torch

from src.model import IrisClassifier, ModelTrainer


def test_model_creation():
    model = IrisClassifier()
    assert isinstance(model, torch.nn.Module)


def test_model_forward_pass():
    model = IrisClassifier()
    batch_size = 10
    input_size = 4
    x = torch.randn(batch_size, input_size)
    output = model(x)
    assert output.shape == (batch_size, 3)


def test_model_save_load():
    model = IrisClassifier()
    trainer = ModelTrainer(model)

    save_path = "test_model.pth"
    trainer.save_model(save_path)
    assert os.path.exists(save_path)

    new_model = IrisClassifier()
    new_trainer = ModelTrainer(new_model)
    new_trainer.load_model(save_path)

    os.remove(save_path)


def test_model_training_input_validation():
    model = IrisClassifier()
    trainer = ModelTrainer(model)

    with pytest.raises(ValueError):
        # Should raise error when trying to train without data
        trainer.train(None, None)


if __name__ == "__main__":
    pytest.main([__file__])
