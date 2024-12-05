import pytest
import torch
from main import get_model, get_device
import json


def test_parameter_count():
    model = get_model(get_device(), requires_summary=False)
    total_params = sum(p.numel() for p in model.parameters())
    assert (
        total_params < 20000
    ), f"Model has {total_params} parameters, should be less than 20000"


def test_batch_normalization():
    """Test if model uses Batch Normalization"""
    model = get_model(get_device(), requires_summary=False)
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model should use Batch Normalization"


def test_dropout():
    """Test if model uses Dropout"""
    model = get_model(get_device(), requires_summary=False)
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use Dropout"


def test_gap_or_fc():
    """Test if model uses either GAP or Fully Connected layer"""
    model = get_model(get_device(), requires_summary=False)
    has_gap = any(
        isinstance(m, torch.nn.AvgPool2d)
        or isinstance(m, torch.nn.Linear)
        or isinstance(m, torch.nn.AdaptiveAvgPool2d)
        for m in model.modules()
    )
    assert has_gap, "Model should use either GAP or Fully Connected layer"


def test_test_accuracy():
    """Test if model achieves accuracy above 99.4%"""
    try:
        with open("metrics.json", "r") as f:
            metrics = json.loads(f.read())
        test_accuracies = metrics["test_accuracy_over_epochs"]
        max_accuracy = max(test_accuracies)  # Get the maximum accuracy value
        assert max_accuracy >= float(
            99.4
        ), f"Model accuracy {max_accuracy}% should be greater than 99.4%"
    except FileNotFoundError:
        assert (
            False
        ), "metrics.json file not found. Make sure the training saves the metrics."
