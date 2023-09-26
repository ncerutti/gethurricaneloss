# tests/test_gethurricaneloss.py

import pytest
from src import gethurricaneloss as hurricane


def test_check_input_positive():
    # should not raise any exception
    hurricane.check_input(1.0, "test value")


def test_check_input_negative():
    with pytest.raises(ValueError, match="test value should be a positive number"):
        hurricane.check_input(-1.0, "test value")


def test_check_input_zero():
    # should not raise any exception
    hurricane.check_input(0.0, "test value")


def test_check_samples_positive():
    # should not raise any exception
    hurricane.check_samples(2, "test sample")


def test_check_samples_negative():
    with pytest.raises(ValueError, match="test sample should be a positive number"):
        hurricane.check_samples(0, "test sample")


def test_check_samples_zero():
    with pytest.raises(ValueError, match="test sample should be a positive number"):
        hurricane.check_samples(0, "test sample")


def test_calculate_loss():
    result = hurricane.calculate_loss(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1000)
    assert isinstance(result, float)
    # you can add more assertions based on expected ranges or values for your model
