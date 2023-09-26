# tests/test_gethurricaneloss.py

import pytest
from src.gethurricaneloss import check_input, check_samples, simulate_loss, compute_loss


def test_check_input_valid():
    """Test that check_input() works for valid input"""
    check_input(5, "Test Value")


def test_check_input_invalid():
    """Test that check_input() raises a ValueError for invalid input"""
    with pytest.raises(ValueError, match="Test Value should be a positive number"):
        check_input(-5, "Test Value")


def test_check_samples_valid():
    """Test that check_samples() works for valid input"""
    check_samples(5, "Samples")


def test_check_samples_invalid():
    """Test that check_samples() raises a ValueError for invalid input"""
    with pytest.raises(ValueError, match="Samples should be a positive number"):
        check_samples(0, "Samples")


def test_simulate_loss():
    """Test that simulate_loss() returns a positive number"""
    result = simulate_loss(1.0, 2.0, 1.0)
    assert result >= 0


def test_compute_loss():
    """Test that compute_loss() returns a positive number"""
    result = compute_loss(1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 100)
    assert result >= 0


def test_compute_loss_2():
    """Test that compute_loss() returns a positive number"""
    result = compute_loss(100.0, 200.0, 100.0, 100.0, 200.0, 100.0, 10000)
    assert result >= 0
