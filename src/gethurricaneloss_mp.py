# src/gethurricaneloss.py

import argparse
import logging
import multiprocessing
from tqdm import tqdm
from numba import jit
import numpy as np


def check_input(value, name):
    """
    Checks that the input value is >= 0.

    Parameters:
    - value (float): The input value to check.
    - name (str): The name/description of the input (used for error messaging).

    Raises:
    - ValueError: If the input value is negative.
    """
    if value < 0:
        logging.error(f"Invalid input for {name}: {value}")
        raise ValueError(f"{name} should be a positive number")


def check_samples(value, name):
    """Validates that the sample value is positive and not zero.

    Parameters:
    - value (int): The input value representing number of samples to check.
    - name (str): The name/description of the input (used for error messaging).

    Raises:
    - ValueError: If the input value is less than 1.
    """
    if value < 1:
        logging.error(f"Invalid input for {name}: {value}")
        raise ValueError(f"{name} should be a positive number")


def worker_function(params):
    (
        florida_rate,
        florida_mean,
        florida_stddev,
        gulf_rate,
        gulf_mean,
        gulf_stddev,
        local_samples,
    ) = params
    local_total_loss = 0
    for _ in range(local_samples):
        simulation_loss = simulate_loss(florida_rate, florida_mean, florida_stddev)
        simulation_loss += simulate_loss(gulf_rate, gulf_mean, gulf_stddev)
        local_total_loss += simulation_loss
    return local_total_loss


@jit(nopython=True)
def simulate_loss(rate: float, mean: float, stddev: float) -> float:
    """
    Simulate hurricane loss for given rate, mean, and stddev.

    Parameters:
    - rate (float): The expected number of hurricanes.
    - mean (float): Mean of the lognormal distribution of hurrican losses.
    - stddev (float): SD of the lognormal distribution of hurrican losses.
    """
    events = np.random.poisson(rate)
    losses = np.random.lognormal(mean, stddev, events)
    return losses.sum()


def compute_loss(
    florida_rate: float,
    florida_mean: float,
    florida_stddev: float,
    gulf_rate: float,
    gulf_mean: float,
    gulf_stddev: float,
    num_samples: int,
) -> float:
    """
    Calculates the average loss due to hurricanes in Florida and in the Gulf states over a number of runs.

    Parameters:
    - florida_rate (float): The expected number of hurricanes in Florida.
    - florida_mean (float): Mean of the lognormal distribution of hurrican losses in Florida.
    - florida_stddev (float): SD of the lognormal distribution of hurrican losses in Florida.
    - gulf_rate (float): The expected number of hurricanes in Gulf states.
    - gulf_mean (float): Mean of the lognormal distribution of hurrican losses in Gulf states.
    - gulf_stddev (float): SD of the lognormal distribution of hurrican losses in Gulf states.
    - num_samples (int): The number of runs.

    Returns:
    - float: The average calculated loss over the number of samples.
    """

    # Number of cores available
    num_cores = multiprocessing.cpu_count()

    # Split work among cores
    samples_per_core = num_samples // num_cores
    params = [
        (
            florida_rate,
            florida_mean,
            florida_stddev,
            gulf_rate,
            gulf_mean,
            gulf_stddev,
            samples_per_core,
        )
        for _ in range(num_cores)
    ]

    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(worker_function, params)

    # Combine results from all processes
    total_loss = sum(results)

    return total_loss / num_samples


def main():
    logging.basicConfig(
        filename="logs.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Calculates the average annual hurricane loss in $Billions for a simple hurricane model."
    )
    parser.add_argument("florida_landfall_rate", type=float)
    parser.add_argument("florida_mean", type=float)
    parser.add_argument("florida_stddev", type=float)
    parser.add_argument("gulf_landfall_rate", type=float)
    parser.add_argument("gulf_mean", type=float)
    parser.add_argument("gulf_stddev", type=float)
    parser.add_argument("-n", "--num_monte_carlo_samples", type=int, default=1000)

    args = parser.parse_args()

    check_input(args.florida_landfall_rate, "Florida landfall rate")
    check_input(args.florida_mean, "Florida mean")
    check_input(args.florida_stddev, "Florida stddev")
    check_input(args.gulf_landfall_rate, "Gulf landfall rate")
    check_input(args.gulf_mean, "Gulf mean")
    check_input(args.gulf_stddev, "Gulf stddev")
    check_samples(args.num_monte_carlo_samples, "Number of Monte Carlo samples")

    logging.info("Starting hurricane loss calculation...")

    try:
        result = compute_loss(
            args.florida_landfall_rate,
            args.florida_mean,
            args.florida_stddev,
            args.gulf_landfall_rate,
            args.gulf_mean,
            args.gulf_stddev,
            args.num_monte_carlo_samples,
        )
    except Exception as e:
        logging.error(f"Error during hurricane loss calculation: {e}")
        raise e

    logging.info("Hurricane loss calculation complete.")

    print(f"Expected annual economic loss: ${result:.4f} billion")


if __name__ == "__main__":
    main()
