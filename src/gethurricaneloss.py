# src/hurricane.py

import argparse
import logging
import timeit
from numba import jit
import numpy as np


def check_input(value, name):
    if value < 0:
        logging.error(f"Invalid input for {name}: {value}")
        raise ValueError(f"{name} should be a positive number")


def check_samples(value, name):
    if value < 1:
        logging.error(f"Invalid input for {name}: {value}")
        raise ValueError(f"{name} should be a positive number")


# @jit(nopython=False)
def calculate_loss(
    florida_rate: float,
    florida_mean: float,
    florida_stddev: float,
    gulf_rate: float,
    gulf_mean: float,
    gulf_stddev: float,
    num_samples: int,
) -> float:
    total_loss = 0

    for _ in range(num_samples):
        simulation_loss = 0

        florida_events = np.random.poisson(florida_rate)
        for _ in range(florida_events):
            simulation_loss += np.random.lognormal(florida_mean, florida_stddev)

        gulf_events = np.random.poisson(gulf_rate)
        for _ in range(gulf_events):
            simulation_loss += np.random.lognormal(gulf_mean, gulf_stddev)

        print(
            f"Simulation loss for current iteration: {simulation_loss}"
        )  # Add this line

        total_loss += simulation_loss

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
        result = calculate_loss(
            args.florida_landfall_rate,
            args.florida_mean,
            args.florida_stddev,
            args.gulf_landfall_rate,
            args.gulf_mean,
            args.gulf_stddev,
            args.num_monte_carlo_samples,
        )
        print(result)
    except Exception as e:
        logging.error(f"Error during hurricane loss calculation: {e}")
        raise e

    logging.info("Hurricane loss calculation complete.")

    print(f"Expected annual economic loss: ${result:.2f} billion")


if __name__ == "__main__":
    main()
