# src/gethurricaneloss_mp_para.py

import argparse
import logging
import multiprocessing
from tqdm import tqdm  # currently unused
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
    """
    Computes the total loss for hurricanes in both Florida and the Gulf states for a given set of samples.

    This function operates in batches to facilitate larger numbers of samples. Each batch computes the loss
    for both Florida and the Gulf states. The function then aggregates the total loss over all batches and any 
    remaining samples (if the number of samples is not perfectly divisible by the batch size).

    Parameters:
    - params (tuple): Contains the following values:
        * florida_rate (float): Expected number of hurricanes in Florida.
        * florida_mean (float): Mean of the lognormal distribution of hurricane losses in Florida.
        * florida_stddev (float): SD of the lognormal distribution of hurricane losses in Florida.
        * gulf_rate (float): Expected number of hurricanes in Gulf states.
        * gulf_mean (float): Mean of the lognormal distribution of hurricane losses in Gulf states.
        * gulf_stddev (float): SD of the lognormal distribution of hurricane losses in Gulf states.
        * local_samples (int): Number of samples to be processed by this worker.

    Returns:
    - float: The total hurricane loss over all the samples processed by this worker.
    """
    # How many rows of data to process at a time
    batch_size = 1000000
    batches = local_samples // batch_size
    local_total_loss = 0.0

    states_rate = np.array([florida_rate, gulf_rate])
    states_mean = np.array([florida_mean, gulf_mean])
    states_stddev = np.array([florida_stddev, gulf_stddev])

    # Compute loss for each batch
    for _ in range(batches):
        local_total_loss += simulate_loss(
            states_rate, states_mean, states_stddev, batch_size
        )

    # Run for the remaining samples if local_samples is not divisible by batch_size
    remaining_samples = local_samples % batch_size
    if remaining_samples > 0:
        local_total_loss += simulate_loss(
            states_rate, states_mean, states_stddev, remaining_samples
        )
    return local_total_loss


@jit(nopython=True)
def simulate_loss(
    rate: np.array, mean: np.array, stddev: np.array, batch_size: int
) -> float:
    """
    Simulates the total loss for two events using given rates, means, and standard deviations.

    Parameters:
    - rate (np.array): 2-element array with rate parameters for the Poisson distribution of each event type.
    - mean (np.array): 2-element array with means of the lognormal distributions for each event's loss.
    - stddev (np.array): 2-element array with standard deviations of the lognormal distributions for each event's loss.
    - batch_size (int): Number of simulation batches to run.

    Returns:
    - float: The total loss accumulated from all events in the simulation.

    Note: the function works with two event areas (as per problem to be solved). Can be adjusted to work with more.
    """
    total_loss = 0.0
    for i in range(2):
        events = np.random.poisson(rate[i], batch_size)
        losses = np.random.lognormal(mean[i], stddev[i], size=events.sum())
        total_loss += losses.sum()
    return total_loss


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

    # Create parameters for each process
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

    # Sum the results from all cores and divide by total number of samples for the average
    total_loss = sum(results)

    return total_loss / num_samples


def main():
    # Set up logging in file logs.log
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

    # Extract and check command-line arguments
    args = parser.parse_args()

    check_input(args.florida_landfall_rate, "Florida landfall rate")
    check_input(args.florida_mean, "Florida mean")
    check_input(args.florida_stddev, "Florida stddev")
    check_input(args.gulf_landfall_rate, "Gulf landfall rate")
    check_input(args.gulf_mean, "Gulf mean")
    check_input(args.gulf_stddev, "Gulf stddev")
    check_samples(args.num_monte_carlo_samples, "Number of Monte Carlo samples")

    logging.info("Starting hurricane loss calculation...")

    # Execute the main computation and handle any exceptions
    # They will be logged and, if this does not work, the whole program does not make much sense
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
