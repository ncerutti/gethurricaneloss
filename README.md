# gethurricaneloss
a command line utility to calculate Florida and Gulf states hurricane losses

# Installation:
## Prerequisites:
- Python 3.9

## Steps:
- **Clone the repository**:
  git clone https://github.com/ncerutti/gethurricaneloss.git ->
  cd gethurricaneloss
- **Install the package**:
  pip install . 
  (or pip install -e . if you are planning to make some changes)


# Usage:

gethurricaneloss [options] florida_landfall_rate florida_mean florida_stddev gulf_landfall_rate gulf_mean gulf_stddev

**CAUTION:** the command as above will run gethurricaneloss_mp_para.py. gethurricaneloss_base may take a long time to compute, depending on the sample size. Please see below (Performance) for more information.

Calculates the average annual hurricane loss in $Billions for a simple hurricane model. The model
is parameterized by:

- florida_landfall_rate – The annual rate of landfalling hurricanes in Florida
- florida_mean, florida_stddev – The LogNormal parameters that describe the economic loss of a landfalling hurricane in Florida.
- gulf_landfall_rate - The annual rate of landfalling hurricanes in the Gulf states
- gulf_mean, gulf_stddev - The LogNormal parameters that describe the economic loss of a landfalling hurricane in the Gulf states.

options:
-n, --num_monte_carlo_samples Number of samples (i.e. simulation years) to run


## A few considerations:

- I do not think it is useful to put in on PyPY, so it currently has to be installed locally as per instructions above.
- It currently includes some input validation (number of inputs and value, all positive), and the argument parser checks that the inputs are indeed numbers.
- I thought about adding further info (SD of losses, 95% CI, min/max). This would be a light calculation (just one float per run), but it goes beyond the task.




## Performance:



Running "pytest" from the root directory will produce a "speed.txt" file, which provides a benchmark of the different versions of the model. The parameters it is tested on are:

florida_landfall_rate: 10.0
florida_mean: 2.0
florida_stddev: 1.0
gulf_landfall_rate: 10.0
gulf_mean: 2.0
gulf_stddev: 1.0
num_samples: 50000000

In principle, a more thorough benchmark (more executions, providing standard deviations) could help better quantifying any differences in performance.
Please note that the times indicated and the differences between them might differ based on the hardware the model is running on.

Also, please note that not all optimizations work regardless of the size of the sample and on the underlying hardware. It might as well be that they should be tailored to specific cases. Therefore it is very important to run the *pytest* benchmark and/or specific benchmarks for the use case.





Beneath a description of the differences between the scripts, their performance as measured on my laptop, and what steps I took in optimizing them:


- **gethurricaneloss_base.py**: 10.2141 seconds with 1/10th of the sample

Base version of the code. No optimization.

- **gethurricaneloss_jit.py**: 8.4518 seconds with 1/10th of the sample

It differs from the previous version only in the use of *numba* in the simulation part. Very large improvement.

- **gethurricaneloss_mp.py**: 24.4029 seconds

This version improves over the previous by using python's *multiprocessing* library.

- **gethurricaneloss_mp_para.py**: 16.3060 seconds

Instead of running the simulations *n* times (where *n* is the number of samples we want, i.e. how many times the simulation will run), it runs it in batches (1000000 currently on my laptop, ideally it should be a tunable parameter). This allows us to leverage the speed of *numpy* in computing matrix operations.

- **gethurricaneloss_two**: 13.8604 seconds

As the previous version, but the losses are calculated once for all events, instead of once per event.




**Steps Forward**:

- Some further optimizations could be, e.g., checking that the event rate and/or mean and standard deviation of the countries are the same before the calculation. If they are, further parallelization optimizations are possible.
- The batching of the matrix operations in the last model is not currently optimized. Searching the optimal number to reduce computing times for the machine would be the next step.
- Further, another straightforward (but potentially costly) step would be to use a more perfoming machine.
- With a similar approach, heavy calculations could be distributed among more nodes using tools like Dask.
- The algorithm might be improved by leveraging GPUs for matrix operations.
- While I have applied a certain number of algorithmic improvements, there are likely other significant improvements that could further reduce runtimes.
- For what concerns algorithmic improvements, further optimizations could emerge for more than 2 countries.