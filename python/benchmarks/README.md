# CPU benchmarks

We install the Bagel wheel to be measured, then run the following from the
repository root. The default run uses NumPy and requires neither a GPU nor PyTorch:

```sh
python python/benchmarks/bench_cpu.py --suite standard --repeats 5 --output python/benchmarks/results/cpu.json --profile python/benchmarks/results/cpu.prof
```

To compare NumPy with PyTorch observations, we install PyTorch and run:

```sh
python python/benchmarks/bench_cpu.py --suite standard --repeats 5 --inputs numpy torch-cpu --output python/benchmarks/results/tensors.json --profile python/benchmarks/results/tensors.prof
```

We use `--suite quick` for a shorter run. With a CUDA-enabled PyTorch build and
a GPU, we can also select `--inputs torch-cuda`. Bagel still runs on the CPU;
that option measures the extra cost of bringing GPU data back to the CPU.
The script reports an error if a requested input type is unavailable.

## What is measured

The starting case is a mean change with 1,000 observations, a 32-particle limit,
and one coefficient before the change. The other cases change one setting
at a time:

| Case | Change from the baseline |
| --- | --- |
| Long series | 5,000 observations |
| Many particles | 128-particle limit |
| Four coefficients | 4 no-change / 8 joint coefficients |
| Sixteen coefficients | 16 no-change / 32 joint coefficients |
| Eight series | Eight independent series processed serially |
| Exact mean | Exact rather than approximate merging distance |
| Unknown variance | Unknown rather than known observation variance |

Each case uses `bagel` pruning, bounded regressors, Gaussian noise from a fixed
random seed, and a mean change halfway through the series. Early stopping is
turned off so every observation is processed. These are synthetic examples;
we also evaluate representative application data when assessing performance.

The JSON file keeps every timing, along with the median, range, throughput,
and details of the machine and software used. Each timed fit creates a new
detector, converts the input, calls the model functions, processes the data,
and returns the results. A warmup runs first, and garbage collection stays
enabled. Imports, data generation, tensor creation, and correctness checks
happen outside the timed part.

Tensor observations use float32 with gradient tracking enabled; NumPy inputs
use float64. The data are rounded to float32 first so both contain exactly
the same values. Results are checked against the NumPy fit before timing.
Both runs use NumPy model functions, which keeps the comparison focused on
how observations are supplied. Tensor model functions are covered by the
integration tests.

The script also measures input conversion separately and runs an additional
NumPy fit to time individual updates and model callbacks. We do not add those
measurements to the main fit time because they come from different runs. Time outside
the callback bodies includes C++ calculations, validation, conversion,
Python calls, and the timers themselves.

When we supply `--profile`, we can inspect a separate profile of the mean
case with:

```sh
python -m pstats python/benchmarks/results/tensors.prof
```

At the prompt, we enter `sort cumulative` followed by `stats 20`. This shows where time
is spent in Python and in calls to the extension. It does not show the work
inside individual C++ functions.

For a controlled before-and-after comparison, we keep the machine, build,
power settings, data, and thread settings fixed. We minimise other workloads
and examine variation between runs. CI saves timings but does not use
them as pass/fail limits. Generated results are ignored by Git; the
[first local measurements](baseline-2026-11-12.md) are saved as a reference.
