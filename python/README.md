# Bagel for Python

Bagel is a Bayesian online changepoint detection algorithm for linear models
based on posterior merging. We can analyse a univariate time series or update
a detector as observations arrive. Each run models at most one changepoint.
The R and Python packages use the same C++ algorithm.

We import the package as `bagel`. The Python distribution is provisionally called
`bagel-changepoint`; a PyPI release is being prepared. The method is described by
[Yang, Fearnhead, and Grose (2026)](https://arxiv.org/abs/2611.03230v1).

## Choosing a model

We select the variance model and merging distance through the following options
to `bagel.fit()`, `bagel.Detector()`, or `BagelDetector`:

| Model | Options | Corresponding fitting function |
| --- | --- | --- |
| Change in mean, known observation variance | `variance="known", distance="exact"` | `bagel_kv_exact()` |
| Known variance, two or more coefficients in the base model | `variance="known", distance="approximate"` | `bagel_kv_approximate()` |
| Unknown observation variance | `variance="unknown", distance="approximate"` | `bagel_uv_approximate()` |

The base model describes the series before a change. The names `exact` and
`approximate` refer to the distance used to select merges. Merging itself
approximates the posterior distribution. For unknown variance, we use the
normal-inverse-gamma KL-based bound.

## Install and use

We can install a compatible wheel file (`.whl`) directly. For example,
with Python 3.14 on 64-bit Windows:

```sh
python -m pip install ./bagel_changepoint-1.8.2-cp314-cp314-win_amd64.whl
```

We select a wheel that matches the Python version, operating system, and
architecture, then run the [example below](#mean-change-example). If a suitable
wheel is unavailable, we build from source as described next.

## Build from source

We require Python 3.10 or newer and NumPy to use Bagel. A source build also
requires a C++20 compiler, Eigen headers (3.4 or newer), and Boost headers
(1.75 or newer). The build installs CMake and pybind11 automatically; R is not required.

On Debian or Ubuntu, we install `libeigen3-dev` and `libboost-math-dev`. On macOS,
we use `brew install eigen boost`. On Windows, we install Visual Studio Build
Tools with the C++ workload, followed by Eigen and Boost, for example through
vcpkg. If the build cannot find the headers, we specify their locations explicitly:

```sh
python -m pip install ./python \
  -Ccmake.define.EIGEN3_INCLUDE_DIR=/path/to/eigen3 \
  -Ccmake.define.BOOST_INCLUDE_DIR=/path/to/boost/include
```

We run this command from the repository root. The two paths should contain `Eigen/` and
`boost/`, respectively. If the compiler can already find those directories,
`python -m pip install ./python` is enough. These are build requirements;
installing a wheel does not need the compiler or headers. CMake does not
download missing source dependencies.

## Mean-change example

In this example, we analyse a series that starts near zero and then rises
to approximately three. We specify the features, coefficient priors, and
transformation used to compare possible changes through three model functions.

```python
import numpy as np
import bagel

def feature(t, tau):
    return np.array([1.0]) if tau == 0 else np.array([1.0, 1.0])

def prior(t):
    if t == 1:
        return {"mu": np.zeros(1), "sigma": np.eye(1)}
    return {"mu": np.zeros(2), "sigma": np.array([[1., -1.], [-1., 2.]])}

def transform(tau):
    return np.array([[1.]]) if tau == 0 else np.array([[1., 1.]])

y = np.array([0., 0.1, -0.1, 0.2, 3., 3.1, 2.9, 3.2])
result = bagel.fit(y, feature, prior, transform,
                   p0=0.9, p=1, sigma=1, max_particles=5,
                   variance="known", distance="exact", threshold=float("inf"))
print(1 - result.no_change_probabilities)
print(result.changepoint())
print(result.active_posteriors)
```

With unknown observation variance, we use the same model functions and specify
an inverse-gamma prior for the variance:

```python
unknown = bagel.fit(y, feature, prior, transform,
                   variance="unknown", distance="approximate",
                   p0=0.9, p=1, nu=3, iota=2, max_particles=5, threshold=0.95)
print(1 - unknown.no_change_probabilities)
print(unknown.changepoint())
```

## Reading the results

| Quantity | Python expression | R expression |
| --- | --- | --- |
| Change probability at each processed observation | `1 - result.no_change_probabilities` | `1 - weights_tau_zero(fit)` |
| Estimated change location and first alarm time | `result.changepoint()` | `changepoint(fit)` |
| Final weights over change locations | `result.weights` | `weights(fit)` |
| Coefficient and variance parameters for each retained group | `result.active_posteriors` | `active_posteriors(fit)` |
| Active group weights and representative locations | `result.active_weights`, `result.taus` | `active_weights(fit)`, `taus(fit)` |
| Relative weights within groups | `result.ratios` | `ratios(fit)` |
| Number of processed observations | `result.time` | `bagelR::time(fit)` |

`result.weights` describes the final distribution over candidate locations;
the first entry is the no-change weight. The history of change probabilities
comes from `1 - result.no_change_probabilities`.

`result.changepoint()` returns a dictionary with `location` and `detected`.
The location is the maximum a posteriori estimate conditional on a change;
it can be available even if no alarm occurred. Ties use the first maximum.
`location` is `None` if there are no positive-weight change candidates.
`detected` is the first observation whose change probability strictly exceeds
the threshold, or `None` if none does. An alarm at the final observation is
included. With the default merging strategy, a change at `location=k` begins
at observation `k+1`; the LP-P exception is described below.

Each dictionary in `result.active_posteriors` contains `tau`, `weight`, the
coefficient mean `mu`, and a covariance factor `sigma`. The no-change particle
comes first. Unknown-variance models also include inverse-gamma shape `nu` and
scale `iota`. Conditional coefficient covariance is the observation variance
times the covariance factor; for known variance, the multiplier is the
constructor's `sigma**2`.

`result.y` contains the full input, including observations after an early alarm.
Probability histories cover only processed observations. Weights and posterior
parameters describe the final processed step.

## Updating a sequential detector

We create the detector once and call `update()` for each new observation. Each
call updates the changepoint weights and parameter posteriors inside that same
detector. The direct return is one number: the current no-change weight.

```python
detector = bagel.Detector(feature, prior, transform,
                         variance="known", distance="exact",
                         p0=0.9, p=1, sigma=1, max_particles=5)
change_probabilities = []
for observation in y:
    no_change = detector.update(observation)
    change_probabilities.append(1 - no_change)
    if 1 - no_change > 0.95:
        break

print(change_probabilities)
print(detector.weights)
print(detector.active_posteriors)
print(detector.time)
```

We retain `detector` for subsequent observations and read its results through
these properties. We apply any stopping rule in the observation loop;
`Detector.update()` does not stop automatically. It stores the current state,
so we record a probability history explicitly, as above. To analyse multiple
changes, we need a procedure for restarting or combining detectors.

## Optional scikit-learn estimator

We use `BagelDetector` for the `fit` and `partial_fit` interface. We install
scikit-learn with `python -m pip install "scikit-learn>=1.6"` alongside the
Bagel wheel. For a source install, we use
`python -m pip install "./python[sklearn]"` with the build requirements above.
Scikit-learn is imported only when we use this optional interface.

We apply this interface to the model functions and NumPy `y` defined above:

```python
from bagel.sklearn import BagelDetector
from sklearn.base import clone

model = BagelDetector(feature, prior, transform,
                      max_particles=5, threshold=1)
model.fit(y)  # Start a fresh analysis.
print(model.result_.changepoint())

stream = clone(model)  # Start another detector with the same settings.
stream.partial_fit(y[:4])
stream.partial_fit(y[4:])
print(stream.result_.no_change_probabilities)
print(stream.n_samples_seen_)
```

`fit` starts a new analysis each time. `partial_fit` retains the current detector
and adds the next chunk; it can also initialise a detector on its first call. We can
continue after `fit` with `partial_fit` as long as the run has not stopped.
Both methods return the estimator itself.

We pass a nonempty array of shape `(n_samples,)` or `(n_samples, 1)`. PyTorch
tensors are also supported, as described below. Chunks must be consecutive parts of the
same series, in time order. The observation indices passed to our model
functions carry on across chunk boundaries. In `fit(X, y=None)` and
`partial_fit(X, y=None)`, we supply the observations in `X`; the optional `y`
argument is ignored because there are no training labels.

### Stopping and continuing

The default `threshold=1` processes every observation. With a threshold in
`[0, 1)`, the detector stops as soon as the change probability is strictly
above it. That can happen partway through a chunk. `stopped_` becomes `True`,
and later calls to `partial_fit` raise an error. We call `fit` to start a new
analysis, or select a threshold of at least one initially to process all data.

We obtain the total number of processed observations from `n_samples_seen_`.
`n_samples_processed_last_` counts those processed from the last chunk. If the
detector stopped early, the remaining data are
`chunk[model.n_samples_processed_last_:]`.

### Reading results and changing settings

When we read `result_`, we obtain a separate copy of the results at that point.
Its `y` contains all observations supplied so far, including any unprocessed
tail of the last chunk. The probability histories cover only the observations
that were processed. The estimator keeps these histories in memory, and each
snapshot copies them. We therefore request snapshots only when needed.

We use `get_params` to inspect the settings and `set_params` to change them.
After changing settings, we start a new run with `fit`; `partial_fit` will
reject changes to a running model. It also checks for direct changes to the
top-level parameters. Changes inside a callback cannot be tracked, so we keep
our model functions deterministic and unchanged during a run. Previously saved
results continue to describe the settings used to produce them.

An invalid input array is rejected before processing that chunk. If a model
callback or the C++ code fails partway through a chunk, the run cannot be
resumed. We resolve the error and call `fit` again.

This interface supports fitting, incremental updates, parameter handling,
and cloning. It does not yet provide `predict`, `transform`, `score`, or saving
and restoring a fitted estimator. Broader scikit-learn pipeline and model
selection support has not been established. The source checkout contains separate compatibility notes for older code.

## PyTorch tensor inputs

We can pass PyTorch tensors directly to Bagel. We install a suitable PyTorch
build using its [installation guide](https://pytorch.org/get-started/locally/),
then use the same model functions as before:

```python
import torch

y = torch.tensor([0., 0.1, -0.1, 0.2, 3., 3.1, 2.9, 3.2],
                 requires_grad=True)
result = bagel.fit(y, feature, prior, transform, max_particles=5)
print(result.no_change_probabilities)  # NumPy array

detector = bagel.Detector(feature, prior, transform, max_particles=5)
detector.update(y[0])  # A zero-dimensional tensor is a scalar observation.
```

The fitting functions take a nonempty, one-dimensional tensor of observations. `Detector.update` takes
one scalar tensor. Our model functions can also return tensors for features,
prior `mu` and `sigma`, and transforms.

Bagel detaches tensors from their gradient graph and converts them to CPU
`float64` data. Calculations therefore run on the CPU even when the input
originates on a GPU, and gradients cannot flow back through Bagel. The input
is unchanged; results use Python and NumPy types, with separate copies
of result arrays. PyTorch is not installed or imported for NumPy-only use.

Dense, real tensors are supported, including integers, booleans, floating-point
types such as `bfloat16`, noncontiguous views, and tensors with
`requires_grad=True`. Complex, sparse, quantized, nested, and meta tensors are
rejected. Inputs must still have the right shape and contain finite values.

A batch of GPU observations is copied to the CPU once. A GPU scalar or callback
result is copied on each call, so we use batch input and CPU model functions when
practical. The conversion follows PyTorch's
[detach, CPU, and NumPy operations](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html),
with a cast to float64 on the CPU to handle types such as bfloat16. The source checkout includes a benchmark guide at `benchmarks/README.md`
for measuring this cost.

## Model functions and results

`Detector(feature_vector, prior, transform, *, p0=0.9, p=1,
max_particles=100, variance="known", distance="exact", sigma=1,
nu=3, iota=2, pruning="bagel")` creates a fresh streaming detector.

- Observation indices start at **one**, matching R. `tau=0` means no change;
  `tau=k` means the change starts at observation `k+1`.
- `feature_vector(t, tau)` returns a finite column vector. One-dimensional
  NumPy arrays are accepted. With `d` no-change and `q` additional change
  coefficients, its length is `d` for `tau=0`, otherwise `d+q`.
- `prior(1)` returns the no-change prior; `prior(t)` for `t>1` returns the
  joint coefficient prior. Return a dictionary with `mu` (column vector)
  and `sigma` (symmetric positive-definite matrix of matching dimension).
  Conditional coefficient covariance is the noise variance times this
  `sigma` matrix. Coefficient dimensions must stay fixed during a run.
- `transform(tau)` returns a two-dimensional matrix with one column per
  coefficient, mapping to the post-change coefficients used for merging.
  All change candidates must have the same transformed dimension.
- Known-variance models take the noise **standard deviation** `sigma`.
  Unknown-variance models use `variance="unknown"` with inverse-gamma
  shape `nu` and scale `iota` for the noise variance.
- For a mean change with known variance, `distance="exact"` uses the scalar
  Gaussian total variation distance. We select `distance="approximate"` for
  the other model settings in the table above; it uses a KL-based bound.
- `pruning` accepts `"bagel"`, `"LP-M"`, and `"LP-P"`. `max_particles` must
  be an integer at least three. `p0` must lie in `(0, 1]`; `p` is retained
  for compatibility but currently does not affect the uniform location prior.
- With a batch `threshold` in `[0, 1)`, processing stops when the change
  probability is strictly above it. `threshold=1` or `float("inf")` processes all data.
  The comparison uses log probabilities. The first no-change weight is `p0`.

When translating code from R, we can use `bagel_kv_exact`,
`bagel_kv_approximate`, and `bagel_uv_approximate`, or their `sequential_`
constructors. We pass options such as `pruning` by name because the Python
fitting functions retain an additional optional argument for older code.
A missing alarm or location is represented by `None` in Python.

Detector properties include `time` (observations processed), `max_particles`,
`no_change_probability`, `log_w0`, `active_weights`, `log_weights`, `weights`,
`taus`, `ratios`, and `active_posteriors`. The posterior dictionaries contain
`tau`, `weight`, `mu`, `sigma`, and, for unknown variance, `nu` and `iota`.
The returned values are copies of the detector's state at the time of access.
Assigning the detector itself to another variable still refers to the same
detector; we create a new one for an independent analysis.

Model callbacks run with the Python GIL held. We keep them deterministic
and avoid modifying the detector inside a callback.

### A note on LP-P pruning

`LP-P` removes candidates without immediately renormalising the remaining
weights. Removed candidates are also absent from `weights`. This means
`result.changepoint()["location"]` gives a position among the surviving
candidates, as it does in R. To find the original observation location, we use
`result.taus[1:][np.argmax(result.active_weights[1:])]` when positive change
weights exist. This limitation does not affect the default `bagel` strategy.

## Development and releases

When changing the algorithm, we edit `../c++/` and run `python tools/sync_core.py`
from the repository root. This updates the copy in `vendor/bagel/`, which lets
the Python source archive build independently. We include that copy with the
changes and make subsequent edits in `c++/` to keep R and Python consistent.

```sh
python -m pip install build pytest
python -m pip install ./python
python -m pytest python/tests
python -m build python
```

`python -m build python` builds a source archive and then builds a wheel from
that archive. We use the same CMake settings above for nonstandard header paths.
We review changes to both packages before releasing them. These commands
build and test locally; publishing is a separate step.
