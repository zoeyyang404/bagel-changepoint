"""Fit a single-changepoint linear model and store its calculated results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import numpy as np

from .detector import Detector
from ._inputs import tensor_to_numpy


def _threshold(value: float) -> tuple[float, float]:
    value = float(value)
    if math.isnan(value) or value < 0:
        raise ValueError("threshold must be nonnegative; use 1 or inf to disable stopping")
    return value, math.inf if value >= 1 else -math.log1p(-value)


@dataclass
class Result:
    """Calculated results from a single-changepoint analysis.

    We obtain a Result through ``fit`` or a named fitting function. The change
    probability at each processed observation is
    ``1 - result.no_change_probabilities``. ``result.changepoint()`` provides
    the estimated location and first alarm time.

    ``weights`` describes the final location weights, with no change first.
    ``active_posteriors`` contains coefficient means and covariance factors
    for each retained group, plus variance parameters for unknown variance.
    Conditional coefficient covariance is observation variance times the
    covariance factor. ``taus`` and ``ratios`` describe group locations and
    relative weights within groups.

    ``y`` contains the full input, including observations after an early alarm.
    Probability histories cover processed observations; posterior quantities
    describe the final processed step. The result is separate from the detector.
    """

    y: np.ndarray
    threshold: float
    max_particles: int
    no_change_probabilities: np.ndarray
    log_no_change_probabilities: np.ndarray
    active_weights: np.ndarray
    active_posteriors: list[dict]
    ratios: list[np.ndarray]
    taus: np.ndarray
    time: int
    trace: list

    @property
    def weights(self) -> np.ndarray:
        """Weights for the change locations, with the no-change weight first."""
        return np.concatenate([w * r for w, r in zip(self.active_weights, self.ratios)])

    def changepoint(self) -> dict[str, int | None]:
        """Return the conditional MAP change location and first alarm time.

        The dictionary contains ``location`` and ``detected``, using one-based
        observation indices. A location can be estimated without an alarm. It
        is ``None`` if no positive-weight change candidates remain. Ties use
        the first maximum.

        ``detected`` is the first observation whose change probability strictly
        exceeds the threshold, including the final observation, or ``None``
        if none does. A threshold of one or greater disables early stopping.
        The comparison uses retained log weights for numerical stability.

        With bagel or LP-M pruning, a change at k begins at observation k+1.
        LP-P removes candidates, so ``location`` is a position among survivors.
        If positive change weights remain, we recover the original boundary
        with ``result.taus[1:][np.argmax(result.active_weights[1:])]``.
        """
        candidates = self.weights[1:]
        location = int(np.argmax(candidates)) + 1 if np.any(candidates > 0) else None
        _, cutoff = _threshold(self.threshold)
        alarms = np.flatnonzero(-self.log_no_change_probabilities > cutoff)
        return {"location": location, "detected": int(alarms[0]) + 1 if alarms.size else None}


def analyse(Y, threshold: float, detector: Detector, tracer: Callable | None = None) -> Result:
    """Process a nonempty, one-dimensional series using a new detector.

    ``tracer(trace, detector)`` runs after each observation and returns a list
    for the next trace. Set threshold >= 1 to process the whole series.
    The supplied detector is updated directly; to continue one that has
    already processed data, use ``detector.update(y)``.
    PyTorch input is detached and converted to CPU float64. Results use
    Python/NumPy types, and gradients cannot flow back through the analysis.
    """
    observations = np.asarray(tensor_to_numpy(Y, "Y"), dtype=float)
    if observations.ndim != 1 or observations.size == 0 or not np.isfinite(observations).all():
        raise ValueError("Y must be a nonempty one-dimensional vector of finite observations")
    if detector.time != 0:
        raise ValueError("analyse requires a fresh detector; use update to continue a stream")
    threshold, cutoff = _threshold(threshold)
    if tracer is not None and not callable(tracer):
        raise TypeError("tracer must be callable or None")
    trace: list = []
    log_w0 = []
    for y in observations:
        detector.update(float(y))
        current = detector.log_w0
        log_w0.append(current)
        if tracer is not None:
            trace = tracer(trace, detector)
            if not isinstance(trace, list):
                raise TypeError("tracer must return a list")
        if -current > cutoff:
            break
    log_w0 = np.asarray(log_w0)
    return Result(
        y=observations.copy(), threshold=threshold, max_particles=detector.max_particles,
        no_change_probabilities=np.exp(log_w0), log_no_change_probabilities=log_w0,
        active_weights=np.asarray(detector.active_weights),
        active_posteriors=detector.active_posteriors,
        ratios=[np.asarray(r) for r in detector.ratios], taus=np.asarray(detector.taus, dtype=int),
        time=detector.time, trace=trace,
    )


def fit(Y, feature_vector, prior, transform, *, threshold=1.0, tracer=None, **detector_options):
    """Fit a linear model with at most one changepoint to a time series.

    We specify ``feature_vector(t, tau)``, ``prior(t)``, and ``transform(tau)``
    as described in ``Detector``. Observations form a nonempty, finite,
    one-dimensional array in time order. NumPy arrays and PyTorch tensors
    are accepted; tensor inputs are detached and converted to CPU float64.

    For a mean change with known variance, we use ``variance="known"``
    and ``distance="exact"``. For a known-variance base model with two or
    more coefficients, we use ``distance="approximate"``. Unknown variance
    uses ``variance="unknown", distance="approximate"``, with inverse-gamma
    shape ``nu`` and scale ``iota``. These options are passed to Detector.

    A threshold in [0, 1) stops at the first change probability strictly
    above it. A threshold of one or greater, including infinity, processes
    the full series. The optional tracer is retained for older Python code;
    probability histories are recorded automatically without it.

    Returns
    -------
    Result
        The fitted result. We obtain the change-probability history from
        ``1 - result.no_change_probabilities``, the location and alarm time
        from ``result.changepoint()``, and the retained-group posterior
        parameters from ``result.active_posteriors``.
    """
    detector = Detector(feature_vector, prior, transform, **detector_options)
    return analyse(Y, threshold, detector, tracer)


def sequential_bagel_kv_exact(feature_vector, prior, transform, p0, p, sigma, max_particles, pruning="bagel"):
    """Create a sequential detector for a mean change with known variance.

    We pass each observation to ``detector.update(y)`` and apply a stopping
    rule in the observation loop. The constructor processes no observations.
    """
    return Detector(feature_vector, prior, transform, p0=p0, p=p, sigma=sigma,
                    max_particles=max_particles, pruning=pruning, variance="known", distance="exact")


def sequential_bagel_kv_approximate(feature_vector, prior, transform, p0, p, sigma, max_particles, pruning="bagel"):
    """Create a known-variance detector using a KL-based merging bound.

    We use this variant for two or more coefficients in the base model.
    The constructor processes no observations; we then call ``update``.
    """
    return Detector(feature_vector, prior, transform, p0=p0, p=p, sigma=sigma,
                    max_particles=max_particles, pruning=pruning, variance="known", distance="approximate")


def sequential_bagel_uv_exact(feature_vector, prior, transform, p0, p, nu, iota, max_particles, pruning="bagel"):
    """Legacy unknown-variance constructor using the KL-based bound.

    We use ``sequential_bagel_uv_approximate`` for new analyses. Both names
    use the same normal-inverse-gamma bound; this name is retained for
    existing Python code.
    """
    return Detector(feature_vector, prior, transform, p0=p0, p=p, nu=nu, iota=iota,
                    max_particles=max_particles, pruning=pruning, variance="unknown", distance="exact")


def sequential_bagel_uv_approximate(feature_vector, prior, transform, p0, p, nu, iota, max_particles, pruning="bagel"):
    """Create a sequential detector for unknown observation variance.

    The detector uses a normal-inverse-gamma KL-based merging bound.
    ``nu`` and ``iota`` are the inverse-gamma shape and scale for variance.
    We process observations with ``update`` and apply stopping in our loop.
    """
    return Detector(feature_vector, prior, transform, p0=p0, p=p, nu=nu, iota=iota,
                    max_particles=max_particles, pruning=pruning, variance="unknown", distance="approximate")


def bagel_kv_exact(Y, feature_vector, prior, transform, p0, p, sigma, max_particles, threshold, tracer=None, pruning="bagel"):
    """Fit a mean-change model with known observation variance.

    ``sigma`` is the observation standard deviation. The model functions
    are described in ``Detector``. We obtain a Result with change
    probabilities, the location and alarm time, and group posteriors.
    A threshold of one or greater processes the full series.
    When translating R calls, we pass ``pruning`` by name: Python retains
    an optional tracer argument before it.
    """
    return analyse(Y, threshold, sequential_bagel_kv_exact(feature_vector, prior, transform, p0, p, sigma, max_particles, pruning), tracer)


def bagel_kv_approximate(Y, feature_vector, prior, transform, p0, p, sigma, max_particles, threshold, tracer=None, pruning="bagel"):
    """Fit a known-variance linear model using a KL-based merging bound.

    We use this variant when the base model has two or more coefficients.
    The arguments and returned Result follow ``bagel_kv_exact``.
    """
    return analyse(Y, threshold, sequential_bagel_kv_approximate(feature_vector, prior, transform, p0, p, sigma, max_particles, pruning), tracer)


def bagel_uv_exact(Y, feature_vector, prior, transform, p0, p, nu, iota, max_particles, threshold, tracer=None, pruning="bagel"):
    """Legacy unknown-variance fit using the normal-inverse-gamma KL bound.

    We use ``bagel_uv_approximate`` for new analyses. Both names use the
    same bound; the older name is retained for existing Python code.
    """
    return analyse(Y, threshold, sequential_bagel_uv_exact(feature_vector, prior, transform, p0, p, nu, iota, max_particles, pruning), tracer)


def bagel_uv_approximate(Y, feature_vector, prior, transform, p0, p, nu, iota, max_particles, threshold, tracer=None, pruning="bagel"):
    """Fit a linear model with unknown observation variance.

    ``nu`` and ``iota`` are the inverse-gamma prior shape and scale for
    variance. Merging uses the normal-inverse-gamma KL-based bound.
    We obtain a Result with change probabilities, the location and alarm
    time, and group posteriors. A threshold of one or greater processes
    the full series. We pass ``pruning`` by name when translating R calls.
    """
    return analyse(Y, threshold, sequential_bagel_uv_approximate(feature_vector, prior, transform, p0, p, nu, iota, max_particles, pruning), tracer)
