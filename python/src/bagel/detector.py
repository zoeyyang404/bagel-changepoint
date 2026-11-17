"""Process observations one at a time, using NumPy or PyTorch inputs."""
from __future__ import annotations

from numpy import ndarray

from ._core import Detector as _NativeDetector
from ._inputs import tensor_to_numpy


class Detector(_NativeDetector):
    """A sequential changepoint detector for a univariate linear model.

    We create the detector once and call ``update(y)`` for observations in
    time order. Each update changes the same detector and returns one number:
    the current no-change weight. Updated location weights and posterior
    parameters are read through ``weights`` and ``active_posteriors``.
    The constructor processes no observations and applies no stopping rule.

    Model functions use one-based observation indices. ``tau=0`` means no
    change; ``tau=k`` means a change beginning at observation k+1.
    ``feature_vector(t, tau)`` returns a column vector, ``prior(t)`` returns
    a dictionary with mean ``mu`` and covariance factor ``sigma``, and
    ``transform(tau)`` maps coefficients to the post-change coefficients
    compared during merging. The README defines their dimensions.

    For a mean change with known variance, we use ``variance="known"`` and
    ``distance="exact"``. We use ``distance="approximate"`` for other
    known-variance models or with ``variance="unknown"``. The constructor's
    ``sigma`` is a noise standard deviation; ``nu`` and ``iota`` describe an
    inverse-gamma prior on unknown variance.

    Model functions may return NumPy arrays or PyTorch tensors. Tensors are
    detached and converted to CPU float64; calculations do not track gradients.
    Results use Python and NumPy types.
    """

    def __init__(self, feature_vector, prior, transform, *, p0=0.9, p=1.0,
                 max_particles=100, variance="known", distance="exact",
                 sigma=1.0, nu=3.0, iota=2.0, pruning="bagel"):
        for name, callback in (("feature_vector", feature_vector),
                               ("prior", prior), ("transform", transform)):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")

        def feature_input(t, tau):
            return tensor_to_numpy(feature_vector(t, tau), "feature_vector result")

        def prior_input(t):
            value = prior(t)
            # Leave malformed return values to the native validator.
            if isinstance(value, dict) and "mu" in value and "sigma" in value:
                if isinstance(value["mu"], ndarray) and isinstance(value["sigma"], ndarray):
                    return value
                return {"mu": tensor_to_numpy(value["mu"], "prior mu"),
                        "sigma": tensor_to_numpy(value["sigma"], "prior sigma")}
            return value

        def transform_input(tau):
            return tensor_to_numpy(transform(tau), "transform result")

        super().__init__(feature_input, prior_input, transform_input, p0=p0, p=p,
                         max_particles=max_particles, variance=variance,
                         distance=distance, sigma=sigma, nu=nu, iota=iota,
                         pruning=pruning)

    def update(self, y):
        """Incorporate one observation and return the current no-change weight.

        ``w0 = detector.update(y)`` assigns one number to ``w0``; ``1 - w0``
        is the current change probability. The same call updates changepoint
        weights and parameter posteriors inside ``detector``. We read these
        through ``detector.weights``, ``detector.active_weights``, and
        ``detector.active_posteriors``, and retain the detector for later data.

        We apply any stopping threshold and record any probability history in
        our observation loop. A tensor observation must be zero-dimensional.
        For a complete series on the GPU, ``fit`` transfers observations to
        the CPU in one operation.
        """
        converted = tensor_to_numpy(y, "y")
        if converted is not y:
            if converted.ndim != 0:
                raise ValueError("y must be a scalar tensor; use fit for a series")
            converted = float(converted)
        return super().update(converted)

    def set_weights(self, weights):
        """Set one nonnegative weight per active particle, summing to one."""
        return super().set_weights(tensor_to_numpy(weights, "weights"))

    def set_log_weights(self, log_weights):
        """Set one log weight per active particle without normalising."""
        return super().set_log_weights(tensor_to_numpy(log_weights, "log_weights"))
