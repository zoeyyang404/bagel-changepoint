"""We provide a Bagel interface with scikit-learn's fit and partial_fit methods."""
from __future__ import annotations

import numpy as np

try:
    from sklearn.base import BaseEstimator
    from sklearn.utils.validation import check_array, check_is_fitted
except ModuleNotFoundError as error:
    if error.name != "sklearn":
        raise
    raise ImportError(
        "bagel.sklearn requires scikit-learn. Install 'bagel-changepoint[sklearn]' "
        "from a wheel or source checkout."
    ) from error

from ._inputs import tensor_to_numpy
from .api import Result, _threshold
from .detector import Detector


class BagelDetector(BaseEstimator):
    """Analyse a series all at once or add new observations in chunks.

    Parameters
    ----------
    feature_vector, prior, transform : callable
        Model functions as described in ``bagel.Detector``. Keep them
        deterministic and unchanged while adding chunks to the same run.
    p0, p, max_particles, variance, distance, sigma, nu, iota, pruning
        Options passed to ``bagel.Detector``, with the same defaults. For
        unknown variance, we set ``variance="unknown", distance="approximate"``.
    threshold : float, default=1.0
        For a value in [0, 1), stop when the change probability is strictly
        above the threshold. Values >= 1 process all observations. Once an
        alarm stops the run, call fit to start again; partial_fit will raise.

    Attributes
    ----------
    result_ : Result
        A separate copy of the results each time we read it. ``y`` contains
        every observation received, including any unprocessed tail of the
        last chunk. Probability histories cover only processed observations.
    n_features_in_ : int
        Always 1, because the input is a univariate time series.
    n_samples_seen_ : int
        Number of observations processed so far in this run.
    n_samples_processed_last_ : int
        Number processed from the last chunk. Use this index to find any
        remaining observations if the detector stopped partway through it.
    stopped_ : bool
        Whether the stopping threshold has been exceeded.

    Notes
    -----
    We pass data with shape (n_samples,) or (n_samples, 1), in time order.
    PyTorch tensors are detached and copied to the CPU when needed. The
    optional ``y`` argument is ignored because there are no training labels.
    Inputs and probability histories are kept, so memory grows with the stream.

    After changing parameters, we start a new analysis with fit. A failure in a
    callback or the C++ code also requires a fresh fit. We can inspect parameters
    and clone an unfitted estimator. Prediction, transformation, scoring, and
    saving/restoring a fitted estimator are not yet supported.
    """

    def __init__(self, feature_vector, prior, transform, *, p0=0.9, p=1.0,
                 max_particles=100, variance="known", distance="exact",
                 sigma=1.0, nu=3.0, iota=2.0, pruning="bagel", threshold=1.0):
        # Store the original arguments so get_params and clone can recover them.
        self.feature_vector = feature_vector
        self.prior = prior
        self.transform = transform
        self.p0 = p0
        self.p = p
        self.max_particles = max_particles
        self.variance = variance
        self.distance = distance
        self.sigma = sigma
        self.nu = nu
        self.iota = iota
        self.pruning = pruning
        self.threshold = threshold

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.one_d_array = True
        return tags

    def __sklearn_is_fitted__(self):
        return hasattr(self, "n_samples_seen_")

    def _ensure_idle(self):
        if getattr(self, "_updating", False):
            raise RuntimeError("Cannot fit or change parameters from inside model callbacks")

    def set_params(self, **params):
        """Change settings, then use fit to start a new run with them."""
        self._ensure_idle()
        # BaseEstimator may partially change nested parameters before raising.
        if params and hasattr(self, "_detector"):
            self._parameters_changed = True
        return super().set_params(**params)

    @staticmethod
    def _observations(X):
        array = check_array(tensor_to_numpy(X, "X"), dtype=np.float64,
                            ensure_2d=False, ensure_all_finite=True)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError("X must have shape (n_samples,) or (n_samples, 1)")
        return array.copy()

    def _clear_run(self):
        for name in ("_detector", "_data_chunks", "_log_history", "_configuration",
                     "_threshold_value", "_cutoff", "_parameters_changed", "_failed",
                     "n_features_in_", "n_samples_seen_", "n_samples_processed_last_",
                     "stopped_"):
            self.__dict__.pop(name, None)

    def _start(self):
        configuration = self.get_params(deep=False)
        detector_options = configuration.copy()
        threshold, cutoff = _threshold(detector_options.pop("threshold"))
        detector = Detector(**detector_options)
        self._clear_run()
        self._detector = detector
        self._configuration = configuration
        self._threshold_value, self._cutoff = threshold, cutoff
        self._data_chunks, self._log_history = [], []

    def _check_configuration(self):
        if getattr(self, "_parameters_changed", False):
            raise ValueError("Parameters changed during a run; call fit to start a fresh analysis")
        for name, value in self.get_params(deep=False).items():
            original = self._configuration[name]
            if value is original:
                continue
            try:
                unchanged = bool(value == original)
            except (TypeError, ValueError):
                unchanged = False
            if not unchanged:
                raise ValueError("Parameters changed during a run; call fit to start a fresh analysis")

    def _consume(self, observations):
        self._updating = True
        consumed = 0
        stopped = False
        try:
            for value in observations:
                self._detector.update(float(value))
                current = self._detector.log_w0
                self._log_history.append(current)
                consumed += 1
                if -current > self._cutoff:
                    stopped = True
                    break
            self._data_chunks.append(observations)
            self.n_features_in_ = 1
            self.n_samples_seen_ = self._detector.time
            self.n_samples_processed_last_ = consumed
            self.stopped_ = stopped
        except BaseException:
            # Some observations may already have changed the C++ detector.
            # Discard the run because we cannot roll those updates back.
            self._clear_run()
            self._failed = True
            raise
        finally:
            self._updating = False
        return self

    def fit(self, X, y=None):
        """Start over, process X in time order, and return this estimator.

        This replaces the previous detector and history. We can add more
        observations with partial_fit afterwards unless an alarm stopped the run.
        """
        self._ensure_idle()
        observations = self._observations(X)
        self._start()
        return self._consume(observations)

    def partial_fit(self, X, y=None):
        """Add the next nonempty chunk, starting a detector on the first call.

        A wrong shape or nonfinite value is rejected before processing the
        chunk. After an alarm, call fit to start again. To keep processing
        without stopping, we set threshold=1 when creating the estimator.
        """
        self._ensure_idle()
        if getattr(self, "_failed", False):
            raise RuntimeError("A previous update failed; call fit to start a fresh analysis")
        if hasattr(self, "_detector"):
            self._check_configuration()
            if self.stopped_:
                raise RuntimeError("The analysis stopped at an alarm; call fit to start a fresh analysis")
        observations = self._observations(X)
        if not hasattr(self, "_detector"):
            self._start()
        return self._consume(observations)

    @property
    def result_(self):
        """Read a separate copy of the received data and results so far."""
        self._ensure_idle()
        check_is_fitted(self)
        detector = self._detector
        log_w0 = np.asarray(self._log_history)
        return Result(
            y=np.concatenate(self._data_chunks), threshold=self._threshold_value,
            max_particles=detector.max_particles,
            no_change_probabilities=np.exp(log_w0), log_no_change_probabilities=log_w0,
            active_weights=np.asarray(detector.active_weights),
            active_posteriors=detector.active_posteriors,
            ratios=[np.asarray(r) for r in detector.ratios],
            taus=np.asarray(detector.taus, dtype=int), time=detector.time, trace=[],
        )
