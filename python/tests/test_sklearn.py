"""Estimator lifecycle, chunk boundaries, and numerical equivalence."""
import numpy as np
import pytest
import bagel

pytest.importorskip("sklearn")
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags
from sklearn.utils.validation import check_is_fitted

from bagel.sklearn import BagelDetector


def assert_results_equal(actual, expected):
    assert actual.time == expected.time
    assert actual.threshold == expected.threshold
    assert actual.max_particles == expected.max_particles
    assert actual.changepoint() == expected.changepoint()
    for name in ("y", "no_change_probabilities", "log_no_change_probabilities",
                 "active_weights", "weights", "taus"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   rtol=2e-9, atol=2e-11)
    assert len(actual.ratios) == len(expected.ratios)
    for left, right in zip(actual.ratios, expected.ratios):
        np.testing.assert_allclose(left, right, rtol=2e-9, atol=2e-11)
    assert len(actual.active_posteriors) == len(expected.active_posteriors)
    for left, right in zip(actual.active_posteriors, expected.active_posteriors):
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_allclose(left[key], right[key], rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize("variance", ["known", "unknown"])
@pytest.mark.parametrize("distance", ["exact", "approximate"])
@pytest.mark.parametrize("pruning", ["bagel", "LP-M", "LP-P"])
def test_batch_chunks_and_single_observations_match(model, variance, distance, pruning):
    data = .2 * np.sin(np.arange(60)) + np.where(np.arange(60) >= 25, 4., 0.)
    options = dict(variance=variance, distance=distance, pruning=pruning, max_particles=5)
    expected = bagel.fit(data, *model, **options)
    whole = BagelDetector(*model, **options)
    assert whole.fit(data[:, None]) is whole
    assert_results_equal(whole.result_, expected)
    chunked = BagelDetector(*model, **options)
    for start, end in ((0, 1), (1, 9), (9, 37), (37, 60)):
        assert chunked.partial_fit(data[start:end]) is chunked
        assert chunked.n_samples_processed_last_ == end - start
    single = BagelDetector(*model, **options)
    for value in data:
        single.partial_fit([value])
    assert_results_equal(chunked.result_, expected)
    assert_results_equal(single.result_, expected)
    assert whole.n_features_in_ == chunked.n_features_in_ == 1
    assert whole.n_samples_seen_ == chunked.n_samples_seen_ == 60
    assert not whole.stopped_ and not chunked.stopped_


def test_fit_resets_and_partial_fit_can_follow_fit(model):
    data = np.array([0., .1, 3., 3.1, 2.9])
    estimator = BagelDetector(*model, max_particles=5)
    estimator.fit(data[:2]).partial_fit(data[2:, None])
    assert_results_equal(estimator.result_, bagel.fit(data, *model, max_particles=5))
    assert estimator.fit([42.], y=[99]) is estimator  # y is ignored.
    assert_results_equal(estimator.result_, bagel.fit([42.], *model, max_particles=5))
    assert estimator.n_samples_seen_ == estimator.n_samples_processed_last_ == 1


def test_callback_indices_do_not_restart_at_chunk_boundaries(model):
    calls = []
    def record(name, callback):
        def wrapped(*args):
            calls.append((name, args))
            return callback(*args)
        return wrapped
    callbacks = tuple(record(name, cb) for name, cb in zip(("feature", "prior", "transform"), model))
    data = np.sin(np.arange(12))
    BagelDetector(*callbacks, max_particles=5).fit(data)
    batch_calls = calls.copy()
    calls.clear()
    estimator = BagelDetector(*callbacks, max_particles=5)
    estimator.partial_fit(data[:3]).partial_fit(data[3:7]).partial_fit(data[7:])
    assert calls == batch_calls


@pytest.mark.parametrize("split", [1, 2, 3, 4])
def test_early_stop_is_sticky_and_reports_unconsumed_tail(model, split):
    data = np.array([0., .1, 1000., 1001., 1002.])
    expected = bagel.fit(data, *model, threshold=.95)
    estimator = BagelDetector(*model, threshold=.95)
    estimator.partial_fit(data[:split])
    if not estimator.stopped_:
        estimator.partial_fit(data[split:])
        assert estimator.n_samples_processed_last_ == 3 - split
    else:
        assert estimator.n_samples_processed_last_ == 3
    snapshot = estimator.result_
    assert estimator.stopped_ and estimator.n_samples_seen_ == 3
    assert snapshot.changepoint()["detected"] == 3
    np.testing.assert_allclose(snapshot.log_no_change_probabilities,
                               expected.log_no_change_probabilities)
    # Received data include the unprocessed tail, but not chunks never sent.
    np.testing.assert_array_equal(snapshot.y, data if split < 3 else data[:split])
    with pytest.raises(RuntimeError, match="stopped"):
        estimator.partial_fit([1003.])
    assert_results_equal(estimator.result_, snapshot)
    estimator.fit([0.])
    assert not estimator.stopped_ and estimator.n_samples_seen_ == 1


@pytest.mark.parametrize("threshold", [0., .95, 1., float("inf")])
def test_batch_threshold_matches_existing_api(model, threshold):
    data = [0., 1000., 1001., 999.]
    estimator = BagelDetector(*model, threshold=threshold).fit(data)
    expected = bagel.fit(data, *model, threshold=threshold)
    assert_results_equal(estimator.result_, expected)
    assert estimator.stopped_ == (threshold < 1)


def test_clone_parameters_and_fitted_detection(model):
    estimator = BagelDetector(*model, max_particles=5)
    assert set(vars(estimator)) == set(estimator.get_params(deep=False))
    with pytest.raises(NotFittedError):
        check_is_fitted(estimator)
    with pytest.raises(NotFittedError):
        _ = estimator.result_
    estimator.fit([0., .1, 3.])
    check_is_fitted(estimator)
    fresh = clone(estimator)
    assert fresh.get_params() == estimator.get_params()
    assert not hasattr(fresh, "n_samples_seen_")
    with pytest.raises(NotFittedError):
        _ = fresh.result_
    assert fresh.set_params(max_particles=7) is fresh
    fresh.fit([4.])
    assert fresh.result_.max_particles == 7 and estimator.result_.max_particles == 5
    tags = get_tags(estimator)
    assert tags.input_tags.one_d_array and tags.input_tags.two_d_array
    assert not tags.input_tags.sparse
    assert tags.estimator_type is None


@pytest.mark.parametrize("change", [
    lambda e: e.set_params(max_particles=7),
    lambda e: e.set_params(threshold=.9),
    lambda e: setattr(e, "sigma", 2.),
])
def test_parameter_changes_require_refit(model, change):
    estimator = BagelDetector(*model, max_particles=5).fit([0., .1])
    snapshot = estimator.result_
    change(estimator)
    with pytest.raises(ValueError, match="Parameters changed"):
        estimator.partial_fit([3.])
    assert_results_equal(estimator.result_, snapshot)
    estimator.fit([0.])
    assert estimator.n_samples_seen_ == 1


@pytest.mark.parametrize("data", [[], np.empty((0, 1)), [[1., 2.]], np.zeros((2, 1, 1)),
                                   [np.nan], [np.inf], [1 + 2j], ["invalid"]])
def test_invalid_chunk_does_not_mutate_fitted_state(model, data):
    estimator = BagelDetector(*model).fit([0., .1])
    snapshot = estimator.result_
    with pytest.raises((ValueError, TypeError)):
        estimator.partial_fit(data)
    assert_results_equal(estimator.result_, snapshot)
    estimator.partial_fit([3.])
    assert estimator.n_samples_seen_ == 3


@pytest.mark.parametrize("options", [{"threshold": -1}, {"threshold": np.nan},
                                    {"max_particles": 2}, {"sigma": 0},
                                    {"pruning": "invalid"}])
def test_configuration_validation_at_fit_time(model, options):
    estimator = BagelDetector(*model, **options)
    assert estimator.get_params()[next(iter(options))] is options[next(iter(options))]
    with pytest.raises((ValueError, TypeError)):
        estimator.fit([0.])
    with pytest.raises(NotFittedError):
        check_is_fitted(estimator)


def test_inputs_and_returned_snapshots_are_independent(model):
    data = np.array([0., .1, 3.])
    estimator = BagelDetector(*model).fit(data)
    saved = estimator.result_
    data[:] = 100
    altered = estimator.result_
    altered.y[:] = 20
    altered.log_no_change_probabilities[:] = 123
    altered.active_posteriors[0]["mu"][:] = 456
    assert_results_equal(estimator.result_, saved)
    estimator.partial_fit([3.1, 2.9])
    assert saved.time == 3
    np.testing.assert_array_equal(saved.y, [0., .1, 3.])
    assert estimator.result_.time == 5


def test_callback_failure_invalidates_partial_run_until_refit(model):
    feature, prior, transform = model
    failing = True
    def feature_callback(t, tau):
        if failing and t == 4:
            raise RuntimeError("model failure")
        return feature(t, tau)
    estimator = BagelDetector(feature_callback, prior, transform).fit([0., .1])
    with pytest.raises(RuntimeError, match="model failure"):
        estimator.partial_fit([.2, .3, .4])
    with pytest.raises(NotFittedError):
        _ = estimator.result_
    with pytest.raises(RuntimeError, match="previous update failed"):
        estimator.partial_fit([.3])
    failing = False
    estimator.fit([0., .1, .2, .3])
    assert estimator.n_samples_seen_ == 4


@pytest.mark.parametrize("operation", ["fit", "partial_fit", "set_params"])
def test_reentrant_estimator_mutation_is_rejected(model, operation):
    feature, prior, transform = model
    def callback(t, tau):
        if operation == "set_params":
            estimator.set_params(sigma=2)
        else:
            getattr(estimator, operation)([0.])
        return feature(t, tau)
    estimator = BagelDetector(callback, prior, transform)
    with pytest.raises(RuntimeError, match="inside model callbacks"):
        estimator.fit([0.])
    with pytest.raises(NotFittedError):
        check_is_fitted(estimator)


def test_tensors_work_with_estimator_and_optional_callbacks(model):
    torch = pytest.importorskip("torch")
    feature, prior, transform = model
    tensor = lambda x: torch.tensor(x, dtype=torch.float64, requires_grad=True)
    callbacks = (lambda t, tau: tensor(feature(t, tau)),
                 lambda t: {k: tensor(v) for k, v in prior(t).items()},
                 lambda tau: tensor(transform(tau)))
    data = torch.tensor([0., .1, -.1, 3., 3.1, 2.9], requires_grad=True)
    estimator = BagelDetector(*callbacks, max_particles=5)
    estimator.partial_fit(data[:2, None]).partial_fit(data[2:])
    expected = bagel.fit(data, *model, max_particles=5)
    assert_results_equal(estimator.result_, expected)
    assert data.grad is None
