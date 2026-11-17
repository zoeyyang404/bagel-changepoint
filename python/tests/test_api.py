import gc
import math

import numpy as np
import pytest
import bagel


@pytest.mark.parametrize("variance", ["known", "unknown"])
@pytest.mark.parametrize("distance", ["exact", "approximate"])
@pytest.mark.parametrize("pruning", ["bagel", "LP-M", "LP-P"])
def test_streaming_matches_batch(model, variance, distance, pruning):
    y = .2 * np.sin(np.arange(1, 61)) + np.where(np.arange(60) >= 25, 4., 0.)
    kwargs = dict(variance=variance, distance=distance, pruning=pruning, max_particles=5)
    result = bagel.fit(y, *model, threshold=math.inf, **kwargs)
    detector = bagel.Detector(*model, **kwargs)
    history = [detector.update(value) for value in y]
    np.testing.assert_allclose(result.no_change_probabilities, history)
    np.testing.assert_allclose(result.weights, detector.weights)
    assert result.time == detector.time == len(y)
    assert len(detector.active_weights) <= 5


def test_log_stopping_and_trace(model):
    y = np.array([0., 1000., 1001., 999., 1000.])
    all_data = bagel.fit(y, *model, threshold=1)
    assert np.isfinite(all_data.log_no_change_probabilities).all()
    assert all_data.changepoint()["detected"] is None
    assert all_data.time == len(y)
    result = bagel.fit(y, *model, threshold=.95,
                       tracer=lambda trace, d: trace + [d.time])
    assert result.changepoint()["detected"] == 2
    assert result.trace == [1, 2]
    assert len(result.y) == len(y)
    assert len(result.no_change_probabilities) == 2


def test_final_observation_alarm_and_single_observation(model):
    result = bagel.fit([0., 1000.], *model, threshold=.95)
    assert result.changepoint() == {"location": 1, "detected": 2}
    assert bagel.fit([0.], *model).changepoint() == {"location": None, "detected": None}


def test_returned_arrays_are_owned_snapshots(model):
    detector = bagel.Detector(*model)
    detector.update(0.)
    snapshot = detector.active_posteriors
    original = snapshot[0]["mu"].copy()
    detector.update(10.)
    del detector
    gc.collect()
    np.testing.assert_array_equal(snapshot[0]["mu"], original)


def test_invalid_callback_does_not_advance_state(model):
    feature, prior, transform = model
    invalid = True
    def sometimes_invalid(t, tau):
        return np.ones(3) if invalid else feature(t, tau)
    detector = bagel.Detector(sometimes_invalid, prior, transform)
    with pytest.raises(ValueError, match="dimensions"):
        detector.update(0.)
    assert detector.time == 0
    invalid = False
    assert detector.update(0.) == .9


def test_callback_lifetime_and_exception(model):
    feature, prior, transform = model
    def exploding(t, tau):
        raise RuntimeError("callback error")
    detector = bagel.Detector(exploding, prior, transform)
    del exploding
    gc.collect()
    with pytest.raises(RuntimeError, match="callback error"):
        detector.update(0.)
    assert detector.time == 0


def test_reentrant_update_rejected(model):
    feature, prior, transform = model
    def callback(t, tau):
        detector.update(1.)
        return feature(t, tau)
    detector = bagel.Detector(callback, prior, transform)
    with pytest.raises(ValueError, match="recursively"):
        detector.update(0.)
    assert detector.time == 0


@pytest.mark.parametrize("kwargs", [
    {"p0": 0}, {"p0": np.nan}, {"max_particles": 2}, {"sigma": 0},
    {"variance": "unknown", "nu": -1}, {"pruning": "invalid"},
    {"variance": "invalid"}, {"distance": "invalid"},
])
def test_invalid_configuration(model, kwargs):
    with pytest.raises((ValueError, TypeError)):
        bagel.Detector(*model, **kwargs)


@pytest.mark.parametrize("y", [[], [np.nan], [np.inf], [[0., 1.]]])
def test_invalid_series(model, y):
    with pytest.raises(ValueError):
        bagel.fit(y, *model)


@pytest.mark.parametrize("threshold", [-1, np.nan])
def test_invalid_threshold(model, threshold):
    with pytest.raises(ValueError):
        bagel.fit([0.], *model, threshold=threshold)


def test_weight_setters_and_uninitialised_access(model):
    detector = bagel.Detector(*model)
    with pytest.raises(ValueError, match="observation"):
        _ = detector.log_w0
    detector.update(0.)
    detector.update(.1)
    original = detector.active_weights
    with pytest.raises(ValueError, match="one value"):
        detector.set_weights([1.])
    with pytest.raises(ValueError, match="nonnegative"):
        detector.set_weights([-1., 2.])
    assert detector.active_weights == original
    detector.set_weights([.25, .75])
    np.testing.assert_allclose(detector.active_weights, [.25, .75])
    detector.set_log_weights([-1000., 0.])
    assert detector.log_w0 == -1000.
    assert detector.no_change_probability == 0.


def test_analyse_rejects_reused_detector(model):
    detector = bagel.Detector(*model)
    detector.update(0.)
    with pytest.raises(ValueError, match="fresh"):
        bagel.analyse([1.], 1, detector)


def test_zero_change_mass_all_pruning_modes(model):
    for pruning in ("bagel", "LP-M", "LP-P"):
        result = bagel.fit(np.sin(np.arange(20)), *model, p0=1, max_particles=3, pruning=pruning)
        assert np.isfinite(result.weights).all()
        assert np.all(result.no_change_probabilities == 1)
        assert result.changepoint() == {"location": None, "detected": None}
