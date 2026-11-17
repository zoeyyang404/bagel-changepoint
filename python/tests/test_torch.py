"""Interop tests use real tensors; CUDA cases run only when available."""
import numpy as np
import pytest
import bagel

torch = pytest.importorskip("torch")


def tensor_model(model, device="cpu"):
    feature, prior, transform = model

    def tensor(value):
        return torch.tensor(value, dtype=torch.float64, device=device,
                            requires_grad=True)

    return (lambda t, tau: tensor(feature(t, tau)),
            lambda t: {key: tensor(value) for key, value in prior(t).items()},
            lambda tau: tensor(transform(tau)))


def assert_results_equal(actual, expected):
    assert actual.time == expected.time
    assert actual.changepoint() == expected.changepoint()
    for name in ("y", "log_no_change_probabilities", "active_weights", "weights", "taus"):
        assert isinstance(getattr(actual, name), np.ndarray)
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   rtol=2e-9, atol=2e-11)
    for left, right in zip(actual.active_posteriors, expected.active_posteriors):
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_allclose(left[key], right[key], rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize("variance", ["known", "unknown"])
@pytest.mark.parametrize("distance", ["exact", "approximate"])
@pytest.mark.parametrize("pruning", ["bagel", "LP-M", "LP-P"])
def test_tensor_observations_and_callbacks_match_numpy(model, variance, distance, pruning):
    # A noncontiguous leaf tensor with gradient tracking enabled.
    y = torch.tensor(np.repeat(np.r_[np.zeros(12), np.ones(12) * 3], 2),
                     dtype=torch.float64)[::2].requires_grad_()
    assert not y.is_contiguous()
    options = dict(variance=variance, distance=distance, pruning=pruning, max_particles=5)
    expected = bagel.fit(y.detach().numpy(), *model, **options)
    actual = bagel.fit(y, *tensor_model(model), **options)
    assert_results_equal(actual, expected)
    assert y.grad is None and y.requires_grad
    saved = actual.y.copy()
    with torch.no_grad():
        y.add_(100)
    np.testing.assert_array_equal(actual.y, saved)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32,
                                   torch.float64, torch.int64, torch.bool])
def test_tensor_dtypes(model, dtype):
    y = torch.tensor([0, 0, 1, 3, 2, 4], dtype=dtype)
    expected = bagel.fit(y.to(torch.float64).numpy(), *model)
    assert_results_equal(bagel.fit(y, *model), expected)


def test_negative_view_and_parameter(model):
    y = torch.nn.Parameter(torch.tensor([0., 0.1, 3., 3.1]))
    assert_results_equal(bagel.fit(y, *model), bagel.fit(y.detach().numpy(), *model))
    negative = torch.complex(torch.zeros_like(y), y).conj().imag
    assert negative.is_neg()
    expected = bagel.fit(-y.detach().numpy(), *model)
    assert_results_equal(bagel.fit(negative, *model), expected)


def test_noncontiguous_tensor_callback_matrices(model):
    feature, prior, transform = model

    def tensor(value):
        return torch.tensor(np.repeat(value, 2, axis=-1), dtype=torch.float64,
                            requires_grad=True)[..., ::2]

    callbacks = (lambda t, tau: tensor(feature(t, tau)),
                 lambda t: {key: tensor(value) for key, value in prior(t).items()},
                 lambda tau: tensor(transform(tau)))
    y = np.array([0., .1, 3., 3.1, 2.9])
    assert_results_equal(bagel.fit(y, *callbacks), bagel.fit(y, *model))


def test_tensor_streaming_weights_and_early_stop(model):
    callbacks = tensor_model(model)
    detector = bagel.Detector(*callbacks)
    detector.update(torch.tensor(0., requires_grad=True))
    detector.update(torch.tensor(0.1, requires_grad=True))
    detector.set_weights(torch.tensor([.25, .75], requires_grad=True))
    np.testing.assert_allclose(detector.active_weights, [.25, .75])
    detector.set_log_weights(torch.tensor([-float("inf"), 0.], requires_grad=True))
    assert detector.no_change_probability == 0
    assert detector.time == 2
    with pytest.raises(ValueError, match="scalar tensor"):
        detector.update(torch.tensor([1.]))
    assert detector.time == 2
    y = torch.tensor([0., 1000., 1001.], requires_grad=True)
    result = bagel.bagel_kv_exact(y, *callbacks, .9, 1, 1, 5, .95)
    assert result.changepoint()["detected"] == 2
    assert result.time == 2 and len(result.y) == 3


@pytest.mark.parametrize("make, error, message", [
    (lambda: torch.tensor([1 + 2j]), TypeError, "real-valued"),
    (lambda: torch.ones(2).to_sparse(), TypeError, "dense"),
    (lambda: torch.quantize_per_tensor(torch.ones(2), .1, 0, torch.qint8), TypeError, "non-quantized"),
    (lambda: torch.empty(2, device="meta"), ValueError, "meta"),
    (lambda: torch.tensor([float("nan")]), ValueError, "finite"),
    (lambda: torch.tensor([float("inf")]), ValueError, "finite"),
    (lambda: torch.ones(2, 2), ValueError, "one-dimensional"),
    (lambda: torch.tensor([]), ValueError, "nonempty"),
])
def test_rejected_tensor_series_does_not_advance(model, make, error, message):
    detector = bagel.Detector(*model)
    with pytest.raises(error, match=message):
        bagel.analyse(make(), 1, detector)
    assert detector.time == 0


@pytest.mark.parametrize("part", ["feature", "mu", "sigma", "transform"])
def test_invalid_tensor_callback_does_not_advance(model, part):
    feature, prior, transform = tensor_model(model)
    invalid = torch.ones(1, dtype=torch.complex64)
    if part == "feature":
        feature = lambda t, tau: invalid
    elif part == "transform":
        transform = lambda tau: invalid
    else:
        original = prior
        prior = lambda t: {**original(t), part: invalid}
    detector = bagel.Detector(feature, prior, transform)
    with pytest.raises(TypeError, match="real-valued"):
        detector.update(0.)
    assert detector.time == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-enabled PyTorch and GPU required")
def test_cuda_tensor_transfer_matches_cpu(model):
    y = torch.tensor([0., .1, -.1, 3., 3.1, 2.9], device="cuda", requires_grad=True)
    expected = bagel.fit(y.detach().cpu().numpy(), *model)
    assert_results_equal(bagel.fit(y, *tensor_model(model, "cuda")), expected)
    detector = bagel.Detector(*tensor_model(model, "cuda"))
    for scalar in y:
        detector.update(scalar)
    np.testing.assert_allclose(detector.active_weights, expected.active_weights)
    detector.set_weights(torch.tensor([1.] + [0.] * (len(detector.taus) - 1), device="cuda"))
    detector.set_log_weights(torch.tensor(detector.log_weights, device="cuda"))
    assert detector.no_change_probability == 1.
    assert y.grad is None
