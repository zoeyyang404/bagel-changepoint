"""Time complete CPU fits using an installed Bagel wheel."""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import numpy as np
import bagel
from bagel._inputs import tensor_to_numpy


@dataclass(frozen=True)
class Case:
    name: str
    length: int
    particles: int
    dimension: int = 1  # No-change dimension; change candidates have twice this.
    series: int = 1
    variance: str = "known"
    distance: str = "approximate"


def cases(suite):
    n, p = (200, 16) if suite == "quick" else (1000, 32)
    return [Case("mean", n, p), Case("long_series", 5 * n, p),
            Case("many_particles", n, 4 * p), Case("four_coefficients", n, p, 4),
            Case("sixteen_coefficients", n, p, 16),
            Case("eight_series_serial", n, p, series=8),
            Case("mean_exact", n, p, distance="exact"),
            Case("unknown_variance", n, p, variance="unknown")]


def make_model(case):
    d = case.dimension
    basis = np.ones((case.length + 1, d))
    # Bounded regressors keep the larger models numerically well-conditioned.
    for j in range(1, d):
        basis[:, j] = np.cos(np.arange(case.length + 1) * (j + 1) * .013)
    identity = np.eye(d)
    baseline = {"mu": np.zeros(d), "sigma": identity}
    changed = {"mu": np.zeros(2 * d),
               "sigma": np.block([[identity, -identity], [-identity, 2 * identity]])}
    projected = np.concatenate((identity, identity), axis=1)

    def feature(t, tau):
        return basis[t] if tau == 0 else np.concatenate((basis[t], basis[t]))

    return feature, lambda t: baseline if t == 1 else changed, lambda tau: identity if tau == 0 else projected


def options(case):
    return dict(max_particles=case.particles, variance=case.variance,
                distance=case.distance, pruning="bagel")


def summary(samples):
    return {"samples_seconds": samples, "median_seconds": statistics.median(samples),
            "min_seconds": min(samples), "max_seconds": max(samples)}


def measure(function, repeats, warmups):
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repeats):
        gc.collect()  # Outside the timed region; normal GC remains enabled.
        start = perf_counter()
        function()
        samples.append(perf_counter() - start)
    return summary(samples)


class CallbackTimer:
    """Time callback bodies in a separate run from the main fit measurements."""
    def __init__(self):
        self.seconds = 0.
        self.calls = 0

    def wrap(self, function):
        def measured(*args):
            start = perf_counter()
            result = function(*args)
            self.seconds += perf_counter() - start
            self.calls += 1
            return result
        return measured


def diagnose(case, y, model):
    timer = CallbackTimer()
    detector = bagel.Detector(*(timer.wrap(callback) for callback in model), **options(case))
    latencies = []
    for value in y:
        start = perf_counter()
        detector.update(float(value))
        latencies.append(perf_counter() - start)
    total = sum(latencies)
    return {
        "series_measured": 1, "update_seconds": total,
        "callback_body_seconds": timer.seconds, "callback_calls": timer.calls,
        # Includes C++ work, validation, conversion, dispatch, and probe overhead.
        # It is deliberately not described as pure numerical-core time.
        "update_excluding_callback_bodies_seconds": total - timer.seconds,
        "update_latency_median_seconds": float(np.median(latencies)),
        "update_latency_p95_seconds": float(np.percentile(latencies, 95)),
        "final_active_particles": len(detector.taus),
    }


def cpu_name():
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def make_data(case, seed):
    data = np.random.default_rng(seed).normal(0, .3, (case.series, case.length))
    data[:, case.length // 2:] += 2.
    # Match values exactly across NumPy float64 and tensor float32 inputs.
    return data.astype(np.float32).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["quick", "standard"], default="standard")
    parser.add_argument("--inputs", nargs="+", choices=["numpy", "torch-cpu", "torch-cuda"], default=["numpy"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, help="Optional cProfile output for the standard mean case")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups must be nonnegative")
    torch = None
    if any(name.startswith("torch") for name in args.inputs):
        import torch
        if "torch-cuda" in args.inputs and not torch.cuda.is_available():
            parser.error("torch-cuda requires a CUDA-enabled PyTorch installation and GPU")
    report = {
        "schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite, "seed": args.seed, "repeats": args.repeats, "warmups": args.warmups,
        "environment": {
            "platform": platform.platform(), "cpu": cpu_name(), "logical_cpus": os.cpu_count(),
            "python": sys.version, "numpy": np.__version__, "bagel": bagel.__version__,
            "bagel_location": str(Path(bagel.__file__).resolve()),
            "package_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(bagel.__file__).parent.iterdir())
                               if p.is_file() and p.suffix in (".py", ".pyd", ".so")},
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch": importlib.metadata.version("torch") if torch is not None else None,
            "torch_threads": torch.get_num_threads() if torch is not None else None,
            "gpu": torch.cuda.get_device_name() if "torch-cuda" in args.inputs else None,
            "thread_environment": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        },
        "notes": ["All Bagel calculations run on CPU, including torch-cuda inputs.",
                  "Input generation and imports are excluded; fit includes input conversion, callbacks and result snapshots.",
                  "Inputs are rounded to float32 before storage so NumPy float64 and torch float32 observations have identical values.",
                  "Multiple series run serially; no parallel or batched backend is implied.",
                  "Conversion and streaming diagnostics are separate measurements, not additive phases of the timed fits.",
                  "Callback diagnostics measure callback bodies only; residual includes native work and adapter overhead."],
        "results": [],
    }
    for case in cases(args.suite):
        data = make_data(case, args.seed)
        model = make_model(case)
        reference = [bagel.fit(y, *model, **options(case)) for y in data]
        diagnostics = diagnose(case, data[0], model)
        for input_name in dict.fromkeys(args.inputs):
            values = list(data) if input_name == "numpy" else [
                torch.tensor(y, dtype=torch.float32, requires_grad=True,
                             device="cuda" if input_name == "torch-cuda" else "cpu") for y in data]
            if input_name == "torch-cuda":
                torch.cuda.synchronize()  # Input construction is outside timed work.

            def run():
                return [bagel.fit(y, *model, **options(case)) for y in values]

            # Correctness is checked separately from timing, against identical
            # observations fitted through NumPy.
            for actual, target in zip(run(), reference):
                assert actual.time == target.time == case.length
                np.testing.assert_allclose(actual.log_no_change_probabilities,
                                           target.log_no_change_probabilities, rtol=2e-9, atol=2e-11)
                np.testing.assert_allclose(actual.weights, target.weights, rtol=2e-9, atol=2e-11)
            timing = measure(run, args.repeats, args.warmups)
            conversion = measure(lambda: [np.asarray(tensor_to_numpy(y, "Y"), dtype=float)
                                          for y in values], args.repeats, args.warmups)
            result = {"case": asdict(case), "input": input_name, "fit": timing,
                      "observations_per_second": case.length * case.series / timing["median_seconds"],
                      "input_conversion": conversion, "streaming_numpy_diagnostics": diagnostics}
            report["results"].append(result)
            print(f"{case.name:24s} {input_name:10s} {timing['median_seconds']:.4f} s "
                  f"{result['observations_per_second']:.0f} obs/s", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.profile:
        case = cases(args.suite)[0]
        y = make_data(case, args.seed)[0]
        model = make_model(case)
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profile = cProfile.Profile()
        profile.runcall(bagel.fit, y, *model, **options(case))
        profile.dump_stats(str(args.profile))
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
