"""Benchmark pysusie against susieR when rpy2 + susieR are available."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from pysusie import SuSiE


@dataclass
class BenchmarkResult:
    backend: str
    n: int
    p: int
    L: int
    runtime_sec: float
    converged: bool
    n_iter: int


def simulate_problem(n: int, p: int, *, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    causal = np.array([p // 6, p // 2, (5 * p) // 6], dtype=int)
    beta[causal] = np.array([0.8, -0.7, 0.6])
    y = X @ beta + rng.normal(size=n)
    return X, y


def run_python_benchmark(n: int, p: int, L: int, *, max_iter: int = 100, seed: int = 123) -> BenchmarkResult:
    X, y = simulate_problem(n, p, seed=seed)
    model = SuSiE(n_effects=L, max_iter=max_iter)

    t0 = time.perf_counter()
    model.fit(X, y)
    runtime = time.perf_counter() - t0

    result = model.result_
    return BenchmarkResult(
        backend="python",
        n=n,
        p=p,
        L=L,
        runtime_sec=float(runtime),
        converged=bool(result.converged),
        n_iter=int(result.n_iter),
    )


def run_r_benchmark_optional(n: int, p: int, L: int, *, max_iter: int = 100, seed: int = 123) -> BenchmarkResult | None:
    try:
        from rpy2 import robjects
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
    except Exception:
        return None

    try:
        susieR = importr("susieR")
    except Exception:
        return None

    X, y = simulate_problem(n, p, seed=seed)

    numpy2ri.activate()
    r = robjects.r
    r.assign("X_py", X)
    r.assign("y_py", y)
    r.assign("L_py", int(L))
    r.assign("max_iter_py", int(max_iter))

    t0 = time.perf_counter()
    fit = r("susieR::susie(X_py, y_py, L=L_py, max_iter=max_iter_py)")
    runtime = time.perf_counter() - t0

    converged = bool(np.array(fit.rx2("converged"))[0])
    n_iter = int(np.array(fit.rx2("niter"))[0])

    return BenchmarkResult(
        backend="r",
        n=n,
        p=p,
        L=L,
        runtime_sec=float(runtime),
        converged=converged,
        n_iter=n_iter,
    )


def compare_backends(n: int, p: int, L: int, *, max_iter: int = 100, seed: int = 123) -> dict:
    py = run_python_benchmark(n, p, L, max_iter=max_iter, seed=seed)
    r = run_r_benchmark_optional(n, p, L, max_iter=max_iter, seed=seed)

    out = {"python": asdict(py), "r": asdict(r) if r is not None else None}
    if r is not None and r.runtime_sec > 0:
        out["speedup_python_vs_r"] = r.runtime_sec / py.runtime_sec
    else:
        out["speedup_python_vs_r"] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--p", type=int, default=1000)
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--python-only", action="store_true")
    args = parser.parse_args()

    if args.python_only:
        result = asdict(
            run_python_benchmark(args.n, args.p, args.L, max_iter=args.max_iter, seed=args.seed)
        )
    else:
        result = compare_backends(args.n, args.p, args.L, max_iter=args.max_iter, seed=args.seed)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
