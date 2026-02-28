"""Bundled simulated datasets for pysusie examples/tests."""

from __future__ import annotations

import numpy as np


def _simulate(n: int, p: int, causal: list[int], effects: list[float], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[np.asarray(causal, dtype=int)] = np.asarray(effects, dtype=float)
    y = X @ beta + rng.normal(scale=1.0, size=n)
    return {"X": X, "y": y, "true_coef": beta, "causal_indices": np.array(causal, dtype=int)}


def load_example(name: str = "N3finemapping") -> dict:
    """Load a bundled example dataset."""
    name = str(name)
    if name == "N3finemapping":
        return _simulate(
            n=574,
            p=1001,
            causal=[100, 300, 700],
            effects=[0.6, -0.5, 0.45],
            seed=12345,
        )
    if name == "small":
        return _simulate(
            n=120,
            p=30,
            causal=[3, 11, 20],
            effects=[0.8, -0.6, 0.7],
            seed=2024,
        )
    raise ValueError("Unknown dataset name. Available: {'N3finemapping', 'small'}")
