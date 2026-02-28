"""Plot helpers for pysusie."""

from __future__ import annotations

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("plotting requires matplotlib") from exc
    return plt


def plot_pip(
    result,
    *,
    credible_sets=None,
    ax=None,
    highlight_cs: bool = True,
    add_legend: bool = True,
    colors=None,
    **kwargs,
):
    """Manhattan-style plot of posterior inclusion probabilities."""
    plt = _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 3)))

    x = np.arange(result.n_variables)
    ax.scatter(x, result.pip, s=12, color=kwargs.pop("color", "tab:blue"), alpha=0.8)
    ax.set_xlabel("Variable")
    ax.set_ylabel("PIP")
    ax.set_ylim(0.0, 1.02)

    if highlight_cs:
        if credible_sets is None:
            credible_sets = result.get_credible_sets()
        if colors is None:
            colors = plt.cm.tab10.colors
        for i, cs in enumerate(credible_sets):
            color = colors[i % len(colors)]
            ax.scatter(cs.variables, result.pip[cs.variables], color=color, s=24, label=f"CS {i}")

        if add_legend and credible_sets:
            ax.legend(frameon=False, fontsize=8)

    return ax


def plot_z(
    result,
    *,
    z_scores: np.ndarray | None = None,
    credible_sets=None,
    ax=None,
    highlight_cs: bool = True,
    add_legend: bool = True,
    colors=None,
    absolute: bool = False,
    **kwargs,
):
    """Manhattan-style plot of z-scores with optional credible-set highlights."""
    plt = _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 3)))

    if z_scores is None:
        sd = result.posterior_sd()
        z = np.zeros(result.n_variables, dtype=float)
        nz = sd > 0
        z[nz] = result.coef[nz] / sd[nz]
    else:
        z = np.asarray(z_scores, dtype=float)
        if z.shape != (result.n_variables,):
            raise ValueError("z_scores must have shape (n_variables,)")

    y = np.abs(z) if absolute else z
    x = np.arange(result.n_variables)
    ax.scatter(x, y, s=12, color=kwargs.pop("color", "tab:orange"), alpha=0.8)
    ax.set_xlabel("Variable")
    ax.set_ylabel("|z-score|" if absolute else "z-score")
    if not absolute:
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.3)

    if highlight_cs:
        if credible_sets is None:
            credible_sets = result.get_credible_sets()
        if colors is None:
            colors = plt.cm.tab10.colors
        for i, cs in enumerate(credible_sets):
            color = colors[i % len(colors)]
            ax.scatter(cs.variables, y[cs.variables], color=color, s=24, label=f"CS {i}")

        if add_legend and credible_sets:
            ax.legend(frameon=False, fontsize=8)

    return ax


def plot_diagnostic(result, *, what: str = "elbo", ax=None):
    """Diagnostic plot for convergence assessment."""
    plt = _require_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    if what == "elbo":
        ax.plot(np.arange(len(result.elbo)), result.elbo, marker="o", lw=1)
        ax.set_ylabel("ELBO")
        ax.set_xlabel("Iteration")
    elif what == "alpha":
        ax.imshow(result.alpha, aspect="auto", interpolation="nearest")
        ax.set_ylabel("Effect")
        ax.set_xlabel("Variable")
    elif what in {"prior_variance", "prior"}:
        ax.plot(result.prior_variance, marker="o", lw=1)
        ax.set_ylabel("Prior variance")
        ax.set_xlabel("Effect")
    else:
        raise ValueError("what must be one of {'elbo', 'alpha', 'prior_variance'}")

    return ax


def plot_changepoint(result, y: np.ndarray, *, ax=None):
    """Simple changepoint-style diagnostic for 1D signals."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    y = np.asarray(y, dtype=float)
    ax.plot(y, color="black", lw=1, label="y")

    cs = result.get_credible_sets()
    for i, c in enumerate(cs):
        if c.variables.size:
            ax.axvline(c.variables[0], color=f"C{i%10}", alpha=0.6, lw=1)

    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    return ax
