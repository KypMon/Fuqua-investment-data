"""Life-cycle simulation utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


class LifeCycleInputError(Exception):
    """Raised when the life-cycle simulation receives invalid input."""


def _ensure_vector(values: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise LifeCycleInputError(f"{label} vector must contain at least one numeric value.")
    if not np.all(np.isfinite(array)):
        raise LifeCycleInputError(f"{label} vector contains non-finite values.")
    return array


def load_vector_from_csv(file_storage, label: str) -> np.ndarray:
    """Load a numeric vector from an uploaded CSV file."""

    if file_storage is None or not getattr(file_storage, "filename", ""):
        raise LifeCycleInputError(f"{label} CSV file is required.")

    try:
        df = pd.read_csv(file_storage, header=None)
    except Exception as exc:  # pragma: no cover - pandas error path
        raise LifeCycleInputError(f"Unable to read {label.lower()} CSV: {exc}") from exc

    if df.empty:
        raise LifeCycleInputError(f"{label} CSV file is empty.")

    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if df.empty:
        raise LifeCycleInputError(f"{label} CSV file does not contain numeric data.")

    series = df.iloc[:, 0].dropna()
    if series.empty:
        raise LifeCycleInputError(f"{label} CSV file does not contain numeric data in the first column.")

    return _ensure_vector(series.to_numpy(dtype=float), label)


def life_cycle_simulation(
    vec_ret: Sequence[float],
    vec_cf: Sequence[float],
    *,
    w0: float = 0.0,
    nsim: int = 1000,
) -> np.ndarray:
    """Run the Monte Carlo life-cycle simulation."""

    returns = _ensure_vector(vec_ret, "Return")
    cashflows = _ensure_vector(vec_cf, "Cash flow")

    if nsim <= 0:
        raise LifeCycleInputError("Number of simulations must be a positive integer.")

    nper = cashflows.size
    rng = np.random.default_rng()
    rmat = rng.choice(returns, size=(nsim, nper), replace=True)

    wmat = np.zeros((nsim, nper), dtype=float)
    wmat[:, 0] = (w0 + cashflows[0]) * (1.0 + rmat[:, 0])

    for idx in range(1, nper):
        wmat[:, idx] = (wmat[:, idx - 1] + cashflows[idx]) * (1.0 + rmat[:, idx])

    return wmat


def _create_histogram(final_wealth: np.ndarray, *, bins: str | int = "auto") -> dict:
    counts, edges = np.histogram(final_wealth, bins=bins)
    table = [
        {
            "from": float(edges[i]),
            "to": float(edges[i + 1]),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]
    return {
        "bins": [float(edge) for edge in edges],
        "counts": [int(value) for value in counts],
        "table": table,
    }


def _compute_below_cutoff(wmat: np.ndarray, wmin_cutoff: float) -> dict:
    counts = (wmat < wmin_cutoff).sum(axis=0)
    periods = np.arange(1, wmat.shape[1] + 1, dtype=int)
    table = [
        {
            "period": int(period),
            "count": int(count),
        }
        for period, count in zip(periods, counts)
    ]
    return {
        "periods": [int(p) for p in periods],
        "counts": [int(value) for value in counts],
        "table": table,
    }


def _summarize_final_wealth(
    final_wealth: np.ndarray,
    wmin_cutoff: float,
    below_cutoff_counts: np.ndarray,
) -> dict:
    summary = {
        "mean": float(np.mean(final_wealth)),
        "median": float(np.median(final_wealth)),
        "min": float(np.min(final_wealth)),
        "max": float(np.max(final_wealth)),
    }
    if final_wealth.size > 1:
        summary["std"] = float(np.std(final_wealth, ddof=1))
    final_below = int(below_cutoff_counts[-1]) if below_cutoff_counts.size else 0
    summary["final_below_cutoff_count"] = final_below
    summary["final_below_cutoff_pct"] = float(final_below / final_wealth.size)
    summary["probability_final_below_cutoff"] = summary["final_below_cutoff_pct"]
    summary["wmin_cutoff"] = float(wmin_cutoff)
    return summary


def run_life_cycle_analysis(
    vec_ret: Sequence[float],
    vec_cf: Sequence[float],
    *,
    w0: float = 0.0,
    wmin_cutoff: float = 0.0,
    nsim: int = 1000,
    bins: str | int = "auto",
) -> dict:
    """Execute the full life-cycle simulation and return structured results."""

    wmat = life_cycle_simulation(vec_ret, vec_cf, w0=w0, nsim=nsim)
    final_wealth = wmat[:, -1]
    histogram = _create_histogram(final_wealth, bins=bins)
    below_cutoff = _compute_below_cutoff(wmat, wmin_cutoff)
    below_counts_array = np.asarray(below_cutoff["counts"], dtype=int)
    summary = _summarize_final_wealth(final_wealth, wmin_cutoff, below_counts_array)

    return {
        "final_wealth_histogram": histogram,
        "wealth_below_cutoff": below_cutoff,
        "summary": summary,
        "metadata": {
            "initial_wealth": float(w0),
            "n_periods": int(wmat.shape[1]),
            "n_simulations": int(wmat.shape[0]),
        },
    }


__all__ = [
    "LifeCycleInputError",
    "load_vector_from_csv",
    "life_cycle_simulation",
    "run_life_cycle_analysis",
]
