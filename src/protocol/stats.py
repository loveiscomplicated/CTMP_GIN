from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import stats


def nadeau_bengio_corrected_t(
    differences: Iterable[float],
    *,
    n_train: int,
    n_test: int,
) -> dict[str, float]:
    values = np.asarray(list(differences), dtype=float)
    if values.size < 2:
        raise ValueError("at least two paired differences are required")
    mean = float(values.mean())
    variance = float(values.var(ddof=1))
    correction = (1.0 / values.size) + (float(n_test) / float(n_train))
    standard_error = math.sqrt(max(variance * correction, 0.0))
    if standard_error == 0.0:
        t_value = math.inf if mean != 0 else 0.0
        p_value = 0.0 if mean != 0 else 1.0
    else:
        t_value = mean / standard_error
        p_value = float(2.0 * stats.t.sf(abs(t_value), df=values.size - 1))
    return {"mean_difference": mean, "t": float(t_value), "p_value": p_value, "n": int(values.size)}


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def bh_fdr_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for rank in range(len(values) - 1, -1, -1):
        index = order[rank]
        running = min(running, values[index] * len(values) / (rank + 1))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def tost(differences: Iterable[float], sesoi: float, alpha: float = 0.05) -> dict[str, float | bool]:
    if sesoi <= 0:
        raise ValueError("SESOI must be positive")
    values = np.asarray(list(differences), dtype=float)
    if values.size < 2:
        raise ValueError("at least two paired differences are required")
    mean = float(values.mean())
    sem = float(stats.sem(values))
    df = values.size - 1
    lower_p = float(stats.t.sf((mean + sesoi) / sem, df)) if sem else 0.0
    upper_p = float(stats.t.cdf((mean - sesoi) / sem, df)) if sem else 0.0
    return {
        "mean_difference": mean,
        "sesoi": float(sesoi),
        "p_lower": lower_p,
        "p_upper": upper_p,
        "p_value": max(lower_p, upper_p),
        "equivalent": bool(max(lower_p, upper_p) < alpha),
    }
