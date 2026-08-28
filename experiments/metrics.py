"""Shared posterior-predictive metrics."""
from __future__ import annotations

import numpy as np


def rmse(estimate, truth) -> float:
    error = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    return float(np.sqrt(np.mean(np.square(error))))


def mae(estimate, truth) -> float:
    return float(np.mean(np.abs(np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float))))


def interval_coverage(draws, truth, level: float = 0.9) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between zero and one")
    lower, upper = np.quantile(np.asarray(draws), [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
    truth = np.asarray(truth)
    return float(np.mean((truth >= lower) & (truth <= upper)))


def interval_width(draws, level: float = 0.9) -> float:
    lower, upper = np.quantile(np.asarray(draws), [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
    return float(np.mean(upper - lower))
