"""Temperature calibration of gallery attribute probabilities."""

from __future__ import annotations

import numpy as np

PROB_EPS = 1e-4


def calibrate_probabilities(
    probabilities: np.ndarray,
    temperature: float | np.ndarray,
) -> np.ndarray:
    """Temperature-scale clipped probabilities via logit space.

    ``temperature`` may be a scalar or a length-A vector (one scale per
    attribute). Values divide the logits (T>1 softens, T<1 sharpens).
    """
    clipped = np.clip(probabilities.astype(np.float32), PROB_EPS, 1.0 - PROB_EPS)
    temps = np.asarray(temperature, dtype=np.float32)
    if temps.ndim == 0:
        if float(temps) == 1.0:
            return clipped
        logits = np.log(clipped) - np.log1p(-clipped)
        scaled = logits / float(temps)
        return 1.0 / (1.0 + np.exp(-scaled))
    if temps.shape != (clipped.shape[1],):
        raise ValueError(
            f"temperature vector shape {temps.shape} != ({clipped.shape[1]},)"
        )
    if np.allclose(temps, 1.0):
        return clipped
    logits = np.log(clipped) - np.log1p(-clipped)
    scaled = logits / np.maximum(temps[None, :], 1e-3)
    return 1.0 / (1.0 + np.exp(-scaled))


def fit_per_attribute_temperatures(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    init: float = 1.0,
    max_iter: int = 50,
) -> np.ndarray:
    """Fit one temperature per attribute by minimising Bernoulli NLL (LBFGS).

    ``probabilities`` and ``targets`` are gallery rows ``(N, A)``. Fitting uses
    ID-val (or in-domain) labels only — never the hidden test set.
    """
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must share shape")
    clipped = np.clip(probabilities.astype(np.float64), PROB_EPS, 1.0 - PROB_EPS)
    labels = targets.astype(np.float64)
    num_attributes = clipped.shape[1]
    temperatures = np.full(num_attributes, float(init), dtype=np.float64)

    def _nll_one(temp: float, attr: int) -> float:
        scale = max(float(temp), 1e-3)
        logits = np.log(clipped[:, attr]) - np.log1p(-clipped[:, attr])
        calibrated = 1.0 / (1.0 + np.exp(-(logits / scale)))
        calibrated = np.clip(calibrated, PROB_EPS, 1.0 - PROB_EPS)
        y = labels[:, attr]
        return float(
            -np.mean(y * np.log(calibrated) + (1.0 - y) * np.log(1.0 - calibrated))
        )

    for attr in range(num_attributes):
        best_t = float(init)
        best_loss = _nll_one(best_t, attr)
        # Coarse grid then local refine (avoids scipy dependency)
        for candidate in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
            loss = _nll_one(candidate, attr)
            if loss < best_loss:
                best_loss = loss
                best_t = candidate
        # Coordinate descent around the best grid point
        step = 0.25
        for _ in range(max_iter):
            improved = False
            for delta in (-step, step):
                trial = max(best_t + delta, 1e-3)
                loss = _nll_one(trial, attr)
                if loss < best_loss - 1e-9:
                    best_loss = loss
                    best_t = trial
                    improved = True
            if not improved:
                step *= 0.5
                if step < 1e-3:
                    break
        temperatures[attr] = best_t
    return temperatures.astype(np.float32)
