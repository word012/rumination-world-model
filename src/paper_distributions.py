"""Reproduce the simulated and theoretical bin probabilities in Figures 4c and 4e."""

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.special import betainc, logit

from paper_fpe import finite_time_probabilities


GAINS = np.array([0.48, 1.04, 1.96, -0.48, -1.04, -1.96])
Q_EDGES = np.linspace(0.0, 1.0, 51)
N_RUNS = 16_384
BLOCK_SIZE = 1_024
SESSIONS = 10
DEFEAT_STEPS = 10
CONTACT_STEPS = 960
DEFEAT_GAIN = 0.136
PROCESS_VARIANCE = 0.01


@njit(inline="always")
def _update(mean, precision, gain, attack):
    # Evaluate both possible sigmoid tails directly to retain small residuals.
    predicted_variance = 1 / precision + PROCESS_VARIANCE
    e = np.exp(-abs(mean))
    if attack:
        residual = e / (1 + e) if mean > 0 else 1 / (1 + e)
    else:
        residual = -1 / (1 + e) if mean >= 0 else -e / (1 + e)
    mean += gain * predicted_variance * residual
    e = np.exp(-abs(mean))
    precision = 1 / predicted_variance + e / ((1 + e) * (1 + e))
    return mean, precision


@njit
def _defeat(mean, precision):
    for _ in range(DEFEAT_STEPS):
        mean, precision = _update(mean, precision, DEFEAT_GAIN, True)
    return mean, precision


@njit(parallel=True)
def _simulate_block(normals, thresholds, entry_mean, entry_precision):
    """Paired runs end at the tenth session (4c) or the first session (4e)."""
    n = len(normals)
    endpoints = np.empty((len(GAINS), n))
    for index in prange(len(GAINS) * n):
        condition, run = index // n, index % n
        gain = GAINS[condition]
        mean, precision = entry_mean, entry_precision
        days = SESSIONS if gain > 0 else 1
        for day in range(days):
            if day:
                mean, precision = _defeat(mean, precision)
            for step in range(CONTACT_STEPS):
                t = day * CONTACT_STEPS + step
                attack = mean + normals[run, t] / np.sqrt(precision) > thresholds[run, t]
                mean, precision = _update(mean, precision, gain, attack)
        endpoints[condition, run] = mean
    return endpoints


def _endpoints(entry_mean, entry_precision):
    endpoints = np.empty((len(GAINS), N_RUNS))
    shape = (BLOCK_SIZE, SESSIONS * CONTACT_STEPS)
    eps = np.finfo(float).eps
    for block in range(N_RUNS // BLOCK_SIZE):
        # Preserve Figure 4c's PCG64 blocks: all Gaussian draws precede the
        # uniforms. The first 960 columns also drive Figure 4e for each run.
        rng = np.random.Generator(np.random.PCG64(990127 + 113 * block))
        normals = rng.standard_normal(shape)
        uniforms = rng.uniform(eps, 1.0 - eps, shape)
        thresholds = np.log(uniforms) - np.log1p(-uniforms)
        start = block * BLOCK_SIZE
        endpoints[:, start:start + BLOCK_SIZE] = _simulate_block(
            normals, thresholds, entry_mean, entry_precision)
    return endpoints


def generate_distributions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate all 50 bin masses for each plotted gain and method at its endpoint."""
    # Infinite outer edges retain both saturated tails without clipping.
    logit_edges = logit(Q_EDGES)
    entry_mean, entry_precision = _defeat(-2.0, 2.0)
    # Each Monte Carlo run enters the FPE after the same ten defeat updates.
    initial = np.full(N_RUNS, entry_mean)
    positive, negative = [], []
    for beta, samples in zip(GAINS, _endpoints(entry_mean, entry_precision), strict=True):
        empirical = np.histogram(samples, bins=logit_edges)[0] / N_RUNS
        if beta > 0:
            theory = np.diff(betainc(1 / beta, 1 / beta, Q_EDGES))
            method, column, table = "stationary_beta", "mass", positive
        else:
            theory = finite_time_probabilities(initial, beta, PROCESS_VARIANCE, CONTACT_STEPS, Q_EDGES)
            method, column, table = "FPE", "probability", negative
        for label, values in (("original", empirical), (method, theory)):
            table.append(pd.DataFrame({"beta": beta, "method": label, "q_left": Q_EDGES[:-1],
                                       "q_right": Q_EDGES[1:], column: values}))
    return pd.concat(positive, ignore_index=True), pd.concat(negative, ignore_index=True)
