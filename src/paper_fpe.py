"""Finite-time Fokker–Planck readout for the first sensory-contact session."""

import numpy as np
from numba import njit
from scipy.special import logit


@njit
def _bernoulli(w):
    """Scharfetter–Gummel flux factor, continuous at zero."""
    if abs(w) < 1e-5:
        return 1 - w / 2 + w * w / 12 - w**4 / 720
    return w / np.expm1(w)


@njit
def _terminal_mass(x, initial, beta, a, dt, steps):
    """Evolve cell masses by implicit Euler with absorbing outer boundaries."""
    n = len(x)
    dx = x[1] - x[0]
    diffusivity = 0.5 * a * beta * beta
    right = np.empty(n)
    left = np.empty(n)
    for j in range(n):
        drift_right = -0.5 * a * beta * np.tanh((x[j] + dx / 2) / 2)
        drift_left = -0.5 * a * beta * np.tanh((x[j] - dx / 2) / 2)
        right[j] = diffusivity / dx**2 * _bernoulli(-drift_right * dx / diffusivity)
        left[j] = diffusivity / dx**2 * _bernoulli(drift_left * dx / diffusivity)

    # Factor the constant tridiagonal backward-Euler system once.
    inverse = np.empty(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    for j in range(n):
        denominator = 1 + dt * (right[j] + left[j])
        if j > 0:
            lower[j] = -dt * right[j - 1]
            denominator -= lower[j] * upper[j - 1]
        inverse[j] = 1 / denominator
        if j + 1 < n:
            upper[j] = -dt * left[j + 1] / denominator

    # Deposit each equally weighted entry between the nearest cell centers.
    mass = np.zeros(n)
    for value in initial:
        position = (value - x[0]) / dx
        j = int(np.floor(position))
        fraction = position - j
        mass[j] += (1 - fraction) / len(initial)
        mass[j + 1] += fraction / len(initial)

    escaped = 0.0
    for _ in range(steps):
        mass[0] *= inverse[0]
        for j in range(1, n):
            mass[j] = (mass[j] - lower[j] * mass[j - 1]) * inverse[j]
        for j in range(n - 2, -1, -1):
            mass[j] -= upper[j] * mass[j + 1]
        escaped += dt * (left[0] * mass[0] + right[-1] * mass[-1])
    if not np.isfinite(mass).all() or mass.min() < 0 or abs(mass.sum() + escaped - 1) > 1e-8:
        raise ArithmeticError("FPE solution failed the nonnegativity or mass-balance check")
    return mass


def finite_time_probabilities(initial, beta, a, duration, q_edges):
    """Return raw q-bin masses from the negative-gain FPE at the session endpoint."""
    dx, dt = 0.025, 0.125
    # Include maximum drift displacement, ten Brownian SDs and ten extra logit units.
    margin = 0.5 * a * abs(beta) * duration + 10 * np.sqrt(a) * abs(beta) * np.sqrt(duration) + 10
    lo = np.floor((initial.min() - margin) / dx) * dx
    hi = np.ceil((initial.max() + margin) / dx) * dx
    x = lo + dx * np.arange(round((hi - lo) / dx) + 1)
    mass = _terminal_mass(x, initial, beta, a, dt, round(duration / dt))
    spacing = x[1] - x[0]
    cell_edges = np.r_[x - spacing / 2, x[-1] + spacing / 2]
    cdf = np.r_[0.0, np.cumsum(mass)]
    return np.diff(np.interp(logit(q_edges), cell_edges, cdf))
