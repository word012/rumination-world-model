"""Simulate the discrete model for Figures 2, 3 and the parameter survey in Figure 4a.

Each replicate has an independent seed for each update. The same replicate is
paired across gain conditions. Sampling its normal and uniform variates once
lets all conditions share the original random stream efficiently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "paper_ensemble.json"


def _variates(seeds, steps):
    normal = np.empty((len(seeds), steps))
    uniform = np.empty_like(normal)
    for row, seed in enumerate(seeds):
        step_seeds = np.random.default_rng(seed).integers(0, 2**63 - 1, size=steps)
        for t, step_seed in enumerate(step_seeds):
            rng = np.random.default_rng(step_seed)
            normal[row, t] = rng.standard_normal()
            uniform[row, t] = rng.random()
    return normal, uniform


@njit(inline="always")
def _sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + np.exp(-x))
    e = np.exp(x)
    return e / (1.0 + e)


@njit(parallel=True)
def _simulate(gains, normal, uniform, x0, y0, process_variance, y_eps, defeat_steps,
              session_steps, defeat_observation, n_display):
    n_conditions = len(gains)
    n_seeds, steps = normal.shape
    endpoints = np.empty((n_conditions, n_seeds))
    paths = np.empty((n_conditions, n_display, steps + 1))
    for run in prange(n_conditions * n_seeds):
        i, j = run // n_seeds, run % n_seeds
        alpha, beta = gains[i]
        x, y = x0, y0
        if j < n_display:
            paths[i, j, 0] = x
        for t in range(steps):
            v = 1.0 / max(y, y_eps)
            if t % session_steps < defeat_steps:
                observation, gain = defeat_observation, alpha
            else:
                z = x + np.sqrt(v) * normal[j, t]
                observation = 1.0 if uniform[j, t] < _sigmoid(z) else 0.0
                gain = beta
            k = v + process_variance
            x = x + gain * k * (observation - _sigmoid(x))
            p = _sigmoid(x)
            y = 1.0 / k + p * (1.0 - p)
            if j < n_display:
                paths[i, j, t + 1] = x
        endpoints[i, j] = x
    return endpoints, paths


def _run(gains, seeds, settings, n_display=0):
    session_steps = settings["defeat_steps"] + settings["post_steps"]
    normal, uniform = _variates(seeds, settings["n_days"] * session_steps)
    return _simulate(
        np.asarray(gains), normal, uniform, settings["x0"], settings["y0"],
        settings["vz"] ** 2, settings["y_eps"], settings["defeat_steps"],
        session_steps, settings["o_defeat"], n_display,
    )


def _readout(x):
    """Apply the model's sigmoid to a NumPy array, including large negative x."""
    positive = x >= 0
    q = np.empty_like(x)
    q[positive] = 1 / (1 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    q[~positive] = e / (1 + e)
    return q


def _representatives(design):
    conditions = design["conditions"]
    gains = [(c["alpha"], c["beta"]) for c in conditions]
    seeds = np.asarray(design["ordered_seeds"])
    displayed = design["display_replicates"]
    if displayed != list(range(len(displayed))):
        raise ValueError("Display replicates must be the first consecutive seeds")
    settings = design["settings"]
    endpoints, paths = _run(gains, seeds, settings, len(displayed))
    steps = np.arange(paths.shape[2])
    session_steps = settings["defeat_steps"] + settings["post_steps"]
    readouts = {"state_index": steps, "time": steps / session_steps}
    frames = []
    for i, condition in enumerate(conditions):
        label = condition["label"]
        readouts[f"{label}_q"] = _readout(paths[i])
        readouts[f"{label}_seeds"] = seeds[:len(displayed)]
        frames.append(pd.DataFrame({"label": label, "replicate": np.arange(len(seeds)),
                                    "seed": seeds, "q_end": _readout(endpoints[i])}))
    # The example in Figure 2a shows the first two sessions of positive condition 2.
    t = steps[:2 * session_steps + 1]
    session, within = np.divmod(t, session_steps)
    defeat = settings["defeat_steps"]
    # Expand the defeat interval to one unit and sensory contact to four units.
    coordinate = 5 * session + np.where(within <= defeat, within / defeat,
                                        1 + 4 * (within - defeat) / settings["post_steps"])
    intro = pd.DataFrame({"display_coordinate": coordinate, "q": readouts["positive_2_q"][0, t]})
    return {"design": design, "readouts": readouts, "intro": intro,
            "endpoints": pd.concat(frames, ignore_index=True)}


def _survey(settings, grid):
    alpha, beta = np.meshgrid(grid["alphas"], grid["betas"], indexing="ij")
    gains = np.column_stack((alpha.ravel(), beta.ravel()))
    seeds = grid["ordered_seeds"]
    endpoints, _ = _run(gains, seeds, settings)
    q = _readout(endpoints)
    # Use the original sample order and pandas group reductions for both maps.
    frame = pd.DataFrame({"alpha": gains[:, 0].repeat(len(seeds)),
                          "beta": gains[:, 1].repeat(len(seeds)), "q": q.ravel()})
    return frame.groupby(["alpha", "beta"]).q.agg(mean_q="mean", sd_q="std", n="size").reset_index()


def generate_ensemble(config_path=CONFIG):
    """Return the representative figure inputs and the full parameter-grid summary."""
    config = json.loads(Path(config_path).read_text())
    survey = config.pop("grid")
    seed = config.pop("representative_root_seed")
    seeds = np.random.default_rng(seed).integers(0, 2**63 - 1, size=config["n_runs_per_condition"])
    config["ordered_seeds"] = seeds.tolist()
    return _representatives(config), _survey(config["settings"], survey)
