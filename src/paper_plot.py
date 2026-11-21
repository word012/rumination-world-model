"""Render the manuscript's three quantitative figures from simulation results."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
import numpy as np
import pandas as pd


FIGURE_FILES = (
    "fig2_positive_observations.pdf",
    "fig3_negative_observations.pdf",
    "fig4_signed_interpretation.pdf",
)
STYLE = {
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "pdf.fonttype": 42, "savefig.facecolor": "white",
    "path.simplify": False, "mathtext.fontset": "dejavusans",
}
INTERPRETATION_STYLE = {
    **STYLE, "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
}
PALETTE = {
    "positive_1": "#fd8d3c", "positive_2": "#e6550d", "positive_3": "#bd0026",
    "negative_1": "#6baed6", "negative_2": "#3182bd", "negative_3": "#08519c",
}
GAIN_MAGNITUDES = (.48, 1.04, 1.96)


def _axes(fig, left, bottom, width, height):
    """Place axes in inches so the trajectory panels match across figures."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


def _heading(fig, x, y, letter, title):
    width, height = fig.get_size_inches()
    fig.text(x / width, y / height, letter, fontsize=11, weight="bold", va="center")
    fig.text((x + .23) / width, y / height, title, fontsize=9, weight="bold", va="center")


def _example_trajectory(fig, intro):
    _heading(fig, .12, 5.75, "a", "Example trajectory")
    ax = _axes(fig, .62, 4.32, 6.20, 1.15)
    for left, right, defeat in ((0, 1, True), (1, 5, False), (5, 6, True), (6, 10, False)):
        label = "Defeat\n10 updates" if defeat else "Sensory contact\n960 updates"
        ax.axvspan(left, right, facecolor="#fbe8e6" if defeat else "#e4f1f9", lw=0, zorder=0)
        ax.text((left + right) / 2, 1.16, label, ha="center", va="center", fontsize=7,
                color="#a23d33" if defeat else "#286589")
    ax.plot(intro.display_coordinate, intro.q, color="black", lw=.85)
    ax.set(xlim=(0, 10), ylim=(-.025, 1.34), yticks=[0, .5, 1], xticks=[0, 1, 5, 6, 10],
           xticklabels=["0", "10", "970", "980", "1940"],
           xlabel="Update (phase-specific scale; defeat expanded)",
           ylabel=r"Perceived threat, $q_t=f(\mu_t)$")
    ax.text(.25, 1.05, "Session 1", transform=ax.transAxes, ha="center", fontsize=7)
    ax.text(.75, 1.05, "Session 2", transform=ax.transAxes, ha="center", fontsize=7)


def _observations(inputs, sign):
    positive = sign == "positive"
    fig = Figure(figsize=(7.1, 5.85 if positive else 4.0))
    if positive:
        _example_trajectory(fig, inputs["intro"])
    _heading(fig, .12, 3.75, "b" if positive else "a", "Individual trajectories")
    _heading(fig, 5.17, 3.75, "c" if positive else "b", "Final distributions")
    design = inputs["design"]
    readouts, endpoints = inputs["readouts"], inputs["endpoints"]
    conditions = [condition for condition in design["conditions"] if condition["label"].startswith(sign)]
    for row, (condition, bottom) in enumerate(zip(conditions, (2.65, 1.55, .45), strict=True)):
        label = condition["label"]
        ax = _axes(fig, .62, bottom, 4.2, .70)
        for replicate, series in enumerate(readouts[f"{label}_q"]):
            highlight = replicate in design["highlight_replicates"]
            ax.plot(readouts["time"], series, color=PALETTE[label], alpha=.95 if highlight else .20,
                    lw=.95 if highlight else .5, zorder=3 if highlight else 2)
        for session in range(10):
            ax.axvspan(session, session + 10 / 970, color="#ad5b49", alpha=.25, lw=0, zorder=0)
        ax.axhline(.5, color=".6", ls=":", lw=.6, zorder=0)
        ax.set(xlim=(0, 10), ylim=(-.025, 1.025), yticks=[0, .5, 1], xticks=[0, 5, 10],
               ylabel=r"Perceived threat, $q_t$")
        if row == 2:
            ax.set_xlabel("Session (update / 970)")

        values = endpoints.loc[endpoints.label.eq(label), "q_end"].to_numpy()
        counts, edges = np.histogram(values, np.linspace(0, 1, 21))
        ax = _axes(fig, 5.34, bottom, 1.48, .70)
        ax.bar(edges[:-1], counts / len(values), width=np.diff(edges), align="edge",
               color=PALETTE[label], edgecolor="white", lw=.45)
        ax.axvline(float(values.mean()), ls="--", lw=1, color=".15")
        ax.set(xlim=(0, 1), ylim=(0, .25 if positive else 1.), xticks=[0, .5, 1],
               yticks=[0, .1, .2] if positive else [0, .5, 1], ylabel="Probability")
        if row == 2:
            ax.set_xlabel(r"Perceived threat, $q_{9700}$")
    return fig


def _cell_edges(centers):
    middle = (centers[1:] + centers[:-1]) / 2
    return np.r_[centers[0] - (middle[0] - centers[0]), middle,
                 centers[-1] + (centers[-1] - middle[-1])]


def _maps(fig, grid, conditions):
    for column, title, left, cmap, maximum, ticks, quantity in (
        ("mean_q", "Mean perceived threat", .72, "Spectral_r", 1., [0, .5, 1], "Mean"),
        ("sd_q", "SD of perceived threat", 4.20, "viridis", .52, [0, .25, .5], "Sample SD"),
    ):
        values = grid.pivot(index="alpha", columns="beta", values=column).sort_index().sort_index(axis=1)
        if values.isna().any().any():
            raise ValueError("Missing parameter combination in the heatmap")
        alphas, betas = values.index.to_numpy(), values.columns.to_numpy()
        ax = _axes(fig, left, 4.85, 2.15, 1.30)
        mesh = ax.pcolormesh(_cell_edges(betas), _cell_edges(alphas), values.to_numpy(),
                             cmap=cmap, vmin=0, vmax=maximum, shading="flat", rasterized=True)
        ax.set(xlim=(betas.min(), betas.max()), ylim=(alphas.min(), alphas.max()),
               xticks=[-2, -1, 0, 1, 2], yticks=[.02, .1, .2, .3],
               xlabel=r"Rumination gain, $\beta$", ylabel=r"Defeat gain, $\alpha$")
        ax.axvline(0, color="white", lw=.8, ls="--", alpha=.8)
        for alpha in sorted({float(c["alpha"]) for c in conditions}):
            ax.axhline(alpha, color="white", lw=.65, ls=":", alpha=.9)
        for condition in conditions:
            ax.scatter(condition["beta"], condition["alpha"], s=31, marker="o",
                       facecolors=PALETTE[condition["label"]], edgecolors="white", lw=.9, zorder=5)
        ax.set_title(title, loc="left", pad=7)
        colorbar = fig.colorbar(mesh, cax=_axes(fig, left + 2.15 + .10, 4.85, .10, 1.30), ticks=ticks)
        colorbar.set_label(rf"{quantity} of $q_T$", fontsize=8.5, labelpad=3)


def _drift(fig, positive, bottom):
    ax = _axes(fig, .72, bottom, 2.42, 1.35)
    ax.axhline(0, color=".70", lw=.6, zorder=0)
    ax.axvline(0, color=".85", lw=.6, zorder=0)
    x = np.linspace(-6., 6., 481)
    prefix = "positive" if positive else "negative"
    for index, magnitude in enumerate(GAIN_MAGNITUDES, start=1):
        beta = magnitude if positive else -magnitude
        drift = -(beta * .01 / 2) * np.tanh(x / 2)
        ax.plot(x, drift, color=PALETTE[f"{prefix}_{index}"], lw=1.5)
    ax.set(xlim=(-6, 6), ylim=(-.011, .011), xticks=[-6, -3, 0, 3, 6],
           yticks=[-.01, -.005, 0, .005, .01], xlabel=r"Mean threat belief, $X$", ylabel="Drift")


def _distributions(fig, table, positive, bottom):
    ax = _axes(fig, 4.20, bottom, 2.42, 1.35)
    prefix, theory, mass = ("positive", "stationary_beta", "mass") if positive else (
        "negative", "FPE", "probability")
    for index, magnitude in enumerate(GAIN_MAGNITUDES, start=1):
        beta = magnitude if positive else -magnitude
        for method in ("original", theory):
            rows = table.loc[table.beta.eq(beta) & table.method.eq(method)].sort_values("q_left")
            if len(rows) != 50:
                raise ValueError(f"Expected 50 readout bins for beta={beta}, method={method}")
            centers = (rows.q_left.to_numpy() + rows.q_right.to_numpy()) / 2
            probability = rows[mass].to_numpy()
            # Zero-probability bins have no position on the logarithmic axis.
            original = method == "original"
            ax.plot(centers, np.where(probability > 0, probability, np.nan),
                    color=PALETTE[f"{prefix}_{index}"], marker="o" if original else None,
                    ls="none" if original else "--", ms=2.8, lw=1.25, zorder=3 if original else 2)
    time = 9700 if positive else 970
    ax.set(xlim=(0, 1), xticks=[0, .5, 1], yscale="log", ylim=(5e-4, .16) if positive else (1e-4, 1.),
           xlabel=rf"Perceived threat, $q_{{{time}}}$", ylabel="Probability")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    handles = [Line2D([], [], color=".25", marker="o", ms=3, ls="none", label="Simulation"),
               Line2D([], [], color=".25", ls="--", lw=1.25,
                      label="Stationary distribution" if positive else "Finite-time FPE")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.035),
              frameon=False, ncol=2, fontsize=8, handlelength=1.6,
              columnspacing=.9, handletextpad=.4, borderaxespad=0)


def _interpretation(conditions, grid, positive, negative):
    fig = Figure(figsize=(7.1, 6.6))
    _maps(fig, grid, conditions)
    _drift(fig, True, 2.60)
    _distributions(fig, positive, True, 2.60)
    _drift(fig, False, .48)
    _distributions(fig, negative, False, .48)
    for x, y, letter in ((.10, 6.48, "a"), (.10, 4.20, "b"), (3.65, 4.20, "c"),
                         (.10, 2.08, "d"), (3.65, 2.08, "e")):
        fig.text(x / 7.1, y / 6.6, letter, fontsize=12, weight="bold", va="top")
    return fig


def render_figures(fig23: dict, grid: pd.DataFrame, positive: pd.DataFrame,
                   negative: pd.DataFrame, output_dir: Path) -> tuple[Path, ...]:
    """Write Figures 2–4 and return their paths in manuscript order.

    fig23 contains design, readouts, intro and endpoints. The grid contains
    the mean and sample SD of final threat for each gain pair. Positive and
    negative distributions contain simulation and theory masses on 50 bins.
    """
    output_dir = Path(output_dir)
    targets = tuple(output_dir / name for name in FIGURE_FILES)
    if any(path.exists() for path in targets):
        raise FileExistsError("Use a figure directory without existing manuscript PDFs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as cleanup:
        for index, path in enumerate(targets):
            with mpl.rc_context(INTERPRETATION_STYLE if index == 2 else STYLE):
                if index == 2:
                    figure = _interpretation(fig23["design"]["conditions"], grid, positive, negative)
                else:
                    figure = _observations(fig23, ("positive", "negative")[index])
                FigureCanvasAgg(figure).draw()
                with path.open("xb") as stream:
                    cleanup.callback(path.unlink)
                    figure.savefig(stream, format="pdf", dpi=300,
                                   metadata={"CreationDate": None, "ModDate": None})
        cleanup.pop_all()
    return targets
