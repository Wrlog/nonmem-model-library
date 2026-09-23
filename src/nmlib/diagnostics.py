"""Diagnostic figures: goodness of fit, visual predictive check, recovery.

These read what `nmlib.estimate` wrote into `fit/results/` and draw it with
the same theme as everything else, so a diagnostic plot and a data plot in
this library look like they came from the same place.

They answer different questions, and the order matters:

* **Goodness of fit** asks whether the model describes the data it was
  fitted to. It is necessary and it is weak: a model can pass all four
  panels and still be wrong about everything that was not observed.
* **The visual predictive check** asks whether data simulated from the
  fitted model looks like the data that was observed -- not just in the
  middle, but in the spread. Goodness of fit plots can look tidy for a
  model that predicts the wrong variability; a VPC cannot.
* **Individual fits** ask whether it holds for single subjects rather than
  only on average, which is the thing a mixed effects model exists to
  describe and the thing every population-level plot averages away.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .theme import (
    SIZE_LABEL,
    SIZE_NOTE,
    SIZE_SUBTITLE,
    SIZE_TITLE,
    Theme,
    finish,
    legend,
    panel,
    plain_log_ticks,
    style_axes,
    title_block,
)


def load_fit(results_dir: Path, key: str) -> dict | None:
    """Everything `estimate` wrote for one model, or None if it was not fitted."""
    est = results_dir / f"{key}_estimates.csv"
    if not est.exists():
        return None
    out: dict = {"estimates": pd.read_csv(est)}
    for name in ("gof", "vpc", "etas"):
        path = results_dir / f"{key}_{name}.csv"
        if path.exists():
            out[name] = pd.read_csv(path)
    status = results_dir / f"{key}_status.json"
    if status.exists():
        out["status"] = json.loads(status.read_text(encoding="utf-8"))
    return out


def _binned_median(x, y, bins: int = 12):
    """A binned median trend: enough to see curvature, no extra dependency."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < bins * 3:
        return None, None
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    mids, meds = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (x >= lo) & (x <= hi)
        if sel.sum() >= 3:
            mids.append(np.median(x[sel]))
            meds.append(np.median(y[sel]))
    return np.array(mids), np.array(meds)


def _unity(ax, values, theme: Theme, log: bool) -> None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0 if log else True)]
    lo, hi = float(np.min(v)), float(np.max(v))
    if log:
        lo, hi = lo * 0.75, hi * 1.35
    else:
        pad = (hi - lo) * 0.06 or 1.0
        lo, hi = lo - pad, hi + pad
    # The only dashed line in the library: it is a reference the eye must
    # separate from the data, not a gridline.
    ax.plot([lo, hi], [lo, hi], color=theme.ink_3, linewidth=1.1,
            linestyle=(0, (4, 3)), zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def gof_panel(fit: dict, theme: Theme, log_scale: bool = False) -> bytes:
    """The conventional four panels, drawn to the house style."""
    d = fit["gof"]
    fig, axes = panel(theme, nrows=2, ncols=2, width=8.4, height=6.8)

    for ax, col, title, sub in (
        (axes[0][0], "PRED", "Observed against population prediction",
         "Spread around unity here is between-subject variability"),
        (axes[0][1], "IPRED", "Observed against individual prediction",
         "Tight around unity: what is left once each subject's etas are in"),
    ):
        style_axes(ax, theme, xgrid=True)
        ax.scatter(d[col], d["DV"], s=11, color=theme.series[0], alpha=0.45,
                   edgecolors="none", zorder=3)
        _unity(ax, np.concatenate([d[col].to_numpy(), d["DV"].to_numpy()]),
               theme, log_scale)
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            plain_log_ticks(ax, "xy")
        ax.set_xlabel(col)
        ax.set_ylabel("DV")
        title_block(ax, theme, title, sub, small=True)

    for ax, col, title, xlabel in (
        (axes[1][0], "TIME", "Conditional weighted residuals against time",
         "Time"),
        (axes[1][1], "PRED", "Conditional weighted residuals against prediction",
         "PRED"),
    ):
        style_axes(ax, theme, xgrid=True)
        ax.axhline(0, color=theme.ink_3, linewidth=1.1, linestyle=(0, (4, 3)),
                   zorder=2)
        for y in (-2, 2):
            ax.axhline(y, color=theme.grid, linewidth=0.8, zorder=1)
        ax.scatter(d[col], d["CWRES"], s=11, color=theme.series[0], alpha=0.45,
                   edgecolors="none", zorder=3)
        mx, my = _binned_median(d[col], d["CWRES"])
        if mx is not None:
            ax.plot(mx, my, color=theme.series[1], linewidth=2, zorder=4,
                    solid_capstyle="round", label="Binned median")
            legend(ax, theme, loc="upper right")
        if log_scale and col == "PRED":
            ax.set_xscale("log")
            plain_log_ticks(ax, "x")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CWRES")
        title_block(ax, theme, title,
                    "Flat on zero, most points inside ±2", small=True)

    fig.tight_layout(h_pad=2.4, w_pad=2.0)
    return finish(fig)


def vpc_plot(fit: dict, observed: pd.DataFrame, theme: Theme,
             n_bins: int = 9, log_scale: bool = False,
             y_label: str = "Observation",
             x_label: str = "Time") -> bytes | None:
    """Visual predictive check.

    For each of the 5th, 50th and 95th percentiles of the data, the shaded
    band is where that percentile fell across the simulated replicates, and
    the line is where it actually fell in the observed data. A line that
    wanders outside its own band means the fitted model is predicting the
    wrong thing at that percentile -- which, for the outer two, is a
    statement about variability that no goodness-of-fit plot makes.
    """
    sim = fit.get("vpc")
    if sim is None or sim.empty:
        return None

    obs = observed[observed["MDV"].astype(int) == 0].copy() \
        if "MDV" in observed else observed.copy()
    obs["DV"] = pd.to_numeric(obs["DV"], errors="coerce")
    obs = obs.dropna(subset=["DV"])

    edges = np.unique(np.quantile(obs["TIME"], np.linspace(0, 1, n_bins + 1)))
    mids = np.array([obs["TIME"][(obs["TIME"] >= lo) & (obs["TIME"] <= hi)].median()
                     for lo, hi in zip(edges[:-1], edges[1:], strict=True)])

    def bin_masks(times: np.ndarray) -> list[np.ndarray]:
        return [(times >= lo) & (times <= hi)
                for lo, hi in zip(edges[:-1], edges[1:], strict=True)]

    def pct_by_bin(times: np.ndarray, values: np.ndarray, q: float):
        return np.array([np.nan if not m.any() else np.quantile(values[m], q)
                         for m in bin_masks(times)])

    # Every replicate shares the same design, so the simulated values form a
    # (replicate, record) rectangle and the percentile of each bin can be
    # taken for all replicates at once. Grouping the long frame by replicate
    # instead means a few tens of thousands of dataframe filters per figure,
    # which dominates the whole build.
    n_rep = int(sim["rep"].nunique())
    sim_time = sim["TIME"].to_numpy(dtype=float)[:len(sim) // n_rep]
    sim_dv = sim["DV"].to_numpy(dtype=float).reshape(n_rep, -1)
    masks = bin_masks(sim_time)

    fig, ax = panel(theme, height=4.6)

    # The simulated interval for each percentile, across replicates.
    outer = theme.ramp[1] if theme.name == "light" else theme.ramp[-2]
    for q, colour in ((0.05, outer), (0.50, theme.series[0]), (0.95, outer)):
        per_rep = np.column_stack([
            np.full(n_rep, np.nan) if not m.any()
            else np.quantile(sim_dv[:, m], q, axis=1) for m in masks])
        lo = np.nanpercentile(per_rep, 2.5, axis=0)
        hi = np.nanpercentile(per_rep, 97.5, axis=0)
        ax.fill_between(mids, lo, hi, color=colour, alpha=0.30, linewidth=0,
                        zorder=2,
                        label=("Simulated interval, median" if q == 0.5 else
                               ("Simulated interval, 5th and 95th"
                                if q == 0.05 else None)))

    obs_time = obs["TIME"].to_numpy(dtype=float)
    obs_dv = obs["DV"].to_numpy(dtype=float)
    for q, style, label in ((0.05, (0, (3, 2)), "Observed 5th and 95th"),
                            (0.50, "solid", "Observed median"),
                            (0.95, (0, (3, 2)), None)):
        ax.plot(mids, pct_by_bin(obs_time, obs_dv, q), color=theme.ink,
                linewidth=1.7, linestyle=style, zorder=4, label=label)

    if log_scale:
        ax.set_yscale("log")
        plain_log_ticks(ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    n_rep = int(sim["rep"].nunique())
    title_block(ax, theme, "Visual predictive check",
                f"Observed percentiles against {n_rep} replicates simulated "
                "from the fitted model")
    # Below the axes: four entries will land on the data wherever they go.
    legend(ax, theme, loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, -0.19))
    fig.tight_layout()
    return finish(fig)


# --------------------------------------------------------------------------
# Parameter recovery
# --------------------------------------------------------------------------

def recovery_rows(estimates: pd.DataFrame) -> pd.DataFrame:
    """Estimate, 95% interval and simulated value, as ratios to the truth.

    Ratios rather than raw values, because a clearance in L/h and a
    coefficient of variation cannot share an axis but their ratios to the
    truth can. One is the target for all of them.

    A parameter is only included when the question can actually be asked of
    it: there has to be a value to compare against, and a standard error to
    build the interval from. Keeping a parameter whose standard error failed
    to compute would score it as *not* recovered, which reports a numerical
    problem as a statistical result.
    """
    d = estimates.dropna(subset=["truth", "se"]).copy()
    d = d[d["truth"] != 0]
    d = d[np.isfinite(d["se"]) & np.isfinite(d["estimate"])]
    d["ratio"] = d["estimate"] / d["truth"]
    d["lo"] = (d["estimate"] - 1.96 * d["se"]) / d["truth"]
    d["hi"] = (d["estimate"] + 1.96 * d["se"]) / d["truth"]
    # A negative truth flips the interval, so put the ends back in order.
    lo, hi = np.minimum(d["lo"], d["hi"]), np.maximum(d["lo"], d["hi"])
    d["lo"], d["hi"] = lo, hi
    d["covers"] = (d["lo"] <= 1.0) & (d["hi"] >= 1.0)
    return d


# --------------------------------------------------------------------------
# Individual fits
# --------------------------------------------------------------------------

def individual_fits(fit: dict, theme: Theme, log_scale: bool = False,
                    n_subjects: int = 8, y_label: str = "Observation",
                    x_label: str = "Time") -> bytes | None:
    """A sample of subjects, each with their own fit drawn through their data.

    Population-level plots average over exactly the thing a mixed effects
    model exists to describe. This is the panel that shows whether the
    structure holds for individual people rather than only on average, and
    it is where a model that is wrong in a way the averages hide -- a shape
    it cannot bend to, a subject it cannot reach -- shows it.

    The subjects are chosen at even quantiles of their own median
    observation, so the panel spans the range of the data instead of
    showing eight typical subjects.
    """
    d = fit.get("gof")
    if d is None or d.empty:
        return None

    order = d.groupby("ID")["DV"].median().sort_values()
    if len(order) == 0:
        return None
    picks = order.index[
        np.unique(np.linspace(0, len(order) - 1, n_subjects).astype(int))]

    ncols = 4
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = panel(theme, nrows=nrows, ncols=ncols, width=8.4,
                      height=2.05 * nrows + 0.8, sharex=True)
    flat = axes.ravel()

    for ax, subject in zip(flat, picks, strict=False):
        g = d[d["ID"] == subject].sort_values("TIME")
        ax.plot(g["TIME"], g["PRED"], color=theme.ink_3, linewidth=1.3,
                linestyle=(0, (4, 3)), zorder=2, label="Population")
        ax.plot(g["TIME"], g["IPRED"], color=theme.series[0], linewidth=1.8,
                zorder=3, solid_capstyle="round", label="Individual")
        ax.scatter(g["TIME"], g["DV"], s=17, color=theme.ink,
                   edgecolors=theme.surface, linewidths=1.0, zorder=4,
                   label="Observed")
        if log_scale:
            ax.set_yscale("log")
            plain_log_ticks(ax)
        ax.set_title(f"Subject {int(subject)}", fontsize=SIZE_NOTE,
                     loc="left", color=theme.ink_3, pad=4)

    for ax in flat[len(picks):]:
        ax.set_visible(False)
    for ax in flat[:len(picks)][-ncols:]:
        ax.set_xlabel(x_label)
    for row in range(nrows):
        flat[row * ncols].set_ylabel(y_label)

    handles, labels_ = flat[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels_, frameon=False, ncol=3,
                     loc="upper center", bbox_to_anchor=(0.5, 0.055),
                     fontsize=SIZE_LABEL)
    for text in leg.get_texts():
        text.set_color(theme.ink_2)

    # Both lines are placed explicitly. suptitle positions itself relative to
    # the axes, which puts it on top of a subtitle drawn in figure
    # coordinates.
    fig.text(0.0, 1.0, f"{len(picks)} subjects, spanning the range of the data",
             ha="left", va="top", fontsize=SIZE_TITLE, color=theme.ink,
             fontweight="600")
    fig.text(0.0, 0.962,
             "Dashed is the population prediction, solid the individual fit; "
             "the gap between them is that subject's random effects",
             ha="left", va="top", fontsize=SIZE_SUBTITLE, color=theme.ink_3)
    fig.tight_layout(rect=(0, 0.07, 1, 0.925), h_pad=1.9, w_pad=1.4)
    return finish(fig)


# --------------------------------------------------------------------------
# Does the distinguishing feature earn its place?
# --------------------------------------------------------------------------

#: Change in objective function at p = 0.05 for one and two parameters.
CHI2_95 = {1: 3.84, 2: 5.99, 3: 7.81}


def comparison_plot(rows: list[dict], theme: Theme) -> bytes | None:
    """How much worse the simpler model fits, for every model in the library.

    Each bar is the increase in objective function when that model's
    distinguishing feature is switched off and everything else is
    re-estimated. The marker is the 95% threshold for the number of
    parameters given up, so a bar reaching past it is a feature that pays
    for itself on this data.

    The scale is logarithmic because the answers are not remotely the same
    size, and that is itself the finding: the tumour model's resistance
    term is not a marginal improvement, it is the difference between a
    model that can bend the way the data bends and one that cannot.
    """
    rows = [r for r in rows if r and r.get("delta_ofv") is not None]
    if not rows:
        return None

    fig, ax = panel(theme, width=8.4, height=0.72 * len(rows) + 2.0)
    style_axes(ax, theme, xgrid=True, ygrid=False)

    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows, strict=True):
        delta = max(float(r["delta_ofv"]), 0.01)
        threshold = CHI2_95.get(int(r["df"] or 0))
        beats = threshold is not None and delta > threshold
        colour = theme.series[0] if (beats or not r["nested"]) else theme.warning
        ax.plot([0.01, delta], [y, y], color=colour, linewidth=3.0,
                solid_capstyle="round", alpha=0.85, zorder=3)
        ax.scatter([delta], [y], s=40, color=colour, zorder=4,
                   edgecolors=theme.surface, linewidths=1.4)
        label = f"+{delta:.0f}"
        if r["nested"] and r.get("p_value") is not None:
            label += f"   p {'<' if r['p_value'] < 1e-4 else '='} " + (
                "0.0001" if r["p_value"] < 1e-4 else f"{r['p_value']:.3g}")
        else:
            label += "   same parameter count, so no p-value"
        ax.annotate(label, xy=(delta, y), xytext=(10, 0),
                    textcoords="offset points", va="center",
                    fontsize=SIZE_NOTE, color=theme.ink_2)
        if threshold is not None:
            ax.scatter([threshold], [y], marker="|", s=150,
                       color=theme.ink_3, zorder=5, linewidths=1.4)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["feature"] for r in rows], fontsize=SIZE_LABEL)
    ax.set_xscale("log")
    ax.set_xlim(0.8, max(float(r["delta_ofv"]) for r in rows) * 9)
    plain_log_ticks(ax, "x")
    ax.set_xlabel("Increase in objective function without the feature "
                  "(log scale)")
    ax.set_ylim(-0.8, len(rows) - 0.2)
    title_block(ax, theme, "Does each model's distinguishing feature earn it?",
                "Bar is the cost of dropping it; the tick is the 95% "
                "threshold for the parameters given up")
    fig.tight_layout()
    return finish(fig)
