"""Diagnostic figures: goodness of fit, visual predictive check, recovery.

These read what `nmlib.estimate` wrote into `fit/results/` and draw it with
the same theme as everything else, so a diagnostic plot and a data plot in
this library look like they came from the same place.

The three answer different questions, and the order matters:

* **Goodness of fit** asks whether the model describes the data it was
  fitted to. It is necessary and it is weak: a model can pass all four
  panels and still be wrong about everything that was not observed.
* **The visual predictive check** asks whether data simulated from the
  fitted model looks like the data that was observed -- not just in the
  middle, but in the spread. Goodness of fit plots can look tidy for a
  model that predicts the wrong variability; a VPC cannot.
* **Recovery** asks whether estimation found the parameters the data was
  simulated from. Nothing in the first two can answer that, and it is the
  only one of the three that needs a simulated dataset to be askable at
  all. It is the reason this library simulates rather than shipping a real
  dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .theme import (
    SIZE_LABEL,
    SIZE_NOTE,
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
    for name, attr in (("gof", "gof"), ("vpc", "vpc")):
        path = results_dir / f"{key}_{name}.csv"
        if path.exists():
            out[attr] = pd.read_csv(path)
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


def recovery_plot(per_model: dict[str, tuple[str, pd.DataFrame]],
                  theme: Theme) -> bytes:
    """Every estimated parameter in the library against the value behind it.

    This is the figure the rest of the library exists to make possible.
    Goodness of fit and a predictive check both ask whether the model
    agrees with the data; only a simulated dataset lets you ask whether
    estimation recovered the answer, and that is a different question with
    a different failure mode.

    The mark is the estimate divided by the value the data was simulated
    from, so one is the target for every row whatever its units, with the
    95% confidence interval as a whisker. Whether that whisker crosses one
    is the test; the distance from one is only how far off this particular
    dataset landed, and a tight interval sitting slightly off one is a
    better result than a wide interval centred on it.
    """
    blocks = []
    for key, (title, est) in per_model.items():
        rows = recovery_rows(est)
        if not rows.empty:
            blocks.append((key, title, rows))

    n_rows = sum(len(r) for _, _, r in blocks)
    height = 1.4 + 0.27 * n_rows + 0.30 * len(blocks)
    fig, ax = panel(theme, width=8.4, height=height)
    style_axes(ax, theme, xgrid=True, ygrid=False)

    y = 0.0
    ticks, labels, group_marks = [], [], []
    for _, title, rows in reversed(blocks):
        group_start = y
        for _, r in rows.iloc[::-1].iterrows():
            covered = bool(r["covers"])
            colour = theme.series[0] if covered else theme.warning
            if np.isfinite(r["lo"]) and np.isfinite(r["hi"]):
                ax.plot([r["lo"], r["hi"]], [y, y], color=colour, linewidth=1.6,
                        alpha=0.75, solid_capstyle="round", zorder=3)
            ax.scatter([r["ratio"]], [y], s=34, color=colour, zorder=4,
                       edgecolors=theme.surface, linewidths=1.4)
            if not covered:
                # A status colour never carries meaning on its own.
                ax.annotate("! interval excludes the simulated value",
                            xy=(r["hi"], y), xytext=(8, 0),
                            textcoords="offset points", va="center",
                            fontsize=SIZE_NOTE, color=theme.warning)
            ticks.append(y)
            labels.append(str(r["parameter"]))
            y += 1.0
        group_marks.append((group_start - 0.55, y - 1.0, title))
        y += 1.5

    ax.axvline(1.0, color=theme.ink_3, linewidth=1.2, zorder=2)
    for lo, hi, title in group_marks:
        ax.annotate(title, xy=(0, hi + 0.62), xycoords=("axes fraction", "data"),
                    xytext=(2, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=SIZE_LABEL, color=theme.ink,
                    fontweight="600")
        ax.axhline(lo, color=theme.grid, linewidth=0.8, zorder=1)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=SIZE_NOTE)
    ax.set_ylim(-1.0, y + 0.1)
    ax.set_xscale("log")
    # Fixed ticks on a log axis, and matplotlib's minor decade labels turned
    # off -- left on, they put a stray "6 x 10^-1" among the "0.67x" labels.
    ax.set_xticks([0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0])
    ax.set_xticklabels(["0.5x", "0.67x", "0.8x", "1x", "1.25x", "1.5x", "2x"])
    ax.minorticks_off()
    spread = np.concatenate([r[["lo", "hi", "ratio"]].to_numpy().ravel()
                             for _, _, r in blocks])
    spread = spread[np.isfinite(spread) & (spread > 0)]
    ax.set_xlim(min(0.62, float(spread.min()) * 0.92),
                max(1.6, float(spread.max()) * 1.08))
    ax.set_xlabel("Estimate divided by the value the data was simulated from")
    covered = sum(int(r["covers"].sum()) for _, _, r in blocks)
    title_block(ax, theme, f"{covered} of {n_rows} parameters recovered",
                "Each estimate divided by the value its data was simulated "
                "from, with a 95% confidence interval")
    fig.tight_layout()
    return finish(fig)
