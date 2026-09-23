"""Goodness-of-fit and visual predictive check figures.

These read what the nlmixr2 fit wrote in fit/results/ and draw it here, so
every figure in the library comes out of the same plotting code and looks
like the others. If a model has not been fitted the functions return None
and the dashboard simply omits the panel rather than inventing one.

The four GOF panels are the conventional set: observations against
population and individual predictions, and conditional weighted residuals
against time and against prediction. A VPC follows, because GOF plots can
look acceptable for a model that still predicts the wrong spread.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .theme import Theme, finish, legend, style_axes


def load_fit(results_dir: Path, key: str) -> dict | None:
    """Return the fit artefacts for a model, or None if it was not fitted."""
    gof = results_dir / f"{key}_gof.csv"
    if not gof.exists():
        return None
    out: dict = {"gof": pd.read_csv(gof)}
    est = results_dir / f"{key}_estimates.csv"
    if est.exists():
        out["estimates"] = pd.read_csv(est)
    status = results_dir / f"{key}_status.json"
    if status.exists():
        import json
        out["status"] = json.loads(status.read_text(encoding="utf-8"))
    vpc = results_dir / f"{key}_vpc.csv"
    if vpc.exists():
        out["vpc"] = pd.read_csv(vpc)
    return out


def _unity(ax, values, theme: Theme) -> None:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    pad = (hi - lo) * 0.05 or 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color=theme.ink_3, linewidth=1, linestyle="--", zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def _loess_ish(x, y, bins=12):
    """A binned median trend: enough to see curvature, no extra dependency."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < bins * 3:
        return None, None
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    mids, meds = [], []
    for i in range(len(edges) - 1):
        sel = (x >= edges[i]) & (x <= edges[i + 1])
        if sel.sum() >= 3:
            mids.append(np.median(x[sel]))
            meds.append(np.median(y[sel]))
    return np.array(mids), np.array(meds)


def gof_panel(fit: dict, theme: Theme, log_scale: bool = False) -> bytes:
    """The standard four-panel goodness of fit."""
    d = fit["gof"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))

    # DV vs PRED and DV vs IPRED
    for ax, col, title in ((axes[0][0], "PRED", "Population prediction"),
                           (axes[0][1], "IPRED", "Individual prediction")):
        style_axes(ax, theme, xgrid=True)
        if col not in d:
            ax.set_visible(False)
            continue
        ax.scatter(d[col], d["DV"], s=14, color=theme.series[0], alpha=0.55,
                   edgecolors="none", zorder=2)
        _unity(ax, np.concatenate([d[col].to_numpy(), d["DV"].to_numpy()]), theme)
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(col)
        ax.set_ylabel("DV")
        ax.set_title(title, fontsize=10.5, loc="left", pad=6)

    # CWRES vs TIME and vs PRED
    for ax, col, title in ((axes[1][0], "TIME", "CWRES against time"),
                           (axes[1][1], "PRED", "CWRES against prediction")):
        style_axes(ax, theme, xgrid=True)
        if "CWRES" not in d or col not in d:
            ax.set_visible(False)
            continue
        ax.axhline(0, color=theme.ink_3, linewidth=1, linestyle="--", zorder=1)
        for y in (-2, 2):
            ax.axhline(y, color=theme.grid, linewidth=1, zorder=1)
        ax.scatter(d[col], d["CWRES"], s=14, color=theme.series[0], alpha=0.55,
                   edgecolors="none", zorder=2)
        mx, my = _loess_ish(d[col], d["CWRES"])
        if mx is not None:
            ax.plot(mx, my, color=theme.series[1], linewidth=2, zorder=3,
                    label="Binned median")
            legend(ax, theme, loc="upper right")
        ax.set_xlabel(col)
        ax.set_ylabel("CWRES")
        ax.set_title(title, fontsize=10.5, loc="left", pad=6)

    fig.tight_layout()
    return finish(fig)


def vpc_plot(fit: dict, observed: pd.DataFrame, theme: Theme,
             n_bins: int = 10, log_scale: bool = False) -> bytes | None:
    """Visual predictive check.

    Percentile bands come from the simulated replicates the fit produced;
    the lines are the same percentiles of the observed data. A model whose
    observed percentiles wander outside the simulated bands is predicting
    the wrong spread even if its GOF plots look tidy.
    """
    sim = fit.get("vpc")
    if sim is None or sim.empty:
        return None

    cols = {c.lower(): c for c in sim.columns}
    t_col = cols.get("time")
    y_col = cols.get("sim") or cols.get("dv")
    rep_col = cols.get("sim.id") or cols.get("sim_id")
    if not (t_col and y_col and rep_col):
        return None

    obs = observed[observed.get("MDV", 0).astype(int) == 0].copy() \
        if "MDV" in observed else observed.copy()
    obs["DV"] = pd.to_numeric(obs["DV"], errors="coerce")
    obs = obs.dropna(subset=["DV"])

    edges = np.unique(np.quantile(obs["TIME"], np.linspace(0, 1, n_bins + 1)))
    mids = (edges[:-1] + edges[1:]) / 2

    def pct_by_bin(frame, tcol, ycol, q):
        out = []
        for i in range(len(edges) - 1):
            sel = frame[(frame[tcol] >= edges[i]) & (frame[tcol] <= edges[i + 1])]
            out.append(np.nan if sel.empty else np.quantile(sel[ycol], q))
        return np.array(out)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    style_axes(ax, theme)

    # For each percentile, the spread of that percentile across replicates.
    for q, shade, name in ((0.05, 0.12, "5th"), (0.5, 0.20, "median"),
                           (0.95, 0.12, "95th")):
        per_rep = []
        for _, rep in sim.groupby(rep_col):
            per_rep.append(pct_by_bin(rep, t_col, y_col, q))
        per_rep = np.vstack(per_rep)
        lo = np.nanpercentile(per_rep, 2.5, axis=0)
        hi = np.nanpercentile(per_rep, 97.5, axis=0)
        colour = theme.series[0] if q == 0.5 else theme.series[2]
        ax.fill_between(mids, lo, hi, color=colour, alpha=shade, linewidth=0,
                        label=f"Simulated {name} (95% interval)")

    for q, style in ((0.05, ":"), (0.5, "-"), (0.95, ":")):
        ax.plot(mids, pct_by_bin(obs, "TIME", "DV", q), color=theme.ink,
                linewidth=1.6, linestyle=style,
                label="Observed median" if q == 0.5 else None)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Observation")
    ax.set_title("Visual predictive check", fontsize=11, loc="left", pad=8)
    legend(ax, theme, loc="best", ncol=2)
    fig.tight_layout()
    return finish(fig)


def estimates_vs_truth(fit: dict, truth: dict[str, float]) -> pd.DataFrame | None:
    """Join the estimates to the parameters the data was simulated from.

    The join is by name and is deliberately conservative: a parameter whose
    name cannot be matched is shown with a blank truth column rather than
    being paired with a guess.
    """
    est = fit.get("estimates")
    if est is None or est.empty:
        return None

    alias = {
        "tvcl": ("TVCL", 1), "tvv1": ("TVV1", 1), "tvq": ("TVQ", 1),
        "tvv2": ("TVV2", 1), "tvv": ("TVV", 1),
        "tvy0": ("TVY0", 1), "tvkl": ("TVKL", 1), "tvkd": ("TVKD", 1),
        "tvlam": ("TVLAMBDA", 1),
        "tvkin": ("TVKIN", 1), "tvkout": ("TVKOUT", 1), "tvic50": ("TVIC50", 1),
        "prop.err": ("PROP_ERR", 1),
    }
    rows = []
    for _, r in est.iterrows():
        name = str(r.get("parameter", "")).strip()
        key = alias.get(name.lower(), (None, 1))[0]
        true_value = truth.get(key) if key else None
        # nlmixr2 reports the back-transformed value in "Back-transformed"
        # when the parameter was estimated on the log scale.
        value = None
        for col in ("Back-transformed(95%CI)", "Back-transformed", "Estimate"):
            if col in est.columns and pd.notna(r.get(col)):
                value = r.get(col)
                break
        rows.append({
            "parameter": name,
            "estimate": value,
            "rse_pct": r.get("%RSE") if "%RSE" in est.columns else None,
            "truth": true_value,
        })
    return pd.DataFrame(rows)
