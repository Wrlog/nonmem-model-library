"""Figures for the dashboard, one per model, drawn from the simulated data.

Each figure shows the behaviour the model exists to describe, not every
column that exists: the concentration profile for PK, tumour trajectories by
arm for TGI, the lag between exposure and response for the indirect response
model, Kaplan-Meier by exposure for time to event, and observed response
against the fitted curve for the logistic model.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .theme import Theme, finish, legend, style_axes  # noqa: E402


def _obs(df: pd.DataFrame) -> pd.DataFrame:
    """Observation records only, with DV numeric."""
    out = df[df["MDV"].astype(int) == 0].copy()
    out["DV"] = pd.to_numeric(out["DV"], errors="coerce")
    return out.dropna(subset=["DV"])


def _quantile_band(ax, g, theme, colour, label_text):
    med = g.groupby("TIME")["DV"].median()
    lo = g.groupby("TIME")["DV"].quantile(0.1)
    hi = g.groupby("TIME")["DV"].quantile(0.9)
    ax.fill_between(med.index, lo, hi, color=colour, alpha=0.16, linewidth=0)
    ax.plot(med.index, med.values, color=colour, linewidth=2, label=label_text)


def pk_2cmt_iv(df: pd.DataFrame, theme: Theme) -> bytes:
    obs = _obs(df)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    style_axes(ax, theme)
    for _, g in obs.groupby("ID"):
        ax.plot(g["TIME"], g["DV"], color=theme.muted, linewidth=0.6, alpha=0.45)
    _quantile_band(ax, obs, theme, theme.series[0], "Median with 10th-90th percentile")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.set_yscale("log")
    ax.set_title("Simulated concentrations, three 500 mg infusions",
                 fontsize=11, loc="left", pad=8)
    legend(ax, theme, loc="upper right")
    fig.tight_layout()
    return finish(fig)


def tgi_claret(df: pd.DataFrame, theme: Theme) -> bytes:
    obs = _obs(df)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    style_axes(ax, theme)
    names = {0: "Placebo", 1: "Low exposure", 2: "High exposure"}
    for i, (arm, g) in enumerate(obs.groupby("ARM")):
        _quantile_band(ax, g, theme, theme.series[i % len(theme.series)],
                       names.get(int(arm), f"Arm {arm}"))
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Tumour size (mm)")
    ax.set_title("Tumour trajectories by exposure arm", fontsize=11, loc="left", pad=8)
    ax.text(0.99, 0.04,
            "Regrowth on treatment is the resistance term, not noise",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=theme.ink_3)
    legend(ax, theme, loc="upper left")
    fig.tight_layout()
    return finish(fig)


def pkpd_idr_inhibition(df: pd.DataFrame, theme: Theme) -> bytes:
    obs = _obs(df)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    style_axes(ax, theme)
    for i, (dose, g) in enumerate(obs.groupby("DOSE")):
        _quantile_band(ax, g, theme, theme.series[i % len(theme.series)],
                       f"{dose:g} mg")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Biomarker (units)")
    ax.set_title("Response lags exposure, and returns on its own clock",
                 fontsize=11, loc="left", pad=8)
    legend(ax, theme, loc="lower right", ncol=3)
    fig.tight_layout()
    return finish(fig)


def _km(times: np.ndarray, events: np.ndarray):
    """Kaplan-Meier estimate, written out rather than pulled from a package."""
    order = np.argsort(times)
    t, e = times[order], events[order]
    surv, step_t, step_s = 1.0, [0.0], [1.0]
    n = len(t)
    for i, ti in enumerate(t):
        at_risk = n - i
        if e[i] == 1 and at_risk > 0:
            surv *= 1 - 1 / at_risk
            step_t.append(ti)
            step_s.append(surv)
    step_t.append(t[-1] if len(t) else 0.0)
    step_s.append(surv)
    return np.array(step_t), np.array(step_s)


def tte_weibull(df: pd.DataFrame, theme: Theme) -> bytes:
    last = df[df["TIME"] > 0].copy()
    last["DV"] = pd.to_numeric(last["DV"])
    cuts = last["EXPO"].quantile([0, 1 / 3, 2 / 3, 1.0]).values
    labels = ["Low exposure", "Middle", "High exposure"]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    style_axes(ax, theme)
    for i in range(3):
        sel = last[(last["EXPO"] >= cuts[i]) & (last["EXPO"] <= cuts[i + 1])]
        t, s = _km(sel["TIME"].to_numpy(), sel["DV"].to_numpy())
        ax.step(t, s, where="post", color=theme.series[i], linewidth=2,
                label=f"{labels[i]} (n={len(sel)})")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Event-free probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("Kaplan-Meier by exposure tertile", fontsize=11, loc="left", pad=8)
    ax.text(0.99, 0.06, "Higher exposure, lower hazard in this simulation",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=theme.ink_3)
    legend(ax, theme, loc="lower left")
    fig.tight_layout()
    return finish(fig)


def logistic_binary(df: pd.DataFrame, theme: Theme, truth: dict | None = None) -> bytes:
    d = df.copy()
    d["DV"] = pd.to_numeric(d["DV"])
    bins = np.linspace(d["EXPO"].min(), d["EXPO"].max(), 9)
    mid = (bins[:-1] + bins[1:]) / 2
    idx = np.digitize(d["EXPO"], bins) - 1
    idx = np.clip(idx, 0, len(mid) - 1)
    rate = [d["DV"][idx == k].mean() if (idx == k).any() else np.nan
            for k in range(len(mid))]
    n_bin = [int((idx == k).sum()) for k in range(len(mid))]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    style_axes(ax, theme)
    sizes = 18 + 120 * np.array(n_bin) / max(max(n_bin), 1)
    ax.scatter(mid, rate, s=sizes, color=theme.series[0],
               edgecolors=theme.surface, linewidths=1.2, zorder=3,
               label="Observed rate per bin (area is n)")
    if truth:
        xs = np.linspace(d["EXPO"].min(), d["EXPO"].max(), 200)
        logit = truth["BASE"] + truth["SLOPE"] * xs
        ax.plot(xs, 1 / (1 + np.exp(-logit)), color=theme.series[1],
                linewidth=2, label="Simulated population curve")
    ax.set_xlabel("Exposure")
    ax.set_ylabel("Probability of response")
    ax.set_ylim(0, 1)
    ax.set_title("Exposure-response for a binary endpoint",
                 fontsize=11, loc="left", pad=8)
    legend(ax, theme, loc="upper left")
    fig.tight_layout()
    return finish(fig)


FIGURES = {
    "pk_2cmt_iv": pk_2cmt_iv,
    "tgi_claret": tgi_claret,
    "pkpd_idr_inhibition": pkpd_idr_inhibition,
    "tte_weibull": tte_weibull,
    "logistic_binary": logistic_binary,
}
