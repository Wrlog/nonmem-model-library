"""One figure per model, drawn from the simulated data.

Each figure shows the behaviour its model exists to describe, not every
column that happens to be in the file: the concentration-time profile for
PK, tumour shrinkage and regrowth for TGI, the lag between exposure and
response for the indirect response model, survival by exposure for time to
event, and the exposure-response curve for the binary endpoint.

Rules that hold across all of them:

* Ordered groupings -- dose levels, exposure tertiles, treatment arms --
  take steps of one hue rather than categorical colours, because the order
  is information the reader should get for free.
* Every series is labelled at its right-hand end, so a colour never has to
  be decoded against a legend to be read.
* Where subject-level variation in the baseline would otherwise swamp the
  effect, the endpoint is shown relative to each subject's own baseline.
  That is how these endpoints are reported in practice, and here it is the
  difference between a figure that shows the model's behaviour and one that
  shows the spread of starting values.
* The subtitle states what the figure shows, so the caption underneath does
  not repeat the axis labels.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .theme import (  # noqa: E402
    SIZE_NOTE,
    Theme,
    finish,
    label_point,
    legend,
    note,
    panel,
    plain_log_ticks,
    style_axes,
    title_block,
)


def _obs(df: pd.DataFrame) -> pd.DataFrame:
    """Observation records only, with DV numeric."""
    out = df[df["MDV"].astype(int) == 0].copy() if "MDV" in df else df.copy()
    out["DV"] = pd.to_numeric(out["DV"], errors="coerce")
    return out.dropna(subset=["DV"])


def _relative_to_baseline(obs: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """Each subject's DV as a percentage of (or change from) their own first value.

    Between-subject variability in the baseline is real, but on an absolute
    axis it is often the largest thing in the picture and it hides the
    effect the model is about. Dividing it out is what these endpoints do in
    practice.
    """
    out = obs.sort_values(["ID", "TIME"]).copy()
    base = out.groupby("ID")["DV"].transform("first")
    out["DV"] = (out["DV"] / base - 1.0) * 100.0 if pct else out["DV"] / base
    return out


def _band(ax, g, colour, label_text, alpha=0.14, lw=2.0, ribbon=True):
    """Median with a 10th-90th percentile ribbon, the house summary mark."""
    med = g.groupby("TIME")["DV"].median()
    if ribbon:
        lo = g.groupby("TIME")["DV"].quantile(0.10)
        hi = g.groupby("TIME")["DV"].quantile(0.90)
        ax.fill_between(med.index, lo, hi, color=colour, alpha=alpha,
                        linewidth=0, zorder=2)
    ax.plot(med.index, med.values, color=colour, linewidth=lw, zorder=4,
            solid_capstyle="round", label=label_text)
    return med


def _end_labels(ax, theme: Theme, items, dx: float = 8) -> None:
    """Direct labels at the right-hand ends, nudged apart so none collide.

    Series ends often sit within a few pixels of each other, which is
    exactly where a direct label stops being an improvement on a legend.
    Each label is pushed up to a minimum spacing from the one below it, and
    the whole stack slides back down if that ran it off the top.
    """
    # Spacing is worked out in axes fractions rather than data units, so the
    # same rule holds on a log axis as on a linear one.
    to_axes = ax.transAxes.inverted()
    ordered = sorted(items, key=lambda it: it[1])
    fractions = [float(to_axes.transform(ax.transData.transform((x, y)))[1])
                 for x, y, _, _ in ordered]

    min_gap = 0.075
    placed: list[float] = []
    for frac in fractions:
        placed.append(frac if not placed else max(frac, placed[-1] + min_gap))
    overflow = placed[-1] - 0.97
    if overflow > 0:
        placed = [p - overflow for p in placed]

    for (x, _, colour, text), y_frac in zip(ordered, placed, strict=True):
        ax.annotate(text, xy=(x, y_frac), xycoords=("data", "axes fraction"),
                    xytext=(dx, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=SIZE_NOTE, color=colour,
                    annotation_clip=False)


def _headroom(ax, right: float = 0.16) -> None:
    """Leave room on the right for the direct labels."""
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, lo + (hi - lo) * (1 + right))


def _pct_ticks(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))


# --------------------------------------------------------------------------
# Pharmacokinetics
# --------------------------------------------------------------------------

def pk_2cmt_iv(df: pd.DataFrame, theme: Theme) -> bytes:
    obs = _obs(df)
    fig, ax = panel(theme, height=4.4)

    for _, g in obs.groupby("ID"):
        ax.plot(g["TIME"], g["DV"], color=theme.muted, linewidth=0.6,
                alpha=0.5, zorder=1)
    med = _band(ax, obs, theme.series[0], None)

    for t in (0.0, 24.0, 48.0):
        ax.axvline(t, color=theme.grid, linewidth=0.7, zorder=0)
    ax.annotate("Doses at 0, 24 and 48 h", xy=(24, 1.0),
                xycoords=("data", "axes fraction"), xytext=(6, -8),
                textcoords="offset points", fontsize=SIZE_NOTE,
                color=theme.ink_3, va="top")

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.set_yscale("log")
    plain_log_ticks(ax)
    title_block(ax, theme, "Concentration after three 500 mg infusions",
                f"One faint line per subject ({obs['ID'].nunique()} of them), "
                "median in blue with its 10th-90th percentile band")
    label_point(ax, theme, med.index[-1], med.values[-1], "Median",
                colour=theme.series[0])
    _headroom(ax, 0.10)
    fig.tight_layout()
    return finish(fig)


# --------------------------------------------------------------------------
# Oncology
# --------------------------------------------------------------------------

def tgi_claret(df: pd.DataFrame, theme: Theme) -> bytes:
    """Fold change from each subject's own baseline, on a log axis.

    Two choices, both forced by the data. Against baseline, because the
    spread of starting tumour sizes is otherwise the largest thing in the
    picture. On a log axis, because the growth is exponential: untreated
    tumours reach thirteen times baseline, and on a linear axis that ceiling
    squashes the treated arms -- whose shrink-and-regrow shape is the entire
    reason this model exists -- into the bottom two percent of the plot.
    Exponential growth is a straight line here, so a change in slope is a
    change in growth rate and can be read directly.
    """
    obs = _relative_to_baseline(_obs(df), pct=False)
    fig, ax = panel(theme, height=4.5)

    names = {0: "Placebo", 1: "Low exposure", 2: "High exposure"}
    arms = sorted(obs["ARM"].unique())
    colours = theme.ordinal(len(arms))       # exposure is ordered, not nominal
    ends = []
    for colour, arm in zip(colours, arms, strict=True):
        g = obs[obs["ARM"] == arm]
        label = names.get(int(arm), f"Arm {arm}")
        med = _band(ax, g, colour, label, alpha=0.11)
        ends.append((med.index[-1], med.values[-1], colour, label))

    ax.set_yscale("log")
    ax.axhline(1.0, color=theme.border, linewidth=1.0, zorder=3)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_ylim(0.45, 25)

    top_arm = _band_medians(obs[obs["ARM"] == arms[-1]])
    nadir_y, nadir_t = min((v, k) for k, v in top_arm.items())
    ax.annotate(f"High exposure bottoms out at {nadir_y:.2f}x on day "
                f"{nadir_t:g}, then regrows on continuing treatment",
                xy=(nadir_t, nadir_y), xytext=(14, -18),
                textcoords="offset points", fontsize=SIZE_NOTE,
                color=theme.ink_3, va="center",
                arrowprops=dict(arrowstyle="-", color=theme.border,
                                linewidth=0.9, shrinkA=0, shrinkB=4))

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Tumour size, relative to baseline")
    title_block(ax, theme, "Tumour size against each subject's own baseline",
                "The high-exposure arm shrinks, then regrows while treatment "
                "continues - that turn is the resistance term, not noise")
    _headroom(ax, 0.22)
    _end_labels(ax, theme, ends)
    legend(ax, theme, loc="upper left")
    fig.tight_layout()
    return finish(fig)


def _band_medians(g: pd.DataFrame) -> dict[float, float]:
    med = g.groupby("TIME")["DV"].median()
    return dict(zip(med.index.tolist(), med.values.tolist(), strict=True))


# --------------------------------------------------------------------------
# PK/PD
# --------------------------------------------------------------------------

def pkpd_idr_inhibition(df: pd.DataFrame, theme: Theme,
                        truth: dict | None = None) -> bytes:
    """Driving concentration above, biomarker response below, one shared axis.

    Two stacked panels rather than two y-scales on one plot: a second axis
    would let the curves be slid against each other until they look however
    the author wants, and *when* the response happens relative to the
    exposure is the entire point of the figure.
    """
    obs = _obs(df)
    rel = _relative_to_baseline(obs)
    fig, (top, bottom) = panel(theme, nrows=2, height=5.9, sharex=True,
                               gridspec_kw={"height_ratios": [1, 2.1]})

    doses = sorted(obs["DOSE"].unique())
    colours = theme.ordinal(len(doses))
    t_max = float(obs["TIME"].max())

    # Dose times come from the data rather than being assumed, so the figure
    # cannot drift out of step with the design it is drawing.
    dose_times = np.unique(
        df.loc[df["EVID"].astype(int) == 1, "TIME"].to_numpy(dtype=float))
    last_dose = float(dose_times.max()) if dose_times.size else 0.0

    if truth:
        ke = truth["TVCL"] / truth["TVV"]
        t = np.linspace(0, t_max, 1200)
        floor = truth["TVIC50"] / 50.0
        elapsed = t[:, None] - dose_times[None, :]
        shape = np.sum(np.where(elapsed >= 0, np.exp(-ke * elapsed), 0.0), axis=1)
        for colour, dose in zip(colours, doses, strict=True):
            c = (dose / truth["TVV"]) * shape
            top.plot(t, np.maximum(c, floor / 2), color=colour, linewidth=1.5,
                     solid_capstyle="round")
        top.axhline(truth["TVIC50"], color=theme.ink_3, linewidth=1.0,
                    linestyle=(0, (4, 3)), zorder=1)
        top.set_yscale("log")
        top.set_ylim(floor, (max(doses) / truth["TVV"]) * np.max(shape) * 1.6)
        plain_log_ticks(top)
        top.set_ylabel("Concentration\n(mg/L)")
        # Above the line, not on it: at this width the label would otherwise
        # sit across the dashes and read as struck through.
        label_point(top, theme, t_max, truth["TVIC50"],
                    f"IC50 {truth['TVIC50']:g} mg/L", dx=-4, dy=7,
                    ha="right", va="bottom", colour=theme.ink_3)

    for ax in (top, bottom):
        ax.axvline(last_dose, color=theme.grid, linewidth=0.9, zorder=0)
    top.annotate("Last of 7 daily doses", xy=(last_dose, 1.0),
                 xycoords=("data", "axes fraction"), xytext=(6, -8),
                 textcoords="offset points", fontsize=SIZE_NOTE,
                 color=theme.ink_3, va="top")

    title_block(top, theme, "Response lags exposure, and leaves on its own clock",
                "Concentration reaches steady state within days; the "
                "biomarker is still falling when dosing stops, and still "
                "recovering weeks later")

    # Medians only. Four overlapping percentile ribbons on one axis turn into
    # a single grey wash that hides the very ordering the figure is about,
    # and the spread has a figure of its own: the predictive check below.
    # Labels go at each curve's nadir rather than at its right-hand end,
    # because by the end of follow-up every dose has returned to baseline and
    # end labels would be four names on four indistinguishable points.
    for colour, dose in zip(colours, doses, strict=True):
        g = rel[rel["DOSE"] == dose]
        med = _band(bottom, g, colour, f"{dose:g} mg", ribbon=False)
        nadir_t = med.idxmin()
        label_point(bottom, theme, nadir_t, med.loc[nadir_t], f"{dose:g} mg",
                    dx=9, ha="left", va="center", colour=colour)

    bottom.axhline(0, color=theme.border, linewidth=1.0, zorder=3)
    bottom.set_xlabel("Time (h)")
    bottom.set_ylabel("Biomarker, change from baseline")
    _pct_ticks(bottom)
    note(bottom, theme,
         "Washout is set by the biomarker's own turnover, not by the drug's "
         "half-life", y=0.04, x=0.985)
    legend(bottom, theme, loc="upper center", ncol=len(doses),
           bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    return finish(fig)


# --------------------------------------------------------------------------
# Survival
# --------------------------------------------------------------------------

def _km(times: np.ndarray, events: np.ndarray):
    """Kaplan-Meier estimate with Greenwood standard errors.

    Written out rather than pulled from a package: it is a dozen lines, and
    the survival figure is the one place a reader is most likely to want to
    check what was actually computed.
    """
    order = np.argsort(times)
    t, e = times[order], events[order]
    n = len(t)
    step_t, step_s, step_se = [0.0], [1.0], [0.0]
    surv, gw = 1.0, 0.0
    for i, ti in enumerate(t):
        at_risk = n - i
        if e[i] == 1 and at_risk > 1:
            surv *= 1 - 1 / at_risk
            gw += 1 / (at_risk * (at_risk - 1))
            step_t.append(ti)
            step_s.append(surv)
            step_se.append(surv * np.sqrt(gw))
    step_t.append(float(t[-1]) if n else 0.0)
    step_s.append(surv)
    step_se.append(step_se[-1])
    return np.array(step_t), np.array(step_s), np.array(step_se)


def tte_weibull(df: pd.DataFrame, theme: Theme) -> bytes:
    last = df[df["TIME"] > 0].copy()
    last["DV"] = pd.to_numeric(last["DV"])
    cuts = last["EXPO"].quantile([0, 1 / 3, 2 / 3, 1.0]).to_numpy()
    labels = ["Low exposure", "Middle tertile", "High exposure"]
    colours = theme.ordinal(3)

    fig, (ax, risk) = panel(theme, nrows=2, height=5.3, sharex=True,
                            gridspec_kw={"height_ratios": [4.0, 1.0]})

    groups, ends = [], []
    for i in range(3):
        sel = last[(last["EXPO"] >= cuts[i]) & (last["EXPO"] <= cuts[i + 1])]
        groups.append(sel)
        t, s, se = _km(sel["TIME"].to_numpy(), sel["DV"].to_numpy())
        ax.fill_between(t, np.clip(s - 1.96 * se, 0, 1),
                        np.clip(s + 1.96 * se, 0, 1), step="post",
                        color=colours[i], alpha=0.14, linewidth=0, zorder=2)
        ax.step(t, s, where="post", color=colours[i], linewidth=2, zorder=3,
                label=f"{labels[i]} (n={len(sel)})")
        ends.append((t[-1], s[-1], colours[i], labels[i]))

    ax.set_ylabel("Event-free probability")
    ax.set_ylim(0, 1.02)
    title_block(ax, theme, "Time to first event, by exposure tertile",
                "Bands are 95% Greenwood intervals; the tertiles separate "
                "steadily, so the exposure effect is on the hazard itself")
    legend(ax, theme, loc="lower left")
    _headroom(ax, 0.15)
    _end_labels(ax, theme, ends)

    # Numbers at risk: a survival plot is incomplete without them, because
    # the right-hand tail of a KM curve can rest on very few subjects.
    style_axes(risk, theme, ygrid=False)
    ticks = np.linspace(0, float(last["TIME"].max()), 6)
    for side in ("left", "bottom"):
        risk.spines[side].set_visible(False)
    risk.set_yticks(range(3))
    risk.set_yticklabels(labels[::-1], fontsize=SIZE_NOTE)
    risk.set_xticks(ticks)
    risk.set_ylim(-0.6, 2.6)
    for i, sel in enumerate(groups):
        for x in ticks:
            risk.text(x, 2 - i, str(int((sel["TIME"] >= x).sum())),
                      ha="center", va="center", fontsize=SIZE_NOTE,
                      color=theme.ink_2)
    risk.set_xlabel("Time (days)")
    risk.annotate("Number still at risk", xy=(0, 1), xycoords="axes fraction",
                  xytext=(0, 7), textcoords="offset points",
                  fontsize=SIZE_NOTE, color=theme.ink_3)
    fig.tight_layout()
    return finish(fig)


# --------------------------------------------------------------------------
# Exposure-response
# --------------------------------------------------------------------------

def _population_probability(base: float, slope: float, sd: float,
                            x: np.ndarray, nodes: int = 41) -> np.ndarray:
    """Average response probability at each exposure, over the random effect.

    Not `expit(base + slope*x)`: that is the curve for the *typical* subject,
    and the points it would be compared against are population averages. A
    random effect on the logit flattens the average curve relative to the
    typical one, so plotting the typical curve against observed rates would
    show a mismatch the model does not actually have.
    """
    z, w = np.polynomial.hermite.hermgauss(nodes)
    eta = np.sqrt(2.0) * sd * z
    logits = base + slope * x[:, None] + eta[None, :]
    return (w / np.sqrt(np.pi) * (1.0 / (1.0 + np.exp(-logits)))).sum(axis=1)


def logistic_binary(df: pd.DataFrame, theme: Theme,
                    truth: dict | None = None) -> bytes:
    d = df.copy()
    d["DV"] = pd.to_numeric(d["DV"])

    # Equal-count bins, so every point carries about the same weight and the
    # interval widths are comparable across the exposure range.
    n_bins = 8
    edges = np.unique(np.quantile(d["EXPO"], np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(d["EXPO"], edges) - 1, 0, len(edges) - 2)
    mid, rate, lo, hi, counts = [], [], [], [], []
    for k in range(len(edges) - 1):
        sel = d[idx == k]
        if sel.empty:
            continue
        n, y = len(sel), sel["DV"].sum()
        p = y / n
        # Wilson interval: it behaves at the ends of the range, where a
        # normal approximation would put the limits outside (0, 1).
        z = 1.96
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = (z / (1 + z * z / n)) * np.sqrt(p * (1 - p) / n
                                               + z * z / (4 * n * n))
        mid.append(sel["EXPO"].mean())
        rate.append(p)
        lo.append(max(0.0, centre - half))
        hi.append(min(1.0, centre + half))
        counts.append(n)

    fig, ax = panel(theme, height=4.4)
    mid = np.array(mid)
    ax.vlines(mid, lo, hi, color=theme.series[0], linewidth=1.6, alpha=0.5,
              zorder=2)
    ax.scatter(mid, rate, s=44, color=theme.series[0],
               edgecolors=theme.surface, linewidths=1.6, zorder=4,
               label="Observed rate per bin, 95% Wilson interval")

    if truth:
        xs = np.linspace(float(d["EXPO"].min()), float(d["EXPO"].max()), 300)
        ys = _population_probability(truth["BASE"], truth["SLOPE"],
                                     truth["IIV_SD"], xs)
        ax.plot(xs, ys, color=theme.series[1], linewidth=2, zorder=3,
                solid_capstyle="round",
                label="Population curve the data was simulated from")
        label_point(ax, theme, xs[-1], ys[-1], "Simulated", dx=6,
                    colour=theme.series[1])

    ax.set_xlabel("Exposure")
    ax.set_ylabel("Probability of response")
    ax.set_ylim(0, 1)
    subjects = d["ID"].nunique()
    title_block(ax, theme, "Exposure-response for a binary endpoint",
                f"{subjects} subjects scored at {len(d) // subjects} visits "
                f"each; each point pools about {int(np.median(counts))} "
                "assessments")
    _headroom(ax, 0.10)
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

#: Figures that also want the truth parameters, not only the data.
NEEDS_TRUTH = {"pkpd_idr_inhibition", "logistic_binary"}
