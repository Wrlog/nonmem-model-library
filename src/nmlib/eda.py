"""Exploratory analysis: what you look at before fitting anything.

Three things, in the order they get done in practice.

**Table 1** describes who is in the dataset and how much data each of them
contributed. It is the first thing a reviewer asks for and the first thing
that catches a data assembly error: a weight of 700 kg, a subject with one
sample, a covariate that is missing for a third of the study.

**A covariate screen** asks which subject characteristics the structural
model has not already accounted for. It is run on the empirical Bayes
random effects, so it can only be read where shrinkage is low -- at high
shrinkage the etas have been pulled toward the population and a screen on
them finds nothing whatever is there.

**Individual profiles**, raw, before any model is involved. Averages hide
the shape of single subjects, and the shape of single subjects is what
decides how many compartments the model needs.
"""

from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd

from .theme import (
    SIZE_LABEL,
    SIZE_NOTE,
    SIZE_SUBTITLE,
    SIZE_TITLE,
    Theme,
    finish,
    panel,
    plain_log_ticks,
    style_axes,
)

# --------------------------------------------------------------------------
# Table 1
# --------------------------------------------------------------------------

def _continuous_row(name: str, values: pd.Series, unit: str) -> dict:
    v = pd.to_numeric(values, errors="coerce")
    missing = int(v.isna().sum())
    v = v.dropna()
    return {
        "label": name + (f" ({unit})" if unit else ""),
        "summary": f"{v.median():g}",
        "spread": f"{v.min():g} to {v.max():g}",
        "missing": missing,
        "n": len(values),
    }


def _categorical_row(name: str, values: pd.Series,
                     levels: dict[int, str]) -> dict:
    v = pd.to_numeric(values, errors="coerce")
    missing = int(v.isna().sum())
    counts = v.dropna().astype(int).value_counts()
    total = int(counts.sum()) or 1
    parts = [f"{levels.get(k, k)} {int(counts.get(k, 0))} "
             f"({counts.get(k, 0) / total * 100:.0f}%)"
             for k in sorted(levels)]
    return {
        "label": name,
        "summary": "; ".join(parts),
        "spread": "",
        "missing": missing,
        "n": len(values),
    }


def table_one(data: pd.DataFrame, spec: dict) -> list[dict]:
    """Baseline characteristics, one row per covariate, one row per subject.

    Covariates are summarised once per subject rather than once per record,
    which is the difference between "the median weight of the people in this
    study" and "the median weight of the blood samples in this study". The
    second is not a number anyone wants, and reporting it is a common way to
    make a heavily sampled subject look like several people.
    """
    per_subject = data.groupby("ID").first().reset_index()
    obs = data[data["MDV"].astype(int) == 0] if "MDV" in data else data
    per_subject_obs = obs.groupby("ID").size()

    rows = [{
        "label": "Subjects",
        "summary": f"{data['ID'].nunique()}",
        "spread": "",
        "missing": 0,
        "n": data["ID"].nunique(),
    }, {
        "label": "Observations per subject",
        "summary": f"{per_subject_obs.median():g}",
        "spread": f"{per_subject_obs.min()} to {per_subject_obs.max()}",
        "missing": 0,
        "n": int(per_subject_obs.sum()),
    }]

    for column, meta in spec.items():
        if column not in per_subject:
            continue
        if meta.get("levels"):
            rows.append(_categorical_row(meta["label"], per_subject[column],
                                         meta["levels"]))
        else:
            rows.append(_continuous_row(meta["label"], per_subject[column],
                                        meta.get("unit", "")))
    return rows


def table_one_html(rows: list[dict]) -> str:
    cells = []
    for r in rows:
        missing = r["missing"]
        if missing:
            share = missing / max(r["n"], 1) * 100
            missing_text = f"{missing} ({share:.0f}%)"
        else:
            missing_text = "0"
        cells.append(
            f'<tr><td>{html.escape(r["label"])}</td>'
            f'<td>{html.escape(r["summary"])}</td>'
            f'<td>{html.escape(r["spread"])}</td>'
            f"<td>{missing_text}</td></tr>")
    return ("<table><thead><tr><th>Characteristic</th>"
            "<th>Number or median</th><th>Range</th>"
            "<th>Missing</th></tr></thead>"
            f"<tbody>{''.join(cells)}</tbody></table>")


# --------------------------------------------------------------------------
# Covariate screen
# --------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, written out to avoid leaning on scipy.stats here."""
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(1, len(v) + 1)
        # Average the ranks of ties, or a covariate like AGE with repeats
        # gets a correlation that depends on the sort order.
        _, inverse, counts = np.unique(v, return_inverse=True,
                                       return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inverse, r)
        return (sums / counts)[inverse]

    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom else 0.0


def covariate_screen(etas: pd.DataFrame, data: pd.DataFrame,
                     spec: dict) -> pd.DataFrame:
    """Every random effect against every covariate, ranked by strength.

    Continuous covariates get a Spearman rank correlation, which does not
    assume the relationship is a straight line on this scale. Categorical
    ones get the difference in median eta between groups, expressed in
    standard deviations of that eta so it is comparable with the
    correlations beside it.
    """
    per_subject = data.groupby("ID").first().reset_index()
    merged = etas.merge(per_subject, on="ID", how="inner")
    eta_columns = [c for c in etas.columns if c != "ID"]

    rows = []
    for eta in eta_columns:
        values = merged[eta].to_numpy(dtype=float)
        sd = values.std(ddof=1) or 1.0
        for column, meta in spec.items():
            if column not in merged or meta.get("skip_screen"):
                continue
            covariate = pd.to_numeric(merged[column], errors="coerce")
            ok = covariate.notna().to_numpy()
            if ok.sum() < 8:
                continue
            if meta.get("levels"):
                groups = [values[ok & (covariate == k).to_numpy()]
                          for k in sorted(meta["levels"])]
                groups = [g for g in groups if len(g) >= 3]
                if len(groups) < 2:
                    continue
                strength = (max(np.median(g) for g in groups)
                            - min(np.median(g) for g in groups)) / sd
                kind = "group difference"
            else:
                strength = abs(_spearman(covariate[ok].to_numpy(), values[ok]))
                kind = "rank correlation"
            rows.append({
                "eta": eta, "covariate": column,
                "label": meta["label"], "kind": kind,
                "strength": float(strength),
                "in_model": bool(meta.get("in_model")),
            })
    return pd.DataFrame(rows).sort_values("strength", ascending=False)


#: Above this, a relationship is worth taking to the next model run. It is a
#: screening threshold, not a test: the point is to rank what to try, and
#: whether it belongs in the model is settled by fitting it, not here.
SCREEN_FLAG = 0.30


def covariate_plot(etas: pd.DataFrame, data: pd.DataFrame, spec: dict,
                   theme: Theme, eta_label: str | None = None) -> bytes | None:
    """The screen, drawn: one panel per covariate for one random effect."""
    per_subject = data.groupby("ID").first().reset_index()
    merged = etas.merge(per_subject, on="ID", how="inner")
    eta_columns = [c for c in etas.columns if c != "ID"]
    if not eta_columns:
        return None
    eta = eta_label or eta_columns[0]

    columns = [(c, m) for c, m in spec.items()
               if c in merged and not m.get("skip_screen")]
    if not columns:
        return None

    values = merged[eta].to_numpy(dtype=float)
    sd = values.std(ddof=1) or 1.0
    ncols = min(len(columns), 4)
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = panel(theme, nrows=nrows, ncols=ncols, width=8.4,
                      height=2.5 * nrows + 1.0)
    flat = np.atleast_1d(axes).ravel()

    for ax, (column, meta) in zip(flat, columns, strict=False):
        style_axes(ax, theme)
        covariate = pd.to_numeric(merged[column], errors="coerce")
        ok = covariate.notna().to_numpy()
        ax.axhline(0, color=theme.border, linewidth=1.0, zorder=2)

        if meta.get("levels"):
            levels = sorted(meta["levels"])
            groups = [values[ok & (covariate == k).to_numpy()] for k in levels]
            parts = ax.boxplot(groups, positions=range(len(levels)),
                               widths=0.55, patch_artist=True,
                               medianprops=dict(color=theme.surface,
                                                linewidth=1.6),
                               flierprops=dict(marker="o", markersize=3,
                                               markerfacecolor=theme.ink_3,
                                               markeredgecolor="none"))
            for box in parts["boxes"]:
                box.set(facecolor=theme.series[0], edgecolor="none", alpha=0.75)
            for whisker in parts["whiskers"] + parts["caps"]:
                whisker.set(color=theme.border, linewidth=1.0)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels([meta["levels"][k] for k in levels],
                               fontsize=SIZE_NOTE)
            medians = [np.median(g) for g in groups if len(g)]
            strength = (max(medians) - min(medians)) / sd if medians else 0.0
            stat = f"gap {strength:.2f} SD"
        else:
            ax.scatter(covariate[ok], values[ok], s=16, alpha=0.6,
                       color=theme.series[0], edgecolors=theme.surface,
                       linewidths=0.8, zorder=3)
            x = covariate[ok].to_numpy(dtype=float)
            y = values[ok]
            if len(np.unique(x)) > 2:
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, slope * xs + intercept, color=theme.series[1],
                        linewidth=1.8, zorder=4)
            strength = abs(_spearman(x, y))
            stat = f"rho {strength:.2f}"

        flagged = strength >= SCREEN_FLAG
        colour = theme.warning if flagged else theme.ink_3
        note = stat + ("  worth a run" if flagged else "")
        ax.annotate(note, xy=(0.5, 1.0), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    va="bottom", fontsize=SIZE_NOTE, color=colour,
                    fontweight="600" if flagged else "normal")
        ax.set_xlabel(meta["label"], fontsize=SIZE_NOTE)

    for ax in flat[len(columns):]:
        ax.set_visible(False)
    for row in range(nrows):
        flat[row * ncols].set_ylabel(f"eta on {eta}", fontsize=SIZE_LABEL)

    fig.text(0.0, 1.0, f"Covariate screen on the random effect for {eta}",
             ha="left", va="top", fontsize=SIZE_TITLE, color=theme.ink,
             fontweight="600")
    fig.text(0.0, 0.955,
             "A covariate already in the model should show no trend here; "
             "one that isn't in the model but trends is the next to try",
             ha="left", va="top", fontsize=SIZE_SUBTITLE, color=theme.ink_3)
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=2.6, w_pad=1.6)
    return finish(fig)


# --------------------------------------------------------------------------
# Individual profiles, before any model
# --------------------------------------------------------------------------

def individual_profiles(data: pd.DataFrame, theme: Theme,
                        n_subjects: int = 12, log_scale: bool = True,
                        y_label: str = "Concentration (mg/L)",
                        x_label: str = "Time (h)",
                        annotate: tuple[str, ...] = ()) -> bytes | None:
    """Raw observations, subject by subject, with nothing fitted to them.

    This is the plot that decides how many compartments to start with. On a
    log axis a one-compartment profile is a straight line and a
    two-compartment profile bends, and no amount of looking at the mean
    profile will tell you which you have -- averaging curves with different
    clearances produces a bend whether or not any individual has one.
    """
    obs = data[data["MDV"].astype(int) == 0].copy() if "MDV" in data else data.copy()
    obs["DV"] = pd.to_numeric(obs["DV"], errors="coerce")
    obs = obs.dropna(subset=["DV"])
    if obs.empty:
        return None

    ids = np.sort(obs["ID"].unique())
    picks = ids[np.unique(np.linspace(0, len(ids) - 1,
                                      min(n_subjects, len(ids))).astype(int))]

    ncols = 4
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = panel(theme, nrows=nrows, ncols=ncols, width=8.4,
                      height=1.95 * nrows + 0.9, sharex=True)
    flat = np.atleast_1d(axes).ravel()

    for ax, subject in zip(flat, picks, strict=False):
        g = obs[obs["ID"] == subject].sort_values("TIME")
        ax.plot(g["TIME"], g["DV"], color=theme.series[0], linewidth=1.4,
                zorder=3, solid_capstyle="round")
        ax.scatter(g["TIME"], g["DV"], s=14, color=theme.ink,
                   edgecolors=theme.surface, linewidths=0.9, zorder=4)
        if log_scale:
            ax.set_yscale("log")
            plain_log_ticks(ax)
        bits = [f"ID {int(subject)}"]
        for column in annotate:
            if column in g:
                bits.append(f"{column} {g[column].iloc[0]:g}")
        ax.set_title("  ".join(bits), fontsize=SIZE_NOTE, loc="left",
                     color=theme.ink_3, pad=4)

    for ax in flat[len(picks):]:
        ax.set_visible(False)
    for ax in flat[:len(picks)][-ncols:]:
        ax.set_xlabel(x_label, fontsize=SIZE_NOTE)
    for row in range(nrows):
        flat[row * ncols].set_ylabel(y_label, fontsize=SIZE_NOTE)

    fig.text(0.0, 1.0, f"{len(picks)} individual profiles, no model fitted",
             ha="left", va="top", fontsize=SIZE_TITLE, color=theme.ink,
             fontweight="600")
    fig.text(0.0, 0.958,
             "Log axis, so a single disposition phase would be a straight "
             "line; these bend, which is the case for two compartments",
             ha="left", va="top", fontsize=SIZE_SUBTITLE, color=theme.ink_3)
    fig.tight_layout(rect=(0, 0, 1, 0.915), h_pad=1.9, w_pad=1.3)
    return finish(fig)
