"""Tests that every figure actually contains its data.

A plotting bug does not usually raise. It produces a PNG of the right size,
with a title, axes and gridlines, and nothing plotted between them -- which
is exactly what shipped when `_unity` set an axis limit of `min - pad`
(negative, for concentration data) and a log scale was applied afterwards.
The panel looked plausible in the build log and was empty on the page.

So the check here is on the pixels: a figure that drew its data has marks
in a series colour, and one that did not has only grey chrome. The
threshold sits between the two by a wide margin -- the emptiest real figure
in the library is eighty times above it, and the reproduced bug is twenty
times below -- and the last test reproduces that bug to show the detector
can fail.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from nmlib import diagnostics, figures
from nmlib.build import CATALOGUE
from nmlib.theme import DARK, LIGHT, finish, panel, style_axes

#: Below this fraction of coloured pixels, a figure has not drawn its data.
INK_FLOOR = 0.0015

THEMES = [LIGHT, DARK]


def coloured_fraction(png: bytes) -> float:
    """Fraction of pixels carrying a data colour rather than grey chrome.

    Chrome -- axes, gridlines, text, the muted spaghetti lines -- is grey or
    near-grey, so it has almost no chroma. Every series colour in the
    palette has plenty.
    """
    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=int)
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    return float((chroma > 25).mean())


def _data(repo_root: Path, key: str):
    df = pd.read_csv(repo_root / "data" / f"{key}.csv")
    truth = json.loads(
        (repo_root / "data" / f"{key}.truth.json").read_text(encoding="utf-8")
    )["parameters"]
    return df, truth


@pytest.mark.parametrize("key", [m["key"] for m in CATALOGUE])
@pytest.mark.parametrize("theme", THEMES, ids=lambda t: t.name)
def test_model_figure_contains_its_data(key, theme, repo_root):
    df, truth = _data(repo_root, key)
    fn = figures.FIGURES[key]
    png = fn(df, theme, truth) if key in figures.NEEDS_TRUTH else fn(df, theme)
    ink = coloured_fraction(png)
    assert ink > INK_FLOOR, f"{key} ({theme.name}) looks empty: {ink:.4%} coloured"


@pytest.mark.parametrize("model", [m for m in CATALOGUE],
                         ids=lambda m: m["key"])
@pytest.mark.parametrize("theme", THEMES, ids=lambda t: t.name)
def test_diagnostic_figures_contain_their_data(model, theme, repo_root):
    """The goodness-of-fit and predictive-check panels, where the bug was."""
    key = model["key"]
    fit = diagnostics.load_fit(repo_root / "fit" / "results", key)
    if fit is None:
        pytest.skip(f"{key} has not been fitted in this tree")
    df, _ = _data(repo_root, key)

    if "gof" in fit:
        ink = coloured_fraction(
            diagnostics.gof_panel(fit, theme, bool(model.get("log"))))
        assert ink > INK_FLOOR, f"{key} GOF ({theme.name}) is empty: {ink:.4%}"

    vpc = diagnostics.vpc_plot(fit, df, theme, log_scale=bool(model.get("log")),
                               y_label=model["y_label"],
                               x_label=model["x_label"])
    if vpc is not None:
        ink = coloured_fraction(vpc)
        assert ink > INK_FLOOR, f"{key} VPC ({theme.name}) is empty: {ink:.4%}"


@pytest.mark.parametrize("theme", THEMES, ids=lambda t: t.name)
def test_recovery_figure_contains_its_data(theme, repo_root):
    results = repo_root / "fit" / "results"
    per_model = {}
    for m in CATALOGUE:
        fit = diagnostics.load_fit(results, m["key"])
        if fit:
            per_model[m["key"]] = (m["title"], fit["estimates"])
    if not per_model:
        pytest.skip("nothing has been fitted in this tree")
    ink = coloured_fraction(diagnostics.recovery_plot(per_model, theme))
    assert ink > INK_FLOOR


def test_a_log_axis_padded_the_old_way_is_detected_as_empty(repo_root):
    """The detector must be able to fail, on the bug it exists for.

    Additive padding puts the lower limit below zero for concentration
    data; applying a log scale afterwards collapses the view and the
    observations vanish, leaving only grey chrome.
    """
    fit = diagnostics.load_fit(repo_root / "fit" / "results", "pk_2cmt_iv")
    if fit is None:
        pytest.skip("pk_2cmt_iv has not been fitted in this tree")
    d = fit["gof"]

    fig, ax = panel(LIGHT)
    style_axes(ax, LIGHT, xgrid=True)
    ax.scatter(d["PRED"], d["DV"], s=11, color=LIGHT.series[0], alpha=0.45)
    values = np.concatenate([d["PRED"].to_numpy(), d["DV"].to_numpy()])
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = (hi - lo) * 0.05
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xscale("log")
    ax.set_yscale("log")

    assert coloured_fraction(finish(fig)) < INK_FLOOR

    # ... and the current code, on the same data, must pass.
    assert coloured_fraction(diagnostics.gof_panel(fit, LIGHT, True)) > INK_FLOOR


def test_unity_limits_stay_positive_on_a_log_axis():
    """The direct statement of the fix, independent of any rendering."""
    fig, ax = panel(LIGHT)
    values = np.array([0.0151, 0.4, 12.0, 51.95])
    diagnostics._unity(ax, values, LIGHT, log=True)
    lo, hi = ax.get_xlim()
    assert lo > 0 and hi > 0
    assert lo < values.min() and hi > values.max()
