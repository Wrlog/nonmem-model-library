"""Tests for the exploratory analysis.

The covariate screen is the piece worth testing hardest, because it is the
one that can be confidently wrong. The PK dataset is simulated with a
covariate structure built for the purpose: weight drives clearance and is
already in the model, age and sex do nothing at all, and a metaboliser
genotype has a real effect that the control stream leaves out. A screen
that works has to find the genotype, leave weight alone, and say nothing
about age or sex -- and a screen that returned the same ranking for any
input would pass none of those three.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import INK_FLOOR, coloured_fraction
from nmlib import diagnostics, eda
from nmlib.build import COVARIATES
from nmlib.simulate import simulate_pk_2cmt
from nmlib.theme import DARK, LIGHT

SPEC = COVARIATES["pk_2cmt_iv"]


@pytest.fixture(scope="module")
def pk():
    return simulate_pk_2cmt()


@pytest.fixture(scope="module")
def pk_etas(repo_root):
    fit = diagnostics.load_fit(repo_root / "fit" / "results", "pk_2cmt_iv")
    if fit is None or "etas" not in fit:
        pytest.skip("pk_2cmt_iv has not been fitted in this tree")
    return fit["etas"]


# --- Table 1 --------------------------------------------------------------

def test_table_one_summarises_subjects_not_records(pk):
    """The distinction that makes Table 1 mean anything.

    Every subject here contributes twelve rows, so a per-record summary
    would report the study as twelve times its real size.
    """
    rows = eda.table_one(pk.data, SPEC)
    subjects = next(r for r in rows if r["label"] == "Subjects")
    assert subjects["summary"] == str(pk.data["ID"].nunique())
    assert int(subjects["summary"]) < len(pk.data)


def test_table_one_reports_every_covariate_in_the_spec(pk):
    rows = eda.table_one(pk.data, SPEC)
    labels = {r["label"] for r in rows}
    for meta in SPEC.values():
        expected = meta["label"] + (f" ({meta['unit']})" if meta.get("unit")
                                    else "")
        assert expected in labels


def test_table_one_weight_matches_the_per_subject_median(pk):
    per_subject = pk.data.groupby("ID")["WT"].first()
    rows = eda.table_one(pk.data, SPEC)
    weight = next(r for r in rows if r["label"].startswith("Weight"))
    assert weight["summary"] == f"{per_subject.median():g}"


def test_table_one_counts_missing_values():
    data = pd.DataFrame({
        "ID": [1, 1, 2, 2], "MDV": [0, 0, 0, 0],
        "DV": [1.0, 2.0, 3.0, 4.0], "WT": [70.0, 70.0, None, None],
    })
    rows = eda.table_one(data, {"WT": {"label": "Weight", "unit": "kg"}})
    weight = next(r for r in rows if r["label"].startswith("Weight"))
    assert weight["missing"] == 1
    assert "50%" in eda.table_one_html(rows)


# --- the covariate screen -------------------------------------------------

def test_screen_finds_the_covariate_that_was_left_out(pk, pk_etas):
    screen = eda.covariate_screen(pk_etas, pk.data, SPEC)
    top = screen.iloc[0]
    assert top["covariate"] == "CYP"
    assert top["eta"] == "CL"
    assert top["strength"] >= eda.SCREEN_FLAG
    assert not top["in_model"]


def test_screen_leaves_alone_the_covariate_already_in_the_model(pk, pk_etas):
    """Weight drives clearance, and the model knows. Nothing should remain."""
    screen = eda.covariate_screen(pk_etas, pk.data, SPEC)
    weight = screen[(screen["covariate"] == "WT") & (screen["eta"] == "CL")]
    assert len(weight) == 1
    assert weight["strength"].iloc[0] < eda.SCREEN_FLAG


def test_screen_is_quiet_about_the_covariates_that_do_nothing(pk, pk_etas):
    screen = eda.covariate_screen(pk_etas, pk.data, SPEC)
    inert = screen[screen["covariate"].isin(["AGE", "SEX"])]
    assert len(inert) == 4                      # two covariates, two etas
    assert (inert["strength"] < eda.SCREEN_FLAG).all()


def test_screen_covers_every_eta_and_covariate_pair(pk, pk_etas):
    screen = eda.covariate_screen(pk_etas, pk.data, SPEC)
    etas = [c for c in pk_etas.columns if c != "ID"]
    assert len(screen) == len(etas) * len(SPEC)


def test_spearman_matches_a_known_value():
    """A monotone but non-linear relationship must come back as 1."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert eda._spearman(x, [1.0, 4.0, 9.0, 16.0, 25.0]) == pytest.approx(1.0)
    assert eda._spearman(x, [25.0, 16.0, 9.0, 4.0, 1.0]) == pytest.approx(-1.0)
    # Ties must be ranked by their average, not by input order.
    assert eda._spearman([1.0, 1.0, 2.0, 2.0],
                         [1.0, 1.0, 2.0, 2.0]) == pytest.approx(1.0)


# --- the figures ----------------------------------------------------------

@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=lambda t: t.name)
def test_covariate_plot_draws_its_data(pk, pk_etas, theme):
    png = eda.covariate_plot(pk_etas, pk.data, SPEC, theme, "CL")
    assert png is not None
    assert coloured_fraction(png) > INK_FLOOR


@pytest.mark.parametrize("theme", [LIGHT, DARK], ids=lambda t: t.name)
def test_individual_profiles_draw_their_data(pk, theme):
    png = eda.individual_profiles(pk.data, theme, annotate=("WT",))
    assert png is not None
    assert coloured_fraction(png) > INK_FLOOR


def test_individual_profiles_handle_a_dataset_with_no_observations():
    data = pd.DataFrame({"ID": [1], "TIME": [0.0], "DV": ["."], "MDV": [1]})
    assert eda.individual_profiles(data, LIGHT) is None
