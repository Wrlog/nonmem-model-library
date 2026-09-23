"""Tests for the simulators and the control-stream checks.

The simulators are checked against the model equations they claim to
implement, not just for running without error: the closed-form PK against
numerical integration, the indirect response baseline against its steady
state, the time-to-event generator against the Weibull survivor function,
and the logistic generator against its own link.

The checker is tested by breaking a control stream on purpose and asserting
it complains, which is the only way to know a linter is doing anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

from nmlib.check import check_control_stream
from nmlib.simulate import (
    SIMULATORS,
    _two_cmt_conc,
    simulate_idr_inhibition,
    simulate_logistic,
    simulate_tte_weibull,
)

MODELS = ["pk_2cmt_iv", "tgi_claret", "pkpd_idr_inhibition",
          "tte_weibull", "logistic_binary"]
EXAMPLES = sorted(p.stem for p in
                  (Path(__file__).resolve().parents[1] / "examples").glob("*.mod"))


# --- simulators -----------------------------------------------------------

def test_two_cmt_matches_numerical_integration():
    """The closed form must agree with RK4 on the same ODEs."""
    cl, v1, q, v2 = 5.0, 15.0, 3.0, 25.0
    amount, interval, inf_dur, n_doses = 500.0, 24.0, 1.0, 3
    k10, k12, k21 = cl / v1, q / v1, q / v2

    def rate(t):
        r = 0.0
        for i in range(n_doses):
            t0 = i * interval
            if t0 <= t < t0 + inf_dur:
                r += amount / inf_dur
        return r

    def rhs(t, y):
        return [rate(t) - (k10 + k12) * y[0] + k21 * y[1],
                k12 * y[0] - k21 * y[1]]

    t_eval = np.linspace(0.2, n_doses * interval, 60)
    # max_step keeps the solver from stepping over an infusion boundary.
    sol = solve_ivp(rhs, (0, t_eval[-1]), [0.0, 0.0], t_eval=t_eval,
                    rtol=1e-10, atol=1e-12, max_step=0.25)
    numeric = sol.y[0] / v1
    closed = _two_cmt_conc(t_eval, amount, n_doses, interval, inf_dur,
                           cl, v1, q, v2)
    assert np.max(np.abs(closed - numeric)) / np.max(numeric) < 1e-5


def test_idr_starts_at_its_steady_state():
    """Baseline must be kin/kout, not an independently drifting value."""
    sim = simulate_idr_inhibition(n_subjects=12, seed=1)
    obs = sim.data[(sim.data["MDV"] == 0) & (sim.data["TIME"] == 0)]
    baseline = pd.to_numeric(obs["DV"]).median()
    expected = sim.truth["TVKIN"] / sim.truth["TVKOUT"]
    # Only residual error and IIV on kout separate the two.
    assert abs(baseline - expected) / expected < 0.25


def test_tte_event_times_follow_the_weibull_survivor():
    """With the exposure effect switched off, the KM should track exp(-(lam t)^k)."""
    sim = simulate_tte_weibull(n_subjects=4000, seed=7, follow_up=1e6)
    rec = sim.data[sim.data["TIME"] > 0]
    # Undo the exposure effect: t_adj has the baseline hazard exactly.
    hr = np.exp(sim.truth["BETA_EXPO"] * rec["EXPO"].to_numpy())
    t_adj = rec["TIME"].to_numpy() * hr ** (1 / sim.truth["SHAPE"])
    lam, shape = sim.truth["LAMBDA"], sim.truth["SHAPE"]
    for q in (0.25, 0.5, 0.75):
        empirical = np.quantile(t_adj, q)
        theoretical = (-np.log(1 - q)) ** (1 / shape) / lam
        assert abs(empirical - theoretical) / theoretical < 0.08


def test_tte_has_both_events_and_censoring():
    sim = simulate_tte_weibull(n_subjects=300, seed=404)
    assert sim.notes["events"] > 30
    assert sim.notes["censored"] > 30


def test_logistic_response_rate_rises_with_exposure():
    sim = simulate_logistic(n_subjects=3000, seed=11)
    d = sim.data
    low = d[d["EXPO"] < 15]["DV"].mean()
    high = d[d["EXPO"] > 45]["DV"].mean()
    assert high > low + 0.15


@pytest.mark.parametrize("key", MODELS)
def test_simulator_is_reproducible(key):
    a = SIMULATORS[key]()
    b = SIMULATORS[key]()
    pd.testing.assert_frame_equal(a.data, b.data)


@pytest.mark.parametrize("key", MODELS)
def test_dataset_has_no_missing_ids_or_times(key):
    df = SIMULATORS[key]().data
    assert df["ID"].notna().all()
    assert df["TIME"].notna().all()


# --- control streams -------------------------------------------------------

@pytest.mark.parametrize("key", MODELS)
def test_control_stream_passes_checks(key, tmp_path, repo_root):
    result = check_control_stream(repo_root / "models" / f"{key}.mod",
                                  repo_root / "data")
    assert result.ok, f"{key}: {result.errors}"


@pytest.mark.parametrize("key", EXAMPLES)
def test_example_passes_checks_apart_from_its_data(key, repo_root):
    """Examples are templates with no dataset; everything else must hold."""
    result = check_control_stream(repo_root / "examples" / f"{key}.mod",
                                  repo_root / "data")
    errors = [e for e in result.errors if not e.startswith("data file not found")]
    assert not errors, f"{key}: {errors}"
    assert not result.warnings, f"{key}: {result.warnings}"


def test_checker_counts_etas_across_block_and_same_records(tmp_path, repo_root):
    """Each $OMEGA BLOCK(1) SAME adds an ETA; dropping one must be caught."""
    src = (repo_root / "examples" / "pk_2cmt_iv_iov.mod").read_text(encoding="utf-8")
    broken = src.replace("$OMEGA BLOCK(1) SAME   ; 6 occasion 4\n", "")
    assert broken != src
    mod = tmp_path / "broken_iov.mod"
    mod.write_text(broken, encoding="utf-8")
    result = check_control_stream(mod, repo_root / "data")
    assert any("ETA(6) referenced but only 5" in e for e in result.errors)


def test_checker_catches_a_reordered_input(tmp_path, repo_root):
    """A permuted $INPUT is the failure mode this checker exists for."""
    src = (repo_root / "models" / "logistic_binary.mod").read_text(encoding="utf-8")
    broken = src.replace("$INPUT ID TIME DV MDV EVID EXPO",
                         "$INPUT ID TIME DV MDV EXPO EVID")
    mod = tmp_path / "broken.mod"
    mod.write_text(broken.replace("../data/", str(repo_root / "data") + "/"),
                   encoding="utf-8")
    result = check_control_stream(mod, repo_root / "data")
    assert not result.ok
    assert any("$INPUT" in e for e in result.errors)


def test_checker_catches_an_undeclared_eta(tmp_path, repo_root):
    src = (repo_root / "models" / "logistic_binary.mod").read_text(encoding="utf-8")
    broken = src.replace("LOGIT = BASE + SLOPE*EXPO + ETA(1)",
                         "LOGIT = BASE + SLOPE*EXPO + ETA(1) + ETA(2)")
    mod = tmp_path / "broken2.mod"
    mod.write_text(broken.replace("../data/", str(repo_root / "data") + "/"),
                   encoding="utf-8")
    result = check_control_stream(mod, repo_root / "data")
    assert not result.ok
    assert any("ETA(2)" in e for e in result.errors)


def test_checker_catches_an_initial_estimate_outside_its_bounds(tmp_path, repo_root):
    src = (repo_root / "models" / "logistic_binary.mod").read_text(encoding="utf-8")
    broken = src.replace("(-1, 0.055, 1)", "(-1, 2.0, 1)")
    mod = tmp_path / "broken3.mod"
    mod.write_text(broken.replace("../data/", str(repo_root / "data") + "/"),
                   encoding="utf-8")
    result = check_control_stream(mod, repo_root / "data")
    assert not result.ok
    assert any("upper bound" in e for e in result.errors)


def test_checker_catches_a_missing_likelihood_flag(tmp_path, repo_root):
    src = (repo_root / "models" / "tte_weibull.mod").read_text(encoding="utf-8")
    broken = src.replace("METHOD=1 LAPLACE LIKELIHOOD", "METHOD=1 INTERACTION")
    mod = tmp_path / "broken4.mod"
    mod.write_text(broken.replace("../data/", str(repo_root / "data") + "/"),
                   encoding="utf-8")
    result = check_control_stream(mod, repo_root / "data")
    assert not result.ok
    assert any("LIKELIHOOD" in e for e in result.errors)


def test_theta_and_eta_are_not_confused(repo_root):
    """THETA(1) must not be counted as an ETA reference."""
    result = check_control_stream(repo_root / "models" / "pk_2cmt_iv.mod",
                                  repo_root / "data")
    assert not any("ETA(" in e and "referenced" in e for e in result.errors)
