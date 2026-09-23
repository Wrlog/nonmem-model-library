"""Tests for the population estimation in `nmlib.estimate`.

An estimator is the one component here that cannot be checked by reading
it. These tests pin down the three things that would make its numbers
wrong without making it crash:

* the marginal likelihood, against brute-force numerical integration of the
  same integral;
* the indirect response solver, against an adaptive ODE integrator on the
  same equations;
* the whole pipeline, by fitting a small dataset simulated from known
  parameters and asking whether the confidence intervals cover them.

The first two are exact comparisons against an independent method. The
third is the claim the dashboard makes, tested on a smaller problem so it
runs in seconds.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.special import expit

from nmlib.estimate import (
    IDRInhibition,
    LogisticBinary,
    PK2Cmt,
    conc_2cmt,
    fit_model,
    marginal_loglik,
)
from nmlib.simulate import (
    IDR_DOSE_TIMES,
    simulate_idr_inhibition,
    simulate_logistic,
    simulate_pk_2cmt,
    simulate_tte_weibull,
)


def _truth_vector_logistic(truth):
    return np.array([truth["BASE"], truth["SLOPE"], np.log(truth["IIV_SD"])])


# --- the integral ---------------------------------------------------------

def test_quadrature_matches_brute_force_integration():
    """Adaptive Gauss-Hermite against a general-purpose integrator.

    The marginal likelihood is an integral over the random effect, and
    getting it subtly wrong biases every estimate without ever raising an
    error. Here the same integral is computed a second way, by scipy's
    adaptive quadrature over the one random effect, and the two must agree
    to far better than any difference that could matter.
    """
    sim = simulate_logistic(n_subjects=8, seed=3)
    model = LogisticBinary(sim.data, sim.truth)
    x = _truth_vector_logistic(sim.truth)
    sd = float(np.exp(x[2]))

    def subject_loglik(i):
        linear = x[0] + x[1] * model.expo[i]
        y = model.y[i]

        def integrand(z):
            p = expit(linear + z)
            joint = np.sum(y * np.log(p) + (1 - y) * np.log1p(-p))
            prior = np.exp(-0.5 * (z / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            return np.exp(joint) * prior

        return np.log(quad(integrand, -12 * sd, 12 * sd, limit=500)[0])

    brute = sum(subject_loglik(i) for i in range(model.n_subj))
    assert marginal_loglik(model, x, nodes=15) == pytest.approx(brute, abs=1e-8)


def test_quadrature_converges_as_nodes_increase():
    """More nodes must change the answer less, and one node must be Laplace."""
    sim = simulate_logistic(n_subjects=40, seed=5)
    model = LogisticBinary(sim.data, sim.truth)
    x = _truth_vector_logistic(sim.truth)
    values = [marginal_loglik(model, x, nodes=n) for n in (1, 3, 9, 15, 25)]
    gaps = [abs(b - a) for a, b in zip(values[:-1], values[1:], strict=True)]
    assert gaps[0] > gaps[-1]                 # it is actually converging
    assert values[-2] == pytest.approx(values[-1], abs=1e-6)


# --- the solvers ----------------------------------------------------------

def test_idr_solver_matches_an_adaptive_integrator():
    """The exponential integrator against solve_ivp on the same ODE."""
    sim = simulate_idr_inhibition(n_subjects=6, seed=13)
    model = IDRInhibition(sim.data, sim.truth)
    t = sim.truth
    x = np.array([np.log(t["TVKIN"]), np.log(t["TVKOUT"]),
                  np.log(t["TVIMAX"] / (1 - t["TVIMAX"])), np.log(t["TVIC50"]),
                  np.log(0.3), np.log(0.5), np.log(0.1)])
    fast = model._solve(x, np.zeros((model.n_subj, 2)))

    kin, kout = t["TVKIN"], t["TVKOUT"]
    imax, ic50, v, ke = t["TVIMAX"], t["TVIC50"], model.v, model.ke
    for i in range(model.n_subj):
        dose = model.dose[i]

        def conc(tt, dose=dose):
            elapsed = tt - IDR_DOSE_TIMES
            return np.sum(np.where(elapsed >= 0,
                                   (dose / v) * np.exp(-ke * elapsed), 0.0))

        sol = solve_ivp(
            lambda tt, y: [kin * (1 - imax * conc(tt) / (ic50 + conc(tt)))
                           - kout * y[0]],
            (0, float(model.times[-1])), [kin / kout], t_eval=model.times,
            rtol=1e-11, atol=1e-13, max_step=3.0)
        assert np.allclose(fast[i], sol.y[0], rtol=2e-6)


def test_predictions_are_batched_over_quadrature_nodes_consistently():
    """A stack of eta sets must give exactly what evaluating each one gives.

    Every model is evaluated on a leading axis of quadrature nodes, and a
    broadcasting mistake there would quietly pair one node's random effects
    with another node's, which changes the likelihood without changing the
    shape of anything. This pins the leading axis down.
    """
    sim = simulate_pk_2cmt(n_subjects=5, seed=2)
    model = PK2Cmt(sim.data, sim.truth)
    x = np.array([np.log(5.0), np.log(15.0), np.log(3.0), np.log(25.0),
                  np.log(0.3), np.log(0.25), np.log(0.15)])
    rng = np.random.default_rng(0)
    nodes = [rng.normal(0, 0.3, (model.n_subj, 2)) for _ in range(4)]

    stacked = model._pk(x, np.stack(nodes))
    assert stacked.shape == (4, model.n_subj, len(model.times))
    for i, eta in enumerate(nodes):
        assert np.allclose(stacked[i], model._pk(x, eta), rtol=1e-12)

    # The same must hold for the log-likelihood the quadrature actually sums.
    ll = model.data_loglik(x, np.stack(nodes))
    for i, eta in enumerate(nodes):
        assert np.allclose(ll[i], model.data_loglik(x, eta), rtol=1e-12)


def test_conc_2cmt_superposes_doses():
    """Three doses must equal the sum of three single-dose profiles."""
    t = np.array([0.5, 5.0, 26.0, 50.0, 70.0])
    args = dict(cl=5.0, v1=15.0, q=3.0, v2=25.0)
    three = conc_2cmt(t, 500.0, 3, 24.0, 1.0, **args)
    one = sum(conc_2cmt(t - d * 24.0, 500.0, 1, 24.0, 1.0, **args)
              * (t - d * 24.0 >= 0) for d in range(3))
    assert np.allclose(three, one, rtol=1e-10)


# --- the whole pipeline ---------------------------------------------------

def test_fit_recovers_the_simulated_logistic_parameters():
    """A displaced start must still land with the truth inside the interval."""
    sim = simulate_logistic(n_subjects=400, seed=21)
    fit = fit_model("logistic_binary", sim.data, sim.truth, verbose=False)
    est = fit.estimates
    assert fit.converged
    for _, r in est.iterrows():
        lo = r["estimate"] - 1.96 * r["se"]
        hi = r["estimate"] + 1.96 * r["se"]
        lo, hi = min(lo, hi), max(lo, hi)
        assert lo <= r["truth"] <= hi, (
            f"{r['parameter']}: {lo:.4g} to {hi:.4g} misses {r['truth']:.4g}")


def test_fit_recovers_the_time_to_event_parameters():
    sim = simulate_tte_weibull(n_subjects=900, seed=22)
    fit = fit_model("tte_weibull", sim.data, sim.truth, verbose=False)
    for _, r in fit.estimates.iterrows():
        lo = r["estimate"] - 1.96 * r["se"]
        hi = r["estimate"] + 1.96 * r["se"]
        lo, hi = min(lo, hi), max(lo, hi)
        assert lo <= r["truth"] <= hi, r["parameter"]


def test_the_optimiser_does_not_simply_return_its_starting_values():
    """The guard against a fit that silently did nothing.

    An objective that fails everywhere hands back the starting vector, and
    a table of starting values looks exactly like a table of estimates.
    """
    sim = simulate_logistic(n_subjects=250, seed=23)
    fit = fit_model("logistic_binary", sim.data, sim.truth, verbose=False)
    model = LogisticBinary(sim.data, sim.truth)
    started_at = model.to_natural(model.start())
    moved = np.abs(fit.estimates["estimate"].to_numpy() - started_at)
    assert np.all(moved > 1e-6)


def test_objective_is_better_at_the_estimate_than_at_the_truth():
    """A maximum-likelihood fit must beat the truth on its own objective."""
    sim = simulate_logistic(n_subjects=300, seed=24)
    model = LogisticBinary(sim.data, sim.truth)
    fit = fit_model("logistic_binary", sim.data, sim.truth, verbose=False)
    est = fit.estimates["estimate"].to_numpy()
    x_hat = np.array([est[0], est[1], np.log(est[2])])
    at_truth = marginal_loglik(model, _truth_vector_logistic(sim.truth), 15)
    assert marginal_loglik(model, x_hat, 15) >= at_truth - 1e-6


# --- the artefacts the dashboard reads ------------------------------------

def test_fit_writes_what_the_dashboard_expects(tmp_path):
    root = tmp_path
    (root / "data").mkdir()
    sim = simulate_logistic(n_subjects=120, seed=25)
    sim.data.to_csv(root / "data" / "logistic_binary.csv", index=False)
    (root / "data" / "logistic_binary.truth.json").write_text(
        json.dumps({"parameters": sim.truth}), encoding="utf-8")

    from nmlib.estimate import fit_all
    fit_all(root, keys=["logistic_binary"], verbose=False)

    results = root / "fit" / "results"
    est = pd.read_csv(results / "logistic_binary_estimates.csv")
    assert set(est.columns) >= {"parameter", "estimate", "se", "rse_pct",
                                "truth", "pct_diff"}
    status = json.loads(
        (results / "logistic_binary_status.json").read_text(encoding="utf-8"))
    assert status["engine"] == "nmlib.estimate"
    assert status["nodes"] >= 1


def test_load_fit_returns_none_when_nothing_was_fitted(tmp_path):
    from nmlib.diagnostics import load_fit
    assert load_fit(tmp_path, "pk_2cmt_iv") is None


def test_recovery_rows_flag_an_interval_that_misses(tmp_path):
    """The coverage flag must be capable of coming out False."""
    from nmlib.diagnostics import recovery_rows
    est = pd.DataFrame({
        "parameter": ["good", "bad"],
        "estimate": [1.0, 2.0],
        "se": [0.1, 0.1],
        "truth": [1.05, 1.0],
        "rse_pct": [10.0, 5.0],
        "pct_diff": [-4.8, 100.0],
    })
    rows = recovery_rows(est)
    assert list(rows["covers"]) == [True, False]


def test_recovery_rows_handle_a_negative_truth():
    """A negative parameter must not come back with its interval inverted."""
    from nmlib.diagnostics import recovery_rows
    est = pd.DataFrame({
        "parameter": ["beta"], "estimate": [-0.025], "se": [0.004],
        "truth": [-0.025], "rse_pct": [16.0], "pct_diff": [0.0],
    })
    rows = recovery_rows(est)
    assert rows["lo"].iloc[0] <= 1.0 <= rows["hi"].iloc[0]
    assert bool(rows["covers"].iloc[0])


def test_path_with_a_space_still_resolves(tmp_path, repo_root):
    """A data path containing a space must not read as 'file not found'."""
    from nmlib.check import check_control_stream
    spaced = tmp_path / "a folder"
    spaced.mkdir()
    (spaced / "logistic_binary.csv").write_bytes(
        (repo_root / "data" / "logistic_binary.csv").read_bytes())
    src = (repo_root / "models" / "logistic_binary.mod").read_text(encoding="utf-8")
    mod = tmp_path / "m.mod"
    mod.write_text(src.replace("../data/", str(spaced) + "/"), encoding="utf-8")
    result = check_control_stream(mod, spaced)
    assert not any("not found" in e for e in result.errors), result.errors


def test_recovery_rows_skip_a_parameter_with_no_standard_error():
    """A failed standard error is not the same as a failed recovery."""
    from nmlib.diagnostics import recovery_rows
    est = pd.DataFrame({
        "parameter": ["ok", "no_se"],
        "estimate": [1.0, 2.0],
        "se": [0.1, np.nan],
        "truth": [1.0, 2.0],
        "rse_pct": [10.0, np.nan],
        "pct_diff": [0.0, 0.0],
    })
    rows = recovery_rows(est)
    assert list(rows["parameter"]) == ["ok"]
