"""Simulators for each model in the library.

Every dataset in this repository is generated here from known parameters, so
each control stream has data to run against and a true answer to be judged
against. Nothing is real patient data.

Each simulator returns a NONMEM-ready long data frame and the parameter
values it was generated from. The parameter values are written out beside the
data, so an estimation run can be compared with the truth rather than only
with itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def _lognormal(rng, cv: float) -> float:
    """A log-normal multiplier with the given coefficient of variation."""
    return float(np.exp(rng.normal(0, np.sqrt(np.log(1 + cv ** 2)))))


@dataclass(frozen=True)
class Simulated:
    """A simulated dataset and the parameters behind it."""

    data: pd.DataFrame
    truth: dict[str, float]
    notes: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Shared PK: two-compartment IV, closed form
# --------------------------------------------------------------------------

def _two_cmt_conc(t, amount, n_doses, interval, inf_dur, cl, v1, q, v2):
    """Central concentration, superposed over identical doses.

    Exact for a linear system, so no integrator is needed. Verified against
    RK4 integration in tests.
    """
    k10, k12, k21 = cl / v1, q / v1, q / v2
    s = k10 + k12 + k21
    disc = np.sqrt(s * s - 4 * k10 * k21)
    alpha, beta = 0.5 * (s + disc), 0.5 * (s - disc)
    a = (alpha - k21) / (alpha - beta)
    b = (k21 - beta) / (alpha - beta)

    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    for i in range(n_doses):
        u = t - i * interval
        on = u >= 0
        uu = np.where(on, u, 0.0)
        if inf_dur > 0:
            rate = amount / inf_dur
            during = np.minimum(uu, inf_dur)
            after = uu - during
            c = (rate / v1) * (
                (a / alpha) * (1 - np.exp(-alpha * during)) * np.exp(-alpha * after)
                + (b / beta) * (1 - np.exp(-beta * during)) * np.exp(-beta * after)
            )
        else:
            c = (amount / v1) * (a * np.exp(-alpha * uu) + b * np.exp(-beta * uu))
        out += np.where(on, c, 0.0)
    return out


def simulate_pk_2cmt(n_subjects=60, seed=101) -> Simulated:
    """Two-compartment IV infusion with allometric weight and IIV on CL and V1."""
    rng = np.random.default_rng(seed)
    truth = dict(TVCL=5.0, TVV1=15.0, TVQ=3.0, TVV2=25.0,
                 WT_EXP_CL=0.75, WT_EXP_V=1.0,
                 IIV_CL_CV=0.30, IIV_V1_CV=0.25, PROP_ERR=0.15)

    times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 36.0, 48.0])
    interval, inf_dur, n_doses = 24.0, 1.0, 3
    rows = []
    for i in range(1, n_subjects + 1):
        wt = float(rng.uniform(45, 110))
        cl = (truth["TVCL"] * (wt / 70) ** truth["WT_EXP_CL"]
              * _lognormal(rng, truth["IIV_CL_CV"]))
        v1 = (truth["TVV1"] * (wt / 70) ** truth["WT_EXP_V"]
              * _lognormal(rng, truth["IIV_V1_CV"]))
        q = truth["TVQ"] * (wt / 70) ** truth["WT_EXP_CL"]
        v2 = truth["TVV2"] * (wt / 70) ** truth["WT_EXP_V"]
        dose = 500.0

        # Dosing records, one per administration.
        for d in range(n_doses):
            rows.append(dict(ID=i, TIME=d * interval, AMT=dose, RATE=dose / inf_dur,
                             DV=".", MDV=1, EVID=1, CMT=1, WT=round(wt, 1)))
        # Observations, sampled after the last dose as well as within it.
        obs_t = np.concatenate([times, times[-3:] + 2 * interval])
        ipred = _two_cmt_conc(obs_t, dose, n_doses, interval, inf_dur, cl, v1, q, v2)
        dv = ipred * (1 + rng.normal(0, truth["PROP_ERR"], ipred.size))
        for tt, y in zip(obs_t, np.maximum(dv, 1e-4), strict=True):
            rows.append(dict(ID=i, TIME=float(tt), AMT=".", RATE=".",
                             DV=round(float(y), 4), MDV=0, EVID=0, CMT=1,
                             WT=round(wt, 1)))

    data = pd.DataFrame(rows).sort_values(["ID", "TIME", "EVID"],
                                          ascending=[True, True, False])
    return Simulated(data.reset_index(drop=True), truth,
                     {"structure": "2-compartment, IV infusion",
                      "doses": n_doses, "interval_h": interval})


# --------------------------------------------------------------------------
# Tumour growth inhibition (Claret)
# --------------------------------------------------------------------------

def simulate_tgi_claret(n_subjects=80, seed=202) -> Simulated:
    """Claret TGI: exponential growth, drug-induced kill that decays with time.

        dY/dt = KL*Y - KD*C(t)*exp(-LAMBDA*t)*Y

    LAMBDA is the resistance term: the kill effect fades as the tumour adapts,
    which is what separates this from a plain kill model and why it can
    reproduce regrowth on treatment.
    """
    rng = np.random.default_rng(seed)
    truth = dict(TVY0=50.0, TVKL=0.006, TVKD=0.012, TVLAMBDA=0.015,
                 IIV_Y0_CV=0.45, IIV_KL_CV=0.50, IIV_KD_CV=0.60,
                 PROP_ERR=0.12)

    # Exposure: a steady average concentration per arm, as a trial would give.
    arms = {0: 0.0, 1: 15.0, 2: 40.0}
    days = np.array([0, 21, 42, 63, 84, 126, 168, 210, 252])

    rows = []
    for i in range(1, n_subjects + 1):
        arm = int(rng.integers(0, 3))
        expo = arms[arm]
        y0 = truth["TVY0"] * _lognormal(rng, truth["IIV_Y0_CV"])
        kl = truth["TVKL"] * _lognormal(rng, truth["IIV_KL_CV"])
        kd = truth["TVKD"] * _lognormal(rng, truth["IIV_KD_CV"])
        lam = truth["TVLAMBDA"]

        # Loop variables are bound as defaults: the closure is used inside
        # this iteration, but binding makes that explicit rather than lucky.
        def rhs(t, y, kl=kl, kd=kd, expo=expo, lam=lam):
            return [kl * y[0] - kd * expo * np.exp(-lam * t) * y[0]]

        sol = solve_ivp(rhs, (0, float(days[-1])), [y0], t_eval=days.astype(float),
                        rtol=1e-8, atol=1e-10)
        ipred = sol.y[0]
        dv = ipred * (1 + rng.normal(0, truth["PROP_ERR"], ipred.size))
        for tt, y in zip(days, np.maximum(dv, 0.1), strict=True):
            rows.append(dict(ID=i, TIME=float(tt), DV=round(float(y), 3),
                             MDV=0, EVID=0, EXPO=expo, ARM=arm))

    return Simulated(pd.DataFrame(rows), truth,
                     {"structure": "Claret tumour growth inhibition",
                      "arms_mg_per_L": arms,
                      "endpoint": "sum of longest diameters (mm)"})


# --------------------------------------------------------------------------
# Indirect response (Dayneka/Jusko model I: inhibition of production)
# --------------------------------------------------------------------------

def simulate_idr_inhibition(n_subjects=60, seed=303) -> Simulated:
    """Indirect response with drug inhibiting production of the biomarker.

        dR/dt = kin*(1 - Imax*C/(IC50 + C)) - kout*R

    The response lags exposure because the biomarker turns over on its own
    clock; washout after stopping is set by kout, not by the drug's half-life.
    """
    rng = np.random.default_rng(seed)
    truth = dict(TVCL=4.0, TVV=30.0,
                 TVKIN=10.0, TVKOUT=0.10, TVIMAX=0.80, TVIC50=8.0,
                 # The model estimates IMAX on the logit scale to keep it in
                 # (0, 1), so the comparable truth is logit(0.8).
                 TVIMAX_LOGIT=float(np.log(0.80 / 0.20)),
                 IIV_KOUT_CV=0.30, IIV_IC50_CV=0.50, PROP_ERR=0.10)

    times = np.array([0, 6, 12, 24, 48, 72, 96, 120, 168, 240, 336])
    rows = []
    for i in range(1, n_subjects + 1):
        dose = float(rng.choice([100.0, 300.0, 900.0]))
        kout = truth["TVKOUT"] * _lognormal(rng, truth["IIV_KOUT_CV"])
        ic50 = truth["TVIC50"] * _lognormal(rng, truth["IIV_IC50_CV"])
        kin = truth["TVKIN"]
        imax = truth["TVIMAX"]
        r0 = kin / kout  # baseline is the steady state of the untreated system

        # Driving concentration: one-compartment IV bolus, cleared first order.
        cl, v = truth["TVCL"], truth["TVV"]
        ke = cl / v

        def conc(t, dose=dose, v=v, ke=ke):
            return (dose / v) * np.exp(-ke * t)

        def rhs(t, y, imax=imax, ic50=ic50, kin=kin, kout=kout):
            c = conc(t)
            inhib = 1 - imax * c / (ic50 + c)
            return [kin * inhib - kout * y[0]]

        sol = solve_ivp(rhs, (0, float(times[-1])), [r0], t_eval=times.astype(float),
                        rtol=1e-8, atol=1e-10)
        ipred = sol.y[0]
        dv = ipred * (1 + rng.normal(0, truth["PROP_ERR"], ipred.size))

        rows.append(dict(ID=i, TIME=0.0, AMT=dose, DV=".", MDV=1, EVID=1, CMT=1,
                         DOSE=dose))
        for tt, y in zip(times, np.maximum(dv, 1e-3), strict=True):
            rows.append(dict(ID=i, TIME=float(tt), AMT=".", DV=round(float(y), 4),
                             MDV=0, EVID=0, CMT=2, DOSE=dose))

    data = pd.DataFrame(rows).sort_values(["ID", "TIME", "EVID"],
                                          ascending=[True, True, False])
    return Simulated(data.reset_index(drop=True), truth,
                     {"structure": "Indirect response, inhibition of production",
                      "baseline": "kin/kout"})


# --------------------------------------------------------------------------
# Time to event (Weibull hazard with an exposure effect)
# --------------------------------------------------------------------------

def simulate_tte_weibull(n_subjects=300, seed=404, follow_up=365.0) -> Simulated:
    """Parametric time-to-event, Weibull baseline hazard, exposure on the hazard.

        h(t) = LAMBDA*SHAPE*(LAMBDA*t)^(SHAPE-1) * exp(BETA*EXPO)

    Event times are drawn by inverse transform of the cumulative hazard, so
    the simulation is exact rather than a fine-grid approximation. Subjects
    without an event by the end of follow-up are censored, which is the
    normal state of a survival dataset and the reason the likelihood has two
    branches.
    """
    rng = np.random.default_rng(seed)
    truth = dict(LAMBDA=0.0025, SHAPE=1.35, BETA_EXPO=-0.025)

    rows = []
    n_events = 0
    for i in range(1, n_subjects + 1):
        expo = float(rng.uniform(0, 40))
        hr = np.exp(truth["BETA_EXPO"] * expo)
        u = rng.uniform()
        # Cumulative hazard H(t) = (LAMBDA*t)^SHAPE * hr; invert H(t) = -log(u).
        t_event = (-np.log(u) / hr) ** (1 / truth["SHAPE"]) / truth["LAMBDA"]
        if t_event <= follow_up:
            dv, time = 1, t_event
            n_events += 1
        else:
            dv, time = 0, follow_up

        # NONMEM TTE layout: a record at time 0 opening the interval, then the
        # event or censoring record.
        rows.append(dict(ID=i, TIME=0.0, DV=0, EVID=0, MDV=0, EXPO=round(expo, 2)))
        rows.append(dict(ID=i, TIME=round(float(time), 3), DV=dv, EVID=0, MDV=0,
                         EXPO=round(expo, 2)))

    return Simulated(pd.DataFrame(rows), truth,
                     {"structure": "Weibull time to event",
                      "events": n_events, "censored": n_subjects - n_events,
                      "follow_up_days": follow_up})


# --------------------------------------------------------------------------
# Binary response (logistic regression with exposure)
# --------------------------------------------------------------------------

def simulate_logistic(n_subjects=400, seed=505) -> Simulated:
    """Binary endpoint with an exposure effect and between-subject variability.

        logit(P) = BASE + SLOPE*EXPO + ETA
    """
    rng = np.random.default_rng(seed)
    truth = dict(BASE=-1.20, SLOPE=0.055, IIV_SD=0.60)

    rows = []
    for i in range(1, n_subjects + 1):
        expo = float(rng.uniform(0, 60))
        eta = rng.normal(0, truth["IIV_SD"])
        logit = truth["BASE"] + truth["SLOPE"] * expo + eta
        p = 1 / (1 + np.exp(-logit))
        dv = int(rng.uniform() < p)
        rows.append(dict(ID=i, TIME=0, DV=dv, MDV=0, EVID=0,
                         EXPO=round(expo, 2)))

    data = pd.DataFrame(rows)
    return Simulated(data, truth,
                     {"structure": "Binary logistic with IIV",
                      "responders": int(data["DV"].sum()),
                      "n": len(data)})


SIMULATORS = {
    "pk_2cmt_iv": simulate_pk_2cmt,
    "tgi_claret": simulate_tgi_claret,
    "pkpd_idr_inhibition": simulate_idr_inhibition,
    "tte_weibull": simulate_tte_weibull,
    "logistic_binary": simulate_logistic,
}
