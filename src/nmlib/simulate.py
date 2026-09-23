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


def simulate_pk_2cmt(n_subjects=100, seed=101) -> Simulated:
    """Two-compartment IV infusion with allometric weight and IIV on CL and V1.

    The subjects carry four covariates, and only one of them does anything.
    That is deliberate: a covariate screen is only worth showing if it has
    something to find and something to correctly reject.

    * **WT** drives clearance and volume allometrically, and the control
      stream already includes it. A screen should therefore find *no*
      residual relationship between weight and the random effects -- the
      covariate model has already taken it.
    * **AGE** and **SEX** are recorded and have no effect at all. They are
      the negative controls.
    * **CYP** is a metaboliser genotype that genuinely lowers clearance,
      and the control stream does **not** include it. It is what the screen
      is supposed to catch.

    Leaving CYP out of the base model has a consequence worth being honest
    about: the between-subject variability the model can see on clearance
    is not the 30% that was drawn, but that inflated by the genotype split
    it cannot explain. `IIV_CL_CV_APPARENT` below is that combined figure,
    worked out in closed form, and it is what the estimate should match.
    Recovering 30% here would mean something had gone wrong.
    """
    rng = np.random.default_rng(seed)
    truth = dict(TVCL=5.0, TVV1=15.0, TVQ=3.0, TVV2=25.0,
                 WT_EXP_CL=0.75, WT_EXP_V=1.0,
                 IIV_CL_CV=0.30, IIV_V1_CV=0.25, PROP_ERR=0.15,
                 CYP_PM_FRACTION=0.22, CYP_PM_CL_RATIO=0.55)

    # Apparent variability on clearance once the unmodelled genotype split
    # is folded into it: var(log CL) = p(1-p)*log(ratio)^2 + omega^2.
    p, ratio = truth["CYP_PM_FRACTION"], truth["CYP_PM_CL_RATIO"]
    omega2 = np.log(1 + truth["IIV_CL_CV"] ** 2)
    apparent = p * (1 - p) * np.log(ratio) ** 2 + omega2
    truth["IIV_CL_CV_APPARENT"] = float(np.sqrt(np.expm1(apparent)))
    # The typical value moves too. A model without the genotype estimates
    # the geometric mean clearance over both groups, which is TVCL pulled
    # down by the poor metabolisers in proportion to how many there are.
    truth["TVCL_APPARENT"] = float(truth["TVCL"] * ratio ** p)

    times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 36.0, 48.0])
    interval, inf_dur, n_doses = 24.0, 1.0, 3
    rows = []
    for i in range(1, n_subjects + 1):
        wt = float(rng.uniform(45, 110))
        age = int(rng.integers(18, 81))
        sex = int(rng.integers(0, 2))
        cyp = int(rng.uniform() < truth["CYP_PM_FRACTION"])
        covariates = dict(WT=round(wt, 1), AGE=age, SEX=sex, CYP=cyp)

        cl = (truth["TVCL"] * (wt / 70) ** truth["WT_EXP_CL"]
              * (truth["CYP_PM_CL_RATIO"] if cyp else 1.0)
              * _lognormal(rng, truth["IIV_CL_CV"]))
        v1 = (truth["TVV1"] * (wt / 70) ** truth["WT_EXP_V"]
              * _lognormal(rng, truth["IIV_V1_CV"]))
        q = truth["TVQ"] * (wt / 70) ** truth["WT_EXP_CL"]
        v2 = truth["TVV2"] * (wt / 70) ** truth["WT_EXP_V"]
        dose = 500.0

        # Dosing records, one per administration.
        for d in range(n_doses):
            rows.append(dict(ID=i, TIME=d * interval, AMT=dose, RATE=dose / inf_dur,
                             DV=".", MDV=1, EVID=1, CMT=1, **covariates))
        # Observations, sampled after the last dose as well as within it.
        obs_t = np.concatenate([times, times[-3:] + 2 * interval])
        ipred = _two_cmt_conc(obs_t, dose, n_doses, interval, inf_dur, cl, v1, q, v2)
        dv = ipred * (1 + rng.normal(0, truth["PROP_ERR"], ipred.size))
        for tt, y in zip(obs_t, np.maximum(dv, 1e-4), strict=True):
            rows.append(dict(ID=i, TIME=float(tt), AMT=".", RATE=".",
                             DV=round(float(y), 4), MDV=0, EVID=0, CMT=1,
                             **covariates))

    data = pd.DataFrame(rows).sort_values(["ID", "TIME", "EVID"],
                                          ascending=[True, True, False])
    return Simulated(data.reset_index(drop=True), truth,
                     {"structure": "2-compartment, IV infusion",
                      "doses": n_doses, "interval_h": interval,
                      "covariates": "WT in the model; AGE and SEX inert; "
                                    "CYP genotype real but left out"})


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
    truth = dict(TVY0=50.0, TVKL=0.006, TVKD=0.0004, TVLAMBDA=0.015,
                 IIV_Y0_CV=0.35, IIV_KL_CV=0.35, IIV_KD_CV=0.40,
                 PROP_ERR=0.12, EXPO_CV=0.30)

    # Nominal exposure per arm. Each subject's actual exposure varies around
    # the arm mean, as it would with real PK variability -- and that spread
    # is what identifies the exposure-response slope. Three fixed exposures,
    # one of them zero, would leave the slope estimated from two points.
    arms = {0: 0.0, 1: 15.0, 2: 40.0}
    days = np.array([0, 21, 42, 63, 84, 105, 126, 168, 189, 210, 252])

    rows = []
    for i in range(1, n_subjects + 1):
        arm = int(rng.integers(0, 3))
        expo = arms[arm] * (_lognormal(rng, truth["EXPO_CV"]) if arms[arm] else 1.0)
        expo = round(expo, 2)
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
        # A floor well below any realistic trajectory: if it ever binds the
        # data is censored and the model cannot reproduce it, which is what
        # an earlier version of this simulator got wrong.
        for tt, y in zip(days, np.maximum(dv, 0.5), strict=True):
            rows.append(dict(ID=i, TIME=float(tt), DV=round(float(y), 3),
                             MDV=0, EVID=0, EXPO=expo, ARM=arm))

    return Simulated(pd.DataFrame(rows), truth,
                     {"structure": "Claret tumour growth inhibition",
                      "arms_mg_per_L": arms,
                      "endpoint": "sum of longest diameters (mm)"})


# --------------------------------------------------------------------------
# Indirect response (Dayneka/Jusko model I: inhibition of production)
# --------------------------------------------------------------------------

#: Daily dosing for a week, then four weeks of follow-up. The design is the
#: point of this example: a single dose of a drug with a five-hour half-life
#: perturbs a slow biomarker hardly at all, and the profile that comes back
#: looks like noise. Dosing to a new steady state and then stopping is what
#: makes the two clocks -- the drug's and the biomarker's -- visibly
#: different, which is the whole reason to reach for an indirect response
#: model instead of a direct one.
IDR_DOSE_TIMES = np.arange(7) * 24.0
IDR_OBS_TIMES = np.array([0, 12, 24, 48, 72, 96, 120, 144, 168, 192, 240,
                          288, 336, 432, 528, 672], dtype=float)


def simulate_idr_inhibition(n_subjects=96, seed=303) -> Simulated:
    """Indirect response with drug inhibiting production of the biomarker.

        dR/dt = kin*(1 - Imax*C/(IC50 + C)) - kout*R

    The response lags exposure because the biomarker turns over on its own
    clock; washout after stopping is set by kout, not by the drug's
    Here the drug's half-life is about seventeen hours and the biomarker's
    about three and a half days, so the separation is large enough to see:
    the response is still falling days after the concentration has reached
    steady state, and still recovering weeks after the last dose.
    """
    rng = np.random.default_rng(seed)
    truth = dict(TVCL=4.0, TVV=100.0,
                 # KIN and KOUT together set the untreated baseline KIN/KOUT,
                 # which is 100 units; KOUT alone sets how fast the biomarker
                 # moves, and it is deliberately slow next to the drug -- the
                 # biomarker's half-life is about five times the drug's.
                 TVKIN=0.80, TVKOUT=0.008, TVIMAX=0.80, TVIC50=8.0,
                 # The model estimates IMAX on the logit scale to keep it in
                 # (0, 1), so the comparable truth is logit(0.8).
                 TVIMAX_LOGIT=float(np.log(0.80 / 0.20)),
                 IIV_KOUT_CV=0.30, IIV_IC50_CV=0.50, PROP_ERR=0.10)

    times = IDR_OBS_TIMES
    rows = []
    for i in range(1, n_subjects + 1):
        dose = float(rng.choice([50.0, 150.0, 450.0, 1350.0]))
        kout = truth["TVKOUT"] * _lognormal(rng, truth["IIV_KOUT_CV"])
        ic50 = truth["TVIC50"] * _lognormal(rng, truth["IIV_IC50_CV"])
        kin = truth["TVKIN"]
        imax = truth["TVIMAX"]
        r0 = kin / kout  # baseline is the steady state of the untreated system

        # Driving concentration: one-compartment IV bolus given daily, each
        # dose cleared first order and superposed on what is left of the
        # previous ones.
        cl, v = truth["TVCL"], truth["TVV"]
        ke = cl / v

        def conc(t, dose=dose, v=v, ke=ke):
            elapsed = t - IDR_DOSE_TIMES
            return float(np.sum(np.where(elapsed >= 0,
                                         (dose / v) * np.exp(-ke * elapsed),
                                         0.0)))

        def rhs(t, y, imax=imax, ic50=ic50, kin=kin, kout=kout):
            c = conc(t)
            inhib = 1 - imax * c / (ic50 + c)
            return [kin * inhib - kout * y[0]]

        sol = solve_ivp(rhs, (0, float(times[-1])), [r0], t_eval=times,
                        rtol=1e-9, atol=1e-11, max_step=6.0)
        ipred = sol.y[0]
        dv = ipred * (1 + rng.normal(0, truth["PROP_ERR"], ipred.size))

        for t_dose in IDR_DOSE_TIMES:
            rows.append(dict(ID=i, TIME=float(t_dose), AMT=dose, DV=".", MDV=1,
                             EVID=1, CMT=1, DOSE=dose))
        for tt, y in zip(times, np.maximum(dv, 1e-3), strict=True):
            rows.append(dict(ID=i, TIME=float(tt), AMT=".", DV=round(float(y), 4),
                             MDV=0, EVID=0, CMT=2, DOSE=dose))

    data = pd.DataFrame(rows).sort_values(["ID", "TIME", "EVID"],
                                          ascending=[True, True, False])
    return Simulated(data.reset_index(drop=True), truth,
                     {"structure": "Indirect response, inhibition of production",
                      "baseline": "kin/kout = 100 units",
                      "dosing": "daily for 7 days, then 21 days of follow-up",
                      "biomarker_half_life_h": round(float(np.log(2) / 0.008), 1),
                      "drug_half_life_h": round(float(np.log(2) / (4.0 / 100.0)), 1)})


# --------------------------------------------------------------------------
# Time to event (Weibull hazard with an exposure effect)
# --------------------------------------------------------------------------

def simulate_tte_weibull(n_subjects=800, seed=404, follow_up=365.0) -> Simulated:
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

def simulate_logistic(n_subjects=350, visits=(4, 8, 12, 16, 20, 24),
                      seed=505) -> Simulated:
    """Binary endpoint with an exposure effect and between-subject variability.

        logit(P) = BASE + SLOPE*EXPO + ETA

    The endpoint is scored at every visit, and that is not decoration. With
    one record per subject the random effect on the logit is not
    identifiable at all: a single Bernoulli draw cannot distinguish a
    subject who is prone to respond from a subject who happened to respond,
    so ETA and the residual randomness are the same thing and OMEGA collapses
    to zero however it is estimated. Repeated assessments of the same subject
    are what separate the two, and they are also what a real study does.
    """
    rng = np.random.default_rng(seed)
    truth = dict(BASE=-1.20, SLOPE=0.055, IIV_SD=0.60)

    rows = []
    for i in range(1, n_subjects + 1):
        expo = float(rng.uniform(0, 60))
        eta = rng.normal(0, truth["IIV_SD"])   # one draw, used at every visit
        logit = truth["BASE"] + truth["SLOPE"] * expo + eta
        p = 1 / (1 + np.exp(-logit))
        for week in visits:
            rows.append(dict(ID=i, TIME=float(week),
                             DV=int(rng.uniform() < p), MDV=0, EVID=0,
                             EXPO=round(expo, 2)))

    data = pd.DataFrame(rows)
    return Simulated(data, truth,
                     {"structure": "Binary logistic with IIV, repeated visits",
                      "visits_weeks": list(visits),
                      "subjects": n_subjects,
                      "responses": int(data["DV"].sum()),
                      "n": len(data)})


SIMULATORS = {
    "pk_2cmt_iv": simulate_pk_2cmt,
    "tgi_claret": simulate_tgi_claret,
    "pkpd_idr_inhibition": simulate_idr_inhibition,
    "tte_weibull": simulate_tte_weibull,
    "logistic_binary": simulate_logistic,
}
