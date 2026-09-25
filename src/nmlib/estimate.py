"""Population estimation, in Python, for every model in the library.

Why this exists
---------------
A library of control streams is only worth something if the streams have
been estimated against data and the estimates land where they should. Doing
that through a second toolchain makes the result conditional on that
toolchain installing; doing it here means the numbers on the dashboard are
produced by the same `python -m nmlib.build` that produces everything else,
on any machine, with no licence and no R.

How the likelihood is computed
------------------------------
Each model is a non-linear mixed effects model: fixed effects shared by
everyone, random effects one draw per subject. The marginal likelihood
integrates the random effects out, and that integral has no closed form, so
it is approximated by **adaptive Gauss-Hermite quadrature**: the quadrature
grid is centred on each subject's posterior mode and scaled by the curvature
there, then the integrand is evaluated on that grid.

With a single node the method reduces exactly to the Laplace approximation,
which is what NONMEM's `LAPLACE` does. With more nodes it converges on the
true integral, which is why the binary model -- where Laplace is known to be
biased -- is run with fifteen. The node count per model is stated in
`NODES` below and reported on the dashboard.

Everything is vectorised over subjects and over quadrature nodes -- each
optimiser step evaluates every subject at every node in one array
operation -- which is what brings the whole library down to a few minutes
on one core. Most of that is the indirect response model, the only one
whose structure has no closed form.

What comes out
--------------
For each model, into `fit/results/`:

* `<key>_estimates.csv` -- parameter, estimate, standard error, %RSE, and the
  value the data was simulated from
* `<key>_gof.csv` -- ID TIME DV PRED IPRED CWRES IWRES for the continuous
  models
* `<key>_vpc.csv` -- replicates simulated from the estimates, for the VPC
* `<key>_etas.csv` -- each subject's empirical Bayes random effects
* `<key>_status.json` -- objective function, run time, method, convergence,
  shrinkage, and the nested comparison against the simpler alternative

Starting values are deliberately displaced from the truth (see `START_SCALE`)
so that "the estimates recover the simulated values" means the optimiser
found them, not that it was handed them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, roots_hermite
from scipy.stats import chi2

#: Quadrature nodes per random effect, per model. One node is Laplace.
NODES = {
    "pk_2cmt_iv": 5,
    "tgi_claret": 3,
    "pkpd_idr_inhibition": 5,
    "tte_weibull": 1,      # no random effects; the integral is not needed
    "logistic_binary": 15,  # Laplace is biased for binary data; quadrature is not
}

#: Every structural starting value is the simulated value multiplied by this,
#: alternating up and down, so the optimiser starts a long way from the answer.
START_SCALE = (1.4, 0.7)

LOG2PI = float(np.log(2 * np.pi))

#: Largest exponent allowed anywhere in a model. The optimiser explores, and
#: an unguarded exp() turns one wild trial step into an inf that poisons the
#: objective everywhere downstream; saturating instead keeps the surface
#: finite and the search recoverable.
EXP_CLIP = 40.0


def _exp(z):
    """exp(), saturated so a trial step cannot produce inf."""
    return np.exp(np.clip(z, -EXP_CLIP, EXP_CLIP))


#: Search boxes for the two kinds of parameter every model has. Both are on
#: the log scale, both are far wider than any estimate here lands, and both
#: exist to stop the simplex wandering into the region where the saturated
#: model is flat rather than wrong.
SD_BOX = (float(np.log(0.02)), float(np.log(2.0)))     # a random-effect SD
ERR_BOX = (float(np.log(0.01)), float(np.log(1.0)))    # residual error CV


# ==========================================================================
# Reporting
# ==========================================================================

@dataclass(frozen=True)
class Report:
    """One row of the estimates table.

    `truth_key` names the entry in the model's truth file this estimate is
    comparable with. A parameter with no comparable truth carries None and is
    shown without a difference rather than paired with a guess.
    """

    label: str
    truth_key: str | None
    unit: str = ""


# ==========================================================================
# The model interface
# ==========================================================================

class PopModel:
    """A population model: fixed effects, random effects, a data likelihood.

    Subclasses supply the structure. Everything to do with the integral, the
    optimiser, the standard errors and the diagnostics is handled here.
    """

    key: str = ""
    n_eta: int = 0
    #: Labels for the estimation-scale parameter vector, for debugging only.
    est_names: tuple[str, ...] = ()
    #: Names of the random effects, in order, as the dashboard shows them.
    eta_labels: tuple[str, ...] = ()

    def __init__(self, data: pd.DataFrame, truth: dict[str, float]):
        self.data = data
        self.truth = truth
        self._warm: np.ndarray | None = None

    # -- to be provided by each model ------------------------------------
    def start(self) -> np.ndarray:
        raise NotImplementedError

    def omega_sd(self, x: np.ndarray) -> np.ndarray:
        """Standard deviations of the random effects, on the eta scale."""
        return np.zeros(0)

    def data_loglik(self, x: np.ndarray, etas: np.ndarray) -> np.ndarray:
        """log p(y_i | eta_i, theta), summed within subject.

        `etas` has shape (..., n_subject, n_eta); the return has shape
        (..., n_subject).
        """
        raise NotImplementedError

    def report(self) -> list[Report]:
        raise NotImplementedError

    def bounds(self) -> list[tuple[float, float]]:
        """Box the search on the estimation scale.

        Saturating the exponentials keeps the objective finite, but a finite
        objective on a saturated model is *flat*, and a flat region is
        exactly what a simplex will happily wander off into. The box is what
        actually keeps the search inside the part of the space where the
        model still means something; it is wide enough that no estimate here
        comes near an edge.
        """
        raise NotImplementedError

    def to_natural(self, x: np.ndarray) -> np.ndarray:
        """The reported parameter vector, in the units of the truth file."""
        raise NotImplementedError

    # -- optional: the continuous models also provide predictions ---------
    def predict(self, x: np.ndarray, etas: np.ndarray) -> np.ndarray | None:
        return None

    def simulate(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
        """One replicate of the observations, for the VPC."""
        return None

    def residual_sd(self, x: np.ndarray, ipred: np.ndarray) -> np.ndarray | None:
        """Standard deviation of the residual error at each prediction."""
        return None

    # -- machinery --------------------------------------------------------
    def joint(self, x: np.ndarray, etas: np.ndarray) -> np.ndarray:
        """log p(y_i | eta_i) + log p(eta_i): the integrand, on the log scale."""
        ll = self.data_loglik(x, etas)
        if self.n_eta:
            sd = self.omega_sd(x)
            z = etas / sd
            ll = ll - 0.5 * np.sum(z * z + LOG2PI + 2 * np.log(sd), axis=-1)
        return ll


# ==========================================================================
# Adaptive Gauss-Hermite quadrature
# ==========================================================================

def _tensor_nodes(k: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product Gauss-Hermite nodes and log-weights for k dimensions."""
    x, w = roots_hermite(n)
    grids = np.meshgrid(*([x] * k), indexing="ij")
    nodes = np.stack([g.ravel() for g in grids], axis=-1)          # (n**k, k)
    logw = np.zeros(nodes.shape[0])
    for g in np.meshgrid(*([np.log(w)] * k), indexing="ij"):
        logw = logw + g.ravel()
    return nodes, logw


def _fd_offsets(k: int, h: float) -> tuple[np.ndarray, list]:
    """Offsets for a central-difference gradient and Hessian in k dimensions."""
    offsets = [np.zeros(k)]
    index: list = [("f0",)]
    for i in range(k):
        for s in (+1, -1):
            e = np.zeros(k)
            e[i] = s * h
            offsets.append(e)
            index.append(("g", i, s))
    for i in range(k):
        for j in range(i + 1, k):
            for si in (+1, -1):
                for sj in (+1, -1):
                    e = np.zeros(k)
                    e[i], e[j] = si * h, sj * h
                    offsets.append(e)
                    index.append(("h", i, j, si, sj))
    return np.array(offsets), index


def _mode_and_curvature(model: PopModel, x: np.ndarray, eta0: np.ndarray,
                        iters: int = 25, h: float = 1e-3,
                        tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mode of each subject's random effects, and the curvature there.

    A damped Newton iteration run on every subject at once: the offsets for
    the finite-difference gradient and Hessian are stacked into a single
    batch, so each iteration is one vectorised model evaluation rather than a
    loop over subjects.

    Only the *data* term is differentiated numerically. The prior is Gaussian
    and its derivatives are written down exactly, which matters twice over:
    a second difference of a quantity of order a few hundred loses most of
    its precision to cancellation, and adding the exact prior curvature
    `diag(1/sd^2)` is what guarantees the result is positive definite. A
    subject whose data says nothing about one random effect -- a placebo
    patient and the drug kill rate, say -- then simply gets the prior back,
    which is the correct answer rather than a singular matrix.
    """
    k = model.n_eta
    eta = eta0.copy()
    n_subj = eta.shape[0]
    offsets, index = _fd_offsets(k, h)
    prior_prec = 1.0 / model.omega_sd(x) ** 2          # (k,)

    neg_hess = np.zeros((n_subj, k, k))
    for _ in range(iters):
        vals = model.data_loglik(x, eta[None, :, :] + offsets[:, None, :])
        pick = {tag: vals[i] for i, tag in enumerate(index)}
        f0 = pick[("f0",)]
        grad = np.zeros((n_subj, k))
        hess = np.zeros((n_subj, k, k))
        for i in range(k):
            gp, gm = pick[("g", i, 1)], pick[("g", i, -1)]
            grad[:, i] = (gp - gm) / (2 * h)
            hess[:, i, i] = (gp - 2 * f0 + gm) / (h * h)
        for i in range(k):
            for j in range(i + 1, k):
                cross = (pick[("h", i, j, 1, 1)] - pick[("h", i, j, 1, -1)]
                         - pick[("h", i, j, -1, 1)] + pick[("h", i, j, -1, -1)])
                hess[:, i, j] = hess[:, j, i] = cross / (4 * h * h)

        # Data part of the negative curvature, projected onto the positive
        # semi-definite cone, then the exact prior added on top.
        w, v = np.linalg.eigh(-hess)
        w = np.maximum(w, 0.0)
        neg_hess = (v @ (w[..., None] * np.swapaxes(v, -1, -2))
                    + np.diag(prior_prec))

        grad = grad - eta * prior_prec                 # exact prior gradient
        step = np.linalg.solve(neg_hess, grad[..., None])[..., 0]
        # Cap the step: Newton along a nearly flat direction can propose an
        # enormous move on the first iteration from a poor start.
        norm = np.linalg.norm(step, axis=-1, keepdims=True)
        step = np.where(norm > 1.0, step / np.maximum(norm, 1e-12), step)
        # An eta is a log-scale deviation; +-6 is a factor of 400, past which
        # the search has left the model rather than found an unusual subject.
        eta = np.clip(eta + step, -6.0, 6.0)
        if np.max(np.abs(step)) < tol:
            break

    return eta, neg_hess


def marginal_loglik(model: PopModel, x: np.ndarray, nodes: int) -> float:
    """Total marginal log-likelihood, random effects integrated out."""
    if model.n_eta == 0:
        return float(np.sum(model.data_loglik(x, np.zeros((0, 0)))))

    n_subj = model.n_subj
    eta0 = (model._warm if model._warm is not None
            else np.zeros((n_subj, model.n_eta)))
    eta_hat, neg_hess = _mode_and_curvature(model, x, eta0)
    if np.all(np.isfinite(eta_hat)):
        model._warm = eta_hat

    # Sigma = inverse curvature; L its Cholesky factor. The quadrature grid is
    # placed at the mode and stretched by L, which is what makes it adaptive.
    chol_h = np.linalg.cholesky(neg_hess)
    log_det_sigma_half = -np.sum(np.log(np.diagonal(chol_h, axis1=-2, axis2=-1)),
                                 axis=-1)

    k = model.n_eta
    grid, logw = _tensor_nodes(k, nodes)
    # eta_q = eta_hat + sqrt(2) * L_sigma @ z, with L_sigma = inv(chol_h).T
    scaled = np.linalg.solve(np.swapaxes(chol_h, -1, -2)[None, ...],
                             (np.sqrt(2.0) * grid)[:, None, :, None])[..., 0]
    etas = eta_hat[None, :, :] + scaled

    terms = (logw[:, None] + model.joint(x, etas)
             + np.sum(grid * grid, axis=-1)[:, None])
    per_subject = (0.5 * k * np.log(2.0) + log_det_sigma_half
                   + _logsumexp(terms, axis=0))
    return float(np.sum(per_subject))


def _logsumexp(a: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


# ==========================================================================
# The models
# ==========================================================================

def _grid_from_times(times: np.ndarray, step_max: float) -> np.ndarray:
    """A solver grid that contains every observation time."""
    out = [float(times[0])]
    for a, b in zip(times[:-1], times[1:], strict=True):
        n = max(1, int(np.ceil((b - a) / step_max)))
        out.extend(np.linspace(a, b, n + 1)[1:].tolist())
    return np.array(out)


def conc_2cmt(t, dose, n_doses, interval, inf_dur, cl, v1, q, v2):
    """Central concentration for a two-compartment IV infusion, superposed.

    Broadcasting: `t` indexes observations on the last axis, the parameters
    index subjects (and quadrature nodes) on the axes before it. Exact for a
    linear system, which is why no integrator appears anywhere near the PK.
    """
    k10, k12, k21 = cl / v1, q / v1, q / v2
    s = k10 + k12 + k21
    disc = np.sqrt(np.maximum(s * s - 4 * k10 * k21, 1e-30))
    alpha, beta = 0.5 * (s + disc), 0.5 * (s - disc)
    a = (alpha - k21) / (alpha - beta)
    b = (k21 - beta) / (alpha - beta)

    v1 = np.maximum(v1, 1e-12)
    out = np.zeros(np.broadcast_shapes(np.shape(cl), np.shape(t)))
    rate = dose / inf_dur
    for i in range(n_doses):
        u = t - i * interval
        on = u >= 0
        uu = np.where(on, u, 0.0)
        during = np.minimum(uu, inf_dur)
        after = uu - during
        c = (rate / v1) * (
            (a / alpha) * (1 - np.exp(-alpha * during)) * np.exp(-alpha * after)
            + (b / beta) * (1 - np.exp(-beta * during)) * np.exp(-beta * after))
        out = out + np.where(on, c, 0.0)
    return out


class PK2Cmt(PopModel):
    """Two-compartment IV infusion; random effects on clearance and V1."""

    key = "pk_2cmt_iv"
    n_eta = 2
    eta_labels = ("CL", "V1")
    est_names = ("log CL", "log V1", "log Q", "log V2",
                 "log sd(CL)", "log sd(V1)", "log prop")

    def __init__(self, data, truth):
        super().__init__(data, truth)
        obs = data[data["MDV"].astype(int) == 0].copy()
        obs["DV"] = pd.to_numeric(obs["DV"])
        wide = obs.pivot_table(index="ID", columns="TIME", values="DV")
        self.times = wide.columns.to_numpy(dtype=float)
        self.y = wide.to_numpy()
        self.ids = wide.index.to_numpy()
        self.wt = (obs.groupby("ID")["WT"].first().reindex(self.ids)
                   .to_numpy()[:, None])
        self.n_subj = len(self.ids)
        self.dose, self.interval, self.inf_dur, self.n_doses = 500.0, 24.0, 1.0, 3
        # A proportional error model gives a prediction of zero infinite
        # weight: the residual standard deviation goes to zero with the
        # prediction, so one badly predicted point can dominate the whole
        # likelihood and the information matrix comes back singular. Floor
        # the error scale at a thousandth of the smallest observation. A
        # model that predicts near zero where something was measured is then
        # penalised heavily, which is right, instead of producing a number
        # that cannot be computed -- which is what a one-compartment model
        # fitted to two-compartment data does at the tail.
        self.sd_floor = float(np.min(self.y)) * 1e-3

    def start(self):
        up, down = START_SCALE
        t = self.truth
        return np.array([
            np.log(t["TVCL"] * up), np.log(t["TVV1"] * down),
            np.log(t["TVQ"] * up), np.log(t["TVV2"] * down),
            np.log(0.45), np.log(0.45), np.log(0.30)])

    def bounds(self):
        wide = [(v - 4.0, v + 4.0) for v in self.start()[:4]]
        return wide + [SD_BOX, SD_BOX, ERR_BOX]

    def omega_sd(self, x):
        return _exp(x[4:6])

    def _pk(self, x, etas):
        cl = _exp(x[0] + etas[..., 0]) * (self.wt[:, 0] / 70) ** 0.75
        v1 = _exp(x[1] + etas[..., 1]) * (self.wt[:, 0] / 70)
        q = _exp(x[2]) * (self.wt[:, 0] / 70) ** 0.75
        v2 = _exp(x[3]) * (self.wt[:, 0] / 70)
        return conc_2cmt(self.times, self.dose, self.n_doses, self.interval,
                         self.inf_dur, cl[..., None], v1[..., None],
                         q[..., None], v2[..., None])

    def predict(self, x, etas):
        return self._pk(x, etas)

    def residual_sd(self, x, ipred):
        return _exp(x[6]) * np.maximum(ipred, self.sd_floor)

    def data_loglik(self, x, etas):
        f = self._pk(x, etas)
        sd = self.residual_sd(x, f)
        r = (self.y - f) / sd
        return -0.5 * np.sum(r * r + LOG2PI + 2 * np.log(sd), axis=-1)

    def to_natural(self, x):
        return np.array([_exp(x[0]), _exp(x[1]), _exp(x[2]), _exp(x[3]),
                         _cv(_exp(x[4])), _cv(_exp(x[5])), _exp(x[6])])

    def report(self):
        # CL and its variability are compared against the *apparent*
        # simulated values, not the drawn ones. This model has no genotype
        # term, so what it can estimate is the geometric mean clearance
        # across both metaboliser groups and a between-subject variability
        # that has the genotype split folded into it. Comparing against the
        # drawn values would report a correct fit of a deliberately
        # incomplete model as a recovery failure.
        return [
            Report("CL, apparent", "TVCL_APPARENT", "L/h"),
            Report("V1", "TVV1", "L"),
            Report("Q", "TVQ", "L/h"), Report("V2", "TVV2", "L"),
            Report("IIV on CL, apparent", "IIV_CL_CV_APPARENT", "CV"),
            Report("IIV on V1", "IIV_V1_CV", "CV"),
            Report("Proportional error", "PROP_ERR", "CV"),
        ]

    def simulate(self, x, rng):
        etas = rng.normal(0, self.omega_sd(x), (self.n_subj, 2))
        f = np.maximum(self._pk(x, etas), 1e-10)
        return np.maximum(f * (1 + rng.normal(0, _exp(x[6]), f.shape)), 1e-4)


class TGIClaret(PopModel):
    """Claret tumour growth inhibition; random effects on Y0, KL and KD.

    The ODE integrates in closed form,

        Y(t) = Y0 * exp(KL*t - (KD*EXPO/LAMBDA) * (1 - exp(-LAMBDA*t)))

    so the estimation never calls a solver, which is what makes three random
    effects affordable.
    """

    key = "tgi_claret"
    n_eta = 3
    eta_labels = ("Y0", "KL", "KD")
    est_names = ("log Y0", "log KL", "log KD", "log LAMBDA",
                 "log sd(Y0)", "log sd(KL)", "log sd(KD)", "log prop")

    def __init__(self, data, truth):
        super().__init__(data, truth)
        d = data.copy()
        d["DV"] = pd.to_numeric(d["DV"])
        wide = d.pivot_table(index="ID", columns="TIME", values="DV")
        self.times = wide.columns.to_numpy(dtype=float)
        self.y = wide.to_numpy()
        self.ids = wide.index.to_numpy()
        self.expo = d.groupby("ID")["EXPO"].first().reindex(self.ids).to_numpy()
        self.arm = d.groupby("ID")["ARM"].first().reindex(self.ids).to_numpy()
        self.n_subj = len(self.ids)

    def start(self):
        up, down = START_SCALE
        t = self.truth
        return np.array([
            np.log(t["TVY0"] * down), np.log(t["TVKL"] * up),
            np.log(t["TVKD"] * down), np.log(t["TVLAMBDA"] * up),
            np.log(0.5), np.log(0.5), np.log(0.5), np.log(0.25)])

    def bounds(self):
        wide = [(v - 4.0, v + 4.0) for v in self.start()[:4]]
        return wide + [SD_BOX, SD_BOX, SD_BOX, ERR_BOX]

    def omega_sd(self, x):
        return _exp(x[4:7])

    def _curve(self, x, etas):
        y0 = _exp(x[0] + etas[..., 0])[..., None]
        kl = _exp(x[1] + etas[..., 1])[..., None]
        kd = _exp(x[2] + etas[..., 2])[..., None]
        lam = _exp(x[3])
        expo = self.expo[..., None]
        decay = (1 - _exp(-lam * self.times)) / lam
        return y0 * _exp(kl * self.times - kd * expo * decay)

    def predict(self, x, etas):
        return self._curve(x, etas)

    def residual_sd(self, x, ipred):
        return _exp(x[7]) * np.maximum(ipred, 1e-10)

    def data_loglik(self, x, etas):
        f = np.maximum(self._curve(x, etas), 1e-8)
        sd = _exp(x[7]) * f
        r = (self.y - f) / sd
        return -0.5 * np.sum(r * r + LOG2PI + 2 * np.log(sd), axis=-1)

    def to_natural(self, x):
        return np.array([_exp(x[0]), _exp(x[1]), _exp(x[2]), _exp(x[3]),
                         _cv(_exp(x[4])), _cv(_exp(x[5])), _cv(_exp(x[6])),
                         _exp(x[7])])

    def report(self):
        return [
            Report("Y0, baseline size", "TVY0", "mm"),
            Report("KL, growth rate", "TVKL", "1/day"),
            Report("KD, kill rate", "TVKD", "1/day per unit exposure"),
            Report("LAMBDA, resistance", "TVLAMBDA", "1/day"),
            Report("IIV on Y0", "IIV_Y0_CV", "CV"),
            Report("IIV on KL", "IIV_KL_CV", "CV"),
            Report("IIV on KD", "IIV_KD_CV", "CV"),
            Report("Proportional error", "PROP_ERR", "CV"),
        ]

    def simulate(self, x, rng):
        etas = rng.normal(0, self.omega_sd(x), (self.n_subj, 3))
        f = np.maximum(self._curve(x, etas), 1e-8)
        return np.maximum(f * (1 + rng.normal(0, _exp(x[7]), f.shape)), 0.5)


class IDRInhibition(PopModel):
    """Indirect response, inhibition of production; random effects on KOUT and IC50.

    The driving concentration is a one-compartment IV bolus, given daily and
    superposed, whose CL and V come from the PK model rather than from this
    dataset -- the biomarker records carry no concentration, so they cannot
    identify PK, and fixing it is what a sequential PK/PD analysis does.

    The response ODE has no closed form, so it is integrated with an
    exponential integrator: over each step the loss term `exp(-KOUT*h)` is
    exact and only the production term is quadratured. That keeps the solve
    accurate at a fixed, small number of operations, and the whole thing is
    vectorised over subjects and quadrature nodes at once. The grid breaks
    at every dose time, because the concentration is not smooth there.
    """

    key = "pkpd_idr_inhibition"
    n_eta = 2
    eta_labels = ("KOUT", "IC50")
    est_names = ("log KIN", "log KOUT", "logit IMAX", "log IC50",
                 "log sd(KOUT)", "log sd(IC50)", "log prop")

    #: Solver step ceiling (h) and Gauss-Legendre nodes within each step.
    STEP_MAX = 6.0
    GL_NODES = 6

    def __init__(self, data, truth):
        super().__init__(data, truth)
        obs = data[data["MDV"].astype(int) == 0].copy()
        obs["DV"] = pd.to_numeric(obs["DV"])
        wide = obs.pivot_table(index="ID", columns="TIME", values="DV")
        self.times = wide.columns.to_numpy(dtype=float)
        self.y = wide.to_numpy()
        self.ids = wide.index.to_numpy()
        self.dose = (obs.groupby("ID")["DOSE"].first().reindex(self.ids)
                     .to_numpy())
        self.n_subj = len(self.ids)

        # PK fixed from the PK model, as in a sequential analysis.
        self.cl, self.v = truth["TVCL"], truth["TVV"]
        self.ke = self.cl / self.v
        self.c0 = self.dose / self.v
        dosing = data[data["EVID"].astype(int) == 1]
        self.dose_times = np.unique(dosing["TIME"].to_numpy(dtype=float))

        # Breakpoints: every observation time and every dose time, since the
        # concentration steps discontinuously at a bolus.
        breaks = np.unique(np.concatenate([self.times, self.dose_times]))
        grid = _grid_from_times(breaks, self.STEP_MAX)
        self.grid = grid
        self.obs_index = np.searchsorted(grid, self.times)
        u, w = np.polynomial.legendre.leggauss(self.GL_NODES)
        self._steps = []
        for a, b in zip(grid[:-1], grid[1:], strict=True):
            h = b - a
            nodes = a + 0.5 * h * (u + 1)          # quadrature points in (a, b)
            # Concentration per unit of C0, summed over the doses already
            # given at each quadrature point.
            elapsed = nodes[:, None] - self.dose_times[None, :]
            shape = np.sum(np.where(elapsed >= 0, _exp(-self.ke * elapsed), 0.0),
                           axis=1)
            self._steps.append((h, 0.5 * h * w, nodes - a, shape))

    def start(self):
        up, down = START_SCALE
        t = self.truth
        return np.array([
            np.log(t["TVKIN"] * up), np.log(t["TVKOUT"] * down),
            logit(0.5), np.log(t["TVIC50"] * up),
            np.log(0.45), np.log(0.45), np.log(0.25)])

    def bounds(self):
        wide = [(v - 4.0, v + 4.0) for v in self.start()[:2]]
        return wide + [(-6.0, 6.0), (self.start()[3] - 4.0, self.start()[3] + 4.0),
                       SD_BOX, SD_BOX, ERR_BOX]

    def omega_sd(self, x):
        return _exp(x[4:6])

    def _solve(self, x, etas):
        kin = _exp(x[0])
        kout = _exp(x[1] + etas[..., 0])
        imax = expit(x[2])
        ic50 = _exp(x[3] + etas[..., 1])[..., None]

        r = kin / kout                       # untreated steady state
        out = np.empty(r.shape + (len(self.grid),))
        out[..., 0] = r
        for i, (h, gw, du, conc_shape) in enumerate(self._steps):
            # Concentration at this step's quadrature points. The PK is fixed
            # and the schedule is shared, so the only thing that varies
            # between subjects is the dose size. (n_subject, nodes).
            c = self.c0[:, None] * conc_shape
            prod = kin * (1 - imax * c / (ic50 + c))
            # Exact decay of what is already there, quadrature on what is made.
            weight = _exp(-kout[..., None] * (h - du))
            r = r * _exp(-kout * h) + np.sum(gw * weight * prod, axis=-1)
            out[..., i + 1] = r
        return out[..., self.obs_index]

    def predict(self, x, etas):
        return self._solve(x, etas)

    def residual_sd(self, x, ipred):
        return _exp(x[6]) * np.maximum(ipred, 1e-10)

    def data_loglik(self, x, etas):
        f = np.maximum(self._solve(x, etas), 1e-8)
        sd = _exp(x[6]) * f
        r = (self.y - f) / sd
        return -0.5 * np.sum(r * r + LOG2PI + 2 * np.log(sd), axis=-1)

    def to_natural(self, x):
        return np.array([_exp(x[0]), _exp(x[1]), expit(x[2]), _exp(x[3]),
                         _cv(_exp(x[4])), _cv(_exp(x[5])), _exp(x[6])])

    def report(self):
        return [
            Report("KIN, production rate", "TVKIN", "units/h"),
            Report("KOUT, loss rate", "TVKOUT", "1/h"),
            Report("IMAX", "TVIMAX", ""),
            Report("IC50", "TVIC50", "mg/L"),
            Report("IIV on KOUT", "IIV_KOUT_CV", "CV"),
            Report("IIV on IC50", "IIV_IC50_CV", "CV"),
            Report("Proportional error", "PROP_ERR", "CV"),
        ]

    def simulate(self, x, rng):
        etas = rng.normal(0, self.omega_sd(x), (self.n_subj, 2))
        f = np.maximum(self._solve(x, etas), 1e-8)
        return np.maximum(f * (1 + rng.normal(0, _exp(x[6]), f.shape)), 1e-3)


class TTEWeibull(PopModel):
    """Weibull time to first event with exposure on the hazard.

    No random effects, so there is no integral: the marginal likelihood is
    the likelihood. Censored subjects contribute the survivor function, event
    subjects the survivor times the hazard -- the two branches the control
    stream writes out by hand.
    """

    key = "tte_weibull"
    n_eta = 0
    est_names = ("log LAMBDA", "log SHAPE", "BETA")

    def __init__(self, data, truth):
        super().__init__(data, truth)
        last = data[data["TIME"] > 0].copy()
        self.time = last["TIME"].to_numpy(dtype=float)
        self.event = pd.to_numeric(last["DV"]).to_numpy(dtype=float)
        self.expo = last["EXPO"].to_numpy(dtype=float)
        self.ids = last["ID"].to_numpy()
        self.n_subj = len(self.ids)
        self.follow_up = float(self.time.max())

    def start(self):
        up, down = START_SCALE
        t = self.truth
        return np.array([np.log(t["LAMBDA"] * up), np.log(t["SHAPE"] * down),
                         0.0])

    def bounds(self):
        return [(self.start()[0] - 4.0, self.start()[0] + 4.0),
                (np.log(0.2), np.log(6.0)), (-2.0, 2.0)]

    def data_loglik(self, x, etas):
        lam, shape, beta = _exp(x[0]), _exp(x[1]), x[2]
        hr = _exp(beta * self.expo)
        cum_haz = (lam * self.time) ** shape * hr
        log_haz = (np.log(lam) + np.log(shape)
                   + (shape - 1) * np.log(np.maximum(lam * self.time, 1e-300))
                   + beta * self.expo)
        return np.array([np.sum(self.event * log_haz - cum_haz)])

    def to_natural(self, x):
        return np.array([_exp(x[0]), _exp(x[1]), x[2]])

    def report(self):
        return [
            Report("LAMBDA, hazard scale", "LAMBDA", "1/day"),
            Report("SHAPE", "SHAPE", ""),
            Report("BETA, exposure on log hazard", "BETA_EXPO", "per unit"),
        ]


class LogisticBinary(PopModel):
    """Binary endpoint, logistic on exposure, with a random effect on the logit.

    The endpoint is scored at several visits per subject, and the same
    subject-level random effect applies at each. That repetition is what
    makes the random effect identifiable at all: from a single Bernoulli
    draw per subject there is nothing to separate a subject who is prone to
    respond from a subject who happened to respond.

    Binary data is also where the Laplace approximation is at its weakest,
    so this model is integrated with fifteen quadrature nodes rather than
    one.
    """

    key = "logistic_binary"
    n_eta = 1
    eta_labels = ("logit",)
    est_names = ("BASE", "SLOPE", "log sd(eta)")

    def __init__(self, data, truth):
        super().__init__(data, truth)
        d = data.copy()
        d["DV"] = pd.to_numeric(d["DV"])
        wide = d.pivot_table(index="ID", columns="TIME", values="DV")
        self.times = wide.columns.to_numpy(dtype=float)
        self.y = wide.to_numpy()                       # (subject, visit)
        self.ids = wide.index.to_numpy()
        self.expo = (d.groupby("ID")["EXPO"].first().reindex(self.ids)
                     .to_numpy(dtype=float))
        self.n_subj = len(self.ids)

    def start(self):
        up, down = START_SCALE
        t = self.truth
        return np.array([t["BASE"] * down, t["SLOPE"] * up, np.log(0.25)])

    def bounds(self):
        return [(-8.0, 8.0), (-2.0, 2.0), SD_BOX]

    def omega_sd(self, x):
        return _exp(x[2:3])

    def data_loglik(self, x, etas):
        # One logit per subject; every visit of that subject shares it.
        lin = (x[0] + x[1] * self.expo + etas[..., 0])[..., None]
        p = np.clip(expit(lin), 1e-12, 1 - 1e-12)
        return np.sum(self.y * np.log(p) + (1 - self.y) * np.log1p(-p), axis=-1)

    def to_natural(self, x):
        return np.array([x[0], x[1], _exp(x[2])])

    def report(self):
        return [
            Report("BASE, logit intercept", "BASE", ""),
            Report("SLOPE, logit per unit exposure", "SLOPE", "per unit"),
            Report("IIV on the logit", "IIV_SD", "SD"),
        ]


class PK1Cmt(PK2Cmt):
    """The obvious simpler alternative: one compartment, no distribution phase.

        C(t) = sum over doses of (dose/(V*T)) * (1 - exp(-ke*t_in))
                                              * exp(-ke*t_after)

    Written out as its own closed form rather than obtained by driving Q
    towards zero in the two-compartment model. That limit is numerically
    degenerate -- beta goes to zero while b/beta goes to infinity, and the
    two cancel only in exact arithmetic -- so the objective function comes
    back as garbage rather than as a worse fit. A comparison is only worth
    reporting if the reduced model was actually fitted.

    Q and V2 keep their slots in the parameter vector and are held fixed,
    since nothing here reads them; the comparison gives up two parameters.
    """

    key = "pk_1cmt_iv"

    def _pk(self, x, etas):
        cl = _exp(x[0] + etas[..., 0]) * (self.wt[:, 0] / 70) ** 0.75
        v = _exp(x[1] + etas[..., 1]) * (self.wt[:, 0] / 70)
        ke = (cl / v)[..., None]
        rate = self.dose / self.inf_dur

        out = np.zeros(np.broadcast_shapes(ke.shape, self.times.shape))
        for i in range(self.n_doses):
            u = self.times - i * self.interval
            on = u >= 0
            uu = np.where(on, u, 0.0)
            during = np.minimum(uu, self.inf_dur)
            after = uu - during
            c = (rate / (v[..., None] * ke)) * (1 - _exp(-ke * during)) \
                * _exp(-ke * after)
            out = out + np.where(on, c, 0.0)
        return out


class IDRDirect(IDRInhibition):
    """The obvious simpler alternative: the drug acts on the biomarker itself.

        R(t) = R0 * (1 - IMAX*C(t)/(IC50 + C(t)))

    No turnover, so the response is a function of the concentration *now*
    and nothing else: it falls the moment the drug arrives and returns the
    moment it leaves. That is the whole difference from the indirect model,
    and comparing the two on the same data is what shows whether the
    turnover structure is earning the parameter it costs. KOUT survives only
    as the thing that sets the baseline, R0 = KIN/KOUT.

    It needs no solver at all, which makes the comparison nearly free.
    """

    key = "pkpd_idr_direct"

    def _solve(self, x, etas):
        kin = _exp(x[0])
        kout = _exp(x[1] + etas[..., 0])
        imax = expit(x[2])
        ic50 = _exp(x[3] + etas[..., 1])[..., None]
        # Concentration at each observation time, same PK as the full model.
        elapsed = self.times[:, None] - self.dose_times[None, :]
        shape = np.sum(np.where(elapsed >= 0, _exp(-self.ke * elapsed), 0.0),
                       axis=1)
        c = self.c0[:, None] * shape
        return (kin / kout)[..., None] * (1 - imax * c / (ic50 + c))


#: The one feature that separates each model from the obvious simpler
#: alternative, and how to switch that feature off. `fix` maps an index in
#: the estimation-scale parameter vector to the value that disables it;
#: everything else is re-estimated, so the comparison is between two fitted
#: models rather than between a fit and a guess.
#:
#: `boundary` marks a test where the null sits on the edge of the parameter
#: space -- a rate or a variance held at zero. The chi-square reference
#: distribution is conservative there (the true null is a mixture), so the
#: real p-value is smaller than the one reported and the conclusion only
#: gets stronger.
COMPARISONS = {
    "pk_2cmt_iv": {
        "feature": "The second compartment",
        "against": "a one-compartment model",
        "question": "Does the distribution phase exist, or would one "
                    "compartment describe these data just as well?",
        "alternative": PK1Cmt,
        "fix": {2: 0.0, 3: 0.0},                  # Q and V2 unused there
        "df": 2,
        "boundary": True,
    },
    "tgi_claret": {
        "feature": "The resistance term LAMBDA",
        "against": "a plain kill model, with no loss of drug effect",
        "question": "Without the resistance term the model can't regrow a "
                    "tumour while treatment continues. Does the data show "
                    "regrowth?",
        "fix": {3: float(np.log(1e-8))},          # LAMBDA -> 0
        "boundary": True,
    },
    "pkpd_idr_inhibition": {
        "feature": "Acting on turnover rather than on the biomarker",
        "against": "a direct effect model",
        "question": "In a direct model the response tracks concentration "
                    "exactly. Is there enough lag in the data to justify the "
                    "indirect structure?",
        "alternative": IDRDirect,
        "df": 0,                                   # same parameter count
        "boundary": False,
    },
    "tte_weibull": {
        "feature": "The Weibull shape parameter",
        "against": "a constant hazard (exponential)",
        "question": "Does the hazard change with time, or is a single rate "
                    "enough?",
        "fix": {1: 0.0},                           # SHAPE -> 1
        "boundary": False,
    },
    "logistic_binary": {
        "feature": "Between-subject variability on the logit",
        "against": "plain logistic regression, every subject alike",
        "question": "Six visits per subject make this estimable. Is there "
                    "really variation between subjects?",
        "fix": {2: float(np.log(0.02))},           # OMEGA -> ~0
        "boundary": True,
    },
}


MODELS: dict[str, type[PopModel]] = {
    "pk_2cmt_iv": PK2Cmt,
    "tgi_claret": TGIClaret,
    "pkpd_idr_inhibition": IDRInhibition,
    "tte_weibull": TTEWeibull,
    "logistic_binary": LogisticBinary,
}


def _method_name(nodes: int, n_eta: int) -> str:
    """How the likelihood was actually computed, for the dashboard.

    Calling a model with no random effects "Laplace" would be wrong: there
    is no integral to approximate, so the likelihood is exact.
    """
    if n_eta == 0:
        return "exact maximum likelihood (no random effects to integrate out)"
    if nodes == 1:
        return "Laplace approximation (one quadrature node)"
    return f"adaptive Gauss-Hermite quadrature, {nodes} nodes"


def _cv(sd: float) -> float:
    """Coefficient of variation of a log-normal with this log-scale SD."""
    return float(np.sqrt(np.expm1(sd * sd)))


# ==========================================================================
# Fitting
# ==========================================================================

@dataclass
class FitResult:
    key: str
    objective: float
    seconds: float
    nodes: int
    converged: bool
    message: str
    estimates: pd.DataFrame
    #: The fitted parameter vector on the estimation scale, used to warm
    #: start the reduced model in the nested comparisons.
    x: np.ndarray | None = None
    gof: pd.DataFrame | None = None
    vpc: pd.DataFrame | None = None
    etas: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)


def _numerical_hessian(fn, x: np.ndarray, rel: float = 1e-4) -> np.ndarray:
    n = len(x)
    h = rel * np.maximum(np.abs(x), 1.0)
    out = np.zeros((n, n))
    f0 = fn(x)
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = h[i], h[j]
            if i == j:
                val = (fn(x + ei) - 2 * f0 + fn(x - ei)) / (h[i] * h[i])
            else:
                val = (fn(x + ei + ej) - fn(x + ei - ej) - fn(x - ei + ej)
                       + fn(x - ei - ej)) / (4 * h[i] * h[j])
            out[i, j] = out[j, i] = val
    return out


def _jacobian(fn, x: np.ndarray, rel: float = 1e-5) -> np.ndarray:
    h = rel * np.maximum(np.abs(x), 1.0)
    cols = []
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = h[i]
        cols.append((fn(x + e) - fn(x - e)) / (2 * h[i]))
    return np.stack(cols, axis=1)


def _optimise(objective, x0: np.ndarray, box: list[tuple[float, float]],
              thorough: bool = True):
    """Simplex into the right basin, then a quasi-Newton step to finish.

    The start is deliberately poor and the first few hundred units of
    objective function are where a gradient method is least reliable, so
    Nelder-Mead goes first; the quadrature makes the surface smooth enough
    for L-BFGS-B to sharpen the answer afterwards.

    `thorough=False` is for the reduced models in the nested comparisons.
    Those are only ever read as a difference of a few tens to a few
    thousand objective-function units, so a tolerance that would matter for
    a reported estimate is wasted effort there -- and a deliberately
    mis-specified model is exactly the case where the simplex takes longest
    to satisfy a tight one.
    """
    if thorough:
        nm = {"maxiter": 400 * max(len(x0), 1), "xatol": 1e-4, "fatol": 1e-4}
        lb = {"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8}
    else:
        nm = {"maxiter": 120 * max(len(x0), 1), "xatol": 1e-3, "fatol": 1e-2}
        lb = {"maxiter": 200, "ftol": 1e-9, "gtol": 1e-6}
    coarse = minimize(objective, x0, method="Nelder-Mead", bounds=box,
                      options={**nm, "adaptive": True})
    fine = minimize(objective, coarse.x, method="L-BFGS-B", bounds=box,
                    options=lb)
    return fine if fine.fun <= coarse.fun else coarse


def _objective_for(model: PopModel, n_nodes: int):
    def objective(x):
        try:
            value = -2.0 * marginal_loglik(model, np.asarray(x, dtype=float),
                                           n_nodes)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return 1e12
        return value if np.isfinite(value) else 1e12

    return objective


def fit_reduced(model: PopModel, n_nodes: int, fixed: dict[int, float],
                warm_start: np.ndarray | None = None) -> tuple[float, int]:
    """Refit with some parameters held, and return the objective and the df.

    Used for the nested comparisons: holding a parameter at the value that
    switches a feature off, refitting everything else, and reading how much
    worse the fit gets is what says whether that feature was paying for
    itself. Only the objective is needed, so no standard errors are
    computed and the run is cheap.

    The search is run twice, from the model's usual displaced start and
    from the full model's own solution, and the better of the two is kept.
    A deliberately bad model has an awkward likelihood surface -- a
    one-compartment model fitted to two-compartment data especially -- and
    from a single start the optimiser can settle in a local optimum that
    differs between machines. Reporting whichever start found the lower
    objective makes the comparison a lower bound on the cost of dropping
    the feature, and makes it reproduce.
    """
    full_objective = _objective_for(model, n_nodes)
    base = np.asarray(model.start(), dtype=float)
    free = [i for i in range(len(base)) if i not in fixed]
    box = model.bounds()

    def expand(xf: np.ndarray) -> np.ndarray:
        x = base.copy()
        for i, value in fixed.items():
            x[i] = value
        x[free] = xf
        return x

    starts = [base[free]]
    if warm_start is not None:
        warm = np.clip(np.asarray(warm_start, dtype=float)[free],
                       [box[i][0] for i in free], [box[i][1] for i in free])
        starts.append(warm)

    best = min(
        float(full_objective(expand(np.asarray(
            _optimise(lambda xf: full_objective(expand(xf)), start,
                      [box[i] for i in free], thorough=False).x))))
         for start in starts)
    return best, len(fixed)


def fit_model(key: str, data: pd.DataFrame, truth: dict[str, float],
              nodes: int | None = None, seed: int = 20240,
              verbose: bool = True) -> FitResult:
    """Estimate one model and produce everything the dashboard shows for it."""
    started = time.perf_counter()
    model = MODELS[key](data, truth)
    n_nodes = NODES[key] if nodes is None else nodes
    objective = _objective_for(model, n_nodes)

    best = _optimise(objective, model.start(), model.bounds())
    x = np.asarray(best.x, dtype=float)
    ofv = float(objective(x))

    # Standard errors: the objective is -2 log L, so its Hessian is twice the
    # observed information and the covariance of the estimates is 2 H^-1.
    natural = model.to_natural(x)
    try:
        hess = _numerical_hessian(objective, x)
        cov_est = 2.0 * np.linalg.inv(hess)
        jac = _jacobian(model.to_natural, x)
        cov_nat = jac @ cov_est @ jac.T
        se = np.sqrt(np.clip(np.diag(cov_nat), 0, None))
    except np.linalg.LinAlgError:
        se = np.full_like(natural, np.nan)

    rows = []
    for i, rep in enumerate(model.report()):
        true_value = truth.get(rep.truth_key) if rep.truth_key else None
        value = float(natural[i])
        pct = (None if true_value in (None, 0)
               else (value - float(true_value)) / float(true_value) * 100.0)
        rows.append({
            "parameter": rep.label,
            "unit": rep.unit,
            "estimate": value,
            "se": float(se[i]) if np.isfinite(se[i]) else None,
            "rse_pct": (float(se[i] / abs(value) * 100.0)
                        if np.isfinite(se[i]) and value != 0 else None),
            "truth": None if true_value is None else float(true_value),
            "pct_diff": pct,
        })
    estimates = pd.DataFrame(rows)

    gof = _gof_table(model, x)
    vpc = _vpc_table(model, x, seed)
    etas, shrinkage = _etas_and_shrinkage(model, x, gof)
    seconds = round(time.perf_counter() - started, 1)

    if verbose:
        worst = estimates["pct_diff"].abs().max()
        print(f"  {key}: OFV {ofv:.2f}, {seconds}s, "
              f"worst recovery {worst:.1f}%" if pd.notna(worst)
              else f"  {key}: OFV {ofv:.2f}, {seconds}s")

    return FitResult(
        key=key, objective=ofv, seconds=seconds, nodes=n_nodes,
        converged=bool(best.success), message=str(best.message),
        estimates=estimates, x=x, gof=gof, vpc=vpc, etas=etas,
        extra={"subjects": int(model.n_subj), "eta": model.n_eta,
               "shrinkage": shrinkage},
    )


def _etas_and_shrinkage(model: PopModel, x: np.ndarray,
                        gof: pd.DataFrame | None):
    """Each subject's estimated random effects, and how much they shrank.

    Shrinkage is the number that says whether the individual-level
    diagnostics on this page mean anything. An empirical Bayes estimate is a
    compromise between what a subject's own data says and what the
    population says, and when a subject carries little information the
    compromise lands near the population value: the estimated etas end up
    with less spread than the OMEGA they were drawn from. At high shrinkage
    the "observed against individual prediction" panel looks excellent for
    the wrong reason -- the individual predictions have quietly become
    population predictions -- and etas can no longer be trusted for spotting
    covariate relationships.

        eta shrinkage = 1 - SD(estimated etas) / OMEGA
        epsilon shrinkage = 1 - SD(IWRES)
    """
    if model.n_eta == 0:
        return None, {}

    eta_hat, _ = _mode_and_curvature(
        model, x, model._warm if model._warm is not None
        else np.zeros((model.n_subj, model.n_eta)))
    omega = model.omega_sd(x)
    eta_sh = [float(1.0 - np.std(eta_hat[:, j], ddof=1) / omega[j])
              for j in range(model.n_eta)]

    labels = model.eta_labels or tuple(
        f"ETA({j + 1})" for j in range(model.n_eta))
    frame = pd.DataFrame(eta_hat, columns=list(labels))
    frame.insert(0, "ID", model.ids)

    shrinkage = {"eta": dict(zip(labels, [round(v, 4) for v in eta_sh],
                                 strict=True))}
    if gof is not None and "IWRES" in gof:
        iwres = pd.to_numeric(gof["IWRES"], errors="coerce").dropna()
        shrinkage["epsilon"] = round(float(1.0 - iwres.std(ddof=1)), 4)
    return frame, shrinkage


def _gof_table(model: PopModel, x: np.ndarray) -> pd.DataFrame | None:
    """Predictions and residuals, including proper FOCE-linearised CWRES.

    CWRES is the residual worth plotting: it is standardised by the variance
    the model actually implies for that subject, random effects included, so
    a structural problem shows as a trend rather than being absorbed by the
    individual fit the way IWRES absorbs it.
    """
    zero = np.zeros((model.n_subj, model.n_eta))
    if model.n_eta == 0 or model.predict(x, zero) is None:
        return None

    eta0 = zero
    eta_hat, _ = _mode_and_curvature(model, x, model._warm
                                     if model._warm is not None else eta0)

    ipred = model.predict(x, eta_hat)
    pred = model.predict(x, eta0)
    sd_i = model.residual_sd(x, ipred)
    iwres = (model.y - ipred) / sd_i

    # G = d f / d eta at the mode, by central differences.
    h = 1e-5
    cols = []
    for j in range(model.n_eta):
        e = np.zeros((model.n_subj, model.n_eta))
        e[:, j] = h
        cols.append((model.predict(x, eta_hat + e)
                     - model.predict(x, eta_hat - e)) / (2 * h))
    g = np.stack(cols, axis=-1)                      # (subj, obs, eta)

    omega = np.diag(model.omega_sd(x) ** 2)
    sd_pop = model.residual_sd(x, pred)
    cwres = np.empty_like(iwres)
    for i in range(model.n_subj):
        gi = g[i]
        cov = gi @ omega @ gi.T + np.diag(sd_pop[i] ** 2)
        # The FOCE-linearised population prediction for this subject.
        expect = ipred[i] - gi @ eta_hat[i]
        try:
            chol = np.linalg.cholesky(cov + np.eye(cov.shape[0]) * 1e-12)
            cwres[i] = np.linalg.solve(chol, model.y[i] - expect)
        except np.linalg.LinAlgError:
            cwres[i] = np.nan

    n_obs = model.y.shape[1]
    return pd.DataFrame({
        "ID": np.repeat(model.ids, n_obs),
        "TIME": np.tile(model.times, model.n_subj),
        "DV": model.y.ravel(),
        "PRED": pred.ravel(),
        "IPRED": ipred.ravel(),
        "CWRES": cwres.ravel(),
        "IWRES": iwres.ravel(),
    })


def _vpc_table(model: PopModel, x: np.ndarray, seed: int,
               n_rep: int = 500) -> pd.DataFrame | None:
    """Replicates simulated from the estimates, for the visual predictive check.

    Simulating from the *estimates* rather than from the truth is the point:
    the check asks whether the fitted model reproduces the data it was fitted
    to, which is a different question from whether the estimates are right.
    """
    rng = np.random.default_rng(seed)
    probe = model.simulate(x, rng)
    if probe is None:
        return None
    reps = [probe] + [model.simulate(x, rng) for _ in range(n_rep - 1)]
    stacked = np.stack(reps)                          # (rep, subj, obs)
    n_rep_actual, n_subj, n_obs = stacked.shape
    return pd.DataFrame({
        "rep": np.repeat(np.arange(n_rep_actual), n_subj * n_obs),
        "TIME": np.tile(model.times, n_rep_actual * n_subj),
        "DV": stacked.ravel(),
    })


def run_comparison(key: str, data: pd.DataFrame, truth: dict[str, float],
                   full_ofv: float, full_x: np.ndarray | None = None,
                   verbose: bool = True) -> dict | None:
    """Refit the model with its distinguishing feature switched off.

    Two shapes of comparison come out of this, and they are not the same
    test. Where the simpler model is the full one with a parameter held at
    a fixed value, the models are nested and the difference in objective
    function is a likelihood ratio statistic with a chi-square reference.
    Where the simpler model is a different structure with the same number
    of parameters -- direct effect against indirect response -- nothing is
    nested and there is no p-value to quote; the objective functions are
    simply comparable, and the lower one describes the data better.
    """
    spec = COMPARISONS.get(key)
    if spec is None:
        return None

    started = time.perf_counter()
    n_nodes = NODES[key]
    model = spec.get("alternative", MODELS[key])(data, truth)
    fixed = spec.get("fix")
    if fixed:
        reduced_ofv, n_fixed = fit_reduced(model, n_nodes, fixed, full_x)
    else:
        objective = _objective_for(model, n_nodes)
        box = model.bounds()
        starts = [model.start()]
        if full_x is not None:
            starts.append(np.clip(full_x, [b[0] for b in box],
                                  [b[1] for b in box]))
        reduced_ofv = min(float(_optimise(objective, st, box,
                                          thorough=False).fun)
                          for st in starts)
        n_fixed = 0
    df = int(spec["df"]) if "df" in spec else n_fixed

    delta = reduced_ofv - full_ofv
    p_value = float(chi2.sf(delta, df)) if df > 0 and delta > 0 else None
    out = {
        "feature": spec["feature"],
        "against": spec["against"],
        "question": spec["question"],
        "full_ofv": round(full_ofv, 2),
        "reduced_ofv": round(reduced_ofv, 2),
        "delta_ofv": round(delta, 2),
        "df": df,
        "p_value": p_value,
        "boundary": bool(spec.get("boundary", False)),
        "nested": df > 0,
        "seconds": round(time.perf_counter() - started, 1),
    }
    if verbose:
        tail = (f"p = {p_value:.2g}" if p_value is not None
                else "not nested, compare objective functions directly")
        print(f"    without {spec['feature'].lower()}: "
              f"dOFV {delta:+.1f} on {df} df, {tail}")
    return out


def fit_all(root: Path, keys: list[str] | None = None,
            verbose: bool = True) -> dict[str, FitResult]:
    """Fit every model and write the artefacts the dashboard reads."""
    out_dir = root / "fit" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, FitResult] = {}

    for key in (keys or list(MODELS)):
        data = pd.read_csv(root / "data" / f"{key}.csv")
        truth = json.loads(
            (root / "data" / f"{key}.truth.json").read_text(encoding="utf-8")
        )["parameters"]
        result = fit_model(key, data, truth, verbose=verbose)
        results[key] = result

        result.estimates.to_csv(out_dir / f"{key}_estimates.csv", index=False)
        if result.gof is not None:
            result.gof.to_csv(out_dir / f"{key}_gof.csv", index=False)
        if result.vpc is not None:
            result.vpc.to_csv(out_dir / f"{key}_vpc.csv", index=False)
        if result.etas is not None:
            result.etas.to_csv(out_dir / f"{key}_etas.csv", index=False)

        comparison = run_comparison(key, data, truth, result.objective,
                                    full_x=result.x, verbose=verbose)
        (out_dir / f"{key}_status.json").write_text(json.dumps({
            "model": key,
            "objective": result.objective,
            "seconds": result.seconds,
            "nodes": result.nodes,
            "converged": result.converged,
            "message": result.message,
            "method": _method_name(result.nodes, result.extra["eta"]),
            "engine": "nmlib.estimate",
            "comparison": comparison,
            **result.extra,
        }, indent=2), encoding="utf-8")

    return results


if __name__ == "__main__":
    fit_all(Path("."))
