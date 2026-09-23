# NONMEM model library

Control streams for the model types that come up repeatedly in drug
development. Each one is fitted here, diagnosed, and tested against the
simpler model it would have to beat to be worth writing.

**[View the dashboard](https://wrlog.github.io/nonmem-model-library/)**

```bash
pip install -e ".[dev]"
python -m nmlib.build          # simulate, fit, check, and build the page
python -m nmlib.build --check  # static checks only
python -m nmlib.build --no-fit # rebuild the page from existing fit results
pytest
```

One command does everything, in one language, with no licence: the datasets
are simulated, all five models are estimated, the control streams are
checked, and every figure is drawn from those results. The whole library
fits in about three minutes on one core.

## What is fitted

All five models are estimated back from their own simulated data by
[`src/nmlib/estimate.py`](src/nmlib/estimate.py), which implements
**adaptive Gauss-Hermite quadrature**: the random effects are integrated out
on a grid centred on each subject's posterior mode and scaled by the
curvature there. With a single node that reduces exactly to the Laplace
approximation, which is what NONMEM's `LAPLACE` does; with more nodes it
converges on the true integral, which is why the binary model — where
Laplace is known to be biased — is run with fifteen.

Starting values are deliberately displaced from the values used to simulate,
so "the estimates recover the truth" means the optimiser found it rather
than started on it.

Each model contributes the following to the dashboard, and they answer
different questions in a deliberate order:

- **Does the structure earn its place?** Recovery says estimation found the
  parameters; it does not say they were worth having. Every model here
  exists because of one feature that separates it from an obvious simpler
  alternative — a second compartment, the resistance term, acting on
  turnover rather than directly, the Weibull shape, the between-subject
  variance. Each is switched off, everything else re-estimated, and the
  increase in objective function is what that feature was buying. Where the
  simpler model is nested the difference is a likelihood ratio statistic;
  where it is a different structure with the same parameter count (direct
  effect against indirect response) there is no p-value to quote and the
  page says so. This is the question the page leads with, because it is the
  one that decides whether a model should exist.
- **Estimates**, each with a confidence interval, an %RSE, and — since the
  data is simulated — the value it was generated from beside it.
- **Shrinkage**, stated before the plots it qualifies. An empirical Bayes
  estimate is a compromise between a subject's own data and the population,
  so where a subject carries little information the estimate collapses
  toward the population value. At high shrinkage the individual-prediction
  panel looks excellent for the wrong reason.
- **Goodness of fit** — observations against population and individual
  predictions, and conditional weighted residuals against time and against
  prediction, with a binned median so curvature is visible.
- **Individual fits** for a sample of subjects spanning the range of the
  data, because everything else on the page averages over exactly the thing
  a mixed effects model exists to describe.
- **A visual predictive check** from 500 replicates simulated from the
  fitted model, because goodness-of-fit plots can look tidy for a model that
  predicts the wrong spread.

The estimator is tested against independent methods rather than against
itself: its marginal likelihood is checked against brute-force numerical
integration of the same integral, and its indirect response solver against
an adaptive ODE integrator on the same equations. See
[`tests/test_estimation.py`](tests/test_estimation.py).

## The models

| Model | Family | Why it is in the library |
| --- | --- | --- |
| [`pk_2cmt_iv.mod`](models/pk_2cmt_iv.mod) | Pharmacokinetics | The structural model most exposure work rests on. Allometric weight, IIV on CL and V1, proportional error. `ADVAN3 TRANS4`, closed form, no ODEs to solve |
| [`tgi_claret.mod`](models/tgi_claret.mod) | Oncology | Claret tumour growth inhibition: exponential growth with a kill term that decays. The decay term is the point — without it the model cannot produce regrowth on treatment |
| [`pkpd_idr_inhibition.mod`](models/pkpd_idr_inhibition.mod) | PK/PD | Dayneka and Jusko model I. The drug acts on turnover, so response lags exposure and washout is set by `KOUT`, not by the drug's half-life |
| [`tte_weibull.mod`](models/tte_weibull.mod) | Survival | Time to first event with a Weibull baseline hazard and exposure on the hazard. Fitted on the likelihood: censored records contribute the survivor function, events contribute survivor × hazard |
| [`logistic_binary.mod`](models/logistic_binary.mod) | Exposure–response | Binary endpoint with IIV on the logit. `$PRED`, and again a likelihood rather than a residual |

Each `.mod` carries comments explaining the structure and why it is written
that way, not just what each line does.

## Examples

[`examples/`](examples/) holds control streams for techniques that do not
need a simulator of their own to be useful: M3 for BLQ data, transit
absorption, a covariate model with log-transformed both sides, inter-occasion
variability, sequential PK/PD, and external validation of a published model.
They are templates without datasets. The static checks run on them in CI, and
each was run in NONMEM against simulated data before it was added. See
[`examples/README.md`](examples/README.md).

## The data is simulated

Every dataset in `data/` is generated by
[`src/nmlib/simulate.py`](src/nmlib/simulate.py) from parameters written out
beside it in `<model>.truth.json`. Nothing here is patient data, and none is
needed.

Simulating rather than shipping a real dataset has a practical payoff: the
true parameter values are known, so every estimate can be read beside the
value it was trying to find.

It also means the *design* is part of what the library has to get right, and
in two places that is the whole lesson:

- The **binary endpoint is scored at six visits per subject**. With one
  record per subject, the between-subject variance on the logit is not
  identifiable at all — a single Bernoulli draw cannot separate a subject
  who is prone to respond from one who happened to respond, and `OMEGA`
  collapses to zero however it is estimated.
- The **indirect response model is dosed daily for a week and then
  followed for three**, with a biomarker whose half-life is about five times
  the drug's. Without that separation between the two clocks, a direct
  effect model fits the same data just as well and there is nothing for the
  indirect structure to earn.

The simulators are tested against the equations they claim to implement —
the closed-form PK against numerical integration of the same ODEs, the
indirect response baseline against its steady state `KIN/KOUT`, and the
event times against the Weibull survivor function at three quantiles.

## Static checks

`python -m nmlib.build --check` runs these against every stream:

- **`$INPUT` against the data columns, in order.** With `IGNORE=@` NONMEM
  skips the header and reads positionally, so a `$INPUT` that is merely a
  permutation of the header loads the wrong column into the wrong variable
  and still runs. This is the single most valuable check here.
- **Parameter references.** Every `THETA(n)`, `ETA(n)` and `EPS(n)` used must
  be declared, and anything declared but never referenced is flagged.
- **Initial estimates inside their own bounds.**
- **Compartments** used in `$DES` and `A_0` must exist in `$MODEL`, and every
  declared compartment must have a `DADT`.
- **Likelihood models.** A stream that defines `Y` branch-wise on `DV` is
  being fitted on the likelihood, so `$ESTIMATION` must say `LIKELIHOOD`;
  the check also notes a stray `$SIGMA` on such a model.

The checker is itself tested by breaking a stream on purpose — a reordered
`$INPUT`, an undeclared `ETA`, an out-of-bounds initial estimate, a missing
`LIKELIHOOD` — and asserting it complains. A linter nobody has seen fail is
not known to work.

## Layout

| Path | Contents |
| --- | --- |
| `models/` | The control streams |
| `examples/` | Template control streams for further techniques, without data |
| `data/` | Simulated datasets and their true parameters, regenerated by the build |
| `src/nmlib/simulate.py` | One simulator per model |
| `src/nmlib/estimate.py` | Population estimation by adaptive Gauss-Hermite quadrature |
| `src/nmlib/check.py` | The static checks |
| `src/nmlib/theme.py` | The palette, type scale and spacing every figure uses |
| `src/nmlib/figures.py` | One figure per model, drawn from the datasets |
| `src/nmlib/diagnostics.py` | Goodness of fit, predictive checks, parameter recovery |
| `src/nmlib/build.py` | Runs all of the above and renders the dashboard |
| `tests/` | Tests for the simulators, the estimator and the checker |

## License

MIT — see [LICENSE](LICENSE).
