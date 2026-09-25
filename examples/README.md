# Examples

Control streams for techniques that come up in real analyses but don't fit
the `models/` pattern of one simulator and one estimator per model. They're
templates: no dataset ships with them, and the `$INPUT` in each one lists the
columns it expects. Point `$DATA` at your own file.

| Example | What it shows |
| --- | --- |
| [`pk_1cmt_oral_m3_blq.mod`](pk_1cmt_oral_m3_blq.mod) | Oral one-compartment model with a lag time and combined error, and the M3 method for records below the limit of quantification: they contribute `P(C < LLOQ)` instead of being dropped |
| [`pk_transit_absorption.mod`](pk_transit_absorption.mod) | Transit-compartment absorption (Savic 2007) on a two-compartment model, with a non-integer number of compartments via Stirling's approximation. Sets `F1 = 0` so the dose is not absorbed twice |
| [`pk_2cmt_iv_covariates_ltbs.mod`](pk_2cmt_iv_covariates_ltbs.mod) | A monoclonal-antibody covariate model: power, threshold and categorical relations, missing values coded `-99`, in PsN scm's layout. Log-transformed both sides error and Xpose-style tables |
| [`pk_2cmt_iv_iov.mod`](pk_2cmt_iv_iov.mod) | Inter-occasion variability on CL, one ETA per occasion sharing a variance through `$OMEGA BLOCK(1) SAME` |
| [`pkpd_idr_sequential_ipp.mod`](pkpd_idr_sequential_ipp.mod) | Sequential PK/PD by the IPP approach: individual PK parameters read from the data, an indirect response model on a biomarker, and a covariate on its baseline |
| [`pk_external_validation.mod`](pk_external_validation.mod) | External validation of a published model: every parameter fixed, `MAXEVAL=0` for post hoc ETAs, CWRES and NPDE on new data, and `EVID=2` records for dense individual profiles |

Before being added here, each one was run in NONMEM 7.5 on data simulated
from known parameter values, and it recovered them. CI doesn't repeat that
because it needs a NONMEM licence. What CI does run on every example is the
static checker (`tests/test_library.py`): parameter references, bounds and
compartments, with only the missing dataset allowed.
