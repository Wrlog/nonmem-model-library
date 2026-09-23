"""Generate the datasets, fit the models, run the checks, build the dashboard.

    python -m nmlib.build            # data + fits + checks + site/index.html
    python -m nmlib.build --check    # static checks only, non-zero on failure
    python -m nmlib.build --no-fit   # skip estimation and reuse fit/results/

Everything the page shows is produced by this one command: the datasets are
simulated here, the models are estimated here, and every figure is drawn
here from those results. There is no step that has to be run somewhere else
and no artefact checked in that the build cannot reproduce.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
from pathlib import Path

import pandas as pd

from . import diagnostics, figures
from .check import check_all
from .estimate import fit_all
from .simulate import SIMULATORS
from .theme import DARK, LIGHT

CATALOGUE = [
    {
        "key": "pk_2cmt_iv",
        "title": "Two-compartment IV population PK",
        "family": "Pharmacokinetics",
        "accent": "blue",
        "log": True,
        "y_label": "Concentration (mg/L)",
        "x_label": "Time (h)",
        "why": "The structural PK model most exposure work rests on. "
               "Allometric weight on clearance and volume, between-subject "
               "variability on CL and V1, proportional residual error.",
        "reads": "ADVAN3 TRANS4 - a closed-form two-compartment model, so no "
                 "differential equations are solved and the run is fast.",
    },
    {
        "key": "tgi_claret",
        "title": "Tumour growth inhibition (Claret)",
        "family": "Oncology",
        "accent": "orange",
        "y_label": "Tumour size (mm)",
        "x_label": "Time (days)",
        "why": "Exponential growth with a drug kill term that decays over "
               "time. The decay is the point: without it the model cannot "
               "produce regrowth while treatment continues, which is what is "
               "seen in practice and what drives survival predictions.",
        "reads": "ADVAN13 with $DES. The resistance term LAMBDA is what "
                 "separates this from a plain kill model.",
    },
    {
        "key": "pkpd_idr_inhibition",
        "title": "Indirect response, inhibition of production",
        "family": "PK/PD",
        "accent": "green",
        "y_label": "Biomarker (units)",
        "x_label": "Time (h)",
        "why": "Dayneka and Jusko's model I. The drug acts on the turnover of "
               "a biomarker rather than on the biomarker itself, so response "
               "lags exposure and washout is set by the biomarker's own loss "
               "rate, not by the drug's half-life.",
        "reads": "Baseline is the untreated steady state KIN/KOUT, set through "
                 "A_0(2) rather than estimated separately. The design carries "
                 "as much weight as the structure: the biomarker's half-life "
                 "is about five times the drug's, and without that separation "
                 "a direct-effect model would fit just as well.",
    },
    {
        "key": "tte_weibull",
        "title": "Time to event, Weibull hazard",
        "family": "Survival",
        "accent": "violet",
        "y_label": "Event-free probability",
        "x_label": "Time (days)",
        "why": "Exposure on the hazard of a first event, with a Weibull "
               "baseline so the hazard can rise or fall with time. The shape "
               "of the survival curve is what most dropout and safety "
               "analyses turn on.",
        "reads": "Fitted on the likelihood, not a residual: a censored record "
                 "contributes the survivor function, an event record the "
                 "survivor times the hazard. LAPLACE LIKELIHOOD is required. "
                 "The frailty term is written in but fixed at zero, because "
                 "one event per subject cannot identify it.",
    },
    {
        "key": "logistic_binary",
        "title": "Binary response, logistic on exposure",
        "family": "Exposure-response",
        "accent": "amber",
        "y_label": "Probability of response",
        "x_label": "Exposure",
        "why": "The shape most exposure-safety and exposure-response analyses "
               "take when the endpoint is yes or no, with between-subject "
               "variability on the logit.",
        "reads": "$PRED rather than a compartment model, and again a "
                 "likelihood: Y is the probability of the outcome that was "
                 "actually observed. The endpoint is scored at six visits per "
                 "subject, which is what makes the variance on the logit "
                 "identifiable at all.",
    },
]

CSS = """
:root{
 --surface:#fcfcfb;--plane:#f9f9f7;--sunken:#f4f3ef;--rule:#e1e0d9;
 --ink:#0b0b0b;--ink-2:#52514e;--ink-3:#898781;
 --blue:#2a78d6;--blue-ink:#1c5cab;--blue-bg:#edf3fc;
 --orange-ink:#b44a16;--orange-bg:#fdf1ec;
 --green-ink:#0d7a54;--green-bg:#e8f7f1;
 --violet-ink:#4a3aa7;--violet-bg:#efedf9;
 --amber-ink:#8a6200;--amber-bg:#fdf5e4;
 --good:#0ca30c;--warn:#8a6200;--bad:#d03b3b;
 --shadow:0 1px 2px rgba(11,11,11,.05),0 1px 12px rgba(11,11,11,.04);
 color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --surface:#1a1a19;--plane:#0d0d0d;--sunken:#232322;--rule:#2f2f2d;
 --ink:#fff;--ink-2:#c3c2b7;--ink-3:#898781;
 --blue:#3987e5;--blue-ink:#86b6ef;--blue-bg:#1f2a38;
 --orange-ink:#e08a5c;--orange-bg:#33241c;
 --green-ink:#5cc79b;--green-bg:#173026;
 --violet-ink:#9085e9;--violet-bg:#252138;
 --amber-ink:#d2a441;--amber-bg:#332a16;
 --good:#0ca30c;--warn:#d2a441;--bad:#e66767;
 --shadow:0 1px 2px rgba(0,0,0,.4);
 color-scheme:dark}}
:root[data-theme="dark"]{
 --surface:#1a1a19;--plane:#0d0d0d;--sunken:#232322;--rule:#2f2f2d;
 --ink:#fff;--ink-2:#c3c2b7;--ink-3:#898781;
 --blue:#3987e5;--blue-ink:#86b6ef;--blue-bg:#1f2a38;
 --orange-ink:#e08a5c;--orange-bg:#33241c;
 --green-ink:#5cc79b;--green-bg:#173026;
 --violet-ink:#9085e9;--violet-bg:#252138;
 --amber-ink:#d2a441;--amber-bg:#332a16;
 --good:#0ca30c;--warn:#d2a441;--bad:#e66767;
 --shadow:0 1px 2px rgba(0,0,0,.4);color-scheme:dark}

*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--plane);color:var(--ink-2);
 font:15.5px/1.65 Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 -webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:40px 20px 88px}

h1{font-size:2.05rem;line-height:1.16;margin:0 0 12px;color:var(--ink);
 letter-spacing:-.025em;font-weight:680}
.lede{margin:0;max-width:64ch;font-size:1.06rem;color:var(--ink-2)}
header{padding-bottom:26px}

.controls{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:7px;
 align-items:center;background:var(--plane);border-bottom:1px solid var(--rule);
 padding:11px 0;margin-bottom:8px}
.tab{font:inherit;font-size:.875rem;cursor:pointer;color:var(--ink-2);
 background:var(--surface);border:1px solid var(--rule);border-radius:999px;
 padding:6px 14px;transition:background .12s,border-color .12s}
.tab:hover{border-color:var(--ink-3)}
.tab[aria-pressed="true"]{background:var(--blue-bg);color:var(--blue-ink);
 border-color:var(--blue);font-weight:600}
.tab:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
.spacer{flex:1 1 auto}

.card{background:var(--surface);border:1px solid var(--rule);border-radius:14px;
 padding:26px 28px;margin:20px 0;box-shadow:var(--shadow)}
.card>:first-child{margin-top:0}
.card>:last-child{margin-bottom:0}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
 gap:14px;margin:0 0 22px}
.tile{background:var(--surface);border:1px solid var(--rule);border-radius:12px;
 padding:16px 18px;box-shadow:var(--shadow)}
.tile .value{display:block;font-size:2.05rem;line-height:1.05;color:var(--ink);
 font-weight:640;letter-spacing:-.02em}
.tile .label{display:block;margin-top:6px;font-size:.845rem;color:var(--ink-3)}
.tile .value .of{font-size:1.15rem;color:var(--ink-3);font-weight:500}

section.model{scroll-margin-top:64px}
section.model[hidden]{display:none}
.family{font-size:.7rem;text-transform:uppercase;letter-spacing:.11em;
 font-weight:700;margin:0 0 7px}
h2{font-size:1.42rem;margin:0 0 12px;color:var(--ink);letter-spacing:-.018em;
 font-weight:640;line-height:1.25}
h3{font-size:.76rem;text-transform:uppercase;letter-spacing:.09em;
 color:var(--ink-3);font-weight:700;margin:30px 0 10px}
.why{margin:0 0 16px;max-width:66ch}
.reads{background:var(--sunken);border-radius:10px;padding:14px 17px;
 font-size:.93rem;margin:0 0 18px;max-width:72ch}

figure{margin:14px 0 4px}
figure img{width:100%;height:auto;display:block;border-radius:8px}
figcaption{font-size:.86rem;color:var(--ink-3);margin-top:10px;max-width:72ch}
img.dark-only{display:none}
@media (prefers-color-scheme:dark){
 :root:not([data-theme="light"]) img.light-only{display:none}
 :root:not([data-theme="light"]) img.dark-only{display:block}}
:root[data-theme="dark"] img.light-only{display:none}
:root[data-theme="dark"] img.dark-only{display:block}

table{border-collapse:collapse;width:100%;font-size:.88rem;margin:12px 0}
th,td{text-align:right;padding:8px 11px;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink-3);font-size:.72rem;text-transform:uppercase;
 letter-spacing:.05em;font-weight:700;white-space:nowrap}
td{color:var(--ink-2);font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:none}
.status{font-size:.86rem;color:var(--ink-2);margin:0 0 4px}
.status b{color:var(--ink);font-weight:640}
.unit{color:var(--ink-3);font-size:.9em;font-weight:400}

pre{background:var(--sunken);border:1px solid var(--rule);border-radius:10px;
 padding:15px 17px;overflow-x:auto;font-size:12.5px;line-height:1.55;
 color:var(--ink-2)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
details{margin:18px 0 0}
summary{cursor:pointer;color:var(--blue-ink);font-size:.9rem;font-weight:550}
summary:focus-visible{outline:2px solid var(--blue);outline-offset:3px}

.pass{color:var(--good);font-weight:640}
.warn{color:var(--warn);font-weight:640}
.fail{color:var(--bad);font-weight:640}
.mono{font-variant-numeric:tabular-nums}

footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--rule);
 font-size:.85rem;color:var(--ink-3)}
a{color:var(--blue-ink)}

@media (max-width:640px){
 .wrap{padding:24px 16px 64px}h1{font-size:1.52rem}h2{font-size:1.2rem}
 .card{padding:18px 16px;border-radius:12px}
 .tile .value{font-size:1.7rem}
 table{font-size:.8rem}th,td{padding:7px 7px}}
@media print{.controls{display:none}
 section.model[hidden]{display:revert !important}
 .card{break-inside:avoid;box-shadow:none}}
"""

SCRIPT = """
(function(){
 var tabs=[].slice.call(document.querySelectorAll('.tab[data-target]'));
 var secs=[].slice.call(document.querySelectorAll('section.model'));
 function show(k){
  secs.forEach(function(s){s.hidden=(k!=='all'&&s.dataset.key!==k);});
  tabs.forEach(function(t){
   t.setAttribute('aria-pressed',String(t.dataset.target===k));});}
 tabs.forEach(function(t){
  t.addEventListener('click',function(){show(t.dataset.target);});});
 var th=document.getElementById('theme');
 function cur(){
  var s=document.documentElement.getAttribute('data-theme');
  var dark=window.matchMedia('(prefers-color-scheme: dark)').matches;
  return s?s:(dark?'dark':'light');}
 function paint(){
  var n=cur();th.textContent=n==='dark'?'Light mode':'Dark mode';
  th.setAttribute('aria-pressed',n==='dark'?'true':'false');}
 th.addEventListener('click',function(){
  document.documentElement.setAttribute('data-theme',
   cur()==='dark'?'light':'dark');paint();});
 paint();show('all');
})();
"""


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def generate_data(root: Path) -> dict[str, dict]:
    """Write every dataset and the parameters it was generated from."""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key, fn in SIMULATORS.items():
        sim = fn()
        csv = data_dir / f"{key}.csv"
        sim.data.to_csv(csv, index=False)
        (data_dir / f"{key}.truth.json").write_text(
            json.dumps({"parameters": sim.truth, "notes": sim.notes}, indent=2,
                       default=str),
            encoding="utf-8")
        summary[key] = {
            "rows": len(sim.data),
            "subjects": int(sim.data["ID"].nunique()),
            "truth": sim.truth,
            "notes": sim.notes,
        }
        print(f"  wrote {csv.name}: {len(sim.data)} rows, "
              f"{sim.data['ID'].nunique()} subjects")
    return summary


def _fig_pair(light: bytes, dark: bytes, alt: str, caption: str) -> str:
    """Both themes inlined; CSS shows whichever matches the reader's mode."""
    safe = html.escape(alt)
    return "\n".join([
        "<figure>",
        f'  <img class="light-only" alt="{safe}" loading="lazy" '
        f'src="data:image/png;base64,{b64(light)}">',
        f'  <img class="dark-only" alt="{safe}" loading="lazy" '
        f'src="data:image/png;base64,{b64(dark)}">',
        f"  <figcaption>{caption}</figcaption>",
        "</figure>",
    ])


def _tile(value: str, label: str, of: str = "") -> str:
    suffix = f' <span class="of">{html.escape(of)}</span>' if of else ""
    return (f'<div class="tile"><span class="value">{html.escape(value)}'
            f'{suffix}</span><span class="label">{html.escape(label)}</span>'
            "</div>")


def _num(v, fmt: str = "{:.4g}") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "&mdash;"
    return fmt.format(float(v))


def _estimates_table(est: pd.DataFrame) -> str:
    """Estimate, interval, and the value the data was simulated from.

    The table is the accessible twin of the recovery figure: every number
    the figure encodes as a position is also here as text.
    """
    rows = []
    for _, r in est.iterrows():
        truth = r.get("truth")
        has_truth = truth is not None and pd.notna(truth)
        se = r.get("se")
        if se is not None and pd.notna(se):
            lo, hi = float(r["estimate"]) - 1.96 * float(se), \
                float(r["estimate"]) + 1.96 * float(se)
            lo, hi = min(lo, hi), max(lo, hi)
            ci = f"{_num(lo)} to {_num(hi)}"
            covered = has_truth and lo <= float(truth) <= hi
        else:
            ci, covered = "&mdash;", None

        pct = r.get("pct_diff")
        pct_txt = ("&mdash;" if pct is None or pd.isna(pct)
                   else f"{float(pct):+.0f}%")
        if covered is None:
            verdict = "&mdash;"
        elif covered:
            verdict = '<span class="pass">yes</span>'
        else:
            verdict = '<span class="warn">outside</span>'

        # An empty unit round-trips through the CSV as NaN, which is truthy,
        # so `or ""` is not enough to keep "(nan)" off the page.
        raw_unit = r.get("unit")
        unit = "" if raw_unit is None or pd.isna(raw_unit) else str(raw_unit)
        name = html.escape(str(r["parameter"]))
        if unit:
            name += f' <span class="unit">({html.escape(unit)})</span>'
        rows.append(
            f"<tr><td>{name}</td><td>{_num(r['estimate'])}</td>"
            f"<td>{ci}</td><td>{_num(r.get('rse_pct'), '{:.0f}%')}</td>"
            f"<td>{_num(truth)}</td><td>{pct_txt}</td><td>{verdict}</td></tr>")

    return ("<table><thead><tr><th>Parameter</th><th>Estimate</th>"
            "<th>95% interval</th><th>RSE</th><th>Simulated from</th>"
            "<th>Difference</th><th>Interval covers it</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def _shrinkage_block(status: dict) -> str:
    """Shrinkage, stated before the plots it qualifies.

    It belongs here rather than in a footnote: the panel immediately below
    is "observed against individual prediction", and shrinkage is the
    number that says how much of that agreement is the model fitting each
    subject and how much is each subject's estimate having been pulled back
    to the population.
    """
    shrink = (status or {}).get("shrinkage") or {}
    eta = shrink.get("eta") or {}
    if not eta:
        return ""

    worst_name, worst = max(eta.items(), key=lambda kv: kv[1])
    parts = ", ".join(f"{name} {value * 100:.0f}%"
                      for name, value in eta.items())
    eps = shrink.get("epsilon")
    eps_txt = (f" Residual (epsilon) shrinkage is {eps * 100:.0f}%."
               if eps is not None else "")

    if worst < 0.20:
        reading = ("All low, so the individual predictions below are "
                   "genuinely individual and the etas can be trusted for "
                   "spotting covariate relationships.")
    elif worst < 0.35:
        reading = (f"{html.escape(worst_name)} is the one to watch; the rest "
                   "carry enough subject-level information to be read "
                   "directly.")
    else:
        reading = (
            f"{html.escape(worst_name)} is high enough to matter: those "
            "estimates have been pulled a long way back towards the "
            "population, so the individual-prediction panel below flatters "
            "that parameter, and an eta-versus-covariate plot on it would "
            "be close to meaningless.")

    return (f'<p class="status"><b>Shrinkage</b> &mdash; {parts}.{eps_txt} '
            "Empirical Bayes estimates are a compromise between a subject's "
            "own data and the population, so where a subject carries little "
            "information the estimate collapses toward the population value "
            f"and the spread of the etas understates OMEGA. {reading}</p>")


def _comparison_block(status: dict) -> str:
    """The nested comparison for one model."""
    c = (status or {}).get("comparison")
    if not c:
        return ""
    if c["nested"]:
        p = c.get("p_value")
        verdict = (
            f'dropping it costs <b>{c["delta_ofv"]:.1f}</b> objective function '
            f'on {c["df"]} degree{"s" if c["df"] != 1 else ""} of freedom'
            # &lt; not "<": a bare less-than in HTML text opens a tag, and
            # a lenient browser recovering from it is not the same as the
            # markup being right.
            + (", p &lt; 0.0001" if p is not None and p < 1e-4
               else (f", p = {p:.3g}" if p is not None else "")))
        caveat = (" The null sits on the edge of the parameter space here, "
                  "so the chi-square reference is conservative and the real "
                  "p-value is smaller than the one quoted."
                  if c.get("boundary") else "")
    else:
        better = "better" if c["delta_ofv"] > 0 else "worse"
        verdict = (f'the two have the same number of parameters, and the full '
                   f'model fits <b>{abs(c["delta_ofv"]):.1f}</b> objective '
                   f'function {better}')
        caveat = (" Nothing is nested, so there is no likelihood ratio test "
                  "to run; the objective functions are simply comparable.")

    return (
        "<h3>Does the structure earn its place?</h3>"
        f'<p class="why">{html.escape(c["question"])}</p>'
        f'<p class="status">Refitting as <b>{html.escape(c["against"])}</b> '
        f'and re-estimating everything else: {verdict}.{caveat}</p>')


def _model_section(root: Path, m: dict, info: dict, checks: dict) -> str:
    key = m["key"]
    df = pd.read_csv(root / "data" / f"{key}.csv")
    truth = info["truth"]

    fig_fn = figures.FIGURES[key]
    if key in figures.NEEDS_TRUTH:
        light, dark = fig_fn(df, LIGHT, truth), fig_fn(df, DARK, truth)
    else:
        light, dark = fig_fn(df, LIGHT), fig_fn(df, DARK)

    body = [
        f'<p class="family" style="color:var(--{m["accent"]}-ink)">'
        f'{html.escape(m["family"])}</p>',
        f'<h2>{html.escape(m["title"])}</h2>',
        f'<p class="why">{html.escape(m["why"])}</p>',
        f'<div class="reads">{html.escape(m["reads"])}</div>',
        _fig_pair(light, dark, m["title"],
                  f'Simulated data: {info["subjects"]} subjects, '
                  f'{info["rows"]} records.'),
    ]

    fit = diagnostics.load_fit(root / "fit" / "results", key)
    if fit:
        st = fit.get("status", {})
        est = fit["estimates"]
        rec = diagnostics.recovery_rows(est)
        body.append("<h3>Estimates against the truth</h3>")
        body.append(
            f'<p class="status">Fitted by <b>{html.escape(str(st.get("method", "?")))}'
            f'</b>, objective function <span class="mono">'
            f'{st.get("objective", float("nan")):.1f}</span>, '
            f'{st.get("seconds", "?")} s. '
            f'<b>{int(rec["covers"].sum())} of {len(rec)}</b> parameters have '
            "a 95% confidence interval containing the value the data was "
            "simulated from.</p>")
        body.append(_estimates_table(est))

        if "gof" not in fit:
            body.append(
                '<p class="status">No goodness-of-fit panel or predictive '
                "check here, and that is deliberate rather than missing. "
                "Both are built on residuals, and a likelihood model has "
                "none: the data is a zero or a one, so there is nothing to "
                "subtract a prediction from. The figure above is the check "
                "that applies instead &mdash; observed outcomes grouped by "
                "exposure, against what the model says they should be.</p>")

        body.append(_comparison_block(st))

        if "gof" in fit:
            body.append("<h3>Goodness of fit</h3>")
            body.append(_shrinkage_block(st))
            body.append(_fig_pair(
                diagnostics.gof_panel(fit, LIGHT, bool(m.get("log"))),
                diagnostics.gof_panel(fit, DARK, bool(m.get("log"))),
                "Goodness of fit",
                "The dashed line is unity; the orange line is a binned median "
                "of the residuals, which should sit on zero."))

            ind_l = diagnostics.individual_fits(
                fit, LIGHT, log_scale=bool(m.get("log")),
                y_label=m["y_label"], x_label=m["x_label"])
            ind_d = diagnostics.individual_fits(
                fit, DARK, log_scale=bool(m.get("log")),
                y_label=m["y_label"], x_label=m["x_label"])
            if ind_l and ind_d:
                body.append(_fig_pair(
                    ind_l, ind_d, "Individual fits",
                    "Everything above averages over subjects. This is where a "
                    "model that is wrong in a way the averages hide shows it."))

        vpc_l = diagnostics.vpc_plot(fit, df, LIGHT, log_scale=bool(m.get("log")),
                                     y_label=m["y_label"], x_label=m["x_label"])
        vpc_d = diagnostics.vpc_plot(fit, df, DARK, log_scale=bool(m.get("log")),
                                     y_label=m["y_label"], x_label=m["x_label"])
        if vpc_l and vpc_d:
            # No heading here: the figure titles itself, and a heading
            # above it would only say the same words twice.
            body.append(_fig_pair(
                vpc_l, vpc_d, "Visual predictive check",
                "Goodness of fit can look tidy for a model that predicts the "
                "wrong spread. This is the check that catches it."))

    chk = checks.get(key)
    if chk is None:
        status = '<span class="fail">no control stream found</span>'
        detail = ""
    elif chk.ok and not chk.warnings:
        status = '<span class="pass">passes every static check</span>'
        detail = ""
    elif chk.ok:
        status = f'<span class="warn">passes, {len(chk.warnings)} note(s)</span>'
        detail = "<ul>" + "".join(f"<li>{html.escape(w)}</li>"
                                  for w in chk.warnings) + "</ul>"
    else:
        status = f'<span class="fail">{len(chk.errors)} error(s)</span>'
        detail = "<ul>" + "".join(f"<li>{html.escape(e)}</li>"
                                  for e in chk.errors) + "</ul>"

    mod_text = (root / "models" / f"{key}.mod").read_text(encoding="utf-8")
    body.append("<h3>The control stream</h3>")
    body.append(f'<p class="status">{key}.mod {status}</p>{detail}')
    body.append(f"<details><summary>Show {key}.mod</summary>"
                f"<pre><code>{html.escape(mod_text)}</code></pre></details>")

    return (f'<section class="model card" data-key="{key}">'
            + "".join(body) + "</section>")


def build_site(root: Path, out: Path, summary: dict) -> Path:
    checks = {c.model: c for c in check_all(root / "models", root / "data")}
    results_dir = root / "fit" / "results"
    fits = {m["key"]: diagnostics.load_fit(results_dir, m["key"])
            for m in CATALOGUE}
    fitted = {k: f for k, f in fits.items() if f}

    parts = ["""<header>
  <h1>NONMEM model library</h1>
  <p class="lede">Control streams for the model types that come up repeatedly
  in drug development &mdash; population PK, tumour growth inhibition,
  indirect response, time to event and a binary exposure&ndash;response. Each
  one is fitted here, diagnosed, and tested against the simpler model it
  would have to beat to be worth writing.</p>
</header>"""]

    tabs = ['<button class="tab" data-target="all" aria-pressed="true">'
            "All models</button>"]
    for m in CATALOGUE:
        tabs.append(f'<button class="tab" data-target="{m["key"]}" '
                    f'aria-pressed="false">{html.escape(m["family"])}</button>')
    parts.append('<div class="controls">' + "".join(tabs)
                 + '<div class="spacer"></div>'
                 + '<button class="tab" id="theme" aria-pressed="false">'
                 "Dark mode</button></div>")

    # --- the scoreboard ---------------------------------------------------
    n_pass = sum(1 for c in checks.values() if c.ok)
    seconds = sum(float((f or {}).get("status", {}).get("seconds", 0) or 0)
                  for f in fits.values())

    comparisons = []
    for m in CATALOGUE:
        f = fits[m["key"]]
        c = (f or {}).get("status", {}).get("comparison")
        if c:
            comparisons.append(c)
    earned = sum(1 for c in comparisons
                 if (c["nested"] and c["delta_ofv"]
                     > diagnostics.CHI2_95.get(int(c["df"]), 3.84))
                 or (not c["nested"] and c["delta_ofv"] > 0))

    tiles = [
        _tile(str(len(fitted)), "models estimated from their own data",
              f"of {len(CATALOGUE)}"),
        _tile(str(earned), "models whose distinguishing feature pays for "
                           "itself", f"of {len(comparisons)}"),
        _tile(str(n_pass), "control streams passing every static check",
              f"of {len(checks)}"),
        _tile(f"{seconds / 60:.0f} min", "to fit the whole library, on one core"),
    ]
    parts.append('<div class="tiles">' + "".join(tiles) + "</div>")

    for m in CATALOGUE:
        parts.append(_model_section(root, m, summary[m["key"]], checks))

    generated = dt.datetime.now().strftime("%d %B %Y")
    parts.append(f"""<footer>
  Generated {generated} by <code>python -m nmlib.build</code>: datasets
  simulated, models fitted, control streams checked and every figure drawn by
  that one command. Source:
  <a href="https://github.com/Wrlog/nonmem-model-library">Wrlog/nonmem-model-library</a>.
</footer>""")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NONMEM model library</title>
<meta name="description" content="Control streams for common population
 pharmacometric models, each with a simulated dataset, the parameters behind
 it, and an estimation run that recovers them.">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{"".join(parts)}
</div>
<script>{SCRIPT}</script>
</body>
</html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


def run_checks(root: Path) -> int:
    print("\nchecking control streams")
    failed = 0
    for c in check_all(root / "models", root / "data"):
        print(f"  {'ok  ' if c.ok else 'FAIL'} {c.model}")
        for e in c.errors:
            print(f"       error: {e}")
            failed += 1
        for w in c.warnings:
            print(f"       note:  {w}")
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("site/index.html"))
    ap.add_argument("--check", action="store_true",
                    help="run the static checks only")
    ap.add_argument("--no-fit", action="store_true",
                    help="reuse whatever is already in fit/results/")
    args = ap.parse_args()

    if args.check:
        failed = run_checks(args.root)
        if failed:
            print(f"\n{failed} error(s)")
            return 1
        print("\nall control streams pass")
        return 0

    print("generating datasets")
    summary = generate_data(args.root)

    if args.no_fit:
        print("\nskipping estimation (--no-fit)")
    else:
        print("\nfitting")
        fit_all(args.root)

    if run_checks(args.root):
        return 1

    path = build_site(args.root, args.out, summary)
    print(f"\nwrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
