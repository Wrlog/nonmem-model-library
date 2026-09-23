"""Generate the datasets, run the checks, and build the dashboard.

    python -m nmlib.build            # data + checks + site/index.html
    python -m nmlib.build --check    # checks only, non-zero exit on failure
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
from pathlib import Path

from . import figures
from .check import check_all
from .simulate import SIMULATORS
from .theme import DARK, LIGHT

CATALOGUE = [
    {
        "key": "pk_2cmt_iv",
        "title": "Two-compartment IV population PK",
        "family": "Pharmacokinetics",
        "accent": "blue",
        "why": "The structural PK model most exposure work rests on. "
               "Allometric weight on clearance and volume, between-subject "
               "variability on CL and V1, proportional residual error.",
        "reads": "ADVAN3 TRANS4 — a closed-form two-compartment model, so no "
                 "differential equations are solved and the run is fast.",
    },
    {
        "key": "tgi_claret",
        "title": "Tumour growth inhibition (Claret)",
        "family": "Oncology",
        "accent": "orange",
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
        "why": "Dayneka and Jusko's model I. The drug acts on the turnover of "
               "a biomarker rather than on the biomarker itself, so response "
               "lags exposure and washout is set by the biomarker's own loss "
               "rate, not by the drug's half-life.",
        "reads": "Baseline is the untreated steady state KIN/KOUT, set through "
                 "A_0(2) rather than estimated as a separate parameter.",
    },
    {
        "key": "tte_weibull",
        "title": "Time to event, Weibull hazard",
        "family": "Survival",
        "accent": "violet",
        "why": "Exposure on the hazard of a first event, with a Weibull "
               "baseline so the hazard can rise or fall with time. The shape "
               "of the survival curve is what most dropout and safety "
               "analyses turn on.",
        "reads": "Fitted on the likelihood, not a residual: a censored record "
                 "contributes the survivor function, an event record the "
                 "survivor times the hazard. LAPLACE LIKELIHOOD is required.",
    },
    {
        "key": "logistic_binary",
        "title": "Binary response, logistic on exposure",
        "family": "Exposure-response",
        "accent": "amber",
        "why": "The shape most exposure-safety and exposure-response analyses "
               "take when the endpoint is yes or no, with between-subject "
               "variability on the logit.",
        "reads": "$PRED rather than a compartment model, and again a "
                 "likelihood: Y is the probability of the outcome that was "
                 "actually observed.",
    },
]

CSS = """
:root{--surface:#fff;--sunken:#f7f8fa;--rule:#e4e6ea;--ink:#16181d;
 --ink-2:#454951;--ink-3:#6b7078;--blue:#2a78d6;--blue-ink:#1d5fae;
 --blue-bg:#edf3fc;--orange-ink:#b44a16;--orange-bg:#fdf1ec;
 --green-ink:#0d7a54;--green-bg:#e8f7f1;--violet-ink:#4a3aa7;
 --violet-bg:#efedf9;--amber-ink:#8a6200;--amber-bg:#fdf5e4;--red:#d1453b;
 color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --surface:#1a1a19;--sunken:#232322;--rule:#3a3a37;--ink:#fff;
 --ink-2:#c3c2b7;--ink-3:#95948b;--blue:#3987e5;--blue-ink:#7fb2f0;
 --blue-bg:#1f2a38;--orange-ink:#e08a5c;--orange-bg:#33241c;
 --green-ink:#5cc79b;--green-bg:#173026;--violet-ink:#9085e9;
 --violet-bg:#252138;--amber-ink:#d2a441;--amber-bg:#332a16;
 color-scheme:dark}}
:root[data-theme="dark"]{--surface:#1a1a19;--sunken:#232322;--rule:#3a3a37;
 --ink:#fff;--ink-2:#c3c2b7;--ink-3:#95948b;--blue:#3987e5;--blue-ink:#7fb2f0;
 --blue-bg:#1f2a38;--orange-ink:#e08a5c;--orange-bg:#33241c;
 --green-ink:#5cc79b;--green-bg:#173026;--violet-ink:#9085e9;
 --violet-bg:#252138;--amber-ink:#d2a441;--amber-bg:#332a16;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink-2);
 font:15px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1020px;margin:0 auto;padding:36px 16px 80px}
h1{font-size:1.75rem;margin:0 0 8px;color:var(--ink);letter-spacing:-.02em}
.lede{margin:0;max-width:66ch}
header{border-bottom:1px solid var(--rule);padding-bottom:22px}
.note{background:var(--sunken);border:1px solid var(--rule);
 border-left:3px solid var(--red);border-radius:8px;padding:12px 16px;
 margin:20px 0;font-size:.92rem}
.note strong{color:var(--ink)}
.controls{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:8px;
 align-items:center;background:var(--surface);border-bottom:1px solid var(--rule);
 padding:12px 0;margin-bottom:6px}
.tab{font:inherit;font-size:.9rem;cursor:pointer;color:var(--ink-2);
 background:var(--sunken);border:1px solid var(--rule);border-radius:7px;
 padding:7px 13px}
.tab[aria-pressed="true"]{background:var(--blue-bg);color:var(--blue-ink);
 border-color:var(--blue);font-weight:600}
.tab:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
.spacer{flex:1 1 auto}
section.model{padding-top:32px}
section.model[hidden]{display:none}
.family{font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;
 font-weight:700;margin:0 0 6px}
h2{font-size:1.18rem;margin:0 0 10px;color:var(--ink);letter-spacing:-.01em}
.why{margin:0 0 14px}
.reads{background:var(--sunken);border-radius:8px;padding:12px 15px;
 font-size:.92rem;margin:0 0 16px}
figure{margin:16px 0 6px}
figure img{width:100%;height:auto;display:block;border-radius:8px}
figcaption{font-size:.86rem;color:var(--ink-3);margin-top:8px}
img.dark-only{display:none}
@media (prefers-color-scheme:dark){
 :root:not([data-theme="light"]) img.light-only{display:none}
 :root:not([data-theme="light"]) img.dark-only{display:block}}
:root[data-theme="dark"] img.light-only{display:none}
:root[data-theme="dark"] img.dark-only{display:block}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:14px 0}
th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink-3);font-size:.74rem;text-transform:uppercase;letter-spacing:.04em;font-weight:600}
td{color:var(--ink-2);font-variant-numeric:tabular-nums}
pre{background:var(--sunken);border:1px solid var(--rule);border-radius:8px;
 padding:14px 16px;overflow-x:auto;font-size:12.5px;line-height:1.55;color:var(--ink-2)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
details{margin:14px 0}
summary{cursor:pointer;color:var(--blue-ink);font-size:.92rem;font-weight:500}
.pass{color:var(--green-ink);font-weight:600}
.warn{color:var(--amber-ink);font-weight:600}
.fail{color:var(--red);font-weight:600}
footer{margin-top:52px;padding-top:18px;border-top:1px solid var(--rule);
 font-size:.85rem;color:var(--ink-3)}
@media (max-width:640px){.wrap{padding:20px 16px 60px}h1{font-size:1.4rem}}
@media print{.controls{display:none}section.model[hidden]{display:revert !important}}
"""

SCRIPT = """
(function(){
 var tabs=[].slice.call(document.querySelectorAll('.tab[data-target]'));
 var secs=[].slice.call(document.querySelectorAll('section.model'));
 function show(k){secs.forEach(function(s){s.hidden=(k!=='all'&&s.dataset.key!==k);});
  tabs.forEach(function(t){t.setAttribute('aria-pressed',String(t.dataset.target===k));});}
 tabs.forEach(function(t){t.addEventListener('click',function(){show(t.dataset.target);});});
 var th=document.getElementById('theme');
 function cur(){var s=document.documentElement.getAttribute('data-theme');
  var dark=window.matchMedia('(prefers-color-scheme: dark)').matches;
  return s?s:(dark?'dark':'light');}
 function paint(){var n=cur();th.textContent=n==='dark'?'Light mode':'Dark mode';
  th.setAttribute('aria-pressed',n==='dark'?'true':'false');}
 th.addEventListener('click',function(){
  document.documentElement.setAttribute('data-theme',cur()==='dark'?'light':'dark');paint();});
 paint();show('all');
})();
"""


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def generate_data(root: Path) -> dict[str, dict]:
    """Write every dataset and its true parameters."""
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


def build_site(root: Path, out: Path, summary: dict) -> Path:
    import pandas as pd

    checks = {c.model: c for c in check_all(root / "models", root / "data")}
    parts = []
    generated = dt.datetime.now().strftime("%d %B %Y")

    parts.append("""<header>
  <h1>NONMEM model library</h1>
  <p class="lede">Control streams for the model types that come up repeatedly
  in drug development — population PK, tumour growth inhibition, indirect
  response, time to event and a binary exposure-response — each with a
  simulated dataset to run against and the parameters it was generated
  from.</p>
</header>""")

    tabs = ['<button class="tab" data-target="all" aria-pressed="true">All</button>']
    for m in CATALOGUE:
        tabs.append(f'<button class="tab" data-target="{m["key"]}" '
                    f'aria-pressed="false">{html.escape(m["family"])}</button>')
    parts.append('<div class="controls">' + "".join(tabs)
                 + '<div class="spacer"></div>'
                 + '<button class="tab" id="theme" aria-pressed="false">'
                 + 'Dark mode</button></div>')

    n_fail = sum(1 for c in checks.values() if not c.ok)
    parts.append(f"""<div class="note">
  <strong>These control streams have not been executed.</strong> NONMEM is
  licensed software and was not available where this was built, so every
  figure below is the <em>simulation</em>, not an estimation result. What is
  checked instead is everything that can be checked without running it:
  $INPUT against the data columns in order, parameter references against
  declarations, initial estimates against their bounds, compartments against
  $MODEL, and likelihood models against their $ESTIMATION record.
  {len(checks) - n_fail} of {len(checks)} pass.
</div>""")

    for m in CATALOGUE:
        key = m["key"]
        info = summary[key]
        csv = root / "data" / f"{key}.csv"
        df = pd.read_csv(csv)
        fig_fn = figures.FIGURES[key]
        if key == "logistic_binary":
            light = fig_fn(df, LIGHT, info["truth"])
            dark = fig_fn(df, DARK, info["truth"])
        else:
            light, dark = fig_fn(df, LIGHT), fig_fn(df, DARK)

        truth_rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v:g}</td></tr>"
            for k, v in info["truth"].items())
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

        body = [
            f'<p class="family" style="color:var(--{m["accent"]}-ink)">'
            f'{html.escape(m["family"])}</p>',
            f'<h2>{html.escape(m["title"])}</h2>',
            f'<p class="why">{html.escape(m["why"])}</p>',
            f'<div class="reads">{html.escape(m["reads"])}</div>',
            "\n".join([
                "<figure>",
                f'  <img class="light-only" alt="{html.escape(m["title"])}" '
                f'src="data:image/png;base64,{b64(light)}">',
                f'  <img class="dark-only" alt="{html.escape(m["title"])}" '
                f'src="data:image/png;base64,{b64(dark)}">',
                f'  <figcaption>Simulated data: {info["subjects"]} subjects, '
                f'{info["rows"]} records.</figcaption>',
                "</figure>",
            ]),
            "<table><thead><tr><th>Simulated from</th><th>Value</th></tr></thead>"
            f"<tbody>{truth_rows}</tbody></table>",
            f"<p>Control stream: {status}</p>{detail}",
            f"<details><summary>Show {key}.mod</summary>"
            f"<pre><code>{html.escape(mod_text)}</code></pre></details>",
        ]
        parts.append(f'<section class="model" data-key="{key}">'
                     + "".join(body) + "</section>")

    parts.append(f"""<footer>
  Generated {generated} by <code>python -m nmlib.build</code>. Datasets are
  simulated from the parameters shown; figures are drawn with matplotlib from
  those datasets. Source:
  <a href="https://github.com/Wrlog/nonmem-model-library">Wrlog/nonmem-model-library</a>.
</footer>""")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NONMEM model library</title>
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("site/index.html"))
    ap.add_argument("--check", action="store_true",
                    help="run the static checks only")
    args = ap.parse_args()

    if not args.check:
        print("generating datasets")
        summary = generate_data(args.root)
    else:
        summary = {}

    print("\nchecking control streams")
    failed = 0
    for c in check_all(args.root / "models", args.root / "data"):
        mark = "ok  " if c.ok else "FAIL"
        print(f"  {mark} {c.model}")
        for e in c.errors:
            print(f"       error: {e}")
            failed += 1
        for w in c.warnings:
            print(f"       note:  {w}")
    if failed:
        print(f"\n{failed} error(s)")
        return 1
    if args.check:
        print("\nall control streams pass")
        return 0

    path = build_site(args.root, args.out, summary)
    print(f"\nwrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
