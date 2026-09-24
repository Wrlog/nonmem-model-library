"""Catalogue a collection of published control streams, without their code.

    python -m nmlib.catalogue "path/to/NON code collection"

The collection lives in a private repository. What is written here to
`catalogue/collection.json` is only what can be read off each stream
mechanically -- its structure, its estimation method, the techniques it
uses and what the static checks make of it -- plus the file name, which is
the title of the paper the model comes from. No code and no free text from
inside a stream is copied: a `$PROBLEM` line is whatever its author typed,
and one in this collection is marked confidential.

Detection is by pattern on the comment-stripped code, so it can miss a
technique written in an unusual way; it does not claim a technique the code
does not contain in some recognisable form.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from .check import (
    _count_params,
    _count_random_effects,
    _record,
    _records,
    _strip_comments,
    check_text,
    count_compartments,
)

#: The private repository the catalogue points into.
REPO = "https://github.com/Wrlog/NONMEM-code-collection"
BRANCH = "main"

#: The only keys a catalogue entry may carry. The test suite holds the
#: committed JSON to this list, so adding a field is a deliberate act.
FIELDS = ("title", "path", "topics", "area", "structure", "ode",
          "compartments", "thetas", "etas", "epsilons", "estimation",
          "evaluation_only", "techniques", "check")

SUFFIXES = {".mod", ".ctl"}

#: Streams are sometimes saved as text; one in this collection is. A .txt
#: file is catalogued when it has the records every stream must have.
TEXT_SUFFIXES = {".txt"}

#: Folder name -> (topic as displayed, broader area for filtering). The
#: folders are the collector's own grouping; the areas fold forty of them
#: into something a reader can filter on.
TOPICS = {
    "$ERROR NORMAL SIGMA": ("Residual error on SIGMA", "Methods"),
    "$ERROR WITH SIGMA 1 FIX": ("Residual error with SIGMA 1 FIX", "Methods"),
    "$PRIOR": ("$PRIOR", "Methods"),
    "ABSORPTION MODEL WITH GUT MODELLING": ("Absorption with gut modelling",
                                            "Absorption"),
    "ADVERSE EVENT AS PD": ("Adverse event as PD", "PK/PD and disease"),
    "ANIMAL TO HUMAN PKPD": ("Animal to human PK/PD", "PK/PD and disease"),
    "ANTIBODY DRUG CONJUGATE": ("Antibody-drug conjugate", "Biologics"),
    "ANTIINFECTIVE": ("Anti-infectives", "Anti-infectives"),
    "BILE RELEASE PARAMETER": ("Bile release", "Absorption"),
    "BOX-COX": ("Box-Cox random effects", "Methods"),
    "CHILDREN": ("Children", "Special populations"),
    "CHILDREN WITH AGE MATURATION": ("Children, age maturation",
                                     "Special populations"),
    "CIRCADIAN RHYTM": ("Circadian rhythm", "PK/PD and disease"),
    "CLINICAL ENDPOINT": ("Clinical endpoint", "PK/PD and disease"),
    "DIABETES": ("Diabetes", "PK/PD and disease"),
    "DISEASE PROGRESSION": ("Disease progression", "PK/PD and disease"),
    "DRUG DISSOLUTION": ("Drug dissolution", "Absorption"),
    "IOV": ("Inter-occasion variability", "Methods"),
    "LARGE MOLECULE IMMUNOLOGY": ("Large molecules, immunology", "Biologics"),
    "LLOQ MODEL": ("Below the LLOQ", "Methods"),
    "MAXEVAL=0": ("MAXEVAL=0", "Methods"),
    "MULTIPLE TRANSIT MODEL": ("Transit absorption", "Absorption"),
    "NEONATE": ("Neonates", "Special populations"),
    "NON CANCER PATIENT PK ON CANCER PATIENT": (
        "Non-cancer PK applied to cancer patients", "Special populations"),
    "NON LINEAR PK": ("Nonlinear PK", "Pharmacokinetics"),
    "NON LINEAR DISTRIBUTION ON ERYTHROCYTE": (
        "Nonlinear distribution into erythrocytes", "Pharmacokinetics"),
    "OBESE": ("Obesity", "Special populations"),
    "ONCOLOGY": ("Oncology", "Oncology"),
    "ONCOLOGY SMALL MOLECULE": ("Oncology, small molecule", "Oncology"),
    "PARENT AND METABOLITE": ("Parent and metabolite", "Pharmacokinetics"),
    "PKPD ACROSS AGE RANGE FROM INFANT": ("PK/PD across paediatric ages",
                                          "Special populations"),
    "PREGANNCY WOMAN": ("Pregnancy", "Special populations"),
    "PURE PD MODEL": ("Pure PD", "PK/PD and disease"),
    "SCALING FROM ADULT TO CHILDREN": ("Scaling from adults to children",
                                       "Special populations"),
    "TIME TO EVENT": ("Time to event", "Time to event"),
    "TMDD": ("Target-mediated disposition", "Biologics"),
    "TUBERCULOSIS": ("Tuberculosis", "Anti-infectives"),
    "UNCLASSIFIED": ("Unclassified", "Unclassified"),
}

#: Technique -> test on the comment-stripped, upper-cased code. Each is
#: written to fire on the construct itself, not on a variable name that
#: merely suggests it.
TECHNIQUES: dict[str, callable] = {}


def _technique(name: str):
    def register(fn):
        TECHNIQUES[name] = fn
        return fn
    return register


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.M) is not None


@_technique("Inter-occasion variability")
def _iov(c: dict) -> bool:
    # ETAs switched by occasion, or $OMEGA ... SAME, which only occasion
    # blocks use.
    return (_has(r"\bOCC\w*\s*\.EQ\.", c["code"]) and
            _has(r"\bETA\(\d+\)\s*\*\s*\w+", c["code"])) or \
        _has(r"\bSAME\b", c["omega"])


@_technique("M3 / BLQ likelihood")
def _m3(c: dict) -> bool:
    # PHI() is only the normal CDF; it is M3 when it is taken at a limit of
    # quantification.
    return _has(r"\bPHI\s*\(", c["code"]) and \
        _has(r"\b\w*(LOQ|BLQ|BQL|LOD|CENS)\w*\b", c["code"])


@_technique("Box-Cox transformed ETA")
def _boxcox(c: dict) -> bool:
    return _has(r"\*\*\s*\w+\s*-\s*1\s*\)\s*/\s*\w+", c["code"])


@_technique("Allometric scaling")
def _allometry(c: dict) -> bool:
    return _has(r"\b\w*(WT|BW|WGT|WEIGHT|FFM|LBW|TBW|NFM)\w*\s*/\s*[\d.]+\s*\)"
                r"\s*\*\*", c["code"]) or _has(r"\*\*\s*\(?\s*0?\.75\b", c["code"])


@_technique("Age maturation (Hill)")
def _maturation(c: dict) -> bool:
    return _has(r"\b(PMA|PNA|GA|AGE|PCA)\w*\s*\)?\s*\*\*\s*\w+", c["code"]) and \
        _has(r"\*\*\s*\w+\s*\)?\s*\+", c["code"])


@_technique("Transit compartments")
def _transit(c: dict) -> bool:
    return _has(r"\bKTR\b|\bGAMLN\s*\(|\bLGAM\w*\s*\(|\bMTT\b", c["code"])


@_technique("Michaelis-Menten elimination")
def _mm(c: dict) -> bool:
    return _has(r"\bV?MAX\w*\b|\bVM\b", c["code"]) and \
        _has(r"\/\s*\(\s*\w*KM\w*\s*\+", c["code"])


@_technique("Target-mediated disposition")
def _tmdd(c: dict) -> bool:
    # Binding and turnover of the target, not merely saturable binding.
    return sum(_has(p, c["code"]) for p in
               (r"\bKON\b", r"\bKOFF\b", r"\bKINT\b", r"\bKDEG\b", r"\bKSS\b",
                r"\bRTOT\b")) >= 2


@_technique("Effect compartment")
def _effect_cmt(c: dict) -> bool:
    return _has(r"\bKE0\b|\bKEO\b", c["code"])


@_technique("Indirect response / turnover")
def _idr(c: dict) -> bool:
    return _has(r"\bKIN\b", c["code"]) and _has(r"\bKOUT\b", c["code"])


@_technique("Emax / sigmoid exposure-response")
def _emax(c: dict) -> bool:
    return _has(r"\b\w*(EC50|IC50|EMAX|IMAX|EDK50)\w*\b", c["code"])


@_technique("Circadian rhythm")
def _circadian(c: dict) -> bool:
    return _has(r"\b(COS|SIN)\s*\(", c["code"])


@_technique("Time to event")
def _tte(c: dict) -> bool:
    return _has(r"\b\w*(HAZ|CHZ|CUMH)\w*\b", c["code"]) and \
        _has(r"\bEXP\s*\(\s*-", c["code"])


@_technique("Count / categorical likelihood")
def _categorical(c: dict) -> bool:
    return c["likelihood"] and not _tte(c) and not _m3(c)


@_technique("Markov dependence")
def _markov(c: dict) -> bool:
    return _has(r"\b(PDV|PREV\w*|PRDV|LDV|LAST\w*)\b", c["code"]) and \
        c["likelihood"]


@_technique("Log-transformed both sides")
def _ltbs(c: dict) -> bool:
    return _has(r"\bIPRED\s*=\s*LOG\s*\(|\bY\s*=\s*LOG\s*\(", c["code"])


@_technique("Residual error via THETA (SIGMA 1 FIX)")
def _sigma_fix(c: dict) -> bool:
    return _has(r"^\s*1\s*FIX", c["sigma"])


@_technique("$PRIOR")
def _prior(c: dict) -> bool:
    return bool(c["prior"]) or _has(r"\bNWPRI\b", c["code"])


@_technique("Mixture model")
def _mixture(c: dict) -> bool:
    return bool(c["mix"])


def read_text(path: Path) -> str:
    """Read a stream whatever its encoding, and whatever its path length.

    Five of the files in the collection are Latin-1, and one path runs past
    Windows' 260-character limit, which the extended-length prefix lifts.
    """
    target = str(path)
    if os.name == "nt" and not target.startswith("\\\\?\\"):
        target = "\\\\?\\" + os.path.abspath(target)
    with open(target, "rb") as fh:
        raw = fh.read()
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def is_control_stream(raw: str) -> bool:
    upper = _strip_comments(raw).upper()
    return all(re.search(rf"^[ \t]*\${r}", upper, re.M)
               for r in ("PROB", "INPUT", "DATA", "THETA"))


def _estimation(est_bodies: list[str]) -> tuple[str, bool, bool]:
    """Name each $ESTIMATION step, and say whether any parameter moves.

    Returns (method, evaluation_only, on_likelihood). A stream whose every
    step has MAXEVAL=0 -- or SAEM/IMP with no iterations, or EONLY -- is an
    evaluation of fixed estimates, which is how a published model is
    typically shared.
    """
    steps, moves, likelihood = [], False, False
    for body in est_bodies:
        b = " ".join(body.upper().split())
        m = re.search(r"\bMET\w*\s*=\s*(\w+)", b)
        meth = m.group(1) if m else "0"
        inter = bool(re.search(r"\bINTER\w*|\bCINTER\b", b))
        laplace = bool(re.search(r"\bLAPLAC\w*", b))
        if re.search(r"\bLIKE\w*|-2LL", b):
            likelihood = True
        if meth in ("1", "COND", "CONDITIONAL"):
            name = "Laplace" if laplace else ("FOCE-I" if inter else "FOCE")
        elif meth in ("0", "ZERO"):
            name = "Laplace" if laplace else "FO"
        elif meth.startswith("IMPMAP"):
            name = "IMPMAP"
        elif meth.startswith("IMP"):
            name = "IMP"
        elif meth in ("SAEM", "BAYES", "ITS", "NUTS", "DIRECT"):
            name = meth
        else:
            name = meth
        maxeval = re.search(r"\bMAX\w*\s*=\s*(\d+)", b)
        niter = re.search(r"\bNITER\s*=\s*(\d+)", b)
        evaluates = ((maxeval is not None and int(maxeval.group(1)) == 0)
                     or (niter is not None and int(niter.group(1)) == 0
                         and name in ("SAEM", "BAYES", "ITS"))
                     or bool(re.search(r"\bEONLY\s*=\s*1", b)))
        moves |= not evaluates
        if not steps or steps[-1] != name:
            steps.append(name)
    method = " then ".join(steps) if steps else "none"
    return method, bool(steps) and not moves, likelihood


def describe(raw: str) -> dict:
    """Everything the catalogue says about one stream, read off its code."""
    text = _strip_comments(raw)
    upper = text.upper()
    code = "\n".join(_record(upper, r) for r in
                     ("PK", "PRED", "ERROR", "DES", "MIX", "AES", "INFN"))

    # $SUBS is a common spelling that is not a prefix of $SUBROUTINES, so the
    # record is found by its first three letters alone.
    sub = re.search(r"^[ \t]*\$SUB\w*(.*?)(?=^[ \t]*\$[A-Z]|\Z)", upper,
                    re.S | re.M)
    sub = sub.group(1) if sub else ""
    advan = re.search(r"\bADVAN\s*=?\s*(\d+)", sub)
    trans = re.search(r"\bTRANS\s*=?\s*(\d+)", sub)
    des = bool(_record(upper, "DES").strip())
    if advan:
        structure = f"ADVAN{advan.group(1)}"
        if trans and trans.group(1) != "1":
            structure += f" TRANS{trans.group(1)}"
    elif _record(upper, "PRED").strip():
        structure = "$PRED"
    else:
        structure = "unspecified"

    compartments = count_compartments(_record(upper, "MODEL"))
    fixed = {"1": 1, "2": 2, "3": 2, "4": 3, "11": 3, "12": 4}
    if not compartments and advan and advan.group(1) in fixed:
        # The closed-form ADVANs imply their compartments (plus output),
        # without a $MODEL record to count.
        compartments = fixed[advan.group(1)]

    method, evaluation_only, likelihood = _estimation(_records(upper, "ESTIMATION"))
    if method == "none" and _record(upper, "SIMULATION").strip():
        method = "simulation only"
    if _has(r"\bF_FLAG\s*=", code):
        likelihood = True
    ctx = {
        "code": code,
        "omega": "\n".join(_records(upper, "OMEGA")),
        "sigma": "\n".join(_records(upper, "SIGMA")),
        "prior": _record(upper, "PRIOR").strip(),
        "mix": _record(upper, "MIX").strip(),
        "likelihood": likelihood,
    }
    techniques = [name for name, test in TECHNIQUES.items() if test(ctx)]

    checked = check_text(raw, "stream")
    return {
        "structure": structure,
        "ode": des,
        "compartments": compartments,
        "thetas": sum(_count_params(b) for b in _records(text, "THETA")),
        "etas": _count_random_effects(_records(text, "OMEGA")),
        "epsilons": _count_random_effects(_records(text, "SIGMA")),
        "estimation": method,
        "evaluation_only": evaluation_only,
        "techniques": techniques,
        "check": {"errors": checked.errors, "notes": checked.warnings},
    }


def _title(stem: str) -> str:
    title = re.sub(r"^Executable_", "", stem)
    title = title.replace("_", " ").strip().rstrip(".")
    return re.sub(r"\s+", " ", title)


def _topic(folder: str) -> tuple[str, str]:
    key = folder.strip().upper()
    if key in TOPICS:
        return TOPICS[key]
    return folder.strip().capitalize(), "Other"


def _fingerprint(raw: str) -> str:
    """Identify a stream by its code, so a copy filed twice is listed once."""
    code = " ".join(_strip_comments(raw).upper().split())
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def scan(collection: Path, repo_root: Path | None = None) -> list[dict]:
    """Catalogue every control stream under `collection`.

    `repo_root` is the root of the private repository the collection sits
    in, and fixes the paths the page links to; it defaults to the parent of
    the collection folder.
    """
    collection = collection.resolve()
    repo_root = (repo_root or collection.parent).resolve()
    seen: dict[str, dict] = {}
    for path in sorted(collection.rglob("*"), key=lambda p: str(p).lower()):
        suffix = path.suffix.lower()
        if suffix not in SUFFIXES | TEXT_SUFFIXES or not path.is_file():
            continue
        raw = read_text(path)
        if suffix in TEXT_SUFFIXES and not is_control_stream(raw):
            continue
        key = _fingerprint(raw)
        topic, area = _topic(path.parent.name)
        if key in seen:
            if topic not in seen[key]["topics"]:
                seen[key]["topics"].append(topic)
            continue
        entry = {
            "title": _title(path.stem),
            "path": path.relative_to(repo_root).as_posix(),
            "topics": [topic],
            "area": area,
            **describe(raw),
        }
        seen[key] = {k: entry[k] for k in FIELDS}
    return sorted(seen.values(), key=lambda e: (e["area"], e["title"].lower()))


def link(entry: dict) -> str:
    return f"{REPO}/blob/{BRANCH}/{quote(entry['path'])}"


def load(root: Path) -> list[dict]:
    path = root / "catalogue" / "collection.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["models"]


def technique_counts(entries: list[dict]) -> list[tuple[str, int]]:
    counts = Counter(t for e in entries for t in e["techniques"])
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("collection", type=Path,
                    help="folder of control streams, inside the private repo")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="root of the private repo (default: the folder's parent)")
    ap.add_argument("--out", type=Path, default=Path("catalogue/collection.json"))
    args = ap.parse_args()

    entries = scan(args.collection, args.repo_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"source": REPO, "models": entries},
                                   indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}: {len(entries)} control streams")
    for name, n in technique_counts(entries):
        print(f"  {n:3d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
