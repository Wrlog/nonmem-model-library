"""Static checks on the control streams.

These are the errors that a control stream can carry while still running to
completion and reporting a plausible-looking objective function, which is
what makes them worth catching mechanically rather than by reading:

* `$INPUT` names must match the data file's columns, **in order** — with
  `IGNORE=@` NONMEM skips the header and reads positionally, so a control
  stream whose `$INPUT` is merely a permutation of the header silently reads
  the wrong column into the wrong variable.
* Every `THETA(n)`, `ETA(n)` and `EPS(n)` referenced must exist, and every
  one declared should be referenced; an unreferenced parameter is almost
  always a leftover.
* Initial estimates must sit inside their own bounds.
* Compartment numbers used in `$DES` and `A_0` must exist in `$MODEL`.
* The data must carry the columns the fit needs, with no missing values
  where NONMEM would not tolerate them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class CheckResult:
    model: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _strip_comments(text: str) -> str:
    return "\n".join(line.split(";")[0] for line in text.splitlines())


def _record_pattern(name: str) -> str:
    """Match $NAME and any abbreviation of it down to three letters.

    NONMEM reads `$EST`, `$ESTIM` and `$ESTIMATION` as the same record, and
    control streams in the wild use all three; matching only the full name
    would report a stream with `$EST` as having no estimation step at all.
    """
    head, tail = name[:3], name[3:]
    optional = "".join(f"(?:{c}" for c in tail) + ")?" * len(tail)
    return rf"^[ \t]*\${head}{optional}\b(.*?)(?=^[ \t]*\$[A-Z]|\Z)"


def _record(text: str, name: str) -> str:
    """Return the body of a $RECORD, or '' if absent."""
    m = re.search(_record_pattern(name), text, re.S | re.M | re.I)
    return m.group(1) if m else ""


def _records(text: str, name: str) -> list[str]:
    """Return the body of every $RECORD of this name, in order.

    $OMEGA in particular is routinely split across records -- one per BLOCK,
    or one per occasion with SAME -- and reading only the first undercounts.
    """
    return [m.group(1) for m in
            re.finditer(_record_pattern(name), text, re.S | re.M | re.I)]


def _count_random_effects(bodies: list[str]) -> int:
    """Count the ETAs (or EPSs) declared across $OMEGA (or $SIGMA) records.

    BLOCK(n) declares n effects however many values it lists, and
    BLOCK(n) SAME declares n more with no values at all.
    """
    n = 0
    for body in bodies:
        block = re.match(r"\s*BLOCK\s*\(\s*(\d+)\s*\)", body, re.I)
        if block:
            n += int(block.group(1))
            continue
        body = re.sub(r"\bDIAG(?:ONAL)?\s*\(\s*\d+\s*\)", " ", body, flags=re.I)
        n += _count_params(body)
    return n


def _count_params(body: str) -> int:
    """Count initial estimates in a $THETA/$OMEGA/$SIGMA body.

    Parenthesised entries are one parameter each; bare numbers are one each.
    """
    body = body.strip()
    if not body:
        return 0
    n = 0
    rest = body
    for group in re.findall(r"\([^)]*\)", body):
        n += 1
        rest = rest.replace(group, " ", 1)
    n += len(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", rest))
    return n


def count_compartments(model_body: str) -> int:
    """Compartments declared in $MODEL.

    NCOMPARTMENTS says so outright when present. Otherwise every COMP counts,
    whatever follows it: `COMP=(CENTRAL)`, `COMP=CENTRAL`, `COMP (CENTRAL)`
    and a bare `COMP=` with the name left to a comment are all one
    compartment to NONMEM, as is the shorter `COM`.
    """
    stated = re.search(r"\bNCOMP\w*\s*=\s*(\d+)", model_body, re.I)
    if stated:
        return int(stated.group(1))
    return len(re.findall(r"\bCOMP?\b", model_body, re.I))


def _bounds(body: str) -> list[tuple[float | None, float, float | None]]:
    """Parse (low, init, up) triples where given."""
    out = []
    for group in re.findall(r"\(([^)]*)\)", body):
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", group)]
        if len(nums) == 3:
            out.append((nums[0], nums[1], nums[2]))
        elif len(nums) == 2:
            out.append((nums[0], nums[1], None))
    return out


def check_control_stream(mod_path: Path, data_dir: Path) -> CheckResult:
    raw = mod_path.read_text(encoding="utf-8")
    text = _strip_comments(raw)
    res = CheckResult(model=mod_path.stem)

    # --- $INPUT against the data file ------------------------------------
    input_body = _record(text, "INPUT")
    items = [i for i in re.split(r"[\s,]+", input_body.strip()) if i]
    data_body = _record(text, "DATA").strip()
    # The path runs up to the first option (IGNORE=@, ACCEPT=..., REWIND) or
    # the end of the line. Splitting on whitespace looks simpler and silently
    # truncates any path containing a space, which then reports as "data file
    # not found" and sends the reader looking in the wrong place.
    first_line = data_body.splitlines()[0] if data_body else ""
    option = re.search(r"\s+(?=[A-Za-z]+\s*=|\bREWIND\b|\bNOREWIND\b)",
                       first_line)
    data_ref = (first_line[:option.start()] if option else first_line).strip()
    csv_path = (mod_path.parent / data_ref).resolve()
    if not csv_path.exists():
        alt = data_dir / Path(data_ref).name
        csv_path = alt if alt.exists() else csv_path

    if not items:
        res.errors.append("no $INPUT record")
    if not csv_path.exists():
        res.errors.append(f"data file not found: {data_ref}")
    else:
        cols = list(pd.read_csv(csv_path, nrows=1).columns)
        bare = [i.split("=")[-1] for i in items]
        if bare != cols:
            res.errors.append(
                "$INPUT does not match the data columns in order.\n"
                f"      $INPUT: {bare}\n"
                f"      header: {cols}\n"
                "      With IGNORE=@ NONMEM reads positionally, so this would "
                "load the wrong column into the wrong variable."
            )
        df = pd.read_csv(csv_path)
        for required in ("ID", "DV"):
            if required not in cols:
                res.errors.append(f"data has no {required} column")
        if "ID" in cols and df["ID"].isna().any():
            res.errors.append("data has missing ID values")
        if len(df) == 0:
            res.errors.append("data file is empty")

    structural = check_text(raw, res.model)
    res.errors += structural.errors
    res.warnings += structural.warnings
    return res


def check_text(raw: str, name: str) -> CheckResult:
    """Every check that can be made from the control stream alone.

    This is what can be run on a stream whose dataset is not to hand -- the
    published models catalogued alongside the library ship without theirs.
    """
    text = _strip_comments(raw)
    res = CheckResult(model=name)

    # --- parameter references --------------------------------------------
    code = "\n".join(_record(text, r) for r in
                     ("PK", "PRED", "ERROR", "DES", "MIX", "AES"))
    for kind, record in (("THETA", "THETA"), ("ETA", "OMEGA"), ("EPS", "SIGMA")):
        bodies = _records(text, record)
        declared = (sum(_count_params(b) for b in bodies) if kind == "THETA"
                    else _count_random_effects(bodies))
        # The lookbehind matters: ETA\(\d+\) also matches the tail of
        # THETA(1), which would report every THETA index as an ETA.
        pattern = rf"(?<![A-Z]){kind}\((\d+)\)"
        used = {int(n) for n in re.findall(pattern, code)}
        # $TABLE may also reference ETA(n); those count as used.
        used |= {int(n) for n in re.findall(pattern, _record(text, "TABLE"))}
        if used and max(used) > declared:
            res.errors.append(
                f"{kind}({max(used)}) referenced but only {declared} "
                f"{'values' if declared != 1 else 'value'} declared in ${record}"
            )
        unused = set(range(1, declared + 1)) - used
        if unused:
            res.warnings.append(
                f"${record}: {kind} {sorted(unused)} declared but never referenced"
            )

    # --- initial estimates inside their bounds ---------------------------
    for record in ("THETA",):
        body = "\n".join(_records(text, record))
        for i, (low, init, up) in enumerate(_bounds(body), start=1):
            if low is not None and init <= low:
                res.errors.append(
                    f"${record} {i}: initial estimate {init} is not above its "
                    f"lower bound {low}")
            if up is not None and init >= up:
                res.errors.append(
                    f"${record} {i}: initial estimate {init} is not below its "
                    f"upper bound {up}")

    # --- compartments ------------------------------------------------------
    n_comp = count_compartments(_record(text, "MODEL"))
    if n_comp:
        used_comp = {int(n) for n in re.findall(r"DADT\((\d+)\)", code)}
        used_comp |= {int(n) for n in re.findall(r"A_0\((\d+)\)", code)}
        used_comp |= {int(n) for n in re.findall(r"\bA\((\d+)\)", code)}
        if used_comp and max(used_comp) > n_comp:
            res.errors.append(
                f"compartment {max(used_comp)} used but $MODEL declares {n_comp}")
        missing_ode = set(range(1, n_comp + 1)) - {
            int(n) for n in re.findall(r"DADT\((\d+)\)", _record(text, "DES"))}
        if _record(text, "DES") and missing_ode:
            res.errors.append(
                f"$DES has no DADT for compartment(s) {sorted(missing_ode)}")

    # --- likelihood models need LIKELIHOOD on $ESTIMATION -----------------
    est = "\n".join(_records(text, "ESTIMATION")).upper()
    # LIKE is NONMEM's own abbreviation, and -2LL states the same thing with
    # Y as minus twice the log-likelihood rather than the likelihood.
    on_likelihood = bool(re.search(r"\bLIKE|-2LL", est))
    defines_y_by_branch = bool(re.search(r"IF\s*\(\s*DV\s*\.EQ\.", code, re.I))
    # F_FLAG=1 declares the likelihood record by record, which is how a
    # stream mixes it with continuous data; LIKE on $ESTIMATION is then not
    # needed and would be wrong for the continuous records.
    per_record = bool(re.search(r"\bF_FLAG\s*=", code, re.I))
    if defines_y_by_branch and not (on_likelihood or per_record):
        res.errors.append(
            "Y is defined branch-wise on DV, which means the model is fitted "
            "on the likelihood, but $ESTIMATION does not say LIKELIHOOD")
    if on_likelihood and _record(text, "SIGMA").strip():
        res.warnings.append(
            "$SIGMA is declared on a LIKELIHOOD model; residual error has no "
            "role there")
    if defines_y_by_branch and "LAPLACE" not in est:
        res.warnings.append(
            "likelihood models are normally fitted with LAPLACE")

    return res


def check_all(models_dir: Path, data_dir: Path) -> list[CheckResult]:
    return [check_control_stream(p, data_dir)
            for p in sorted(models_dir.glob("*.mod"))]
