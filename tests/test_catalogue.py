"""Tests for the reference-collection catalogue.

The collection's code is private, so the first job of these tests is to
hold the committed catalogue to the fields it is allowed to carry: nothing
from inside a stream beyond what the scanner derives. The rest check that
the scanner and the checker read the spellings found in real streams --
abbreviated records, `COMP=` with the name in a comment, likelihoods
declared per record -- the way NONMEM does.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from nmlib.catalogue import FIELDS, TECHNIQUES, describe, is_control_stream
from nmlib.check import check_text, count_compartments


def stream(body: str) -> str:
    return textwrap.dedent(body).strip() + "\n"


# --- the committed catalogue ------------------------------------------------

@pytest.fixture(scope="module")
def entries(repo_root):
    path = repo_root / "catalogue" / "collection.json"
    if not path.exists():
        pytest.skip("no catalogue committed")
    return json.loads(path.read_text(encoding="utf-8"))["models"]


def test_catalogue_carries_only_the_allowed_fields(entries):
    for e in entries:
        assert set(e) == set(FIELDS), e["title"]
        assert set(e["check"]) == {"errors", "notes"}


def test_catalogue_names_only_known_techniques(entries):
    for e in entries:
        assert set(e["techniques"]) <= set(TECHNIQUES), e["title"]


def test_catalogue_lists_each_stream_once(entries):
    paths = [e["path"] for e in entries]
    assert len(paths) == len(set(paths))


# --- reading real-world spellings ------------------------------------------

ONE_CMT = stream("""
    $PROBLEM test
    $INPUT ID TIME DV AMT
    $DATA x.csv IGNORE=@
    $SUBS ADVAN13 TOL=6
    $MODEL {model}
    $PK
      CL = THETA(1)*EXP(ETA(1))*(WT/70)**0.75
      V  = THETA(2)
    $DES
      DADT(1) = -CL/V*A(1)
    $ERROR
      IPRED = LOG(A(1)/V)
      Y = IPRED + EPS(1)
    $THETA (0, 1) (0, 10)
    $OMEGA 0.1
    $SIGMA 0.05
    $EST METH=COND INTER MAX=0 POSTHOC
""")


@pytest.mark.parametrize("model, n", [
    ("COMP=(CENTRAL)", 1),
    ("COMP=CENTRAL COMP=PERIPH", 2),
    ("COMP (CENTRAL) COMP (PERIPH)", 2),
    ("NCOMPARTMENTS=3 COMP=(A) COMP= COMP=", 3),
    ("COM = (A1) COM = (A2)", 2),
])
def test_compartments_are_counted_however_they_are_spelled(model, n):
    assert count_compartments(model) == n


def test_bare_comp_with_the_name_in_a_comment_is_a_compartment():
    """The paracetamol stream this was found in declares six this way."""
    model = "NCOMPARTMENTS=2\nCOMP=(CENTRAL,DEFDOSE)\nCOMP=\n"
    text = ONE_CMT.replace("$MODEL {model}", "$MODEL\n" + model)
    text = text.replace("DADT(1) = -CL/V*A(1)",
                        "DADT(1) = -CL/V*A(1)\n  DADT(2) = CL/V*A(1)")
    assert check_text(text, "t").ok


def test_description_reads_abbreviated_records():
    d = describe(ONE_CMT.replace("{model}", "COMP=(CENTRAL)"))
    assert d["structure"] == "ADVAN13"
    assert d["ode"] and d["compartments"] == 1
    assert (d["thetas"], d["etas"], d["epsilons"]) == (2, 1, 1)
    assert d["estimation"] == "FOCE-I"
    assert d["evaluation_only"]
    assert "Allometric scaling" in d["techniques"]
    assert "Log-transformed both sides" in d["techniques"]


def test_a_compartment_with_no_equation_is_flagged():
    text = ONE_CMT.replace("{model}", "COMP=(CENTRAL) COMP=(LEFTOVER)")
    result = check_text(text, "t")
    assert any("no DADT for compartment(s) [2]" in e for e in result.errors)


LIKELIHOOD = stream("""
    $PROBLEM binary
    $INPUT ID DV
    $DATA x.csv IGNORE=@
    $PRED
      P = 1/(1+EXP(-(THETA(1)+ETA(1))))
      IF (DV.EQ.1) Y = P
      IF (DV.EQ.0) Y = 1-P
    $THETA 0.1
    $OMEGA 0.1
    {est}
""")


@pytest.mark.parametrize("est", [
    "$EST METHOD=1 LAPLACE LIKE",
    "$ESTIMATION METHOD=COND LAPLACE -2LL",
])
def test_likelihood_is_recognised_in_its_short_forms(est):
    text = LIKELIHOOD.replace("{est}", est)
    if "-2LL" in est:
        text = text.replace("Y = P", "Y = -2*LOG(P)")
    assert check_text(text, "t").ok


def test_f_flag_declares_the_likelihood_per_record():
    text = LIKELIHOOD.replace("{est}", "$EST METHOD=1 LAPLACE")
    assert not check_text(text, "t").ok
    text = text.replace("$PRED\n", "$PRED\n  F_FLAG = 1\n")
    assert check_text(text, "t").ok


def test_missing_likelihood_is_still_caught():
    text = LIKELIHOOD.replace("{est}", "$EST METHOD=1 INTER")
    assert any("LIKELIHOOD" in e for e in check_text(text, "t").errors)


def test_phi_alone_is_not_m3():
    """PHI() is the normal CDF; only at a limit of quantification is it M3."""
    base = ONE_CMT.replace("{model}", "COMP=(CENTRAL)")
    no_loq = base.replace("Y = IPRED + EPS(1)", "Y = PHI(IPRED) + EPS(1)")
    with_loq = base.replace("Y = IPRED + EPS(1)",
                            "Y = PHI((LLOQ-IPRED)/0.1) + EPS(1)")
    assert "M3 / BLQ likelihood" not in describe(no_loq)["techniques"]
    assert "M3 / BLQ likelihood" in describe(with_loq)["techniques"]


def test_text_files_are_catalogued_only_when_they_are_streams():
    assert is_control_stream(ONE_CMT)
    assert not is_control_stream("Notes on the prolactin model.\n$THETA was fixed.")
