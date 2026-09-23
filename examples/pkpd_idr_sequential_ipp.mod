$PROBLEM Sequential PK/PD: indirect response on a biomarker, individual PK read from data
; Template: no dataset ships with this stream. $INPUT lists what it expects.
;
; The IPP approach (Zhang, Beal and Sheiner 2003). The PK model is fitted
; first; each subject's post hoc CL, V1, Q and V2 are written out, merged
; onto the PD dataset as CLI, V1I, QI and V2I, and held fixed here. Only
; the PD parameters are estimated. It is faster and more robust than fitting
; PK and PD together, and it keeps a misspecified PD model from pulling the
; PK estimates. What it gives up is the uncertainty in the individual PK:
; the PD standard errors are conditional on it being exact, so they are too
; narrow when the PK is sparse.
;
; PD: Dayneka and Jusko model I. The drug inhibits production (KIN) of the
; biomarker, which is lost at the first-order rate KOUT. The response lags
; the concentration, and it recovers at a rate set by KOUT once the drug is
; gone, not by the drug's half-life.
;
; Data layout: dose records into CMT=1, biomarker observations with CMT=3.
; Drug concentrations are left out; they were used in the PK fit. Time is
; in days.

$INPUT ID TIME AMT RATE DV MDV EVID CMT CLI V1I QI V2I CRP
$DATA your_data.csv IGNORE=@

$SUBROUTINE ADVAN13 TOL=9

$MODEL
COMP=(CENTRAL)             ; drug, amount
COMP=(PERIPH)              ; drug, amount
COMP=(BIOMARK)             ; biomarker, concentration units

$PK
; Individual PK from the earlier fit. No THETAs or ETAs.
K10 = CLI/V1I
K12 = QI/V1I
K21 = QI/V2I

; Baseline scales with inflammation. Missing CRP is coded -99 and treated as
; the reference value.
CRPN = CRP
IF (CRP.EQ.-99) CRPN = 5
BASE = THETA(1)*(CRPN/5)**THETA(5)*EXP(ETA(1))

; Starting the biomarker at BASE, with KIN defined from it, puts the system
; at its own steady state before the first dose. Estimate BASE and KOUT, not
; KIN and KOUT: the baseline is what the data measures directly.
KOUT = THETA(2)
KIN  = BASE*KOUT
A_0(3) = BASE

; IMAX is fixed at 1, full inhibition at saturating exposure. Estimate it
; only if the observed exposures reach the plateau; otherwise IMAX and EC50
; trade off against each other and neither is identified.
IMAX = THETA(3)
EC50 = THETA(4)*EXP(ETA(2))

$DES
; Drug concentration from the fixed individual PK. An exposure summary
; computed outside NONMEM (average concentration over each interval, say)
; can be put in its place as a data item. That decouples the PD from the PK
; structure, and loses the shape of the profile within each interval.
CP  = A(1)/V1I
INH = IMAX*CP/(EC50 + CP)

DADT(1) = -(K10 + K12)*A(1) + K21*A(2)
DADT(2) =   K12*A(1)        - K21*A(2)
DADT(3) = KIN*(1 - INH) - KOUT*A(3)

$ERROR
CONC  = A(1)/V1I           ; $DES variables do not reach $TABLE; recompute
IPRED = A(3)
Y     = IPRED*(1 + EPS(1))

$THETA
(0, 500)        ; 1 BASE, biomarker at CRP = 5 mg/L
(0, 0.1)        ; 2 KOUT (1/day); biomarker half-life ln(2)/KOUT
1 FIX           ; 3 IMAX
(0, 5.0)        ; 4 EC50 (mg/L)
(-3, 0.3, 3)    ; 5 BASE~CRP exponent

$OMEGA
0.25            ; 1 IIV on BASE
0.5             ; 2 IIV on EC50

$SIGMA
0.09            ; 1 proportional error (~30%)

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=9999 SIG=3 PRINT=5 NOABORT
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV IPRED PRED CWRES BASE EC50 CONC ETA(1) ETA(2)
       ONEHEADER NOPRINT FILE=sdtab_ipp
