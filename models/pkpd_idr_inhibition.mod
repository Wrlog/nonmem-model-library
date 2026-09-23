$PROBLEM Indirect response, drug inhibiting production of a biomarker
; Data: data/pkpd_idr_inhibition.csv
;   simulated by nmlib.simulate.simulate_idr_inhibition
;
; dR/dt = KIN*(1 - IMAX*C/(IC50 + C)) - KOUT*R
;
; This is Dayneka and Jusko's model I. The response lags exposure because
; the biomarker has its own turnover; washout after stopping is governed by
; KOUT, not by the drug's half-life. Baseline is the untreated steady state
; KIN/KOUT, so it is not a separate parameter.
;
; The design matters as much as the structure here. The drug is given daily
; for a week and then stopped, and the biomarker's half-life (about 3.5
; days) is roughly five times the drug's (about 17 hours). That separation
; is what makes the two clocks distinguishable: the response is still
; falling after the concentration has reached steady state, and still
; recovering weeks after the last dose. With a biomarker that turned over
; as fast as the drug cleared, this model and a direct effect model would
; fit the same data equally well and there would be nothing to estimate.

$INPUT ID TIME AMT DV MDV EVID CMT DOSE
$DATA ../data/pkpd_idr_inhibition.csv IGNORE=@

$SUBROUTINE ADVAN13 TOL=9

$MODEL
COMP=(CENTRAL, DEFDOSE)   ; 1 drug
COMP=(RESPONSE)           ; 2 biomarker

$PK
CL   = THETA(1)
V    = THETA(2)
KIN  = THETA(3)
KOUT = THETA(4)*EXP(ETA(1))
IMAX = THETA(5)
IC50 = THETA(6)*EXP(ETA(2))

S1 = V

; The biomarker starts at its untreated steady state.
A_0(2) = KIN/KOUT

$DES
CONC    = A(1)/V
INH     = 1 - IMAX*CONC/(IC50 + CONC)
DADT(1) = -(CL/V)*A(1)
DADT(2) = KIN*INH - KOUT*A(2)

$ERROR
IPRED = A(2)
Y     = IPRED*(1 + EPS(1))

$THETA
(0, 4.0)          ; 1 CL   (L/h), driving PK, fixed from the PK analysis
(0, 100.0)        ; 2 V    (L), likewise
(0, 0.80)         ; 3 KIN  production rate (units/h)
(0, 0.008)        ; 4 KOUT loss rate constant (1/h); baseline KIN/KOUT = 100
(0, 0.80, 1)      ; 5 IMAX maximum fractional inhibition, bounded at 1
(0, 8.0)          ; 6 IC50 concentration at half maximum inhibition (mg/L)

$OMEGA
0.09              ; 1 IIV on KOUT
0.22              ; 2 IIV on IC50

$SIGMA
0.01              ; 1 proportional residual error (~10%)

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=9999 SIG=3 PRINT=5 NOABORT
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV IPRED PRED CWRES DOSE KOUT IC50
       ONEHEADER NOPRINT FILE=sdtab_idr
