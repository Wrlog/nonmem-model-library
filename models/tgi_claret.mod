$PROBLEM Claret tumour growth inhibition with resistance
; Data: data/tgi_claret.csv, simulated by nmlib.simulate.simulate_tgi_claret
;
; dY/dt = KL*Y - KD*EXPO*exp(-LAMBDA*t)*Y
;
; KL is exponential growth, KD the drug kill rate, and LAMBDA the resistance
; term that lets the kill effect fade with time. Without LAMBDA the model
; cannot reproduce regrowth while treatment continues, which is the
; behaviour the model exists to describe.

$INPUT ID TIME DV MDV EVID EXPO ARM
$DATA ../data/tgi_claret.csv IGNORE=@

$SUBROUTINE ADVAN13 TOL=9

$MODEL
COMP=(TUMOUR, DEFDOSE)

$PK
Y0     = THETA(1)*EXP(ETA(1))
KL     = THETA(2)*EXP(ETA(2))
KD     = THETA(3)*EXP(ETA(3))
LAMBDA = THETA(4)

A_0(1) = Y0

$DES
; EXPO is the subject's average exposure, constant within a subject here.
KILL       = KD*EXPO*EXP(-LAMBDA*T)
DADT(1)    = KL*A(1) - KILL*A(1)

$ERROR
IPRED = A(1)
Y     = IPRED*(1 + EPS(1))

$THETA
(0, 50.0)    ; 1 Y0     baseline tumour size (mm)
(0, 0.006)   ; 2 KL     growth rate (1/day)
(0, 0.0004)  ; 3 KD     kill rate per unit exposure
(0, 0.015)   ; 4 LAMBDA resistance rate (1/day)

$OMEGA
0.18         ; 1 IIV on Y0
0.22         ; 2 IIV on KL
0.30         ; 3 IIV on KD

$SIGMA
0.0144       ; 1 proportional residual error (~12%)

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=9999 SIG=3 PRINT=5 NOABORT
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV IPRED PRED CWRES EXPO ARM Y0 KL KD
       ONEHEADER NOPRINT FILE=sdtab_tgi
