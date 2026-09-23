$PROBLEM Time to event, Weibull baseline hazard with an exposure effect
; Data: data/tte_weibull.csv, simulated by nmlib.simulate.simulate_tte_weibull
;
; h(t) = LAMBDA*SHAPE*(LAMBDA*t)^(SHAPE-1) * exp(BETA*EXPO)
;
; Fitted on the likelihood directly, not on a residual: the contribution of
; a subject is the survivor function at their last time if they were
; censored, and survivor times hazard if they had the event. SHAPE above 1
; means the hazard rises with time, below 1 that it falls.
;
; Data layout: one record at TIME=0 opening the interval with DV=0, then one
; record at the event or censoring time carrying DV=1 or DV=0.

$INPUT ID TIME DV EVID MDV EXPO
$DATA ../data/tte_weibull.csv IGNORE=@

$SUBROUTINE ADVAN13 TOL=9

$MODEL
COMP=(CUMHAZ)

$PK
LAMBDA = THETA(1)*EXP(ETA(1))
SHAPE  = THETA(2)
BETA   = THETA(3)

; A(1) accumulates the cumulative hazard from time zero.
A_0(1) = 0

$DES
; DEL keeps the power term finite at T=0 when SHAPE < 1.
DEL     = 1E-12
HAZ     = LAMBDA*SHAPE*(LAMBDA*(T+DEL))**(SHAPE-1)*EXP(BETA*EXPO)
DADT(1) = HAZ

$ERROR
DEL    = 1E-12
CUMHAZ = A(1)
SURV   = EXP(-CUMHAZ)
HAZNOW = LAMBDA*SHAPE*(LAMBDA*(TIME+DEL))**(SHAPE-1)*EXP(BETA*EXPO)

; Censored record: probability of surviving to here.
; Event record: density, survivor times the hazard at the event time.
IF (DV.EQ.0) Y = SURV
IF (DV.EQ.1) Y = SURV*HAZNOW

$THETA
(0, 0.0025)       ; 1 LAMBDA scale (1/day)
(0, 1.35)         ; 2 SHAPE  Weibull shape, >1 means hazard rises with time
(-3, -0.025, 3)   ; 3 BETA   log hazard ratio per unit exposure

$OMEGA
0.04              ; 1 IIV on LAMBDA (frailty)

$ESTIMATION METHOD=1 LAPLACE LIKELIHOOD MAXEVAL=9999 SIG=3 PRINT=5 NOABORT
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV EXPO CUMHAZ SURV
       ONEHEADER NOPRINT FILE=sdtab_tte
