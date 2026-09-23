$PROBLEM Binary response, logistic regression on exposure with IIV
; Data: data/logistic_binary.csv, simulated by nmlib.simulate.simulate_logistic
;
; logit(P) = BASE + SLOPE*EXPO + ETA
;
; Fitted on the likelihood: Y is the probability of the observed outcome,
; P for a responder and 1-P otherwise. This is the shape most exposure-safety
; and exposure-response analyses take when the endpoint is yes or no.
; LAPLACE is required because the likelihood is not normal.

$INPUT ID TIME DV MDV EVID EXPO
$DATA ../data/logistic_binary.csv IGNORE=@

$PRED
BASE  = THETA(1)
SLOPE = THETA(2)

LOGIT = BASE + SLOPE*EXPO + ETA(1)
P     = EXP(LOGIT)/(1 + EXP(LOGIT))

; Contribution of this record to the likelihood.
IF (DV.EQ.1) Y = P
IF (DV.EQ.0) Y = 1 - P

$THETA
(-10, -1.20, 10)  ; 1 BASE  log odds at zero exposure
(-1, 0.055, 1)    ; 2 SLOPE change in log odds per unit exposure

$OMEGA
0.36              ; 1 IIV on the logit (SD 0.6)

$ESTIMATION METHOD=1 LAPLACE LIKELIHOOD MAXEVAL=9999 SIG=3 PRINT=5 NOABORT
$COVARIANCE UNCONDITIONAL
$TABLE ID EXPO DV P ETA(1)
       ONEHEADER NOPRINT FILE=sdtab_logistic
