$PROBLEM One-compartment oral PK, lag time, combined error, M3 for BLQ records
; Template: no dataset ships with this stream. $INPUT lists what it expects.
;
; The M3 method (Beal 2001; Bergstrand and Karlsson 2009) keeps records
; below the limit of quantification in the fit instead of dropping them. A
; measured concentration contributes the usual normal density; a BLQ record
; contributes the probability that the concentration lies below the LLOQ,
; PHI((LLOQ - IPRED)/W). Dropping BLQ records instead biases CL upwards when
; they cluster in the terminal phase, because the model is then shown only
; the subjects whose concentrations were still high.
;
; Data layout: BLQ=1 on records reported below the LLOQ, with MDV=0 so the
; record still counts. DV on those records is not used; put 0 there.

$INPUT ID TIME AMT DV MDV EVID CMT BLQ
$DATA your_data.csv IGNORE=@

$SUBROUTINE ADVAN2 TRANS2

$PK
CL    = THETA(1)*EXP(ETA(1))
V     = THETA(2)*EXP(ETA(2))
KA    = THETA(3)*EXP(ETA(3))

; Lag time without IIV. An ETA on ALAG1 makes the likelihood non-smooth in
; that ETA wherever a sample falls near the end of the lag, which is exactly
; where FOCE's gradients go wrong. If the delay varies between subjects, the
; transit model in pk_transit_absorption.mod describes it smoothly instead.
ALAG1 = THETA(6)

S2 = V

$ERROR
LLOQ  = 0.1        ; mg/L, assay lower limit of quantification

; Combined error with its two standard deviations as THETAs and a single
; EPS fixed at 1. W is then the residual SD itself, which is what M3 needs
; in PHI((LLOQ - IPRED)/W) and what IWRES divides by.
IPRED = F
W     = SQRT((THETA(4)*IPRED)**2 + THETA(5)**2)
IRES  = DV - IPRED
IWRES = IRES/W

; When simulating (ICALL=4, e.g. for a VPC) every record gets a
; concentration, and whether it was BLQ is decided afterwards against the
; LLOQ. That is why ICALL is part of the condition: NM-TRAN does not allow
; F_FLAG to be set inside a separate IF (ICALL.EQ.4) block.
IF (BLQ.EQ.1.AND.ICALL.NE.4) THEN
  F_FLAG = 1                      ; Y is a likelihood: P(C < LLOQ)
  Y      = PHI((LLOQ - IPRED)/W)
  MDVRES = 1                      ; no residual is defined for this record
ELSE
  F_FLAG = 0                      ; Y is a prediction: normal density
  Y      = IPRED + W*EPS(1)
ENDIF

$THETA
(0, 4.0)     ; 1 CL    (L/h)
(0, 40.0)    ; 2 V     (L)
(0, 1.0)     ; 3 KA    (1/h)
(0, 0.15)    ; 4 proportional error SD
(0, 0.05)    ; 5 additive error SD (mg/L)
(0, 0.3)     ; 6 ALAG1 (h)

$OMEGA
0.09         ; 1 IIV on CL
0.09         ; 2 IIV on V
0.25         ; 3 IIV on KA

$SIGMA
1 FIX        ; scaled by W, so the THETAs carry the magnitude

; M3 needs LAPLACIAN: with F_FLAG=1 the record's Y is a probability, and
; FOCE's linearisation of Y in the ETAs does not apply to it.
$ESTIMATION METHOD=1 LAPLACIAN INTERACTION MAXEVAL=9999 SIG=3 PRINT=5 NOABORT
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV BLQ IPRED PRED CWRES IWRES CL V KA ETA(1) ETA(2) ETA(3)
       ONEHEADER NOPRINT FILE=sdtab_oral_m3
