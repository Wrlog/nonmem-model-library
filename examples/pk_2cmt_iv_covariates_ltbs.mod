$PROBLEM Two-compartment IV infusion, covariates on CL and V1, log-transformed both sides
; Template: no dataset ships with this stream. $INPUT lists what it expects.
;
; The covariate model of a typical monoclonal antibody, with the four kinds
; of covariate relation that come up in practice written out once each:
;
;   power, continuous       weight on CL and V1, albumin on CL
;   threshold               anti-drug antibody titre, only above the cutoff
;   categorical shift       a concomitant medication, as a fractional change
;   fixed allometry         weight on Q and V2, exponents not estimated
;
; Each relation sits between ;;; <NAME>-DEFINITION START/END markers, the
; layout PsN's scm writes, so a model found by stepwise covariate search can
; be read line for line against this one.
;
; Time is in days.

$INPUT ID TIME AMT RATE DV MDV EVID CMT WT ALB ADA CONMED
$DATA your_data.csv IGNORE=@

$SUBROUTINE ADVAN3 TRANS4

$PK
; Time after dose, for the table only; nothing in the model uses it.
IF (AMT.GT.0) TDOS = TIME
TAD = TIME - TDOS

; Missing covariates are coded -99 and replaced by the reference value, so
; the relation evaluates to exactly 1 and the subject is typical for that
; covariate. That is imputation at the median in all but name, and it is
; defensible only while few values are missing.
WTN = WT
IF (WT.EQ.-99) WTN = 70
ALBN = ALB
IF (ALB.EQ.-99) ALBN = 4

;;; CLWT-DEFINITION START
CLWT = (WTN/70)**THETA(5)
;;; CLWT-DEFINITION END

;;; CLALB-DEFINITION START
; Low albumin marks an inflammatory, protein-losing state in which IgG is
; cleared faster, so this exponent is expected to be negative.
CLALB = (ALBN/4)**THETA(6)
;;; CLALB-DEFINITION END

;;; CLADA-DEFINITION START
; Titres at or below the assay's positivity cutoff are treated as negative:
; below it the number is noise, not a smaller amount of antibody.
ADACUT = 20
CLADA  = 1
IF (ADA.GT.ADACUT) CLADA = (ADA/ADACUT)**THETA(7)
;;; CLADA-DEFINITION END

;;; CLCONMED-DEFINITION START
CLCONMED = 1                               ; CONMED=0, and missing
IF (CONMED.EQ.1) CLCONMED = 1 + THETA(8)
;;; CLCONMED-DEFINITION END

;;; CL-RELATION START
CLCOV = CLWT*CLALB*CLADA*CLCONMED
;;; CL-RELATION END

;;; V1WT-DEFINITION START
V1WT = (WTN/70)**THETA(9)
;;; V1WT-DEFINITION END

CL = THETA(1)*CLCOV*EXP(ETA(1))
V1 = THETA(2)*V1WT *EXP(ETA(2))
Q  = THETA(3)*(WTN/70)**0.75
V2 = THETA(4)*(WTN/70)

S1 = V1

$ERROR
; Log-transformed both sides: DV in the data is LOG(concentration) and the
; prediction is compared on the same scale. An additive error there is a
; proportional error on the original scale, but the prediction can never
; go negative and troughs that span orders of magnitude are weighted evenly.
;
; F is 0 before the first dose and LOG(0) would stop the run. Those records
; are doses or MDV=1, so IPRED there only has to be finite.
IPRED = 0
IF (F.GT.0) IPRED = LOG(F)
Y = IPRED + EPS(1)

$THETA
(0, 0.3)        ; 1 CL  (L/day) at 70 kg, albumin 4 g/dL, ADA negative
(0, 3.2)        ; 2 V1  (L)     at 70 kg
(0, 0.5)        ; 3 Q   (L/day) at 70 kg
(0, 2.5)        ; 4 V2  (L)     at 70 kg
(0, 0.75, 2)    ; 5 CL~WT  exponent
(-5, -1.0, 5)   ; 6 CL~ALB exponent
(-2, 0.2, 2)    ; 7 CL~ADA exponent, above the cutoff
(-0.9, -0.1, 2) ; 8 CL~CONMED fractional change
(0, 0.6, 2)     ; 9 V1~WT  exponent

$OMEGA
0.09            ; 1 IIV on CL (~30% CV)
0.04            ; 2 IIV on V1 (~20% CV)

$SIGMA
0.04            ; 1 additive on log scale (~20% proportional)

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=9999 SIG=3 PRINT=5 NOABORT
$COVARIANCE UNCONDITIONAL

; Split the way Xpose reads them: sdtab for diagnostics, patab for
; parameters and ETAs, cotab and catab for continuous and categorical
; covariates.
$TABLE ID TIME TAD DV IPRED PRED CWRES NPDE ONEHEADER NOPRINT FILE=sdtab_cov
$TABLE ID CL V1 Q V2 ETA(1) ETA(2) FIRSTONLY ONEHEADER NOPRINT FILE=patab_cov
$TABLE ID WT ALB ADA FIRSTONLY ONEHEADER NOPRINT FILE=cotab_cov
$TABLE ID CONMED FIRSTONLY ONEHEADER NOPRINT FILE=catab_cov
