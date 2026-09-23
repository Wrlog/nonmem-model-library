$PROBLEM External validation: a published model, every parameter fixed, run on new data
; Template: no dataset ships with this stream. $INPUT lists what it expects.
;
; Nothing is estimated. MAXEVAL=0 skips the population search, and NONMEM
; goes straight to the post hoc step: each subject's ETAs are the mode of
; their posterior given the fixed population values. What comes out is the
; new data's objective function under the old model, PRED and IPRED, CWRES
; and NPDE -- the material for a prediction-error table and an external VPC.
;
; The structure and every value below must be copied exactly from the
; publication, including its covariate reference values and its error
; model. A validation of a model that was quietly re-parameterised on the
; way in is a validation of a different model. The values here are
; placeholders.
;
; To draw full individual profiles rather than points at the sampling
; times, add EVID=2 records on a time grid. They carry no observation, so
; the objective function and the ETAs are unchanged; they only add rows to
; the table. Time is in days.

; Second derivatives with respect to ETA are only used by LAPLACIAN.
; Skipping them makes NM-TRAN generate less code.
$ABBR DERIV2=NO

$INPUT ID TIME AMT RATE DV MDV EVID CMT WT ALB
$DATA your_data.csv IGNORE=@

$SUBROUTINE ADVAN3 TRANS4

$PK
IF (AMT.GT.0) TDOS = TIME
TAD = TIME - TDOS

WTN = WT
IF (WT.EQ.-99) WTN = 70
ALBN = ALB
IF (ALB.EQ.-99) ALBN = 4

CL = THETA(1)*(WTN/70)**THETA(5)*(ALBN/4)**THETA(6)*EXP(ETA(1))
V1 = THETA(2)*(WTN/70)**THETA(7)*EXP(ETA(2))
Q  = THETA(3)*(WTN/70)**0.75
V2 = THETA(4)*(WTN/70)

S1 = V1

$ERROR
; Same error model as the publication: here, log-transformed both sides,
; so DV is LOG(concentration).
IPRED = 0
IF (F.GT.0) IPRED = LOG(F)
Y = IPRED + EPS(1)

$THETA
(0, 0.3)  FIX   ; 1 CL  (L/day) at 70 kg, albumin 4 g/dL
(0, 3.2)  FIX   ; 2 V1  (L)     at 70 kg
(0, 0.5)  FIX   ; 3 Q   (L/day) at 70 kg
(0, 2.5)  FIX   ; 4 V2  (L)     at 70 kg
0.75      FIX   ; 5 CL~WT  exponent
-1.0      FIX   ; 6 CL~ALB exponent
0.6       FIX   ; 7 V1~WT  exponent

$OMEGA
0.09 FIX        ; 1 IIV on CL
0.04 FIX        ; 2 IIV on V1

$SIGMA
0.04 FIX        ; 1 additive on log scale

; METHOD=1 INTERACTION although nothing moves: the post hoc ETAs and CWRES
; are computed under the method named here, so it should be the one the
; model was developed with.
$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 SIG=3 PRINT=5

; No $COVARIANCE: there is nothing estimated for it to describe.

$TABLE ID TIME TAD EVID DV IPRED PRED CWRES NPDE CL V1 ETA(1) ETA(2)
       ONEHEADER NOPRINT FILE=sdtab_extval
