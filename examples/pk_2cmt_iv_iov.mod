$PROBLEM Two-compartment IV infusion with inter-occasion variability on CL
; Template: no dataset ships with this stream. $INPUT lists what it expects.
;
; A subject's clearance is not one number. With long-term dosing it moves
; from one dosing interval to the next -- disease activity, albumin,
; antibodies -- and a model with IIV alone has to explain that movement
; somewhere. It ends up in an inflated OMEGA, or in EBEs that chase the
; latest trough. Both matter when the model is used to individualise the
; next dose, which is exactly what the post hoc CL is used for.
;
; Here each occasion gets its own ETA on CL, and all of them share one
; variance through $OMEGA BLOCK(1) SAME. There is no reason for the third
; occasion to vary more than the second, and a separate variance for each
; would not be estimable from one or two samples per occasion.
;
; Data layout: OCC numbers the occasions 1..4. The trough drawn just before
; a dose belongs to the occasion that dose ends, not the one it starts.
; An occasion without observations contributes nothing, and its ETA stays
; at zero. Time is in days.

$INPUT ID TIME AMT RATE DV MDV EVID CMT OCC WT
$DATA your_data.csv IGNORE=@

$SUBROUTINE ADVAN3 TRANS4

$PK
IOVCL = 0
IF (OCC.EQ.1) IOVCL = ETA(3)
IF (OCC.EQ.2) IOVCL = ETA(4)
IF (OCC.EQ.3) IOVCL = ETA(5)
IF (OCC.EQ.4) IOVCL = ETA(6)

; Allometry on a 70 kg reference with the conventional fixed exponents.
WTCL = (WT/70)**0.75
WTV  = WT/70

; The subject's CL on this occasion. The between-subject part, ETA(1), is
; what carries forward to a future occasion; IOVCL does not.
CL = THETA(1)*WTCL*EXP(ETA(1) + IOVCL)
V1 = THETA(2)*WTV *EXP(ETA(2))
Q  = THETA(3)*WTCL
V2 = THETA(4)*WTV

S1 = V1

$ERROR
IPRED = F
Y     = IPRED*(1 + EPS(1))

$THETA
(0, 0.3)       ; 1 CL (L/day) at 70 kg
(0, 3.2)       ; 2 V1 (L)     at 70 kg
(0, 0.5)       ; 3 Q  (L/day) at 70 kg
(0, 2.5)       ; 4 V2 (L)     at 70 kg

$OMEGA
0.09           ; 1 IIV on CL
0.04           ; 2 IIV on V1
$OMEGA BLOCK(1)
0.04           ; 3 IOV on CL, occasion 1
$OMEGA BLOCK(1) SAME   ; 4 occasion 2
$OMEGA BLOCK(1) SAME   ; 5 occasion 3
$OMEGA BLOCK(1) SAME   ; 6 occasion 4

$SIGMA
0.04           ; 1 proportional error (~20%)

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=9999 SIG=3 PRINT=5 NOABORT
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME OCC DV IPRED PRED CWRES CL V1 ETA(1) ETA(2) ETA(3) ETA(4)
       ETA(5) ETA(6) ONEHEADER NOPRINT FILE=sdtab_iov
