$PROBLEM Two-compartment IV infusion, allometric weight, IIV on CL and V1
; Data: data/pk_2cmt_iv.csv, simulated by nmlib.simulate.simulate_pk_2cmt
; The workhorse structural PK model: everything downstream in this library
; that needs an exposure driver assumes something of this shape.

$INPUT ID TIME AMT RATE DV MDV EVID CMT WT
$DATA ../data/pk_2cmt_iv.csv IGNORE=@

$SUBROUTINE ADVAN3 TRANS4

$PK
; Allometry on a 70 kg reference. Exponents are fixed at the conventional
; 0.75 for flows and 1 for volumes; estimate them instead only if the data
; can carry it, which a single study rarely can.
WTCL = (WT/70)**0.75
WTV  = (WT/70)**1

CL = THETA(1)*WTCL*EXP(ETA(1))
V1 = THETA(2)*WTV *EXP(ETA(2))
Q  = THETA(3)*WTCL
V2 = THETA(4)*WTV

S1 = V1

$ERROR
IPRED = F
Y     = IPRED*(1 + EPS(1))

$THETA
(0, 5.0)    ; 1 CL  (L/h) at 70 kg
(0, 15.0)   ; 2 V1  (L)   at 70 kg
(0, 3.0)    ; 3 Q   (L/h) at 70 kg
(0, 25.0)   ; 4 V2  (L)   at 70 kg

$OMEGA
0.09        ; 1 IIV on CL  (var on log scale, ~30% CV)
0.06        ; 2 IIV on V1  (~25% CV)

$SIGMA
0.0225      ; 1 proportional residual error (~15%)

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=9999 SIG=3 PRINT=5 NOABORT
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV IPRED PRED CWRES CL V1 Q V2 WT ETA(1) ETA(2)
       ONEHEADER NOPRINT FILE=sdtab_pk
