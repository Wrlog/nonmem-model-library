$PROBLEM Two-compartment oral PK with transit-compartment absorption
; Template: no dataset ships with this stream. $INPUT lists what it expects.
;
; Savic et al. 2007. A chain of transit compartments produces the delayed,
; gradual onset of absorption that a lag time can only approximate with a
; step. Rather than writing out N compartments, the chain is replaced by
; the analytical rate at which drug leaves the N-th of them,
;
;   BIO*DOSE*KTR*(KTR*t)^N*exp(-KTR*t)/N!,   KTR = (N+1)/MTT,
;
; fed into the absorption compartment. N no longer has to be an integer and
; is estimated like any other parameter.
;
; Single dose per subject. t here is time since that dose, and the input
; from an earlier dose is not carried forward when a new one is given; for
; repeated dosing, write the transit compartments out as ODEs instead.

$INPUT ID TIME AMT DV MDV EVID CMT
$DATA your_data.csv IGNORE=@

; TOL is the number of accurate digits the ODE solver aims for. Values of
; 2 or 3 are fast and wrong often enough to move the estimates; 9 is safe.
$SUBROUTINE ADVAN13 TOL=9

$MODEL
COMP=(DEPOT)               ; absorption compartment, fed by the transit chain
COMP=(CENTRAL, DEFOBS)
COMP=(PERIPH)

$PK
IF (NEWIND.NE.2) THEN      ; first record of a subject
  PODO = 0
  TDOS = 0
ENDIF
IF (AMT.GT.0.AND.CMT.EQ.1) THEN
  PODO = AMT
  TDOS = TIME
ENDIF

; The dose record only tells the model how much was given and when. The
; input function in $DES is what delivers it, so the bolus itself must not
; enter DEPOT: leave F1 at its default of 1 and every dose is absorbed twice.
F1 = 0

; KA and the transit chain both shape the absorption phase, and they trade
; off against each other. When KA is not much slower than KTR, expect KA
; and its ETA to be poorly determined, and NONMEM to say that "problems
; occurred" even though the fit is good. If the covariance step agrees,
; remove the ETA on KA before anything else.
CL  = THETA(1)*EXP(ETA(1))
V2  = THETA(2)*EXP(ETA(2))
Q   = THETA(3)
V3  = THETA(4)
KA  = THETA(5)*EXP(ETA(3))
MTT = THETA(6)*EXP(ETA(4))
NN  = THETA(7)             ; number of transit compartments, need not be whole
BIO = 1                    ; absolute F is not identifiable from oral data alone

KTR = (NN+1)/MTT
; log(NN!) by Stirling's approximation, which is what lets NN be non-integer.
LNFAC = LOG(2.5066) + (NN+0.5)*LOG(NN) - NN

K20 = CL/V2
K23 = Q/V2
K32 = Q/V3
S2  = V2

$DES
; Written on the log scale because (KTR*t)^NN and NN! overflow long before
; their ratio does. The 1E-10 terms keep LOG finite at t = 0 and before the
; dose, where the input is then effectively zero.
TAD = T - TDOS
IF (TAD.LT.0) TAD = 0
LNSCALE = LOG(BIO*PODO + 1E-10) + LOG(KTR) - LNFAC
INPT    = EXP(LNSCALE + NN*LOG(KTR*TAD + 1E-10) - KTR*TAD)

DADT(1) = INPT - KA*A(1)
DADT(2) = KA*A(1) - (K20 + K23)*A(2) + K32*A(3)
DADT(3) = K23*A(2) - K32*A(3)

$ERROR
IPRED = A(2)/V2
W     = SQRT((THETA(8)*IPRED)**2 + THETA(9)**2)
IWRES = (DV - IPRED)/W
Y     = IPRED + W*EPS(1)

$THETA
(0, 10.0)      ; 1 CL  (L/h)
(0, 30.0)      ; 2 V2  (L), central
(0, 5.0)       ; 3 Q   (L/h)
(0, 50.0)      ; 4 V3  (L), peripheral
(0, 1.5)       ; 5 KA  (1/h)
(0, 1.0)       ; 6 MTT (h), mean transit time
(0.5, 4, 30)   ; 7 NN  number of transit compartments
(0, 0.15)      ; 8 proportional error SD
(0, 0.05)      ; 9 additive error SD (mg/L)

$OMEGA
0.09           ; 1 IIV on CL
0.09           ; 2 IIV on V2
0.16           ; 3 IIV on KA
0.16           ; 4 IIV on MTT

$SIGMA
1 FIX

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=9999 SIG=3 PRINT=5 NOABORT
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV IPRED PRED CWRES IWRES CL V2 KA MTT NN ETA(1) ETA(2) ETA(3)
       ETA(4) ONEHEADER NOPRINT FILE=sdtab_transit
