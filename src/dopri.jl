# Butcher Tableau
const B = [[1/5],
     [3/40, 9/40],
     [44/45, -56/15, 32/9],
     [19372/6561, -25360/2187, 64448/6561, -212/729],
     [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
     [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
     ]

# Step size fractions
const nodes = [1/5, 3/10, 4/5, 8/9, 1, 1]

# Weights for 5th order method
const b5 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
# Weights for 4th order method
const b4 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
# Error estimate
const errest = b5 .- b4

#Interpolation coefficients
const interpC = hcat(
[1,       -183/64,      37/12,       -145/128],
[0,       0,             0,             0],
[0,        1500/371,     -1000/159,   1000/371],
[0,       -125/32,      125/12,      -375/64],
[0,       9477/3392,    -729/106,    25515/6784],
[0,       -11/7,        11/3,        -55/28],
[0,       3/2,          -4,           5/2],
)

# ── Order-5 continuous extension (BACKLOG.md S5 item 3) ─────────────────────
# `interpC` above is provably order 4 (Hairer & Wanner, Solving ODEs I, II.6 —
# the "free" 7-stage interpolant, no extra evaluations, same one MATLAB's
# ntrp45/scipy use). Reaching order 5 needs 2 extra function evaluations per
# interpolated step (Shampine 1986); the extra-stage tableau + basis
# polynomials below are the Calvo, Montijano & Rández construction ("A
# fifth-order interpolant for the Dormand and Prince Runge-Kutta method",
# J. Comput. Appl. Math. 29 (1990) 91-100), transcribed from the
# PyNumAl/Python-Numerical-Analysis public tableau collection ("Dormand-
# Prince RK5(4) Pair.txt", https://github.com/PyNumAl/Python-Numerical-Analysis,
# which cites the same paper and PDF at
# https://core.ac.uk/download/pdf/81164751.pdf — the original PDF itself was
# not independently fetchable at the time of writing, see
# docs/dev/native-port/portlog-inbox/dense-order5.md for the sourcing dead-ends).
# Correctness was NOT taken on faith from the transcription: verified two
# ways before use — (a) exact-rational check that `interpC5` reduces to `b5`
# at θ=1 (both extra stages must contribute 0 to the accepted-step value);
# (b) numeric convergence check that the two extra stages combine to a
# genuinely 6th-order-locally-accurate value, and that the full interpolant
# achieves O(h^6) local / O(h^5) global dense-output error vs. `interpC`'s
# O(h^5) local / O(h^4) global — see the portlog-inbox entry for the measured
# orders. Used by both `interpolate(s::PreconStepper, ti)` and
# `interpolate(s::RustNativeStepper, ti)` via the shared `interpC5_weights`/
# `_dp5_extra_stages!` helpers in RK45.jl, so the two steppers cannot
# silently diverge on interpolant math.
const dp5_extra_c = 2/5   # both extra stages sit at the same interior node
const dp5_extra_a7 = [-24018683/8152320000, 25144/43425, -76360723/337557000,
                       349808429/2445696000, -13643731773/144024320000, 1/20,
                       -12268567/254760000]
const dp5_extra_a8 = [2104901/23010000, 0, 54324224/106708875, 134233/2301000,
                       -13268529/406510000, 26972/2013375, -6324/479375, -1737/7670]

# interpC5[power, stage]: coefficient of θ^power (power=1..5) for stage
# (stage=1..9: the original 7 DP5 stages, then the 2 extra stages above).
const interpC5 = hcat(
[1,        -1809809/441792,    1150189/147264,    -2025025/294528,    995255/441792],
[0,        0,                  0,                  0,                  0],
[0,        40963600/2561013,  -14677200/284557,    49017100/853671,   -54769600/2561013],
[0,        259225/73632,      -1161175/73632,       1189575/49088,     -834475/73632],
[0,       -4817961/2601664,    21163599/2601664,   -64133775/5203328,   14882535/2601664],
[0,        36542/48321,       -53416/16107,          323345/64428,      -112475/48321],
[0,       -1901/2301,          8771/2301,            -14140/2301,         7270/2301],
[0,       -120625/18408,       120625/6136,          -120625/6136,        120625/18408],
[0,       -125/18,              125/4,                -125/3,              625/36],
)