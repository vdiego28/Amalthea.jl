# Native-Rust Backend Port — Governing Math

> Status: design doc for the phased port. No phase is implemented yet.
> Companion docs: [ARCHITECTURE.md](ARCHITECTURE.md), [TESTING.md](TESTING.md),
> [PORT_LOG.md](PORT_LOG.md).
>
> **`luna-rust/README.md` is the authoritative equation reference** for the
> already-ported leaf kernels (Zeisberger/Marcatili dispersion, QDHT, PPT
> ionization, Raman ADE, Gauss-Legendre integrating factor). This file adds the
> **loop-level** math the native port must reproduce: the interaction-picture
> stepper, the per-geometry RHS, the FFT conventions, and the response functions.

## 1. The propagated quantity

Luna integrates the unidirectional pulse-propagation equation (UPPE), or its
envelope reduction (GNLSE), in the spectral domain. The state vector `y` is the
frequency-domain field:

- **RealGrid** (carrier-resolved): `y = Eω`, the real-input FFT (`rfft`) of the
  real field `E(t)`. Length `Nω = Nt÷2 + 1`.
- **EnvGrid** (envelope): `y = Eω`, the full complex FFT (`fft`) of the complex
  envelope. Length `Nω = Nt`.

The PDE in `z` has the schematic form

```
∂z Eω = L(ω, z) · Eω  +  N[Eω](z)
```

where `L` is the **diagonal linear operator** (dispersion β(ω) + loss + a moving
reference frame) and `N` is the **nonlinear RHS** (Kerr, plasma, Raman, …),
assembled in the time/space domain and transformed back to ω.

## 2. Interaction-picture Dormand-Prince 5(4)

The stiffness lives in `L`. Luna removes it analytically by integrating in the
**interaction picture**: define `ybar = exp(-L·(z-z₀))·Eω`. The linear part is
then exact and the adaptive RK integrates only the (smooth) nonlinear part.

In code this is the **preconditioned RHS** `fbar!` (`src/RK45.jl:309-319`):

```
fbar!(out, ybar, t1, t2):
    y   ← ybar
    prop!(y,   t1, t2)        # forward:  y *= exp(L·(t2-t1))
    f!(out, y, t2)            # nonlinear RHS at t2
    prop!(out, t1, t2, bwd=true)   # back:  out *= exp(L·(t1-t2))
```

i.e. **`fbar! = prop_back ∘ f! ∘ prop_forward`**. The propagator is a pure
diagonal phase multiply (`src/RK45.jl:281-306`):

```
prop!(y, t1, t2):       y .*= exp.(L .* (t2 - t1))     # const-linop case
```

For a z-dependent `L`, `prop!` integrates `L` to the later time and caches it
(`src/RK45.jl:294-306`). Native code must reproduce the same "evaluate L at the
later time, reuse if unchanged" memoization to stay bit-comparable.

### 2.1 Tableau, FSAL, error estimate
Standard Dormand-Prince 5(4) (7 stages, the coefficients live in
`src/dopri.jl`). Properties the native loop must preserve exactly:

- **FSAL** (First Same As Last): stage 7 of the accepted step is stage 1 of the
  next — one RHS evaluation per step is reused, not recomputed.
- **Embedded 4th-order** solution gives the error estimate `yerr = y5 - y4`.
- **Dense output**: 4th-order interpolation (`src/RK45.jl:260-278`) using the
  stage derivatives `ks[1..7]` and a cubic-in-σ coefficient table, so `saveN`
  output points are produced without extra RHS evals.

### 2.2 Lund PI step controller
The accepted/rejected decision uses a normalized error `err = norm(yerr, y, yn,
rtol, atol)`; the next step size uses a proportional-integral law (Lund / Söderlind):

```
dt_new = dt · safety · err^(-α) · err_prev^(β)
```

with the standard DP5 exponents. Bit-comparability of the *trajectory* requires
the identical `err` reduction (`weaknorm` / `maxnorm`, `src/RK45.jl:321+`),
the same `safety`, and the same accept/reject threshold — otherwise the two
paths take different step sequences and diverge within the tolerance band even
when each RHS is bit-identical. See [TESTING.md](TESTING.md) §nondeterminism.

## 3. The nonlinear RHS `f!` per geometry

`f!` is a `Trans*` functor (`src/NonlinearRHS.jl`). The shared shape:

```
f!(out, Eω, z):
    1. to_time!(Et, Eω)          # ω → t  (inverse FFT, possibly oversampled grid FTo)
    2. fill Pt from responses(Et)  # Kerr / plasma / Raman, in the time (and space) domain
    3. to_freq!(Pω, Pt)          # t → ω  (forward FFT)
    4. out ← prefactor(ω, z) .* Pω   # geometry normalization + i·ω/(2ε₀c) type factor
```

The geometries differ in steps 1/3 (which transform) and step 4 (normalization):

### 3.1 Mode-averaged — `TransModeAvg` (`src/NonlinearRHS.jl:531`) — Phase 1 (RealGrid) / Phase 2 (EnvGrid)
No spatial transform. Pure 1-D FFT pair plus an effective-area scaling
(`norm_mode_average`). The simplest geometry; the reference implementation for
the resident-field architecture. Phase 1 ported the RealGrid (r2c/c2r) Kerr
path (`rhs_mode_avg_real`); Phase 2 extended it to the EnvGrid (c2c) envelope
path (`rhs_mode_avg_env`), which needs the 3/4 SVEA prefactor on `Kerr_env`
that the carrier-resolved Kerr term doesn't (§5.1).

### 3.2 Radial — `TransRadial` (`src/NonlinearRHS.jl:663`) — Phase 3
The field carries a radial axis. Each RHS adds a **QDHT** (`mul!`/`ldiv!` with
the Hankel `T` matrix) on the `r ↔ k` axis around the time transform. The QDHT
is already in Rust (`LUNA_USE_RUST_QDHT`); Phase 3 makes its `T` matrix and
scratch resident in `NativeSim` instead of re-marshalling per call.

### 3.3 Modal — `TransModal` (`src/NonlinearRHS.jl:421`) — Phase 5
The hardest. The nonlinear polarization is projected onto each waveguide mode by
an **overlap integral** evaluated with adaptive cubature (`pointcalc!`,
`src/NonlinearRHS.jl:363-399`; `Erω_to_Prω!`, `:401-419`). Steps per RHS:
mode-field evaluation at cubature nodes → `to_space!` (sum modes → physical
field) → responses → re-project (`Erω_to_Prω!`) → modal normalization. Requires
a Rust adaptive-cubature routine; mode dispersion is already Rust.

> **Concurrency note (do not reintroduce):** the `TransModal` integration loop
> writes shared struct buffers (`t.Erω`, `t.Prω`, `t.Prmω`); it must stay
> **sequential**. A prior `Threads.@threads` parallelization caused a data race
> → corrupted overlaps → every RK step rejected. (BACKLOG "Done" item.) The Rust
> port must keep per-node state thread-local if it ever parallelizes.

### 3.4 Free-space — `TransFree` (`src/NonlinearRHS.jl:826`) — Phase 6
2-D Cartesian spatial axis: a **3-D FFT** (2 spatial + 1 time). Needs 3-D FFTW
plans resident in `NativeSim`. Stays on Julia until Phase 6.

## 4. FFT conventions (must match FFTW.jl exactly)

| Grid | forward (t→ω) | inverse (ω→t) | length |
|------|---------------|---------------|--------|
| RealGrid | `rfft` | `irfft` | `Nt → Nt÷2+1` |
| EnvGrid | `fft` | `ifft` | `Nt → Nt` |

- Normalization is applied explicitly via `copy_scale!` / the `prefactor`, **not**
  folded into the plan. The native side must apply the same scalar at the same
  point so rounding matches.
- The **oversampled grid** `FTo` (`grid.FTo`, larger `Nt`) is used for the
  time-domain product to avoid aliasing of the cubic Kerr term; the native RHS
  needs a second pair of plans for it and the same zero-pad/truncate convention.
- Binding the **same FFTW library** (ARCHITECTURE §4.1) is what makes these
  transforms bit-identical rather than merely close.

## 5. Response functions (time/space domain, step 2)

All live in `src/Nonlinear.jl`. The native port reproduces each in Rust.

### 5.1 Kerr (`src/Nonlinear.jl:81-157`)
Instantaneous χ⁽³⁾. Variants: scalar `E³`; vectorial (polarization-resolved);
third-harmonic (`thg`) vs. Hilbert-envelope (no-thg) forms. For EnvGrid a
`Kerr_env` variant uses `|E|²E`. Coefficient: `χ₃ · ε₀` folded into one scalar.
**Gotcha (bit us in Phase 2a):** `Kerr_env` carries an extra **3/4 SVEA
prefactor** vs. the carrier-resolved (no-thg) form — `rhs_mode_avg_env` must
multiply `kerr_fac` by `0.75`. Missing this passed compilation but showed up
as a single-step error of ~9.5e-6 (not machine epsilon), not a crash.

### 5.2 Plasma — `PlasmaCumtrapz` (`src/Nonlinear.jl:161-250`) — Phase 2, ported
The expensive nonlinearity. Per RHS:
1. ionization rate `W(|E|)` from the PPT LUT (already Rust, `LUNA_USE_RUST_IONISATION`);
2. **three cumulative trapezoidal integrals** (`cumtrapz`): free-electron density
   `ρ(t)` from the rate; the loss/heating current; the polarization current;
3. assemble the plasma current `J(t)` and add to `Pt`.
Ported in Phase 2 as `rhs_plasma` (`luna-rust/src/native.rs`, `cumtrapz_slice_f64`
+ `native_set_plasma_params`), gated on `LUNA_USE_RUST_IONISATION=1` (the native
path silently falls back to Julia plasma physics if the ionization handle isn't
wired — see PORT_LOG 2026-06-30 "Wire plasma" note).

### 5.3 Raman (`src/Nonlinear.jl:357-431`)
Delayed χ⁽³⁾. The carrier-field SDO path (`RamanPolarField`) is already Rust
(`LUNA_USE_RUST_RAMAN`): a single-damped-oscillator ADE stepped by an exact
exponential integrator (`x_{n+1} = A·x_n + B0·Iₙ + B1·I_{n+1}`), O(Nt) and
allocation-free, vs. Julia's FFT convolution. Phase 4 makes that solver resident
in the RHS. The envelope path (`RamanPolarEnv`) and intermediate-broadening
(Gaussian-damped) responses stay Julia.

## 6. Linear operator `L(ω, z)` (Phase 7)

`L` is built in `src/LinearOps.jl`:
- **Constant** (`make_const_linop`): z-independent; built once at setup, copied
  into `NativeSim` as a flat array. Phases 1–6 use this path — cheap, left on
  Julia for assembly, applied in Rust.
- **z-dependent** (`make_linop`, free-space/radial/modal, `LinearOps.jl:77,185,337`):
  rebuilt per step (e.g. changing gas pressure, bending). Phase 7 ports the
  `_fill_linop` assembly so `prop!` never returns to Julia.

`L` includes: the propagation constant β(ω) (from the Rust dispersion kernels),
the loss `α(ω)/2`, and the subtraction of the reference `β₀ + β₁·(ω-ω₀)` that
defines the co-moving frame.

## 7. QDHT normalization caveat (already resolved, preserve in port)

Julia's `Hankel.QDHT` and the Rust `Qdht` use **different T-matrix
normalizations**. The wired path resolves this by passing **Julia's T matrix**
into Rust (transposed to row-major at `init_qdht_ffi`) and using that exact
bit-pattern at runtime — agreement ~1e-13 (BLAS vs. sequential summation order).
The resident-QDHT port (Phase 3) must keep using Julia's T matrix, **not**
recompute T from the Rust convention, or the normalization will silently differ.
