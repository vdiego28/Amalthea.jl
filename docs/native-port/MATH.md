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

### 3.2 Radial — `TransRadial` (`src/NonlinearRHS.jl:663`) — Phase 3, ported
The field carries a radial axis: buffers are `(n_time, n_r)`, and each RHS
wraps the time transform with a **QDHT** (`mul!`/`ldiv!` with the Hankel `T`
matrix) on the `r ↔ k` axis:
```
to_time!(Eto, Eω)        # ω → t, per r-column (batched dim-1 FFT in Julia)
_qdht_ldiv!(Eto, QDHT)   # k → r
Et_to_Pt!(Pto, Eto, ...) # Kerr, pointwise in (t, r) — same E³ formula as Phase 1
Pto .*= towin
_qdht_mul!(Pto, QDHT)    # r → k
to_freq!(nl, Pto)        # t → ω, per r-column
nl .*= ωwin .* (-im·ω) ./ (2 .* normfun(z))
```

**Reuse, don't reinvent — the resident QDHT is already built.** `ffi.rs`'s
`QdhtFfiHandle` (used by the existing `LUNA_USE_RUST_QDHT` FFI wiring) already
holds Julia's `T` matrix in the correct convention and exposes
`apply_real(data, n_time, scale)` / `apply_cplx(data, n_time, scale)` as plain
Rust methods operating on a column-major `(n_time, n_r)` buffer — not just an
FFI entry point. Phase 3 stores a `QdhtFfiHandle` instance directly inside
`NativeSim` (built once in `native_set_radial_params` from Julia's `HT.T`,
`HT.N`, `HT.scaleRK`) and calls `apply_real`/`apply_cplx` directly from
`rhs_radial` — no FFI round-trip per RHS, no reimplementation of the QDHT
math. Do **not** use `diffraction::Qdht` (a different Rust-native struct with
its own T-matrix convention) — it does not match Julia's normalization
(CLAUDE.md gotcha).

**FFT: loop the existing rank-1 plan over columns, don't add `plan_many`
yet.** Julia's `plan_rfft(xt, 1)` is a batched ("many") transform over the
`n_r` columns; the native side reuses the existing single-vector
`ComplexFft1d`/`RealFft1d` (`fftw.rs`) called `n_r` times in a loop instead
of adding a new batched FFTW plan type. This is simpler and the established
~1e-13 tolerance tier is the safety net — only add `plan_many` if single-step
equivalence comes in worse than that tier for a reason traced to the FFT
step specifically.

**Normalization: precompute one complex `(n_ω, n_r)` array, don't port
`norm_radial`'s Bessel/k_z math into Rust.** The tail of `TransRadial`'s
`__call__` is `nl .*= ωwin .* (-im·ω) ./ (2 .* normfun(z))`. Precompute
`M[iω,ir] = ωwin[iω] * (-im*ω[iω]) / (2 * normfun_array[iω,ir])` once in
Julia and pass it to Rust as a flat array; `rhs_radial` just does
`ks[idx][i] *= M[i]` elementwise on the flattened buffer. This captures the
`ω=0`/evanescent (`βsq≤0`) masking and `ωwin` for free, and needs zero of
`norm_radial`'s math ported. **This is only valid for a z-invariant
`normfun`** (i.e. `const_norm_radial`, constant-medium gas cell) — exactly
the same constant-medium restriction Phases 1-6 already carry for the linop
(§6, ARCHITECTURE §4.4). A z-dependent `normfun` (tapered fiber, pressure
gradient) is out of scope until Phase 7 alongside the z-dependent linop.

**The linop is 2-D for radial**, `(n_ω, n_r)` (`LinearOps.make_const_linop`
with a `Hankel.QDHT` grid — `k_z = √((nω/c)² − k_r²)` depends on both `ω` and
`k_r`), not the `Vector{ComplexF64}` Phases 1-2 assumed. `apply_prop` stays
purely elementwise on the flattened column-major buffer, so no Rust change is
needed there — but the Julia-side native-path guard in `RK45.solve_precon`
(`src/RK45.jl:27`, currently `linop isa Vector{ComplexF64}`) must broaden to
also accept `Matrix{ComplexF64}`, or radial never reaches the native
constructor at all.

**Invariant (state explicitly, don't let it drift):** every 2-D buffer touched
by `rhs_radial` — `Eto`/`Pto` (time domain), `Eωo`/`Pωo` (oversampled
frequency), the flattened `ks[i]`/`field` buffers — is **column-major
`(n_time, n_r)`**, i.e. one r-column is `n_time` contiguous elements. This is
what `QdhtFfiHandle::apply_real/apply_cplx` expects, and what makes each
column contiguous for the looped rank-1 FFT. If any buffer ever disagrees with
this layout, the result is silent garbage, not a crash.

**Tolerance heads-up:** the QDHT carries a ~1e-13 BLAS-vs-Rayon summation-order
floor (CLAUDE.md), applied *twice* per RHS call (`ldiv!` + `mul!`) inside the
stepper now. Radial's single-step tier will land around low-1e-13 to ~1e-12,
not the cleaner sub-1e-13 Phase 1/2 FFTW-only tier — that is the QDHT floor,
not a bug; do not chase it lower. Apply the Phase 2 lesson from the start:
build the full-solve equivalence test with `max_dt=min_dt=dt` (fixed step
size) so it measures state accumulation, not adaptive-path agreement.

**Scope implemented:** RealGrid + scalar Kerr only (`Nonlinear.Kerr_field`,
the same `E³` formula already in `rhs_mode_avg_real`, applied per r-column),
`shotnoise=false` (skips the `Et_noise` branch). EnvGrid-radial and
plasma-radial are deferred follow-ups, mirroring how Phase 1 preceded Phase
2's EnvGrid/plasma extension.

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
