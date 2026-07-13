# Native-Rust Backend Port — Governing Math

> Status: design doc for the phased port. No phase is implemented yet.
> Companion docs: [ARCHITECTURE.md](ARCHITECTURE.md), [TESTING.md](TESTING.md),
> [PORT_LOG.md](PORT_LOG.md).
>
> **`amalthea/README.md` is the authoritative equation reference** for the
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
  output points are produced without extra RHS evals. **This is a hard
  requirement, not an optional accuracy nicety**: Phases 0-7's
  `RustNativeStepper` used *linear* interpolation between accepted steps as a
  stopgap (no per-step FFI round-trip needed) — invisible while every
  native-specific test drove the stepper via raw `solve()`/`step!()` calls
  with no dense-output sampling, but responsible for nearly every
  general-purpose test failure once Phase 8 made native the default (any
  `saveN`/`MemoryOutput` config resamples between steps). Fixed in Phase 8 by
  exporting the 7 resident stages (`get_ks_stage`) and porting the same
  `interpC` quartic formula Julia-side — see PORT_LOG's Phase 8 entry.

### 2.1a Per-step external field mutation must be pushed back to Rust (Phase 8)
`Luna.run`'s `stepfun` callback (grid frequency/time windowing) mutates the
field **after** each accepted step, and for `PreconStepper` that mutation is
free (it's the same live array used as the next step's starting state). A
resident-field design has no such freebie: `RustNativeStepper`'s `native_step`
starts every call from Rust's *own* internally-tracked field, not from
whatever Julia last wrote into the FFI-shared buffer. Any code path that
mutates the field between `step!` calls (currently: `stepfun`'s windowing)
**must** be followed by a call that pushes the change back into the resident
state, or it is silently discarded — this bit Phase 8 badly (see PORT_LOG).
The fix is `_native_field_resync!(s)` (`src/RK45.jl`, a no-op for every
stepper except `RustNativeStepper`), called right after `stepfun` in the
generic `solve(s, tmax; stepfun, ...)` loop, via a new `native_resync_field`
FFI. It deliberately does **not** recompute the FSAL stage-0 RHS (unlike the
construction-time `set_field`, which correctly does, for the no-history
initial-condition case) — Julia's own `PreconStepper` doesn't re-evaluate the
nonlinear RHS after windowing either, it only re-propagates the FSAL-carried
last stage linearly into the new frame, and matching that (not "improving" on
it) is what reproduces Julia's number.

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
`QdhtFfiHandle` (used by the existing `AMALTHEA_USE_RUST_QDHT` FFI wiring) already
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

### 3.3 Modal — `TransModal` (`src/NonlinearRHS.jl:421`) — Phase 5, ported (narrow scope)
The nonlinear polarization is projected onto each waveguide mode by an
**overlap integral** evaluated with adaptive cubature (`pointcalc!`,
`src/NonlinearRHS.jl:363-399`; `Erω_to_Prω!`, `:401-419`). Per cubature node
`(r,θ)`: mode-field synthesis (`to_space!`: sum `Emω` weighted by each mode's
`Exy(r,θ,z)` → physical field `Erω`) → `to_time!` → nonlinear response
(`Et_to_Pt!`) → `towin` → `to_freq!` → `ωwin`/`norm!` → re-project onto modes
(`Prω · transpose(Ems)`) → scale by the polar Jacobian `2π·r` → return to the
cubature routine as one component of the vector integrand.

**Reuse, don't reinvent — bind the same C `libcubature`, don't write an
adaptive-cubature algorithm.** `Cubature.jl` is a thin `ccall` wrapper around
Steven Johnson's C library (`Cubature_jll`, symbols `hcubature_v`/
`pcubature_v`), not a pure-Julia reimplementation — confirmed via
`Cubature.Cubature_jll.libcubature` (an artifact path) and
`nm -D libcubature.so` (exports `hcubature_v`/`pcubature_v`/`hcubature`/
`pcubature`). This is exactly `FFTW.FFTW_jll.libfftw3`'s shape: `native.rs`
`dlopen`s the identical binary Julia calls (path passed in from Julia, same
`Library` loader pattern as `fftw.rs`), and Rust's per-node evaluator is
registered as the `integrand_v` C callback
(`int(unsigned ndim, size_t npt, const double *x, void *fdata, unsigned fdim, double *fval)`).
**Reimplementing the adaptive subdivision algorithm was rejected outright**:
region-splitting decisions depend on an FP-summation-order-sensitive error
estimate exactly like the RK45 step controller (§2.2) — a reimplementation
would face the same adaptive-path divergence that took real debugging effort
in Phases 1-2 (TESTING.md §3), except cubature has no `max_dt=min_dt` escape
hatch to pin node placement. Binding the same library makes node placement
bit-identical by construction instead of merely close.

**Scope implemented — narrow on purpose, mirrors the Phase 3/4 pattern:**
- **RealGrid only**, `full=false` only (the default — `Cubature.pcubature_v`,
  a 1-D radial integral; Julia's `pointcalc!` fixes `θ=0` in this branch, since
  the azimuthal dependence is folded into the analytic mode-field formula, not
  integrated numerically). `full=true` (`hcubature_v`, genuine 2-D `(r,θ)`
  integral) is deferred.
- **`MarcatiliMode`, `kind=:HE`, `n=1` only** (the `HE1m` family —
  `HE11`, `HE12`, …, the mode basis used throughout Luna's capillary examples
  and tests). Field formula needs only `besselj(0,·)` and `besselj(1,·)`
  (`src/Capillary.jl:271-288`), which `diffraction.rs`'s existing `j0`/`j1`
  already provide — **verified against Julia `SpecialFunctions.besselj` over
  `x∈[0,6]` (covers `u₀₁≈2.405`, `u₀₂≈5.520`): max absolute error ~1.5e-15**
  (the ~1e-11 *relative* error right at a Bessel zero is an artifact of
  dividing by a near-zero value, not a precision problem — the field itself is
  correctly ~0 there by construction, the boundary condition the zero encodes).
  General `n` (needed for `TE`/`TM` and `HE_{n>1,m}` modes) needs a proper
  general-order Bessel routine (Miller's backward recurrence — the naive
  upward recurrence is unstable for `x<n`) and is deferred; the native guard
  rejects any mode with `n≠1` or `kind≠:HE` and falls back to Julia.
- **Constant radius only** (`MarcatiliMode{<:Number,...}`, i.e. `m.a` is a
  plain number, not a z-dependent closure `m.a(z)` for a tapered capillary).
  Reject/fall back otherwise.
- **Normalization precomputed in Julia, not ported.** `MarcatiliMode` overrides
  the generic (numerically-integrated) `Modes.N` with a **closed form**:
  `N(m,z) = π/2 · a(z)² · besselj(n, unm)² · √(ε₀/μ₀)` (`src/Capillary.jl:285-288`).
  For a constant-radius mode this is a single z-invariant scalar. Julia
  precomputes `1/√N` per mode once and passes it over FFI alongside `unm`,
  `a`, `n`, `φ` — **no `besselj` call happens in Rust for normalization**, only
  for the per-node field synthesis (`besselj(n-1, r·unm/a)`, i.e. `j0` for
  `n=1`).
- **Kerr-only nonlinearity, `npol=1` gated in — `npol=2` implemented but not
  yet verified, gated off.** `rhs_modal_pointcalc` reuses the exact
  `KerrScalar!` (`npol=1`, `E³`) / `KerrVector!` (`npol=2`,
  `(Ex²+Ey²)·(Ex,Ey)`, `src/Nonlinear.jl:81-93`) formulas already ported for
  Phase 1/3. The Julia wiring in `RK45.jl` `error()`s on `npol≠1` — the
  `KerrVector!` code path exists in `native.rs` but the shipped gate test
  only exercises `npol=1` (`components=:y`, the default for
  `polarisation=:linear/:x/:y`); a `npol=2` test needs real energy in
  **both** polarisation components (circular/elliptical, not just `:xy`
  components with a degenerate y-only input, which would leave the
  `Ex²`-cross-term untested) to genuinely check the coupling term, and that
  hasn't been done — do not remove this gate without adding that test first.
  Raman and plasma are **deferred for complexity, not because they are physically
  ill-defined at cubature nodes** — per-node Raman is exactly as well-formed
  as Phase 4's per-column Raman (the ADE solver resets its state every RHS
  call from the current time-domain field, `solve_scalar`; it has no memory
  across z-steps or across spatial location, so a moving cubature node is not
  a problem). A future phase can add it as one more additive `Et_to_Pt!` term,
  exactly as Phase 4 added it to `rhs_mode_avg_real`.
- **`shotnoise=false`** (`Emω_noise = nothing`) — the modified shot-noise
  branch (`Er_noise`/`Er_nl`) is not ported.
- **Any other mode type (`DelegatedMode`, interpolated/numeric modes, or a
  mode tuple mixing an ineligible mode with eligible ones) is a hard
  fallback to Julia, not a deferred scope item** — those modes are arbitrary
  Julia closures with no Rust-portable representation at all (unlike the
  scope items above, which are "not yet ported").

**Multi-mode projection is exercised by the first-cut test, not skipped.**
The gate test uses two co-eligible modes (`HE11` + `HE12`, both `n=1`, both
`j0`-based, different `unm`) specifically so the `to_space!` sum-over-modes
matmul and the back-projection matmul (`Prω · transpose(Ems)`) are genuinely
tested with `nmodes=2`, not left implicitly untested by a single-mode
shortcut. Self-validating per the Phase 4 lesson: the full-solve testset
independently asserts the Kerr-driven HE11→HE12 energy transfer is
non-negligible (2.0e-5 of total energy, far above any noise floor) before
trusting the Rust-vs-Julia comparison — a passing comparison where both
paths trivially stay in mode 1 would prove nothing about the two matmuls.

**Gate achieved:** single-step 1.4e-19 (looser ~1e-10 tier documented below
was the conservative target; the dominant single-step error source, `j0`'s
~1e-15 absolute agreement with `SpecialFunctions.besselj`, turned out not to
be the limiting factor at this problem size), full-solve 4.0e-16 (fixed step
size, `test/test_native_modal.jl`). The looser tier is still worth stating
explicitly for future phases: node placement is bit-identical (same
`libcubature` binary) but mode-field synthesis is not bit-for-bit against
`SpecialFunctions.besselj`, so ~1e-10 (not ~1e-13) is the *documented*
ceiling even though this run landed far under it.

> **Concurrency note (do not reintroduce):** the `TransModal` integration loop
> writes shared struct buffers (`t.Erω`, `t.Prω`, `t.Prmω`); it must stay
> **sequential**. A prior `Threads.@threads` parallelization caused a data race
> → corrupted overlaps → every RK step rejected. (BACKLOG "Done" item.) The Rust
> port keeps the `libcubature` call and its Rust callback strictly
> single-threaded (no `rayon`/`Threads` inside the callback) for the same
> reason.

### 3.4 Free-space — `TransFree` (`src/NonlinearRHS.jl:826`) — Phase 6, ported
2-D Cartesian spatial axis: a genuine **3-D FFT** over `(t,y,x)` jointly
(`FFTW.plan_rfft(x, (1,2,3))`, `src/Luna.jl:295` — all three axes transformed
together, unlike Phase 3's radial QDHT-plus-1-D-FFT). Per RHS
(`TransFree.__call__`, `src/NonlinearRHS.jl:818-838`):
```
copy_scale!(Eωo, Eωk, N, scale)     # zero-pad ω → oversampled, per (y,x) column
ldiv!(Eto, FT, Eωo)                 # (ω,ky,kx) → (t,y,x), ONE joint 3-D transform
Et_to_Pt!(Pto, Eto, resp, ρ, idcs)  # Kerr, pointwise over the whole (t,y,x) volume
Pto .*= towin                       # per t, broadcast over (y,x)
mul!(Pωo, FT, Pto)                  # (t,y,x) → (ω,ky,kx), ONE joint 3-D transform
copy_scale!(nl, Pωo, N, 1/scale)    # truncate oversampled → base, per (y,x) column
nl .*= ωwin .* (-im·ω) ./ (2 .* normfun(z))
```

**Reuse, don't reinvent — same FFTW binary, one new plan-creation call.**
`fftw.rs` already `dlopen`s the identical libfftw3 Julia's `FFTW.jl` uses
(ARCHITECTURE §4.1); the *execute* entry points (`fftw_execute_dft_r2c`/
`_c2r`) are rank-agnostic — they work on a 3-D plan exactly as on the
existing 1-D plans. The only new surface is **plan creation**:
`fftw_plan_dft_r2c_3d`/`fftw_plan_dft_c2r_3d` (`RealFft3d` in `fftw.rs`).
This is much lower-risk than Phase 5's cubature situation (a genuinely new
library) — it is the *same* binary, just a rank-3 plan instead of rank-1.

**Dimension order and normalization were measured, not assumed, before
being trusted.** Julia's buffers are column-major `(n_t, n_y, n_x)` (`n_t`
fastest-varying); FFTW's basic-interface dimension list is given
slowest→fastest, so `RealFft3d::new` passes `(n_x, n_y, n_t)` — **reversed**
— to align FFTW's fastest dimension with Julia's `n_t` axis. This is the
same reversal rule that makes 1-D bit-parity trivial, but for rank 3 it is
easy to get subtly wrong (and a Rust-only forward+inverse round-trip test
*cannot* catch a transposed-axis bug — it would still round-trip correctly
against itself). Verified instead with a literal cross-check against
`FFTW.rfft(reshape(Float64.(1:24),4,3,2), (1,2,3))` computed independently in
Julia (`fftw.rs::tests::r2c_3d_matches_julia_reference`) — confirms both the
dimension order **and** which axis gets conjugate-symmetric-halved (`n_t`,
matching Julia's `size(rfft(x,(1,2,3))) == (n_t÷2+1, n_y, n_x)`).

**The round-trip normalization factor is `1/(n_t·n_y·n_x)`, not `1/n_t`** —
easy to get wrong by copying the 1-D convention, since the transform now
spans all three axes rather than just the time axis. Confirmed by the same
literal test.

**Multi-dimensional c2r destroys its input.** Unlike 1-D c2r (which supports
`FFTW_PRESERVE_INPUT`, used by `RealFft1d`), FFTW does not support
`PRESERVE_INPUT` for rank>1 c2r plans. `rhs_free` follows the same
copy-into-scratch-before-inverse structure every other native RHS already
uses (Phase 1/3's `eoo`/`radial_eoo` scratch buffers), so this is harmless
by construction, not a new precaution.

**No per-column spatial step, unlike radial.** Because the FFT is a single
joint 3-D transform (not a 1-D transform looped over a separate QDHT-style
spatial step), the Kerr response and the final normalization multiply are
**plain flat elementwise operations over the whole `(t,y,x)`/`(ω,ky,kx)`
volume** — no column-dependent logic at all (Kerr's `E³` formula and the
precomputed `M` array are identical at every spatial point). Only the
zero-pad/truncate (`copy_scale!`) and `towin` apodization steps need an
explicit per-`(y,x)`-column loop, since those act along the `t`/`ω` axis
specifically. This makes `rhs_free` mechanically simpler than `rhs_radial`
once the FFT primitive itself is trusted.

**Normalization precomputed as one flat complex array, same pattern as
Phase 3's `M`.** `ωwin[iω]·(-i·ω[iω])/(2·normfun(z)[iω,iky,ikx])` is
precomputed once in Julia (valid only for a z-invariant `normfun`, i.e.
`const_norm_free` — the same constant-medium restriction every other phase
carries) and passed as one flat `(n_spec, n_y, n_x)` array; `rhs_free` does
one elementwise multiply, needing zero of `norm_free`'s `k_z`/evanescent
masking logic ported.

**Scope implemented:** RealGrid + `const_norm_free` (z-invariant normfun)
only, scalar Kerr (reusing the exact `E³` formula already ported), the
modified shot-noise branch (`Et_noise`/`Et_nl`) is **not** ported
(`shotnoise=false` required). EnvGrid free-space (c2c 3-D) and a z-dependent
`normfun` are deferred, mirroring every prior phase's RealGrid-first pattern.

**Gate achieved:** single-step 7.05e-18, full-solve 5.01e-17 (fixed step
size, `test/test_native_free.jl`, `N=8` transverse grid). Both land at the
same clean FFTW-parity floor as Phase 1/2 (no QDHT-style intermediate
summation-order floor, since there is no separate spatial transform step) —
consistent with the dimension-order/normalization risk being fully retired
at the FFT-primitive level (`fftw.rs`'s literal cross-check) rather than
showing up as residual noise in the RHS.

### 3.5 z-dependent linear operator — mode-averaged pressure-gradient capillary, Phase 7, ported (narrow scope)

Phases 1–6 all require `L(ω)` to be **constant in z** (`make_const_linop`):
built once in Julia, copied into `NativeSim.linop`, applied via
`exp(linop·Δt)`. Phase 7 is the first to support `L(ω,z)` resident, for one
concrete, common case: a **pressure-gradient (tapered-fill) capillary**,
`Interface.jl:532-541` → `Capillary.gradient(gas, L, p0, p1)` → a graded-core,
**constant-radius** `MarcatiliMode`. This is the "tapered fiber, pressure
gradient" case flagged as the motivating use in the Phase 6 handoff notes.
Scope restrictions (mirrors every prior phase's narrowing discipline):
mode-averaged (`TransModeAvg`) only, RealGrid only, single non-vector mode,
`kind`/`n`/`model`/`loss` unrestricted (both work — see below), Kerr-only.
Radial/free-space/modal z-dependent `nfun` (arbitrary index profile, not
just a graded-core gas fill) is deferred — narrower but real, matching the
existing wired path (`Interface.jl`'s pressure-tuple branch), not invented.

**The physics is separable in `(ω, z)` — this is the whole design.**
`LinearOps.jl:279-296`'s z-dependent `linop!(out,z)` (mode-averaged case)
needs, at each requested `z`:
```
εco(ω;z) = coren(ω;z)² = 1 + γ(λ(ω))·dens(z)        # Capillary.gradient, :302-341
nwg(ω)                                                # z-independent for constant radius
neff(ω;z) = combine(εco(ω;z), nwg(ω); model)          # sqrt(εco-nwg) or 1+(εco-1)/2-nwg
nc(ω;z)   = conj_clamp(neff(ω;z), ω)                  # LinearOps.jl:246
β1(z)     = Modes.dispersion(mode, 1, ω0; z=z)        # Maths.derivative — adaptive FD
out[iω]   = -i·(ω[iω]/c·nc(ω[iω];z) - ω[iω]·β1(z))    # LinearOps.jl:307
```
`γ(λ)` is the gas's Sellmeier susceptibility-per-density coefficient
(`PhysData.sellmeier_gas`, a *real* function of wavelength only — no
z-dependence at all) and `dens(z)` is the density profile
(`dspl(p(z))`, a cubic spline of pressure over a closed-form or piecewise
pressure ramp). **Critically, `εco(ω;z) - 1` factors as a fixed per-ω array
(`γ(λ(ω))`) times a scalar function of z (`dens(z)`)** — the ω-dependence
and z-dependence never mix. This means:
- `nwg(ω)` is **exactly** the already-resident combine from the existing
  `AMALTHEA_USE_RUST_DISPERSION` wiring (`dispersion.rs::MarcatiliNeff::neff_vector`,
  `Capillary.jl:409-470`) — not re-derived, just re-implemented inline in
  `native.rs` (the opaque handle itself isn't shared with `NativeSim`, but
  the formula — `sqrt(εco-nwg)`/`1+(εco-1)/2-nwg`, then truncate `Im` if
  `!loss` — is copied verbatim from `neff_vector`, which already documents
  "no clamping here" i.e. `conj_clamp` is applied by the *caller*, exactly
  matching where it's applied here too).
- The only two z-dependent quantities the resident stepper needs are
  **scalars**: `dens(z)` and `β1(z)`. Neither needs an O(n_ω) Julia
  round-trip per stage — each is a single number.

**`β1(z)` is an exact closed form, not a LUT — see `BETA1_ANALYTIC.md`.**
`dens(pressure)` is transferred exactly from `PhysData.densityspline`'s own
`(x,y,D)` (a `HermiteSpline` reproducing Julia's `Maths.CSpline` formula
bit-for-bit — re-fitting a *different* spline through sampled `dens` values
was tried first and does not converge, see the postmortem in `PORT_LOG.md`).
`β1(z) = d/dω[ω/c·Re(neff(ω,z))]|_{ω0}` does not need a LUT at all: since
`εco(ω;z)-1 = γ(λ(ω))·dens(z)` is separable and `nwg(ω)` is z-independent,
the chain rule collapses β1(z) to a function of the single scalar `dens(z)`
given 4 z-independent constants (`γ0`, `dγ0`, `nwg0`, `dnwg0`) computed once
at setup. These are computed via `Maths.derivative` fed a `BigFloat`
argument rather than hand-derived symbolically (the gas/glass Sellmeier
closures are opaque — arbitrary materials, not hardcoded coefficients) —
see `BETA1_ANALYTIC.md` for the full derivation, why an earlier `(dens,β1)`
LUT design was replaced, and the resulting tolerance tradeoff (Rust's β1 is
now more accurate than Julia's own adaptive-FD `dispersion`, which means
Rust *deliberately* diverges from the Julia oracle by Julia's own FD error,
amplified over propagation length and bandwidth into the full-solve
tolerance floor below).

**Boundary behavior must match Julia's flat clamp, not spline extrapolation.**
`Capillary.gradient`'s `p(z)` is flat outside `[0,L]`
(`z > L ? p1 : z <= 0 ? p0 : sqrt(...)`, `Capillary.jl:310-312`), and the
adaptive stepper's final accepted step can legitimately evaluate the linop
at a `z` slightly past `flength` (the `while s.tn <= tmax` loop in
`RK45.solve` checks *before* stepping, so the step that crosses `tmax` still
executes). Both LUTs therefore **clamp their query `z` into `[0, flength]`
before the spline lookup** (`z.clamp(0.0, flength)`), exactly reproducing
Julia's flat boundary behavior instead of letting the natural-spline BC
extrapolate/oscillate past the fitted domain — a held-out test point at and
beyond both boundaries is part of the equivalence test, not just interior
points (an all-interior held-out check would be blind to exactly this, the
same way Phase 6's square-grid test was blind to the axis-swap bug).

**`prop!`'s per-stage recompute, not a new call site.** Rust already applies
`exp(linop·Δt)` at 5 call sites per `native_step` (the FSAL re-frame, the
3 forward/backward pairs per RK stage, and the final propagate-to-`tn`) —
see `native.rs::native_step`. Every one of Julia's `prop!(y,t1,t2,bwd)`
calls evaluates `linop!` at `t2` **regardless of `bwd`** (`LinearOps.jl:306-311`,
memoized so a repeated `t2` is a no-op) — i.e. a forward/backward pair
always shares one `linop` evaluation. Phase 7 mirrors this with
`NativeSim::ensure_linop_at(z)`, which overwrites the (now no-longer-const)
`self.linop` buffer in place only when the requested `z` differs from the
last one cached, and is a no-op entirely when the mode is not the
z-dependent case (Phases 1-6 unaffected, zero cost). This keeps every
existing `apply_prop(y, &s.linop, dt)` call site unchanged; only a
`s.ensure_linop_at(z_eval)` call is inserted immediately before each one.

**Scope not ported:** `EnvGrid` (β0-reference-frame variant of the same
formula, `LinearOps.jl:353-380`), radial/free-space/modal z-dependent
`nfun` (arbitrary index profile — the separable-in-`(ω,z)` structure this
design leans on is specific to a graded-core gas fill, not a general
`nfun(ω;z)`), multi-mode (`make_linop(grid,modes,λ0)`, tapered-radius
`MarcatiliMode{<:Function}`), and plasma/Raman-in-z-dependent-linop (Kerr
only, consistent with every prior phase).

**The nonlinear RHS also needs the z-dependence, not just the linop.**
`TransModeAvg`'s `(t::TransModeAvg)(nl,Eω,z)` (`NonlinearRHS.jl:531-545`)
calls `t.densityfun(z)` fresh every RK stage, and `norm_mode_average`'s
`norm!(nl,z)` (`NonlinearRHS.jl:553-565`) calls `βfun!(β,z)` fresh every
stage too — both genuinely vary across a pressure gradient. Phases 1-6's
constant-medium `native_set_mode_avg_params` instead bakes `kerr_fac =
density(0)·ε₀·γ3` and `beta[i] = β(ω_i;0)` in **once**, at construction —
correct there (medium never changes), wrong for Phase 7 as originally
wired. `ensure_linop_at` now also rescales `kerr_fac` by the just-computed
`dens(z)` and overwrites `beta[i]` with `ω_i/c·Re(neff(ω_i,z))` (reusing the
per-ω `neff` already computed for the linop) every call. Missing this was
the actual cause of a ~9% full-solve mismatch during development — traced
by isolating that the z-dependent linop alone already matched Julia to
~1e-8 (via `native_debug_linop_at`), and that a `kerr=false` (pure linear)
full-solve run matched Julia to the same ~1e-8, while `kerr=true` did not:
the divergence had to be in the RHS, not the linear propagator. See
`PORT_LOG.md`'s Phase 7 postmortem.

**Expected tolerance floor.** `dens(z)` is transferred exactly (no LUT, no
interpolation error), and `β1(z)`'s analytic closed form matches Julia's own
`Modes.dispersion` (an adaptive finite difference) to ~1e-11–1e-12 relative
— see `BETA1_ANALYTIC.md`. This is *smaller* than Phases 1-6's floors, but
it is a **systematic**, not random, difference from the Julia oracle (Rust
computes the true derivative; Julia's FD has its own small, repeatable
truncation error), so it accumulates coherently over the full propagation
length and spectral bandwidth rather than averaging out. For a broadband
run (λlims spanning 200nm-4000nm, 0.5m gradient) this measured as ~1e-6 in
the point-wise linop and ~1e-4 in a fixed-step full-solve — both confirmed
(via a `kerr=false` control run showing the *same* magnitude) to be entirely
this β1 method difference, not a bug. Narrower-bandwidth or shorter runs
will show a proportionally smaller floor. Do not chase this number down by
reverting to a Julia-value-reproducing LUT — that trades "Rust is exact"
for "Rust matches Julia's imprecision," which is not what Phase 7 is for.

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
1. ionization rate `W(|E|)` from the PPT LUT (already Rust, `AMALTHEA_USE_RUST_IONISATION`);
2. **three cumulative trapezoidal integrals** (`cumtrapz`): free-electron density
   `ρ(t)` from the rate; the loss/heating current; the polarization current;
3. assemble the plasma current `J(t)` and add to `Pt`.
Ported in Phase 2 as `rhs_plasma` (`amalthea/src/native.rs`, `cumtrapz_slice_f64`
+ `native_set_plasma_params`), gated on `AMALTHEA_USE_RUST_IONISATION=1` (the native
path silently falls back to Julia plasma physics if the ionization handle isn't
wired — see PORT_LOG 2026-06-30 "Wire plasma" note).

### 5.3 Raman (`src/Nonlinear.jl:357-431`) — Phase 4, ported
Delayed χ⁽³⁾. The carrier-field SDO path (`RamanPolarField`) is already Rust
(`AMALTHEA_USE_RUST_RAMAN`): a single-damped-oscillator ADE stepped by an exact
exponential integrator (`x_{n+1} = A·x_n + B0·Iₙ + B1·I_{n+1}`), O(Nt) and
allocation-free, vs. Julia's FFT convolution. Phase 4 makes that solver resident
in the RHS. The envelope path (`RamanPolarEnv`) and intermediate-broadening
(Gaussian-damped) responses stay Julia. *(Update: both later went native —
`RamanPolarEnv` in Phase F item 2, intermediate-broadening as a second
resident FFT-convolution kernel in Phase I item 2 (2026-07-08,
`native_set_raman_fft_params` / `rhs_mode_avg_env` Step 3c) — see
BACKLOG.md. This section's remaining text describes the Phase 4 SDO/ADE
kernel only.)*

**Reuse, don't reinvent — same pattern as Phase 3's QDHT.** `raman.rs`'s
`TimeDomainRamanSolver` is already a self-contained, public Rust struct
(`new`, `reset_state`, `solve`) — the existing `AMALTHEA_USE_RUST_RAMAN` FFI
wiring is a thin `ccall` wrapper around it. `NativeSim` stores one directly
(`Option<TimeDomainRamanSolver>`), built once in a new
`native_set_raman_params` from the same `omega`/`gamma`/`coupling` arrays
Julia already extracts in `Interface._make_rust_raman_handle_from_response`
(`Ω`, `1/τ2ρ(1.0)`, `K` per `RamanRespSingleDampedOscillator`). `solve()`
resets its internal oscillator state at the start of every call
(`solve_scalar`, `raman.rs`), so it is safe to call once per RHS evaluation
exactly like Kerr/plasma — no state carries between RK stages, matching the
Julia FFT-convolution semantics it replaces (each RHS call recomputes the
full convolution over the current trial field, not an incremental
step-to-step integration).

**Additive, not a replacement RHS.** Raman is just another entry in
`TransModeAvg`'s `resp` tuple, called after Kerr in `Et_to_Pt!` and
accumulated into the same `Pt` buffer — `resp!(Pt, Et, density)` for each
response, `fill!(Pt,0)` once at the start. So `rhs_mode_avg_real` gains one
more additive step after the existing Kerr (and optional plasma) step,
computed from the *same* time-domain field buffer (`self.eto`) already in
hand — no new to_time!/to_freq! transform needed:
```
intensity[i] = Eto[i]²                      # thg=true: sqr!(R, E) = E.^2
raman_solver.solve(intensity, raman_p)      # resets state, then ADE-integrates
Pto[i] += ρ · Eto[i] · raman_p[i]            # matches Pout[i]=ρ*E[i]*R.P[i]
```
`ρ` is the same constant-medium density (`f!.densityfun(0.0)`) already used
for Kerr's `kerr_fac` — stored separately (unscaled by `ε₀·γ3`) since Raman's
accumulation formula needs the raw density, not the Kerr-folded constant.

**Scope: `thg=true` only** (the default, and the common case — `E²`
intensity, no Hilbert transform). `thg=false` computes intensity via
`1/2·|hilbert(E)|²` (`Kerr_field_nothg`'s sibling path, `sqr!` in
Nonlinear.jl:342-349) — porting the Hilbert transform to Rust is a separate
piece of work, deferred as a follow-up alongside `RamanPolarEnv` and
intermediate-broadening (Gaussian-damped) responses, none of which are in
this phase's scope.

**Eligibility mirrors the existing `AMALTHEA_USE_RUST_RAMAN` wiring exactly**
(`Interface._make_rust_raman_handle_from_response`, `src/Interface.jl:581`):
`CombinedRamanResponse` whose `Rs` are all `RamanRespSingleDampedOscillator`,
with density-independent `τ2ρ`. Molecular gases (`:N2`, `:H2`, `:D2`, `:N2O`,
`:CH4`, `:SF6`) auto-enable Raman by default in `makeresponse`
(`src/Interface.jl:608`). N2's vibrational line qualifies (constant
`τ2v=6e-12`); its rotational line (`RamanRespRotationalNonRigid`, a
multi-line comb) does not — use `rotation=false, vibration=true` for a
clean single-oscillator test case. H2's vibrational line is *not* eligible
(`Bρv`/`Aρv` make `τ2ρ` density-dependent) — useful as a negative control.

**Gate-test pitfall (found during Phase 4 validation, see PORT_LOG
2026-07-01): verify the feature under test actually changes the reference
result before trusting a passing equivalence assertion.** At the parameters
first tried (N2, 1 atm, 1 μJ, 30 fs, one 1cm z-step), Raman-on vs Raman-off
gave an *exact* `0.0` difference in the Julia oracle alone — not close to
zero, identically zero — because Raman's raw per-step RHS contribution is
~2e-16 relative to Kerr's (at the double-precision floor) for a single small
step; the effect only becomes measurable (1.1e-4 over 5cm/6 fixed steps) once
compounded through the propagation's nonlinear dynamics. A test comparing two
implementations that both silently omit the same feature passes for the
wrong reason. The full-solve testset therefore asserts, in order: (1) Raman
changes the Julia oracle's result by a margin unmistakably above the FP
floor, *then* (2) Rust matches Julia on that changed result. Apply the same
two-part structure to any future feature whose per-step effect might be
small relative to the dominant nonlinearity.

## 6. Linear operator `L(ω, z)` (Phase 7, ported — narrow scope, see §3.5)

`L` is built in `src/LinearOps.jl`:
- **Constant** (`make_const_linop`): z-independent; built once at setup, copied
  into `NativeSim` as a flat array. Phases 1–6 use this path — cheap, left on
  Julia for assembly, applied in Rust.
- **z-dependent** (`make_linop`): rebuilt per step (e.g. changing gas
  pressure, bending). Phase 7 ports the mode-averaged, graded-core
  constant-radius `MarcatiliMode` case (`LinearOps.jl:279-296`) resident —
  see §3.5 for the full design (separable `εco(ω;z)`, LUT'd `dens(z)`/`β1(z)`,
  boundary-clamp discipline). Radial/free-space/modal `make_linop` (arbitrary
  `nfun(ω;z)`, `LinearOps.jl:77,185,337`) and multi-mode are deferred.

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

## 8. Beyond-Luna math options (2026-07-08 audit — not ported behavior, deliberate improvements)

The port has already crossed the "better than the oracle" line
deliberately in a few places: the analytic β1 closed form
(BETA1_ANALYTIC.md) replacing Julia's adaptive-FD noise, Chebyshev-
recurrence dispersion, the O(Nt) ADE Raman integrator replacing
O(Nt·logNt) FFT convolution, and exact Gauss-Legendre integrating
factors. The options below continue that line. **None is a parity
port**: each breaks bit-parity with the Julia oracle, so each needs its
own controlled-divergence verification like β1 got — prove the
divergence is Rust being *more* correct (e.g. against a BigFloat or
closed-form ground truth), not just different. That verification is
most of the cost, not the code. Tracked as SUGGESTIONS.md items 12,
15-17 / BACKLOG.md Phase J.

### 8.1 Direct embedded-error coefficients (SUGGESTIONS.md #15)
Luna computes the DP5(4) embedded error as the difference of the 5th-
and 4th-order solutions — a near-total cancellation (`b5-b4`,
TESTING.md §3), which is why a ~1e-15 summation-order difference can
become a ~20% relative disagreement in `err` and send two adaptive
integrators down different step paths. The standard, numerically
superior formulation computes the estimate directly as
`err = Σᵢ eᵢ·kᵢ` with precomputed `eᵢ = b5ᵢ − b4ᵢ` coefficients (exact
rationals, no runtime cancellation). This could dissolve the entire
fixed-step (`max_dt=min_dt`) test discipline that exists to work around
the cancellation — but note it changes *both* steppers' step sequences,
so it should land on the Julia `PreconStepper` and the native stepper in
the same commit (same rule as the dense-output item).

### 8.2 Direct PPT evaluation (SUGGESTIONS.md #16)
Julia's `IonRatePPTAccel` is itself a spline LUT over the true PPT
series, and the Rust port matched it LUT-for-LUT (parity, per the
port's rules). But the PPT rate can be evaluated directly with good
special-function code — more accurate than *both* current paths, the
same move as β1. It also structurally eliminates the out-of-range
segfault (BACKLOG Phase J item 2): a closed-form evaluation has no
fitted range to fall off of. Verification target: the direct sum
against a BigFloat evaluation of the same series, then the sim-level
triangulating tests unchanged.

### 8.3 5th-order dense output (SUGGESTIONS.md #12, already tracked)
Luna's quartic `interpC` is one order below the integrator; Shampine's
DP5 continuous extension is the textbook-correct choice. Both sides
must change in the same commit (see S5.3).

### 8.4 r2c halving of the FFT-conv Raman (BACKLOG Phase J item 3)
The padded E² and h are mathematically real in both Raman convolution
paths; Julia's `RamanPolarEnv` nonetheless uses a full c2c `plan_fft`
(Nonlinear.jl:327) and the native SiO2 kernel mirrors it for parity. A
Hermitian r2c/c2r pair computes the identical convolution at half the
transform cost. Same answer, different summation order → fixed-step
validation discipline.

### 8.5 Short-kernel (overlap-save) Raman convolution (SUGGESTIONS.md #17)
Luna's own comment (Nonlinear.jl:406-411) concedes the double-length
grid is "safe, until we come up with [something more efficient]". For
strongly damped responses — SiO2's Gaussian damping kills h beyond
~100 fs on a multi-ps grid — an overlap-save convolution with a short
kernel (length set by where |h| falls below f64 noise, checked at
setup, falling back to the full double grid otherwise) is much
cheaper. **Hard boundary**: do not go further and fit recursive/IIR
filters to the Gaussian-damped response — that is the multi-SDO
approximation trap BACKLOG Phase I item 2 explicitly rules out
(feedback: prefer analytic over LUT/fit-the-noise).
