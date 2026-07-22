//! Native resident-field simulation state for the exclusively-Rust backend.
//!
//! See `docs/dev/native-port/ARCHITECTURE.md` (§3 `NativeSim`) and `MATH.md`.
//!
//! Phase 0 goal: a single opaque handle that owns, for the lifetime of one
//! `solve`, the spectral field and every scratch buffer the interaction-picture
//! Dormand-Prince loop touches — so the FFI boundary is crossed a handful of
//! times per `solve` instead of ~14× per step (no Julia callback in the hot
//! loop).
//!
//! ## Build status (keep in sync with `docs/dev/native-port/PORT_LOG.md`)
//! - [x] Phase 0a — handle lifecycle + field set/get round-trip (this file).
//! - [x] Phase 0b — FFTW binding (`fftw.rs`): dlopen the *same* libfftw3 Julia
//!       uses + c2c/r2c/c2r plans, planner-lock-serialized. Round-trip tested.
//!       Plans stored in `NativeSim`; planner flag is a parameter for bit-parity.
//! - [x] Phase 0c — callback-free interaction-picture step (`native_step` FFI);
//!       reproduces `precon_step_inner` / Julia `make_fbar!` + `make_prop!` with
//!       a no-op RHS. Gate: rel_solve < 1e-6 (zero-RHS bit-exact). COMPLETE.
//! - [x] Phase 1 — mode-averaged + scalar Kerr (RealGrid): `rhs_mode_avg_real`,
//!       `native_set_mode_avg_params`, `RustNativeStepper` Julia wiring.
//!       Gate: single-step ≤1e-13, full-solve 5.8e-13. COMPLETE.
//! - [x] Phase 2 — Plasma + EnvGrid Kerr: `rhs_mode_avg_env` (EnvGrid c2c Kerr,
//!       3/4 SVEA prefactor) + `rhs_plasma` (cumtrapz ×3, current assembly via
//!       `native_set_plasma_params`). Replaces `PlasmaCumtrapz` (Nonlinear.jl:161)
//!       + EnvGrid Kerr. Gate: Phase 2a (EnvGrid Kerr) single-step <1e-13,
//!       full-solve 3.2e-17; Phase 2b (RealGrid+plasma) single-step 3.8e-17,
//!       full-solve 2.7e-16 (fixed step size — see PORT_LOG 2026-07-01).
//!       COMPLETE.
//! - [x] Phase 3 — Radial (`TransRadial`) + resident QDHT: `rhs_radial`
//!       (RealGrid scalar Kerr, QDHT wrapped around the time transform via the
//!       resident `QdhtFfiHandle`) + `native_set_radial_params`. Replaces
//!       `TransRadial` (NonlinearRHS.jl:663). See MATH.md §3.2 for the design
//!       (column-major `(n_time, n_r)` buffer invariant, precomputed `M`
//!       normalization array, looped rank-1 FFT rather than a batched plan).
//!       Scope: RealGrid + scalar Kerr only, z-invariant `normfun`
//!       (`const_norm_radial`); EnvGrid-radial and plasma-radial deferred.
//!       Gate: single-step 1.1e-17, full-solve 1.3e-16 (fixed step size).
//!       COMPLETE.
//! - [x] Phase 4 — Raman: resident `TimeDomainRamanSolver` (`raman.rs`,
//!       already-existing ADE solver, reused directly) added as an additive
//!       term in `rhs_mode_avg_real` alongside Kerr/plasma via
//!       `native_set_raman_params`. Replaces `RamanPolarField`
//!       (NonlinearRHS.jl:357-431). Scope: RealGrid, `thg=true` only (E²
//!       intensity, no Hilbert transform), all-SDO density-independent-τ2
//!       eligibility (same as the existing `AMALTHEA_USE_RUST_RAMAN` FFI wiring).
//!       Gate: full-solve Rust-vs-Julia 4.2e-8, with Raman independently
//!       verified to change the Julia oracle by 1.1e-4 (self-validating —
//!       a single 1cm z-step shows Raman's contribution below the FP floor
//!       relative to Kerr; the effect is cumulative over propagation, see
//!       MATH.md §5.3 and PORT_LOG 2026-07-01). COMPLETE.
//! - [x] Phase 5 — Modal (`TransModal`), narrow scope: `libcubature`
//!       (`Cubature_jll`) dlopened at runtime, same binary Julia's
//!       `Cubature.jl` calls, bound via `cubature.rs` (`pcubature_v` only —
//!       `full=false`, the radial modal integral, is what Luna itself selects
//!       for `HE, n=1` mode collections — see `Interface.needfull`). Per-node
//!       evaluation (`rhs_modal_pointcalc`) reuses the existing rank-1 FFT
//!       plans (looped, not batched, mirroring Phase 3) and the existing Kerr
//!       formula. Two-mode (HE11+HE12) gate: single-step 1.4e-19, full-solve
//!       4.0e-16 (fixed step size), with the HE11→HE12 energy transfer
//!       independently verified non-negligible (2.0e-5, self-validating —
//!       see MATH.md §3.3 and PORT_LOG). COMPLETE.
//! - [x] Phase 6 — Free-space (`TransFree`): a genuine 3-D FFTW plan
//!       (`fftw.rs::RealFft3d`, new `fftw_plan_dft_r2c_3d`/`_c2r_3d` symbols —
//!       same libfftw3 binary, not a new library) replaces the QDHT-plus-1-D
//!       pattern Phase 3 used for radial; `rhs_free` has no per-column
//!       spatial step (Kerr and the precomputed normalization are plain flat
//!       elementwise ops over the whole `(t,y,x)`/`(ω,ky,kx)` volume). Dim
//!       order (`(n_x,n_y,n_t)` reversed for Julia's column-major
//!       `(n_t,n_y,n_x)`) and the `1/(n_t·n_y·n_x)` round-trip normalization
//!       were verified against a literal `FFTW.rfft` reference
//!       (`fftw.rs::tests::r2c_3d_matches_julia_reference`) before being
//!       trusted, not assumed from the row/column-major rule alone. Scope:
//!       RealGrid, `const_norm_free` (z-invariant), scalar Kerr,
//!       `shotnoise=false`. Gate: single-step 7.05e-18, full-solve 5.01e-17
//!       (fixed step size). See MATH.md §3.4. COMPLETE.
//! - [x] Phase 7 — z-dependent linop: mode-averaged, graded-core
//!       constant-radius `MarcatiliMode`, two-point pressure gradient
//!       (`Capillary.gradient`/`Interface.jl:532-541`). `εco(ω;z)-1` factors
//!       as a fixed per-ω `γ(λ(ω))` array times a scalar `dens(z)` — the
//!       `nwg(ω)` combine is the same formula as the existing
//!       `dispersion.rs::MarcatiliNeff` (copied inline, handle not shared).
//!       `z→pressure` is a closed form ported exactly (no interpolation).
//!       `dens(pressure)` and `β1(dens)` are both **transferred**
//!       `Maths.CSpline`s (`spline.rs::HermiteSpline`), not spline-fit from
//!       scratch in Rust — an earlier attempt to LUT `dens`/`β1` by
//!       resampling and refitting a *different* (natural-BC) spline never
//!       converged, because real-gas density (via CoolProp) composed
//!       through `dspl` (itself already a spline) isn't smooth enough at
//!       the scale a from-scratch refit needs; transferring the identical
//!       piecewise cubic sidesteps this (see MATH.md §3.5, PORT_LOG). `z` is
//!       clamped to `[0,flength]` before the pressure map to match Julia's
//!       flat pressure-profile boundary. `NativeSim::ensure_linop_at`
//!       recomputes `self.linop` in place only when `z` changes and is a
//!       no-op for Phases 1-6. Scope: RealGrid, mode-averaged, Kerr-only;
//!       EnvGrid and radial/free-space/modal `nfun(ω;z)` deferred. Gate:
//!       single-step/full-solve — see MATH.md §3.5 and PORT_LOG. COMPLETE.

use crate::cubature::CubatureApi;
use crate::diffraction::jn;
use crate::ffi::QdhtFfiHandle;
use crate::fftw::{ComplexFft1d, ComplexFft3d, FftwApi, RealFft1d, RealFft3d};
use crate::raman::{RamanOscillator, TimeDomainRamanSolver};
use crate::spline::HermiteSpline;
use libc::{c_char, c_double, c_int, c_uint, c_void, size_t};
use num_complex::Complex;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::ffi::CStr;

/// Resident simulation state. One per `solve`. Opaque to Julia.
///
/// Buffers are sized to `n` (the spectral length `Nω`) at construction and are
/// never reallocated during the solve — the hot loop is allocation-free.
pub struct CpuNativeSim {
    /// Spectral length `Nω` (`Nt÷2+1` for RealGrid, `Nt` for EnvGrid).
    pub n: usize,
    /// The propagated spectral field `Eω` (the RK state vector `y`).
    pub field: Vec<Complex<f64>>,
    /// Constant diagonal linear operator `L(ω)` (dispersion + loss + frame).
    /// z-dependent assembly is Phase 7; Phases 0–6 use this constant array.
    pub linop: Vec<Complex<f64>>,
    /// Bumped every time `ensure_linop_at`/`ensure_free_norm_at`/
    /// `ensure_modal_linop_at` actually recompute `self.linop` in place (not
    /// on their early-return no-op branches) — invalidates `exp_cache`
    /// entries keyed against a stale `linop`. Constant-linop geometries
    /// (Phases 0-6) never bump this, so their cache entries stay valid for
    /// the whole solve.
    pub linop_version: u64,
    /// `exp(linop[i]*dt)` cache, keyed on the exact `dt` bit pattern and
    /// `linop_version` — docs/dev/BACKLOG.md S1 item 3 ("`exp(L·c_i·dt)` caching").
    /// DP45's stage loop (`step`) calls `apply_prop` with the same handful
    /// of `dt` sub-multiples (the fixed `DP_NODES[ii]*dt` fractions) up to
    /// 14×/step, and the same `dt` recurs across consecutive accepted steps
    /// whenever the adaptive controller doesn't change step size (the common
    /// case once a propagation settles) or under fixed-step testing. See
    /// `apply_prop_cached`.
    pub exp_cache: ExpCache,
    /// RK stage derivatives k1..k7 (FSAL reuses k7→k1 across steps).
    pub ks: [Vec<Complex<f64>>; 7],
    /// Embedded-pair error estimate buffer.
    pub yerr: Vec<Complex<f64>>,
    /// Stage / accumulation scratch (`y + dt·Σ b_i k_i`).
    pub ystage: Vec<Complex<f64>>,

    // ── Phase 0b: FFTW ────────────────────────────────────────────────────────
    pub fftw_api: Option<FftwApi>,
    pub fft_c2c: Option<ComplexFft1d>,
    pub fft_r2c: Option<RealFft1d>,
    pub fft_c2c_over: Option<ComplexFft1d>,
    pub fft_r2c_over: Option<RealFft1d>,
    pub is_real: bool,
    pub fft_norm: f64,
    pub fft_norm_over: f64,

    // ── Phase 1: RealGrid mode-averaged RHS buffers ──────────────────────────────
    pub eto: Vec<f64>,
    pub pto: Vec<f64>,
    pub eoo: Vec<Complex<f64>>,
    pub poo: Vec<Complex<f64>>,
    pub towin: Vec<f64>,
    pub owin: Vec<f64>,
    pub sidx: Vec<bool>,
    pub pre: Vec<Complex<f64>>,
    pub beta: Vec<f64>,
    /// De-branched `pre[i]/beta[i]*sqrt_aeff` at `sidx` positions, identity
    /// (`1+0i`) elsewhere — precomputed once so `rhs_mode_avg_{real,env}`'s
    /// norm+freq-window steps become a plain unconditional multiply loop
    /// (compiler auto-vectorizes; the branchy per-element `if sidx[i]` form
    /// did not — confirmed via `objdump`, docs/dev/BACKLOG.md S1 item 4). `owin`
    /// itself is folded to `1.0` outside `sidx` at construction time for the
    /// same reason (only ever read at these two sidx-gated call sites).
    /// `x*1.0 == x` exactly in IEEE 754 for any finite/NaN/Inf `x`, so this
    /// is bit-identical to the branchy original, not merely equivalent.
    pub norm_pre_beta: Vec<Complex<f64>>,
    pub kerr_fac: f64,
    pub nlscale: f64,
    pub sqrt_aeff: f64,
    pub n_time: usize,
    pub n_time_over: usize,
    pub n_spec: usize,
    pub n_spec_over: usize,
    /// Modified shot-noise time-domain field, length `n_time_over`, or empty
    /// if `has_noise` is false. Set once at construction (Julia:
    /// `NonlinearRHS.jl:534-535`) and added to `eto` (after the 1/sc scaling,
    /// scaled by the same `inv_sc`) every RHS call, matching
    /// `TransModeAvg`'s `Et_nl = Eto + Et_noise/sc` (NonlinearRHS.jl:561-562).
    /// `eto` is pure per-call scratch here (never read after this RHS
    /// evaluation returns), so adding the noise in place is equivalent to
    /// Julia's separate `Et_nl` buffer without the extra allocation/copy.
    pub et_noise: Vec<f64>,
    pub has_noise: bool,
    /// EnvGrid (complex) counterpart of `et_noise`, length `n_time_over`.
    /// Same injection point/formula as `et_noise`, added to `eto_cplx`
    /// (also gated by `has_noise`).
    pub et_noise_cplx: Vec<Complex<f64>>,

    // ── Phase 2: EnvGrid complex time-domain buffers (c2c path) ─────────────────
    pub eto_cplx: Vec<Complex<f64>>,
    pub pto_cplx: Vec<Complex<f64>>,
    pub eoo_cplx: Vec<Complex<f64>>,
    pub poo_cplx: Vec<Complex<f64>>,

    // ── Phase 2: Plasma (RealGrid only — Julia has no EnvGrid plasma yet) ────────
    /// Rate, fraction, phase, J, P scratch buffers for cumtrapz ×3.
    pub plas_rate: Vec<f64>,
    pub plas_fraction: Vec<f64>,
    pub plas_phase: Vec<f64>,
    pub plas_j: Vec<f64>,
    pub plas_p: Vec<f64>,
    /// Raw ptr to Julia's `PptIonizationRate` handle. Julia owns the pointee;
    /// valid for the lifetime of the enclosing `RustNativeStepper`.
    pub plasma_ion_ptr: *const crate::ionization::PptIonizationRate,
    /// docs/dev/BACKLOG.md Phase I item 3: `ionisation=:ADK` handle, set instead of
    /// `plasma_ion_ptr` when `native_set_plasma_params_adk` is called.
    /// Exactly one of `plasma_ion_ptr`/`plasma_adk_ptr` is non-null at a
    /// time — `plasma_is_adk` selects which `apply_plasma_*` reads.
    pub plasma_adk_ptr: *const crate::ionization::AdkIonizationRate,
    pub plasma_is_adk: bool,
    pub plasma_ionpot: f64,
    pub plasma_e_ratio: f64,
    pub plasma_preionfrac: f64,
    pub plasma_dt: f64,
    /// Gas density (m⁻³), multiplied into the plasma polarisation before
    /// accumulating into `pto`/`radial_pto` — reproduces
    /// `(Plas::PlasmaCumtrapz)(out, Et, ρ)`'s `out .+= ρ .* Plas.P`
    /// (`Nonlinear.jl:237-249`). Found missing entirely (2026-07-08) while
    /// verifying Phase I item 3 (ADK): both the PPT and ADK native paths
    /// computed a correct per-sample ionization rate/polarisation `plas_p`
    /// but never scaled it by density before adding to `pto`, silently
    /// under-computing plasma's contribution for every native
    /// mode-averaged/radial plasma sim since this was first wired
    /// (docs/dev/BACKLOG.md Phase C) — undetected because existing equivalence
    /// tests only exercised weak-field/near-zero-ionization regimes.
    pub plasma_density: f64,
    pub has_plasma: bool,

    // ── Phase 3: Radial (TransRadial) — resident QDHT + scalar Kerr ──────────────
    pub is_radial: bool,
    /// Modified shot-noise time-domain field, column-major `(n_time_over,
    /// n_r)`, length `n_time_over*n_r`, or empty if `has_radial_noise` is
    /// false. Added to `radial_eto` in place right after the QDHT ldiv!
    /// step, unscaled — matches `TransRadial`'s `Et_nl = Eto + Et_noise`
    /// (NonlinearRHS.jl:691-692; no 1/sc factor here, unlike mode-averaged).
    pub radial_et_noise: Vec<f64>,
    pub has_radial_noise: bool,
    /// EnvGrid (complex) counterpart of `radial_et_noise`, same layout,
    /// gated by the same `has_radial_noise` flag.
    pub radial_et_noise_cplx: Vec<Complex<f64>>,
    /// Number of radial grid points (`Hankel.QDHT.N`). Buffers are column-major
    /// `(n_time, n_r)`: column `r` is `n_time` contiguous elements.
    pub n_r: usize,
    pub qdht: Option<QdhtFfiHandle>,
    pub qdht_scale_fwd: f64,
    pub qdht_scale_inv: f64,
    /// Precomputed `M[iω,ir] = ωwin[iω]·(-i·ω[iω])/(2·normfun_array[iω,ir])`
    /// (flattened, column-major `(n_spec, n_r)`) — folds `norm_radial` + `ωwin`
    /// into one elementwise multiply. Valid only for a z-invariant `normfun`
    /// (`const_norm_radial`) — see MATH.md §3.2.
    pub radial_m: Vec<Complex<f64>>,
    pub radial_eto: Vec<f64>,
    pub radial_pto: Vec<f64>,
    pub radial_eoo: Vec<Complex<f64>>,
    pub radial_poo: Vec<Complex<f64>>,
    /// EnvGrid (c2c) counterparts of `radial_eto`/`radial_pto` — only
    /// allocated (non-empty) when `is_real == false`. Time-domain buffers
    /// are complex for EnvGrid (unlike RealGrid's real `radial_eto`/`radial_pto`),
    /// so they need their own storage rather than reinterpreting the real
    /// ones. `radial_eoo`/`radial_poo` (frequency-domain) are already
    /// `Complex<f64>` and are reused as-is for both grid types — see
    /// MATH.md §3.2's EnvGrid follow-up note.
    pub radial_eto_c: Vec<Complex<f64>>,
    pub radial_pto_c: Vec<Complex<f64>>,

    // ── Phase 4: Raman (RealGrid, additive term in rhs_mode_avg_real) ──
    pub has_raman: bool,
    pub raman_solver: Option<TimeDomainRamanSolver>,
    /// Constant-medium density (unscaled, unlike `kerr_fac` which folds in ε₀·γ3).
    pub raman_density: f64,
    pub raman_intensity: Vec<f64>,
    pub raman_p: Vec<f64>,
    /// `true` (default): intensity is `E²` (`sqr!`'s `thg=true` branch).
    /// `false` (Phase F.1): intensity is `1/2·|hilbert(E)|²`
    /// (`sqr!`'s `thg=false` branch, `Nonlinear.jl:342-349`) — needs the
    /// resident c2c plan/buffers below.
    pub raman_thg: bool,
    /// Resident c2c FFT plan for the Hilbert transform (`thg=false` only).
    /// RealGrid has no c2c plan otherwise (`fft_c2c`/`fft_c2c_over` are only
    /// built for EnvGrid by `set_fftw_plans`), so this is its own plan, sized
    /// `n_time_over`, built lazily in `set_raman_params` once `is_real` and
    /// `n_time_over` are already known.
    pub raman_hilbert_fft: Option<ComplexFft1d>,
    /// Two scratch buffers (`n_time_over` each) — `ComplexFft1d::forward`/
    /// `inverse` require distinct in/out buffers (the plan is built
    /// out-of-place, see `fftw.rs`), reused sequentially across r-columns
    /// (radial) or quadrature nodes (modal) exactly like `raman_solver`.
    pub raman_hilbert_a: Vec<Complex<f64>>,
    pub raman_hilbert_b: Vec<Complex<f64>>,

    // ── Phase I item 2: intermediate-broadening (`:SiO2`) Raman, resident
    // FFT convolution — `rhs_mode_avg_env` only (`RamanPolarEnv`, EnvGrid).
    // `RamanRespIntermediateBroadening`'s Gaussian-damped impulse response
    // (`Raman.jl:97-105`) has no finite-SDO decomposition, so it cannot
    // reuse `raman_solver`'s ADE path (see docs/dev/BACKLOG.md Phase I item 2) — this
    // reproduces `(R::RamanPolar)(out, Et, ρ)`'s generic Julia FFT-conv path
    // (`Nonlinear.jl:395-417`) instead: precompute the padded impulse
    // response's spectrum once (`raman_fft_hw`, scaled by `dt/n_over` so no
    // further scaling is needed per step), then each step does
    // `P = irfft(hw .* rfft(E2_padded))` via a resident length-`2·n_time_over`
    // r2c/c2r plan pair (Phase J.3, docs/dev/BACKLOG.md, MATH.md §8.4): both
    // `E2_padded` and `h` are mathematically real (E² and a real impulse
    // response), so a Hermitian r2c/c2r pair computes the identical
    // convolution as the earlier full c2c plan at roughly half the FFT cost
    // (measured 1.8-2.8x on the isolated forward+multiply+inverse step across
    // n_time_over=1024..65536, `amalthea/benches/raman_fft_r2c_bench.rs`; the
    // real `:SiO2` config lands at n_time_over=4096 -> ~2.2x) — see PORT_LOG
    // for the full measurement. c2r also drops the c2c path's spurious
    // imaginary-noise component of P (real inputs convolved via c2c still
    // carry ~1e-16 imaginary garbage from FFT rounding; c2r forces it exactly
    // zero), which is why the equivalence tolerance does not widen. Mutually
    // exclusive with `has_raman` (only one Raman response is ever present in
    // `f!.resp`).
    pub has_raman_fft: bool,
    /// Constant-medium density (unscaled, like `raman_density`).
    pub raman_fft_density: f64,
    /// Precomputed `rfft(h_padded) .* (dt / n_over)`, length
    /// `n_over/2+1` (`n_over = 2·n_time_over`).
    pub raman_fft_hw: Vec<Complex<f64>>,
    /// Resident r2c/c2r plan pair, real length `2·n_time_over` / spectral
    /// length `n_time_over+1` (zero-padded convolution).
    pub raman_fft_plan: Option<RealFft1d>,
    /// Zero-padded `E²` scratch (second half stays zero forever), length
    /// `2·n_time_over`. Real (not `Complex`) since `sqr!`'s `1/2·|E|²` is
    /// real by construction.
    pub raman_fft_e2: Vec<f64>,
    /// Spectrum scratch, length `n_time_over+1` (distinct buffer from
    /// `raman_fft_e2`, `RealFft1d` requires out-of-place in/out).
    pub raman_fft_ew: Vec<Complex<f64>>,

    // ── Phase 5: Modal (TransModal), narrow scope — see MATH.md §3.3 ────────────
    pub is_modal: bool,
    /// Number of modes (`sim.n == n_spec * n_modes`).
    pub n_modes: usize,
    /// Number of polarisation columns (1 or 2 — `Modes.ToSpace.npol`).
    pub npol: usize,
    /// Shared physical core radius (constant across z and across modes —
    /// scope restriction, see MATH.md §3.3).
    pub modal_a: f64,
    /// Per-mode `unm` (Bessel zero), length `n_modes`.
    pub modal_unm: Vec<f64>,
    /// Per-mode `1/√N(m)` (closed-form Marcatili normalization, precomputed
    /// in Julia — no `besselj` call in Rust for this), length `n_modes`.
    pub modal_inv_sqrt_n: Vec<f64>,
    /// Per-mode Bessel order for the radial field factor `J_order(r·unm/a)`
    /// — `n-1` for `:HE`, `1` for `:TE`/`:TM` (docs/dev/BACKLOG.md Phase E.1: general
    /// mode orders, generalizing the Phase 5 `kind=:HE,n=1`-only `J0`
    /// scope). Length `n_modes`.
    pub modal_order: Vec<i32>,
    /// Per-mode kind: 0=`:HE`, 1=`:TE`, 2=`:TM` — selects the angular
    /// formula in `mode_angle_xy` (docs/dev/BACKLOG.md Phase E.3: general θ,
    /// generalizing E.1/E.2's fixed-`θ=0` shortcut). Length `n_modes`.
    pub modal_kind: Vec<u8>,
    /// Per-mode rotation angle `ϕ` — only meaningful for `:HE` (`:TE`/`:TM`
    /// have no `ϕ` dependence, see `Capillary.field`). Length `n_modes`.
    pub modal_phi: Vec<f64>,
    /// `full=true`: genuine 2-D `(r,θ)` cubature (`hcubature_v`) instead of
    /// the `full=false` radial-only 1-D integral at fixed `θ=0`
    /// (`pcubature_v`) — docs/dev/BACKLOG.md Phase E.3.
    pub modal_full: bool,
    /// Per-polarisation-column selector: 0 → x component, 1 → y component
    /// (see `mode_angle_xy`). Length `npol`.
    pub modal_pol_select: Vec<u8>,
    /// Precomputed `ωwin[iω]·normfunc(ω[iω])` (folds the window and whatever
    /// `TransModal.norm!` does — extracted numerically from the Julia
    /// closure, not re-derived — see MATH.md §3.3), length `n_spec`.
    pub modal_nlfac: Vec<Complex<f64>>,
    pub modal_kerr_fac: f64,
    pub cubature: Option<CubatureApi>,
    pub modal_rtol: f64,
    pub modal_atol: f64,
    pub modal_maxevals: usize,
    /// Per-worker scratch for the modal per-node integrand (`modal_pointcalc`).
    /// Entry 0 is always present (sized in `native_set_modal_params`) and is
    /// the sole scratch used on the sequential path (`n_threads == 1`).
    /// Entries 1.. are allocated lazily by `ensure_modal_pool` when
    /// `n_threads > 1` so each rayon worker owns a disjoint `ModalScratch`
    /// (docs/dev/BACKLOG.md S2 Phase 4 — modal RHS threading). Every buffer a
    /// node writes lives here, so the read-only `ModalRO` view of `self` can be
    /// shared `&` across threads while each worker mutates only its own entry.
    pub modal_scratch_pool: Vec<ModalScratch>,
    /// Snapshot of the trial field (`Emω`, flattened column-major
    /// `(n_spec, n_modes)`) for the duration of one `rhs_modal` call — the
    /// `libcubature` callback only receives a `void*`, so the field to
    /// integrate against has to live on `self` rather than be passed
    /// alongside it.
    pub modal_emega: Vec<Complex<f64>>,

    // ── Phase 6: Free-space (TransFree), RealGrid + scalar Kerr — see MATH.md §3.4 ──
    pub is_free: bool,
    pub n_y: usize,
    pub n_x: usize,
    pub fft_r2c_3d: Option<RealFft3d>,
    /// EnvGrid (c2c) counterpart of `fft_r2c_3d` — Phase D.3 (docs/dev/BACKLOG.md).
    /// Only one of `fft_r2c_3d`/`fft_c2c_3d` is `Some`, selected by `is_real`
    /// at `native_set_free_params` time, same split as the radial pair.
    pub fft_c2c_3d: Option<ComplexFft3d>,
    /// Precomputed `ωwin[iω]·(-i·ω[iω])/(2·normfun(z)[iω,iky,ikx])`, flattened
    /// column-major `(n_spec, n_y, n_x)` — folds `norm_free` + `ωwin` into one
    /// elementwise multiply, exactly like Phase 3's radial `M` array. Valid
    /// only for a z-invariant `normfun` (`const_norm_free`).
    pub free_m: Vec<Complex<f64>>,
    /// `1/(n_time_over·n_y·n_x)` — **not** the 1-D `fft_norm_over` (which is
    /// only `1/n_time_over`): the free-space transform spans all three axes,
    /// so reusing the 1-D factor would silently under-scale by `1/(n_y·n_x)`.
    /// See MATH.md §3.4.
    pub free_fft_norm_over: f64,
    pub free_eto: Vec<f64>,
    pub free_pto: Vec<f64>,
    pub free_eoo: Vec<Complex<f64>>,
    pub free_poo: Vec<Complex<f64>>,
    /// EnvGrid (c2c) counterparts of `free_eto`/`free_pto` — only allocated
    /// (non-empty) when `is_real == false`, same split as `radial_eto_c`/
    /// `radial_pto_c` (Phase D.1). `free_eoo`/`free_poo` (frequency-domain)
    /// are already `Complex<f64>` and reused as-is for both grid types.
    pub free_eto_c: Vec<Complex<f64>>,
    pub free_pto_c: Vec<Complex<f64>>,

    // ── Phase 7 (+ docs/dev/BACKLOG.md Phase F item 3): z-dependent linop — mode-
    // averaged, graded-core constant-radius MarcatiliMode, two-point OR
    // general multi-point piecewise pressure gradient — see MATH.md §3.5.
    pub is_zdep_mode_avg: bool,
    /// Pressure-gradient breakpoints (`Capillary.TwoPointGradient`'s `[0,L]`
    /// or `MultiPointGradient`'s general `Z`, sorted ascending, length >= 2).
    /// `z` queries are clamped into `[zdep_grad_z[0], zdep_grad_z[last]]`
    /// before converting to pressure, reproducing `Capillary.gradient`'s flat
    /// pressure profile outside the breakpoint range instead of extrapolating.
    pub zdep_grad_z: Vec<f64>,
    /// Pressures at each breakpoint (same length as `zdep_grad_z`). Within a
    /// segment `[Z[i],Z[i+1]]`, `p(z) = sqrt(P[i]² + t·(P[i+1]²-P[i]²))`
    /// (`t` the fractional position) is a closed form, ported exactly (no
    /// interpolation) — only `dens`/`β1` as functions of the resulting
    /// *pressure* are LUT'd (see below for why pressure, not z, is the LUT
    /// axis). `ensure_linop_at` selects the segment containing `z`.
    pub zdep_grad_p: Vec<f64>,
    /// `dens(pressure)` — **transferred**, not re-fit: `Maths.CSpline`'s own
    /// `(x,y,D)` from `PhysData.densityspline`, evaluated in Rust with the
    /// identical Hermite-cubic formula. Re-fitting a *different* (natural
    /// cubic) spline through samples of `dspl` was tried first and failed
    /// to converge — the refit chases `dspl`'s own knot-to-knot 3rd-derivative
    /// jumps (real-gas density via CoolProp is not perfectly smooth at that
    /// scale), which only shrinks like `O(h)`, not `O(h⁴)`, so no amount of
    /// resampling closes the gap. Transferring the *same* piecewise cubic
    /// sidesteps the problem entirely (see MATH.md §3.5, PORT_LOG postmortem).
    pub zdep_dens_lut: Option<HermiteSpline>,
    /// `γ(λ(ω))` per spectral bin (0 outside `sidx`) — z-independent gas
    /// Sellmeier coefficient, computed once in Julia.
    pub zdep_gamma: Vec<f64>,
    /// `nwg(ω)` per spectral bin (0 outside `sidx`) — z-independent for
    /// constant core radius, same combine as `dispersion.rs::MarcatiliNeff`.
    pub zdep_nwg_re: Vec<f64>,
    pub zdep_nwg_im: Vec<f64>,
    /// Angular frequency grid — needed for the linop assembly and for
    /// `conj_clamp`'s `3000·c/ω` bound; not stored elsewhere in `NativeSim`.
    pub zdep_omega: Vec<f64>,
    /// 0 = `:full` (`sqrt(εco-nwg)`), 1 = `:reduced` (`1+(εco-1)/2-nwg`).
    pub zdep_model: u8,
    pub zdep_loss_on: bool,
    /// Reference frequency `ω0` — where `β1(z) = d/dω[ω/c·Re(neff(ω,z))]`
    /// is evaluated. Closed-form β1 constants (see `docs/dev/native-port/
    /// BETA1_ANALYTIC.md`): `εco(ω;z)-1 = γ(λ(ω))·dens(z)` is separable and
    /// `nwg(ω)` is z-independent (constant radius), so β1(z) reduces to a
    /// function of the single scalar `dens(z)` given these 4 z-independent
    /// constants (computed once in Julia via `Maths.derivative` fed a
    /// `BigFloat` argument — exact to far below `Float64` epsilon, replacing
    /// an earlier `(dens,β1)` LUT that chased Julia's own adaptive-FD noise
    /// floor instead of the true derivative).
    pub zdep_omega0: f64,
    pub zdep_gamma0: f64,
    pub zdep_dgamma0: f64,
    pub zdep_nwg0: Complex<f64>,
    pub zdep_dnwg0: Complex<f64>,
    /// `ε₀·γ3` — density factored out of `kerr_fac = density(z)·ε₀·γ3` so
    /// `ensure_linop_at` can rescale `kerr_fac` by the just-computed `dens(z)`
    /// every call. Julia's `TransModeAvg` calls `t.densityfun(z)` fresh at
    /// every RK stage (`NonlinearRHS.jl`'s `(t::TransModeAvg)(nl,Eω,z)`) — the
    /// non-z-dependent phases (0-6) instead bake a single `kerr_fac` in at
    /// construction (`density(0)`), which is wrong for a pressure gradient
    /// where density varies ~10× over the fibre. Missing this update was the
    /// actual cause of the ~9% full-solve mismatch seen during development —
    /// the z-dependent *linop* was already accurate to ~1e-8 at that point,
    /// so the divergence had to be in the RHS, not the linear propagator (see
    /// PORT_LOG's Phase 7 postmortem).
    pub zdep_kerr_fac_per_dens: f64,
    /// Memoizes the last `z` `ensure_linop_at` computed `self.linop` for —
    /// mirrors Julia's `make_prop!`'s `lastt2` memoization (a
    /// forward/backward `prop!` pair always shares one evaluation).
    pub zdep_last_z: f64,

    // ── Phase D.5: z-dependent linop + normfun — free-space (`TransFree`),
    // two-point pressure gradient — see `LinearOps.ZDepLinopFree` /
    // `NonlinearRHS.ZDepNormFree`. No waveguide term (unlike Phase 7's
    // Marcatili case): `n(ω;z) = sqrt(1+γ(λ(ω))·ρ(z))` is the whole index,
    // so both `self.linop` (`LinearOps._fill_linop_xy!`) and `self.free_m`
    // (`NonlinearRHS.norm_free`) are recomputed here from the same
    // per-(ω,k⊥) `k²(ω;z)-k⊥²` quantity every RK stage.
    pub is_zdep_free: bool,
    pub zdep_free_flength: f64,
    pub zdep_free_p0: f64,
    pub zdep_free_p1: f64,
    pub zdep_free_dens_lut: Option<HermiteSpline>,
    /// `γ(λ(ω))` per spectral bin (0 outside `sidx`), length `n_spec`.
    pub zdep_free_gamma: Vec<f64>,
    pub zdep_free_omega: Vec<f64>,
    /// `grid.ωwin` per spectral bin — free-space doesn't otherwise store
    /// this (the constant-linop path folds it into `free_m` once and
    /// discards it), so it needs its own field here for the per-stage recompute.
    pub zdep_free_omegawin: Vec<f64>,
    /// `k⊥² = kx²+ky²`, flattened column-major `(n_y,n_x)` (iy fastest) —
    /// matches `free_m`'s own `(n_spec,n_y,n_x)` flatten order.
    pub zdep_free_kperp2: Vec<f64>,
    /// `grid.sidx`, length `n_spec` — free-space's own `self.sidx` field is
    /// never populated (only `native_set_mode_avg_params` fills it), so
    /// this needs its own copy for the sidx-gated `k2` default.
    pub zdep_free_sidx: Vec<bool>,
    /// β1(z) reference frequency (`wlfreq(grid.referenceλ)`, matching
    /// `LinearOps.make_linop`'s own β1 evaluation point).
    pub zdep_free_omega0: f64,
    pub zdep_free_gamma0: f64,
    pub zdep_free_dgamma0: f64,
    /// `ε₀·γ3`, density factored out — see `zdep_kerr_fac_per_dens`'s doc
    /// for why this must be rescaled by `dens(z)` every call, not baked in
    /// once at construction.
    pub zdep_free_kerr_fac_per_dens: f64,
    pub zdep_free_last_z: f64,

    // ── Phase E.2: z-dependent modal linop — tapered/per-mode radius
    // (`TransModal`, `full=false`, constant density, `a(z)` a shared
    // Function across all modes) — see `Capillary.ZDepLinopModalTaper`.
    // Unlike Phase 7 (constant radius, varying density) this is the mirror
    // case: density is constant (checked via `_check_density_zindependent`
    // in RK45.jl, so `kerr_fac`/`modal_nlfac` stay fixed, unlike Phase 7's
    // `zdep_kerr_fac_per_dens`) but the core radius `a(z)` varies, so
    // `nwg(ω;z)` (not `εco`) is the z-dependent piece of each mode's neff.
    // `nwg(ω,a) = unm²·(φ(ω)/a)²·(1 - i·vn(ω)·φ(ω)/a)²` with `φ(ω)=c/ω` —
    // separable in `a` and `ω`, so it is evaluated exactly (not LUT'd) at
    // any `z` from `unm` (already have it, `modal_unm`) and the per-mode
    // per-ω complex `vn(ω)` array below (`get_vn(cladn(ω)²,kind)`,
    // z-independent — same physical cladding regardless of core radius).
    pub is_zdep_modal: bool,
    pub zdep_modal_flength: f64,
    /// `a(z)` — an arbitrary Julia `Function`, transferred as a natural
    /// cubic spline (`Maths.CSpline`, dense z-sampling of the ground-truth
    /// function, NOT a re-fit of an existing spline — no spline-of-spline
    /// risk, unlike the density LUTs elsewhere in this file).
    pub zdep_modal_a_lut: Option<HermiteSpline>,
    /// `a(0)` — since `N(m,z) ∝ a(z)²` (`Capillary.N`), the per-mode
    /// normalization `1/√N(m,z) = 1/√N(m,0)·a(0)/a(z)` is a simple rescale
    /// of the z=0 baseline (`zdep_modal_inv_sqrt_n0`), needing no per-z
    /// `besselj` recompute (`unm`, hence `besselj(np1,unm)`, is
    /// radius-independent).
    pub zdep_modal_a0: f64,
    pub zdep_modal_inv_sqrt_n0: Vec<f64>,
    pub zdep_modal_omega0: f64,
    /// `grid.ω`, length `n_spec` — needed for the per-(mode,ω) neff/linop
    /// assembly and `conj_clamp`.
    pub zdep_modal_omega: Vec<f64>,
    pub zdep_modal_sidx: Vec<bool>,
    pub zdep_modal_model: u8,
    pub zdep_modal_loss_on: bool,
    /// `εco(ω)` per mode (flattened `(n_spec, n_modes)`, 0 outside sidx) —
    /// each mode keeps its own gas/core-index closure in Julia (not
    /// necessarily `===`-shared even for a single-gas fibre), so this is
    /// stored per-mode rather than assumed common, at negligible extra cost.
    pub zdep_modal_eco: Vec<f64>,
    /// `vn(ω) = get_vn(cladn(ω)²,kind)` per mode (flattened `(n_spec,
    /// n_modes)` complex, 0 outside sidx) — the cladding-Sellmeier piece of
    /// `nwg`, z-independent (same cladding material regardless of `a(z)`).
    pub zdep_modal_vn_re: Vec<f64>,
    pub zdep_modal_vn_im: Vec<f64>,
    /// Per-mode `εco(ω0)`/`dεco/dω(ω0)` and `vn(ω0)`/`dvn/dω(ω0)` — the
    /// BigFloat-differentiated constants `β1(z)` needs (same
    /// `Maths.derivative`-fed-`BigFloat` pattern as Phase 7/D.5's `γ0`/`dγ0`).
    pub zdep_modal_eco0: Vec<f64>,
    pub zdep_modal_deco0: Vec<f64>,
    pub zdep_modal_v0_re: Vec<f64>,
    pub zdep_modal_v0_im: Vec<f64>,
    pub zdep_modal_dv0_re: Vec<f64>,
    pub zdep_modal_dv0_im: Vec<f64>,
    /// Reference mode index (0-based) `β1(z)` is evaluated from — matches
    /// `LinearOps.make_linop(grid,modes,λ0;ref_mode)`'s `ref_mode`.
    pub zdep_modal_ref_mode: usize,
    pub zdep_modal_last_z: f64,

    // ── S2: threading the native RHS (docs/dev/BACKLOG.md) ────────────────────────────
    /// Rayon thread count for later-phase parallel RHS seams. `<= 1` means
    /// sequential — the default, matching pre-S2 behavior exactly.
    pub n_threads: usize,
    /// Lazily built the first time a parallel seam actually runs with
    /// `n_threads > 1` — never touches rayon's process-global pool (see
    /// `set_threads`'s doc on the trait for why).
    pub thread_pool: Option<rayon::ThreadPool>,

    // ── S5.2: deterministic mode (docs/dev/BACKLOG.md) ─────────────────────────────────
    /// When `true`, skip the QDHT BLAS-3 path (`AMALTHEA_QDHT_BLAS`) even if a
    /// BLAS library was loaded, forcing the row-parallel Rayon fallback
    /// instead — see `set_deterministic`'s doc on the trait for why this is
    /// the one real lever today (no `NativeBackend` RHS code reads
    /// `n_threads` yet; S2 Phase 3, which would have, was reverted).
    pub deterministic: bool,
}

/// Envelope-Raman intensity `0.5·|E|²` (`sqr!(R::RamanPolarEnv, E)`,
/// `Nonlinear.jl:387-390`). Shared by `rhs_mode_avg_env`'s Step 3b (ADE) and
/// Step 3c (FFT-conv) — both fed the same formula from the same time-domain
/// field into different destination buffers before this was pulled out
/// (Phase J.5 dedup, docs/dev/BACKLOG.md). Writes exactly `out.len()`
/// elements starting at index 0 of both slices; callers own any additional
/// zero-padding (Step 3c's `raman_fft_e2` tail).
#[inline]
fn raman_intensity_half_env(eto: &[Complex<f64>], out: &mut [f64]) {
    for i in 0..out.len() {
        out[i] = 0.5 * eto[i].norm_sqr();
    }
}

/// Envelope-Raman polarisation accumulation `pto[i] += eto[i] · (ρ · p[i])`
/// (`Pout[i] = ρ*E[i]*R.P[i]`, `Nonlinear.jl:457-459`). Shared by Step 3b
/// (`p` = ADE-integrated `raman_p`) and Step 3c (`p` = FFT-convolved
/// `raman_fft_e2`, real since Phase J.3's r2c/c2r halving) — same formula,
/// same accumulation target, different `p`/`ρ` source (Phase J.5 dedup,
/// docs/dev/BACKLOG.md). All three slices must have equal length.
#[inline]
fn raman_accumulate_env(pto: &mut [Complex<f64>], eto: &[Complex<f64>], p: &[f64], rho: f64) {
    for i in 0..pto.len() {
        pto[i] += eto[i] * (rho * p[i]);
    }
}

/// Analytic-signal intensity `1/2·|hilbert(field)|²`, matching
/// `Maths.hilbert`'s FFTW bin convention bit-for-bit (`sqr!`'s `thg=false`
/// branch, `Nonlinear.jl:342-349`): forward c2c FFT, double bins `1..n/2`
/// (0-indexed), zero bins `n/2..n`, DC (bin 0) untouched, inverse c2c FFT
/// normalized by `1/n`. `buf_a`/`buf_b` are scratch (length == `field.len()`)
/// — `ComplexFft1d::forward`/`inverse` require distinct in/out buffers (the
/// plan is built out-of-place, see `fftw.rs`).
fn hilbert_intensity(
    fft: &ComplexFft1d,
    buf_a: &mut [Complex<f64>],
    buf_b: &mut [Complex<f64>],
    field: &[f64],
    out: &mut [f64],
    norm: f64,
) {
    let n = field.len();
    for i in 0..n {
        buf_a[i] = Complex::new(field[i], 0.0);
    }
    fft.forward(buf_a, buf_b);
    let n1 = n / 2;
    for k in 1..n1 {
        buf_b[k] *= 2.0;
    }
    for k in n1..n {
        buf_b[k] = Complex::new(0.0, 0.0);
    }
    fft.inverse(buf_b, buf_a);
    for i in 0..n {
        let a = buf_a[i] * norm;
        out[i] = 0.5 * a.norm_sqr();
    }
}

/// Closed-form z -> pressure for a piecewise pressure-gradient fill
/// (`Capillary.TwoPointGradient`/`MultiPointGradient`, docs/dev/BACKLOG.md Phase F
/// item 3). `zs` (ascending, length >= 2) and `ps` are the breakpoint
/// positions/pressures; flat-clamps outside `[zs[0], zs[last]]` (reproducing
/// `Capillary.gradient`'s own boundary behavior) and otherwise selects the
/// segment `[zs[i],zs[i+1]]` containing `z`, applying the same
/// `sqrt(p0² + t·(p1²-p0²))` interpolation `MultiPointGradient`'s Julia-side
/// `p(z)` uses within that segment. A two-point gradient is just the
/// single-segment case (`zs=[0,L]`).
fn zdep_pressure_at(z: f64, zs: &[f64], ps: &[f64]) -> f64 {
    let n = zs.len();
    if z <= zs[0] {
        return ps[0];
    }
    if z >= zs[n - 1] {
        return ps[n - 1];
    }
    // findlast(x -> x < z, zs): last breakpoint strictly below z.
    let mut i = 0;
    for k in 0..n - 1 {
        if zs[k] < z {
            i = k;
        }
    }
    let (z0, z1, p0, p1) = (zs[i], zs[i + 1], ps[i], ps[i + 1]);
    (p0 * p0 + (z - z0) / (z1 - z0) * (p1 * p1 - p0 * p0)).sqrt()
}

impl CpuNativeSim {
    fn new(n: usize, linop: &[Complex<f64>]) -> Self {
        let z = || vec![Complex::new(0.0, 0.0); n];
        CpuNativeSim {
            n,
            field: z(),
            linop: linop.to_vec(),
            linop_version: 0,
            exp_cache: ExpCache::new(),
            ks: [z(), z(), z(), z(), z(), z(), z()],
            yerr: z(),
            ystage: z(),
            fftw_api: None,
            fft_c2c: None,
            fft_r2c: None,
            fft_c2c_over: None,
            fft_r2c_over: None,
            is_real: false,
            fft_norm: 1.0,
            fft_norm_over: 1.0,
            eto: Vec::new(),
            pto: Vec::new(),
            eoo: Vec::new(),
            poo: Vec::new(),
            towin: Vec::new(),
            owin: Vec::new(),
            sidx: Vec::new(),
            pre: Vec::new(),
            beta: Vec::new(),
            norm_pre_beta: Vec::new(),
            kerr_fac: 0.0,
            nlscale: 0.0,
            sqrt_aeff: 1.0,
            n_time: 0,
            n_time_over: 0,
            n_spec: n,
            n_spec_over: 0,
            et_noise: Vec::new(),
            has_noise: false,
            et_noise_cplx: Vec::new(),
            eto_cplx: Vec::new(),
            pto_cplx: Vec::new(),
            eoo_cplx: Vec::new(),
            poo_cplx: Vec::new(),
            plas_rate: Vec::new(),
            plas_fraction: Vec::new(),
            plas_phase: Vec::new(),
            plas_j: Vec::new(),
            plas_p: Vec::new(),
            plasma_ion_ptr: std::ptr::null(),
            plasma_adk_ptr: std::ptr::null(),
            plasma_is_adk: false,
            plasma_ionpot: 0.0,
            plasma_e_ratio: 0.0,
            plasma_preionfrac: 0.0,
            plasma_dt: 1.0,
            plasma_density: 0.0,
            has_plasma: false,
            is_radial: false,
            radial_et_noise: Vec::new(),
            has_radial_noise: false,
            radial_et_noise_cplx: Vec::new(),
            n_r: 0,
            qdht: None,
            qdht_scale_fwd: 1.0,
            qdht_scale_inv: 1.0,
            radial_m: Vec::new(),
            radial_eto: Vec::new(),
            radial_pto: Vec::new(),
            radial_eoo: Vec::new(),
            radial_poo: Vec::new(),
            radial_eto_c: Vec::new(),
            radial_pto_c: Vec::new(),
            has_raman: false,
            raman_solver: None,
            raman_density: 0.0,
            raman_intensity: Vec::new(),
            raman_p: Vec::new(),
            raman_thg: true,
            raman_hilbert_fft: None,
            raman_hilbert_a: Vec::new(),
            raman_hilbert_b: Vec::new(),
            has_raman_fft: false,
            raman_fft_density: 0.0,
            raman_fft_hw: Vec::new(),
            raman_fft_plan: None,
            raman_fft_e2: Vec::new(),
            raman_fft_ew: Vec::new(),
            is_modal: false,
            n_modes: 0,
            npol: 0,
            modal_a: 0.0,
            modal_unm: Vec::new(),
            modal_inv_sqrt_n: Vec::new(),
            modal_order: Vec::new(),
            modal_kind: Vec::new(),
            modal_phi: Vec::new(),
            modal_full: false,
            modal_pol_select: Vec::new(),
            modal_nlfac: Vec::new(),
            modal_kerr_fac: 0.0,
            cubature: None,
            modal_rtol: 1e-3,
            modal_atol: 0.0,
            modal_maxevals: 512,
            modal_scratch_pool: Vec::new(),
            modal_emega: Vec::new(),
            is_free: false,
            n_y: 0,
            n_x: 0,
            fft_r2c_3d: None,
            fft_c2c_3d: None,
            free_m: Vec::new(),
            free_fft_norm_over: 1.0,
            free_eto: Vec::new(),
            free_pto: Vec::new(),
            free_eoo: Vec::new(),
            free_poo: Vec::new(),
            free_eto_c: Vec::new(),
            free_pto_c: Vec::new(),
            is_zdep_mode_avg: false,
            zdep_grad_z: Vec::new(),
            zdep_grad_p: Vec::new(),
            zdep_dens_lut: None,
            zdep_gamma: Vec::new(),
            zdep_nwg_re: Vec::new(),
            zdep_nwg_im: Vec::new(),
            zdep_omega: Vec::new(),
            zdep_model: 0,
            zdep_loss_on: false,
            zdep_omega0: 0.0,
            zdep_gamma0: 0.0,
            zdep_dgamma0: 0.0,
            zdep_nwg0: Complex::new(0.0, 0.0),
            zdep_dnwg0: Complex::new(0.0, 0.0),
            zdep_kerr_fac_per_dens: 0.0,
            zdep_last_z: f64::NAN,

            is_zdep_free: false,
            zdep_free_flength: 0.0,
            zdep_free_p0: 0.0,
            zdep_free_p1: 0.0,
            zdep_free_dens_lut: None,
            zdep_free_gamma: Vec::new(),
            zdep_free_omega: Vec::new(),
            zdep_free_omegawin: Vec::new(),
            zdep_free_kperp2: Vec::new(),
            zdep_free_sidx: Vec::new(),
            zdep_free_omega0: 0.0,
            zdep_free_gamma0: 0.0,
            zdep_free_dgamma0: 0.0,
            zdep_free_kerr_fac_per_dens: 0.0,
            zdep_free_last_z: f64::NAN,

            is_zdep_modal: false,
            zdep_modal_flength: 0.0,
            zdep_modal_a_lut: None,
            zdep_modal_a0: 0.0,
            zdep_modal_inv_sqrt_n0: Vec::new(),
            zdep_modal_omega0: 0.0,
            zdep_modal_omega: Vec::new(),
            zdep_modal_sidx: Vec::new(),
            zdep_modal_model: 0,
            zdep_modal_loss_on: true,
            zdep_modal_eco: Vec::new(),
            zdep_modal_vn_re: Vec::new(),
            zdep_modal_vn_im: Vec::new(),
            zdep_modal_eco0: Vec::new(),
            zdep_modal_deco0: Vec::new(),
            zdep_modal_v0_re: Vec::new(),
            zdep_modal_v0_im: Vec::new(),
            zdep_modal_dv0_re: Vec::new(),
            zdep_modal_dv0_im: Vec::new(),
            zdep_modal_ref_mode: 0,
            zdep_modal_last_z: f64::NAN,
            n_threads: 1,
            thread_pool: None,
            deterministic: false,
        }
    }

    fn rhs_mode_avg_real(&mut self, idx: usize, eomega: &[Complex<f64>]) {
        // ── Step 1: to_time! (RealGrid) ─────────────────────────────────────────
        let scale_fwd = (self.n_spec_over - 1) as f64 / (self.n_spec - 1) as f64;
        self.eoo.fill(Complex::new(0.0, 0.0));
        for i in 0..self.n_spec {
            self.eoo[i] = eomega[i] * scale_fwd;
        }
        if let Some(ref fft) = self.fft_r2c_over {
            fft.inverse(&mut self.eoo, &mut self.eto);
            let inv_nto = self.fft_norm_over;
            for v in &mut self.eto {
                *v *= inv_nto;
            }
        }

        // ── Step 2: scale by 1/(nlscale·sqrt(Aeff)) ─────────────────────────────
        let sc = self.nlscale * self.sqrt_aeff;
        let inv_sc = 1.0 / sc;
        for v in &mut self.eto {
            *v *= inv_sc;
        }

        // ── Step 2b: modified shot-noise injection (Et_nl = Eto + Et_noise/sc) ──
        // `eto` is scratch for this call only (never read again after this RHS
        // evaluation returns), so adding the noise in place matches Julia's
        // separate Et_nl buffer without needing a second allocation.
        if self.has_noise {
            for i in 0..self.n_time_over {
                self.eto[i] += self.et_noise[i] * inv_sc;
            }
        }

        // ── Step 3: Kerr RHS (KerrScalar!) Pto += kerr_fac · Eto³ ───────────────
        self.pto.fill(0.0);
        for i in 0..self.n_time_over {
            let e = self.eto[i];
            self.pto[i] += self.kerr_fac * e * e * e;
        }

        // ── Step 3b: Plasma polarisation (if enabled) ───────────────────────────
        if self.has_plasma {
            self.apply_plasma_real();
        }

        // ── Step 3c: Raman polarisation (if enabled) ────────────────────────────
        if self.has_raman {
            self.apply_raman_real();
        }

        // ── Step 4: time-window apodization Pto *= towin ────────────────────────
        for i in 0..self.n_time_over {
            self.pto[i] *= self.towin[i];
        }

        // ── Step 5: to_freq! (RealGrid) rfft(Pto) → Pωo, crop+scale → self.ks[idx] ──────
        let scale_inv = (self.n_spec - 1) as f64 / (self.n_spec_over - 1) as f64;
        if let Some(ref fft) = self.fft_r2c_over {
            fft.forward(&mut self.pto, &mut self.poo);
            for i in 0..self.n_spec {
                self.ks[idx][i] = self.poo[i] * scale_inv;
            }
        }

        // ── Step 6: norm! — pre[i]/β[i]*sqrt_aeff, precomputed (identity outside
        // sidx) so this is a plain vectorizable multiply, not a per-element branch.
        for i in 0..self.n_spec {
            self.ks[idx][i] *= self.norm_pre_beta[i];
        }

        // ── Step 7: freq-window apodization — owin already folded to 1.0 outside
        // sidx at construction time (set_mode_avg_params).
        for i in 0..self.n_spec {
            self.ks[idx][i] *= self.owin[i];
        }
    }

    /// Dispatches to whichever ionization handle is wired (PPT spline LUT
    /// or ADK closed form — docs/dev/BACKLOG.md Phase I item 3), matching each
    /// handle's own out-of-domain fallback: PPT clamps to `rate(e_max)`
    /// (see `PptIonizationRate::rate`'s doc); ADK never errors (no upper
    /// domain limit), so its `unwrap_or_else` branch is unreachable.
    ///
    /// # Safety
    /// Exactly one of `plasma_ion_ptr`/`plasma_adk_ptr` must be non-null
    /// and valid, selected by `plasma_is_adk`.
    fn plasma_rate(&self, e_abs: f64) -> f64 {
        if self.plasma_is_adk {
            let ion = unsafe { &*self.plasma_adk_ptr };
            ion.rate(e_abs).unwrap_or(0.0)
        } else {
            let ion = unsafe { &*self.plasma_ion_ptr };
            ion.rate(e_abs)
                .unwrap_or_else(|_| ion.rate(ion.e_max).unwrap_or(0.0))
        }
    }

    /// Plasma polarisation via cumtrapz ×3, added to `self.pto` in-place.
    /// Reproduces `Nonlinear.PlasmaScalar!` (src/Nonlinear.jl:194-206).
    ///
    /// # Safety
    /// `plasma_ion_ptr` must be non-null and valid.
    fn apply_plasma_real(&mut self) {
        let n = self.n_time_over;
        let dt = self.plasma_dt;

        // 1. ionization rate W(|E(t)|)
        for i in 0..n {
            let e_abs = self.eto[i].abs();
            self.plas_rate[i] = self.plasma_rate(e_abs);
        }

        // 2. cumtrapz(fraction, rate, dt) → raw integral ∫₀ᵗ W dτ
        cumtrapz_slice_f64(&self.plas_rate, &mut self.plas_fraction, dt);

        // 3. ρ(t) = preionfrac + 1 − exp(−∫W)
        for i in 0..n {
            self.plas_fraction[i] = self.plasma_preionfrac + 1.0 - (-self.plas_fraction[i]).exp();
        }

        // 4. phase[i] = ρ[i] * e_ratio * E[i]
        for i in 0..n {
            self.plas_phase[i] = self.plas_fraction[i] * self.plasma_e_ratio * self.eto[i];
        }

        // 5. cumtrapz(J, phase, dt) → free-electron current
        cumtrapz_slice_f64(&self.plas_phase, &mut self.plas_j, dt);

        // 6. ionization loss current: J[i] += Ip * W[i] * (1−ρ[i]) / E[i]
        for i in 0..n {
            let e = self.eto[i];
            if e.abs() > 0.0 {
                self.plas_j[i] +=
                    self.plasma_ionpot * self.plas_rate[i] * (1.0 - self.plas_fraction[i]) / e;
            }
        }

        // 7. cumtrapz(P, J, dt) → plasma polarisation
        cumtrapz_slice_f64(&self.plas_j, &mut self.plas_p, dt);

        // 8. add ρ·P(t) to pto — reproduces `out .+= ρ .* Plas.P`
        // (Nonlinear.jl:237-249); previously missing the ρ factor entirely.
        let rho = self.plasma_density;
        for i in 0..n {
            self.pto[i] += rho * self.plas_p[i];
        }
    }

    /// Radial counterpart of `apply_plasma_real` — Phase D.2 (docs/dev/BACKLOG.md),
    /// MATH.md §3.2's deferred plasma-radial follow-up. Same cumtrapz ×3
    /// formula (`PlasmaScalar!`, Nonlinear.jl:194-206), applied independently
    /// per r-column: `NonlinearRHS.jl`'s `Et_to_Pt!` for `TransRadial` calls
    /// each response with a 1-D view of one r-column (`idcs` iterates the r
    /// dimension), so `PlasmaCumtrapz` always sees a scalar field here —
    /// there is no coupling between radial nodes in the plasma response
    /// itself (only the QDHT couples r-columns, and that happens before/after
    /// this step, not within it). Scratch buffers (`plas_*`) are sized
    /// `n_time_over*n_r` by `native_set_plasma_params` when `is_radial`.
    ///
    /// # Safety
    /// `plasma_ion_ptr` must be non-null and valid.
    fn apply_plasma_radial(&mut self) {
        let n_time_over = self.n_time_over;
        let n_r = self.n_r;
        let dt = self.plasma_dt;
        let n_threads = self.n_threads;

        if n_threads > 1 {
            // docs/dev/BACKLOG.md S2 item 2: `plas_rate`/`plas_fraction`/`plas_phase`/
            // `plas_j`/`plas_p` are already `n_time_over*n_r` matrices (see
            // struct doc) — the per-column body below only ever reads/writes
            // its own disjoint `[r*n_time_over, (r+1)*n_time_over)` slice of
            // each, so this is safe-by-construction, not a new scratch
            // layout. `plasma_rate`'s `&self` method can't be called from
            // inside a rayon closure that also holds disjoint `&mut` field
            // borrows (the borrow checker can't see that it only touches
            // `plasma_is_adk`/`plasma_adk_ptr`/`plasma_ion_ptr`), so the raw
            // pointers are copied out first and `plasma_rate_at` (a free
            // function, no `self`) is called instead — identical logic to
            // `plasma_rate`.
            let is_adk = self.plasma_is_adk;
            let adk_ptr = ReadOnlyPtr(self.plasma_adk_ptr);
            let ion_ptr = ReadOnlyPtr(self.plasma_ion_ptr);
            let preionfrac = self.plasma_preionfrac;
            let e_ratio = self.plasma_e_ratio;
            let ionpot = self.plasma_ionpot;

            let pool = self.thread_pool.get_or_insert_with(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .build()
                    .expect(
                        "failed to build rayon thread pool for native RHS (docs/dev/BACKLOG.md S2)",
                    )
            });
            let eto = &self.radial_eto;
            let rate = &mut self.plas_rate;
            let frac = &mut self.plas_fraction;
            let phase = &mut self.plas_phase;
            let j = &mut self.plas_j;
            let p = &mut self.plas_p;
            pool.install(|| {
                eto.par_chunks(n_time_over)
                    .zip(rate.par_chunks_mut(n_time_over))
                    .zip(frac.par_chunks_mut(n_time_over))
                    .zip(phase.par_chunks_mut(n_time_over))
                    .zip(j.par_chunks_mut(n_time_over))
                    .zip(p.par_chunks_mut(n_time_over))
                    .for_each(|(((((eto_c, rate_c), frac_c), phase_c), j_c), p_c)| {
                        // 1. ionization rate W(|E(t)|)
                        for i in 0..eto_c.len() {
                            rate_c[i] = plasma_rate_at(is_adk, adk_ptr, ion_ptr, eto_c[i].abs());
                        }
                        // 2. cumtrapz(fraction, rate, dt) → raw integral ∫₀ᵗ W dτ
                        cumtrapz_slice_f64(rate_c, frac_c, dt);
                        // 3. ρ(t) = preionfrac + 1 − exp(−∫W)
                        for i in 0..frac_c.len() {
                            frac_c[i] = preionfrac + 1.0 - (-frac_c[i]).exp();
                        }
                        // 4. phase[i] = ρ[i] * e_ratio * E[i]
                        for i in 0..phase_c.len() {
                            phase_c[i] = frac_c[i] * e_ratio * eto_c[i];
                        }
                        // 5. cumtrapz(J, phase, dt) → free-electron current
                        cumtrapz_slice_f64(phase_c, j_c, dt);
                        // 6. ionization loss current: J[i] += Ip*W[i]*(1−ρ[i])/E[i]
                        for i in 0..j_c.len() {
                            let e = eto_c[i];
                            if e.abs() > 0.0 {
                                j_c[i] += ionpot * rate_c[i] * (1.0 - frac_c[i]) / e;
                            }
                        }
                        // 7. cumtrapz(P, J, dt) → plasma polarisation
                        cumtrapz_slice_f64(j_c, p_c, dt);
                    });
            });
        } else {
            // Sequential path — bit-identical to pre-S2 behavior, untouched.
            for r in 0..n_r {
                let start = r * n_time_over;
                let end = start + n_time_over;

                // 1. ionization rate W(|E(t)|)
                for i in start..end {
                    let e_abs = self.radial_eto[i].abs();
                    self.plas_rate[i] = self.plasma_rate(e_abs);
                }

                // 2. cumtrapz(fraction, rate, dt) → raw integral ∫₀ᵗ W dτ, per column
                cumtrapz_slice_f64(
                    &self.plas_rate[start..end],
                    &mut self.plas_fraction[start..end],
                    dt,
                );

                // 3. ρ(t) = preionfrac + 1 − exp(−∫W)
                for i in start..end {
                    self.plas_fraction[i] =
                        self.plasma_preionfrac + 1.0 - (-self.plas_fraction[i]).exp();
                }

                // 4. phase[i] = ρ[i] * e_ratio * E[i]
                for i in start..end {
                    self.plas_phase[i] =
                        self.plas_fraction[i] * self.plasma_e_ratio * self.radial_eto[i];
                }

                // 5. cumtrapz(J, phase, dt) → free-electron current, per column
                cumtrapz_slice_f64(
                    &self.plas_phase[start..end],
                    &mut self.plas_j[start..end],
                    dt,
                );

                // 6. ionization loss current: J[i] += Ip * W[i] * (1−ρ[i]) / E[i]
                for i in start..end {
                    let e = self.radial_eto[i];
                    if e.abs() > 0.0 {
                        self.plas_j[i] +=
                            self.plasma_ionpot * self.plas_rate[i] * (1.0 - self.plas_fraction[i])
                                / e;
                    }
                }

                // 7. cumtrapz(P, J, dt) → plasma polarisation, per column
                cumtrapz_slice_f64(&self.plas_j[start..end], &mut self.plas_p[start..end], dt);
            }
        }

        // 8. add ρ·P(t,r) to radial_pto (see apply_plasma_real's step 8 doc)
        let rho = self.plasma_density;
        for i in 0..(n_time_over * n_r) {
            self.radial_pto[i] += rho * self.plas_p[i];
        }
    }

    /// Raman polarisation via the resident time-domain ADE solver (`thg=true`:
    /// intensity = E², no Hilbert transform), added to `self.pto` in-place.
    /// Reproduces `(R::RamanPolar)(out, Et, ρ)` for `RamanPolarField`
    /// (src/Nonlinear.jl:357-431) — see MATH.md §5.3.
    fn apply_raman_real(&mut self) {
        let n = self.n_time_over;
        if self.raman_thg {
            for i in 0..n {
                let e = self.eto[i];
                self.raman_intensity[i] = e * e;
            }
        } else {
            let fft = self.raman_hilbert_fft.take();
            if let Some(ref fft) = fft {
                let norm = self.fft_norm_over;
                hilbert_intensity(
                    fft,
                    &mut self.raman_hilbert_a,
                    &mut self.raman_hilbert_b,
                    &self.eto[..n],
                    &mut self.raman_intensity[..n],
                    norm,
                );
            }
            self.raman_hilbert_fft = fft;
        }
        if let Some(ref mut solver) = self.raman_solver {
            solver.solve(&self.raman_intensity, &mut self.raman_p);
        }
        let rho = self.raman_density;
        for i in 0..n {
            self.pto[i] += rho * self.eto[i] * self.raman_p[i];
        }
    }

    /// Radial counterpart of `apply_raman_real` — Phase D.4 (docs/dev/BACKLOG.md).
    /// Same `thg=true` scalar-intensity ADE solve, applied independently per
    /// r-column: `Et_to_Pt!` for `TransRadial` calls each response with a
    /// 1-D view of one r-column, so `RamanPolarField` sees a scalar field
    /// here exactly like `PlasmaCumtrapz` (see `apply_plasma_radial`'s doc).
    /// `solve()` resets its oscillator state at entry, so the single
    /// resident `raman_solver` can be reused sequentially across columns —
    /// no per-column solver instance is needed. Scratch buffers
    /// (`raman_intensity`/`raman_p`) are sized `n_time_over*n_r` by
    /// `native_set_raman_params` when `is_radial`.
    fn apply_raman_radial(&mut self) {
        let n_time_over = self.n_time_over;
        let n_r = self.n_r;
        let rho = self.raman_density;
        let thg = self.raman_thg;
        let n_threads = self.n_threads;

        if n_threads > 1 && n_r > 1 {
            // docs/dev/BACKLOG.md S2 Phase 4 item 1: parallelize the per-r-column
            // Raman ADE solve. Unlike plasma (stateless per column), the solver
            // and the Hilbert scratch (`raman_hilbert_a`/`_b`) are *shared
            // mutable* state on the sequential path, so naive `par_chunks_mut`
            // would race exactly as the S2 plan warns. Fix: partition the
            // r-columns into <= n_threads *contiguous* groups and give each
            // parallel task its OWN cloned solver and OWN Hilbert a/b scratch —
            // no `current_thread_index`, no interior mutability. Each task
            // provably touches only disjoint column slices of every buffer AND
            // owns its own solver/scratch. `solve()` resets oscillator state at
            // entry (raman.rs), so a cloned solver produces output *bit-identical*
            // to the shared one — the n_threads=1-vs-N exact-equality the S2 plan
            // mandates (there is no cross-column reduction either: the final
            // `pto += rho·eto·p` accumulate is elementwise/per-column). The FFT
            // plan is `Sync` and shared read-only (fftw_execute is thread-safe
            // against distinct buffers, which the per-task a/b are).
            let norm = self.fft_norm_over;
            let group_len = n_r.div_ceil(n_threads);
            let chunk = group_len * n_time_over;
            let n_chunks = n_r.div_ceil(group_len);

            let base_solver = self.raman_solver.clone();
            let scratch_len = if thg { 0 } else { n_time_over };
            let scratch: Vec<(
                Option<TimeDomainRamanSolver>,
                Vec<Complex<f64>>,
                Vec<Complex<f64>>,
            )> = (0..n_chunks)
                .map(|_| {
                    (
                        base_solver.clone(),
                        vec![Complex::new(0.0, 0.0); scratch_len],
                        vec![Complex::new(0.0, 0.0); scratch_len],
                    )
                })
                .collect();

            let pool = self.thread_pool.get_or_insert_with(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n_threads)
                    .build()
                    .expect(
                        "failed to build rayon thread pool for native RHS (docs/dev/BACKLOG.md S2)",
                    )
            });

            let fft_ref = self.raman_hilbert_fft.as_ref();
            let eto = &self.radial_eto;
            let pto = &mut self.radial_pto;
            let inten = &mut self.raman_intensity;
            let pol = &mut self.raman_p;

            pool.install(|| {
                eto.par_chunks(chunk)
                    .zip(pto.par_chunks_mut(chunk))
                    .zip(inten.par_chunks_mut(chunk))
                    .zip(pol.par_chunks_mut(chunk))
                    .zip(scratch.into_par_iter())
                    .for_each(
                        |(
                            (((eto_c, pto_c), inten_c), pol_c),
                            (mut solver_opt, mut buf_a, mut buf_b),
                        )| {
                            let ncol = eto_c.len() / n_time_over;
                            for col in 0..ncol {
                                let s = col * n_time_over;
                                let e = s + n_time_over;
                                if thg {
                                    for i in s..e {
                                        let ev = eto_c[i];
                                        inten_c[i] = ev * ev;
                                    }
                                } else if let Some(fft) = fft_ref {
                                    hilbert_intensity(
                                        fft,
                                        &mut buf_a,
                                        &mut buf_b,
                                        &eto_c[s..e],
                                        &mut inten_c[s..e],
                                        norm,
                                    );
                                }
                                if let Some(ref mut solver) = solver_opt {
                                    solver.solve(&inten_c[s..e], &mut pol_c[s..e]);
                                }
                                for i in s..e {
                                    pto_c[i] += rho * eto_c[i] * pol_c[i];
                                }
                            }
                        },
                    );
            });
        } else {
            // Sequential path — bit-identical to pre-S2 behavior, untouched.
            let fft = self.raman_hilbert_fft.take();
            for r in 0..n_r {
                let start = r * n_time_over;
                let end = start + n_time_over;
                if thg {
                    for i in start..end {
                        let e = self.radial_eto[i];
                        self.raman_intensity[i] = e * e;
                    }
                } else if let Some(ref fft) = fft {
                    let norm = self.fft_norm_over;
                    hilbert_intensity(
                        fft,
                        &mut self.raman_hilbert_a,
                        &mut self.raman_hilbert_b,
                        &self.radial_eto[start..end],
                        &mut self.raman_intensity[start..end],
                        norm,
                    );
                }
                if let Some(ref mut solver) = self.raman_solver {
                    solver.solve(
                        &self.raman_intensity[start..end],
                        &mut self.raman_p[start..end],
                    );
                }
                for i in start..end {
                    self.radial_pto[i] += rho * self.radial_eto[i] * self.raman_p[i];
                }
            }
            self.raman_hilbert_fft = fft;
        }
    }

    /// Free-space counterpart of `apply_plasma_radial` — docs/dev/BACKLOG.md Phase I
    /// item 6. `TransFree`'s `Et_to_Pt!` dispatches each response over
    /// `t.idcs` (`CartesianIndices((Ny,Nx))`), so `PlasmaCumtrapz` sees a
    /// scalar field per `(y,x)` column exactly like the radial `TransRadial`
    /// case — same per-column cumtrapz-×3 formula, just column-major over
    /// `n_y*n_x` columns instead of `n_r`. RealGrid only (Julia's `PlasmaCumtrapz`
    /// is inherently field-resolved; EnvGrid free-space, `rhs_free_env`, has no
    /// plasma step, matching the fact that no EnvGrid geometry gets native
    /// plasma anywhere in this port).
    fn apply_plasma_free(&mut self) {
        let n_time_over = self.n_time_over;
        let n_cols = self.n_y * self.n_x;
        let dt = self.plasma_dt;

        for c in 0..n_cols {
            let start = c * n_time_over;
            let end = start + n_time_over;

            for i in start..end {
                let e_abs = self.free_eto[i].abs();
                self.plas_rate[i] = self.plasma_rate(e_abs);
            }
            cumtrapz_slice_f64(
                &self.plas_rate[start..end],
                &mut self.plas_fraction[start..end],
                dt,
            );
            for i in start..end {
                self.plas_fraction[i] =
                    self.plasma_preionfrac + 1.0 - (-self.plas_fraction[i]).exp();
            }
            for i in start..end {
                self.plas_phase[i] = self.plas_fraction[i] * self.plasma_e_ratio * self.free_eto[i];
            }
            cumtrapz_slice_f64(
                &self.plas_phase[start..end],
                &mut self.plas_j[start..end],
                dt,
            );
            for i in start..end {
                let e = self.free_eto[i];
                if e.abs() > 0.0 {
                    self.plas_j[i] +=
                        self.plasma_ionpot * self.plas_rate[i] * (1.0 - self.plas_fraction[i]) / e;
                }
            }
            cumtrapz_slice_f64(&self.plas_j[start..end], &mut self.plas_p[start..end], dt);
        }

        let rho = self.plasma_density;
        for i in 0..(n_time_over * n_cols) {
            self.free_pto[i] += rho * self.plas_p[i];
        }
    }

    /// Free-space counterpart of `apply_raman_radial` — docs/dev/BACKLOG.md Phase I
    /// item 6. Same per-column reuse of the resident `raman_solver` (its
    /// `solve()` resets internal oscillator state at entry, so no cross-column
    /// leakage); RealGrid only (`RamanPolarField`, either `thg` value —
    /// `RamanPolarEnv` is not wired for any geometry but mode-averaged EnvGrid).
    fn apply_raman_free(&mut self) {
        let n_time_over = self.n_time_over;
        let n_cols = self.n_y * self.n_x;
        let rho = self.raman_density;

        let thg = self.raman_thg;
        let fft = self.raman_hilbert_fft.take();
        for c in 0..n_cols {
            let start = c * n_time_over;
            let end = start + n_time_over;
            if thg {
                for i in start..end {
                    let e = self.free_eto[i];
                    self.raman_intensity[i] = e * e;
                }
            } else if let Some(ref fft) = fft {
                let norm = self.fft_norm_over;
                hilbert_intensity(
                    fft,
                    &mut self.raman_hilbert_a,
                    &mut self.raman_hilbert_b,
                    &self.free_eto[start..end],
                    &mut self.raman_intensity[start..end],
                    norm,
                );
            }
            if let Some(ref mut solver) = self.raman_solver {
                solver.solve(
                    &self.raman_intensity[start..end],
                    &mut self.raman_p[start..end],
                );
            }
            for i in start..end {
                self.free_pto[i] += rho * self.free_eto[i] * self.raman_p[i];
            }
        }
        self.raman_hilbert_fft = fft;
    }

    /// EnvGrid mode-averaged RHS: Kerr_env (|E|²·E) via c2c FFTW plans.
    /// Reproduces `TransModeAvg` for `ComplexF64` fields (src/NonlinearRHS.jl:531).
    fn rhs_mode_avg_env(&mut self, idx: usize, eomega: &[Complex<f64>]) {
        let n = self.n_spec; // = n_time for EnvGrid (c2c: Nω = Nt)
        let no = self.n_time_over;
        let half = n / 2;

        // ── Step 1: to_time! (EnvGrid c2c) ──────────────────────────────────────
        // copy_scale_both: first and last N÷2 elements; zero-pad middle.
        // scale = No/N  (NonlinearRHS.jl:124)
        let scale_fwd = no as f64 / n as f64;
        self.eoo_cplx.fill(Complex::new(0.0, 0.0));
        for i in 0..half {
            self.eoo_cplx[i] = eomega[i] * scale_fwd;
        }
        for i in 0..half {
            self.eoo_cplx[no - half + i] = eomega[n - half + i] * scale_fwd;
        }

        if let Some(ref fft) = self.fft_c2c_over {
            fft.inverse(&mut self.eoo_cplx, &mut self.eto_cplx);
            let inv_nto = self.fft_norm_over;
            for v in &mut self.eto_cplx {
                *v *= inv_nto;
            }
        }

        // ── Step 2: scale by 1/(nlscale·√Aeff) ──────────────────────────────────
        let inv_sc = 1.0 / (self.nlscale * self.sqrt_aeff);
        for v in &mut self.eto_cplx {
            *v *= inv_sc;
        }

        // ── Step 2b: modified shot-noise injection (Et_nl = Eto + Et_noise/sc) ──
        if self.has_noise {
            for i in 0..no {
                self.eto_cplx[i] += self.et_noise_cplx[i] * inv_sc;
            }
        }

        // ── Step 3: Kerr_env: pto_cplx[i] += (3/4)*kerr_fac · |E|² · E ──────────
        // Factor 3/4: SVEA averaging of E³ over carrier oscillation
        // (matches KerrScalarEnv! in Nonlinear.jl:121).
        self.pto_cplx.fill(Complex::new(0.0, 0.0));
        let kf = Complex::new(0.75 * self.kerr_fac, 0.0);
        for i in 0..no {
            let e = self.eto_cplx[i];
            self.pto_cplx[i] += kf * e.norm_sqr() * e;
        }

        // ── Step 3b: Raman polarisation (envelope) — Phase F.2 ──────────────────
        // Reproduces `(R::RamanPolar)(out, Et, ρ)` for `RamanPolarEnv`
        // (Nonlinear.jl:357-430): `sqr!(R::RamanPolarEnv, E) = 1/2·|E|²`
        // (Nonlinear.jl:351-354) has no `thg` branch at all (unlike
        // `RamanPolarField`), so `raman_thg` is irrelevant here — always the
        // same real-intensity formula. Reuses the identical resident
        // `raman_solver` (real f64 in/out arrays) already wired for the
        // RealGrid path; only the intensity source (complex envelope, not
        // real field) and the final accumulation (complex × real, not
        // real × real) differ.
        if self.has_raman {
            raman_intensity_half_env(&self.eto_cplx[..no], &mut self.raman_intensity[..no]);
            if let Some(ref mut solver) = self.raman_solver {
                solver.solve(&self.raman_intensity[..no], &mut self.raman_p[..no]);
            }
            raman_accumulate_env(
                &mut self.pto_cplx[..no],
                &self.eto_cplx[..no],
                &self.raman_p[..no],
                self.raman_density,
            );
        }

        // ── Step 3c: Raman polarisation, intermediate-broadening (`:SiO2`) —
        // Phase I item 2. Resident FFT-conv counterpart of the block above,
        // for `RamanRespIntermediateBroadening` (no finite-SDO form, see
        // `set_raman_fft_params`'s doc comment). `has_raman`/`has_raman_fft`
        // are mutually exclusive (only one Raman response ever present).
        // r2c/c2r (Phase J.3, docs/dev/BACKLOG.md, MATH.md §8.4): `raman_fft_e2`
        // is real (both the padded intensity and the convolution result `P`
        // are mathematically real), so `raman_fft_plan` is a `RealFft1d` and
        // `raman_fft_ew`'s spectrum is the shorter `n_over/2+1` Hermitian half.
        if self.has_raman_fft {
            raman_intensity_half_env(&self.eto_cplx[..no], &mut self.raman_fft_e2[..no]);
            // The upper half must be genuinely zero-padded every step:
            // `plan.inverse` below writes the full 2N-length P back into
            // this same buffer, so from the second RHS call onward
            // `[no..2no)` would otherwise hold the previous step's
            // convolution tail. Julia never hits this (`R.E2`'s upper half
            // is a separate, never-written buffer), and the double-length
            // grid exists precisely to prevent truncation/wrap-around
            // artefacts (Nonlinear.jl:406-411) — don't rely on h's tail
            // happening to be zero at the wrap distance.
            for i in no..self.raman_fft_e2.len() {
                self.raman_fft_e2[i] = 0.0;
            }
            if let Some(ref plan) = self.raman_fft_plan {
                plan.forward(&mut self.raman_fft_e2, &mut self.raman_fft_ew);
                for k in 0..self.raman_fft_ew.len() {
                    self.raman_fft_ew[k] *= self.raman_fft_hw[k];
                }
                // ifft is unnormalized; `raman_fft_hw` already folds in the
                // `1/n_over` factor (see `set_raman_fft_params`), so the
                // result of `inverse` here is already the true `P`, and is
                // exactly real (c2r admits no imaginary component, unlike
                // the earlier c2c path's ~1e-16 imaginary rounding noise).
                plan.inverse(&mut self.raman_fft_ew, &mut self.raman_fft_e2);
            }
            raman_accumulate_env(
                &mut self.pto_cplx[..no],
                &self.eto_cplx[..no],
                &self.raman_fft_e2[..no],
                self.raman_fft_density,
            );
        }

        // ── Step 4: time-window apodization ─────────────────────────────────────
        for i in 0..no {
            self.pto_cplx[i] *= self.towin[i];
        }

        // ── Step 5: to_freq! (c2c): fft → copy_scale_both_back → ks[idx] ────────
        // scale = N/No  (NonlinearRHS.jl:146)
        let scale_inv = n as f64 / no as f64;
        if let Some(ref fft) = self.fft_c2c_over {
            fft.forward(&mut self.pto_cplx, &mut self.poo_cplx);
            // zero out ks[idx] first (positions outside first/last half stay 0)
            self.ks[idx].fill(Complex::new(0.0, 0.0));
            for i in 0..half {
                self.ks[idx][i] = self.poo_cplx[i] * scale_inv;
            }
            for i in 0..half {
                self.ks[idx][n - half + i] = self.poo_cplx[no - half + i] * scale_inv;
            }
        }

        // ── Step 6: norm! and freq-window — precomputed, identity outside sidx
        // (see set_mode_avg_params / rhs_mode_avg_real's Step 6/7 comment).
        for i in 0..n {
            self.ks[idx][i] *= self.norm_pre_beta[i];
        }
        for i in 0..n {
            self.ks[idx][i] *= self.owin[i];
        }
    }

    /// Radial (`TransRadial`) RHS: QDHT-wrapped scalar Kerr (`E³`) for a
    /// RealGrid field. Reproduces `TransRadial.__call__`
    /// (src/NonlinearRHS.jl:663) for `Nonlinear.Kerr_field` only (no plasma,
    /// no shot-noise — see MATH.md §3.2 for scope).
    ///
    /// All 2-D buffers here are column-major `(n_time, n_r)`: column `r` is
    /// `n_time` contiguous elements — required by `QdhtFfiHandle::apply_real`
    /// and what makes each column contiguous for the looped rank-1 FFT.
    fn rhs_radial(&mut self, idx: usize, eomega: &[Complex<f64>]) {
        let n_r = self.n_r;
        let n_spec = self.n_spec;
        let n_spec_over = self.n_spec_over;
        let n_time_over = self.n_time_over;

        // ── Step 1: to_time! per r-column (RealGrid r2c) ─────────────────────────
        let scale_fwd = (n_spec_over - 1) as f64 / (n_spec - 1) as f64;
        self.radial_eoo.fill(Complex::new(0.0, 0.0));
        for r in 0..n_r {
            let in_col = &eomega[r * n_spec..(r + 1) * n_spec];
            let out_col = &mut self.radial_eoo[r * n_spec_over..(r + 1) * n_spec_over];
            for i in 0..n_spec {
                out_col[i] = in_col[i] * scale_fwd;
            }
        }
        if let Some(ref fft) = self.fft_r2c_over {
            // docs/dev/BACKLOG.md S2 item 2: each r-column's inverse FFT reads/writes
            // a disjoint slice of `radial_eoo`/`radial_eto` — FFTW's
            // `fftw_execute_dft_r2c`/`_c2r` (what `RealFft1d::inverse` calls)
            // is documented thread-safe against one shared plan with
            // distinct buffers (see `fftw.rs`'s `RealFft1d`-`Sync` doc), so
            // this is safe without any per-worker scratch.
            if self.n_threads > 1 {
                let n_threads = self.n_threads;
                let pool = self.thread_pool.get_or_insert_with(|| {
                    rayon::ThreadPoolBuilder::new().num_threads(n_threads).build()
                        .expect("failed to build rayon thread pool for native RHS (docs/dev/BACKLOG.md S2)")
                });
                let eoo = &mut self.radial_eoo;
                let eto = &mut self.radial_eto;
                pool.install(|| {
                    eoo.par_chunks_mut(n_spec_over)
                        .zip(eto.par_chunks_mut(n_time_over))
                        .for_each(|(spec_col, time_col)| fft.inverse(spec_col, time_col));
                });
            } else {
                for r in 0..n_r {
                    let spec_start = r * n_spec_over;
                    let time_start = r * n_time_over;
                    let spec_col = &mut self.radial_eoo[spec_start..spec_start + n_spec_over];
                    let time_col = &mut self.radial_eto[time_start..time_start + n_time_over];
                    fft.inverse(spec_col, time_col);
                }
            }
            let inv_nto = self.fft_norm_over;
            for v in &mut self.radial_eto {
                *v *= inv_nto;
            }
        }

        // ── Step 2: QDHT ldiv! (k → r) ────────────────────────────────────────────
        let scale_inv = self.qdht_scale_inv;
        if let Some(ref mut qdht) = self.qdht {
            qdht.apply_real(&mut self.radial_eto, n_time_over, scale_inv);
        }

        // ── Step 2b: modified shot-noise injection (Et_nl = Eto + Et_noise) ──────
        // `radial_eto` is scratch for this call only, so adding in place
        // matches Julia's separate Et_nl buffer without an extra allocation.
        if self.has_radial_noise {
            for i in 0..(n_time_over * n_r) {
                self.radial_eto[i] += self.radial_et_noise[i];
            }
        }

        // ── Step 3: Kerr (KerrScalar!): Pto += kerr_fac · Eto³, pointwise (t,r) ───
        self.radial_pto.fill(0.0);
        for i in 0..(n_time_over * n_r) {
            let e = self.radial_eto[i];
            self.radial_pto[i] += self.kerr_fac * e * e * e;
        }

        // ── Step 3b: Plasma polarisation (if enabled), per r-column ──────────────
        if self.has_plasma {
            self.apply_plasma_radial();
        }

        // ── Step 3c: Raman polarisation (if enabled), per r-column ────────────────
        if self.has_raman {
            self.apply_raman_radial();
        }

        // ── Step 4: time-window apodization per r-column (reuses 1-D towin) ──────
        for r in 0..n_r {
            let start = r * n_time_over;
            for t in 0..n_time_over {
                self.radial_pto[start + t] *= self.towin[t];
            }
        }

        // ── Step 5: QDHT mul! (r → k) ─────────────────────────────────────────────
        let scale_fwd_q = self.qdht_scale_fwd;
        if let Some(ref mut qdht) = self.qdht {
            qdht.apply_real(&mut self.radial_pto, n_time_over, scale_fwd_q);
        }

        // ── Step 6: to_freq! per r-column (RealGrid r2c) ─────────────────────────
        let scale_inv2 = (n_spec - 1) as f64 / (n_spec_over - 1) as f64;
        if let Some(ref fft) = self.fft_r2c_over {
            // Same disjoint-slices argument as Step 1 above.
            if self.n_threads > 1 {
                let n_threads = self.n_threads;
                let pool = self.thread_pool.get_or_insert_with(|| {
                    rayon::ThreadPoolBuilder::new().num_threads(n_threads).build()
                        .expect("failed to build rayon thread pool for native RHS (docs/dev/BACKLOG.md S2)")
                });
                let pto = &mut self.radial_pto;
                let poo = &mut self.radial_poo;
                pool.install(|| {
                    pto.par_chunks_mut(n_time_over)
                        .zip(poo.par_chunks_mut(n_spec_over))
                        .for_each(|(time_col, spec_col)| fft.forward(time_col, spec_col));
                });
            } else {
                for r in 0..n_r {
                    let time_start = r * n_time_over;
                    let spec_start = r * n_spec_over;
                    let time_col = &mut self.radial_pto[time_start..time_start + n_time_over];
                    let spec_col = &mut self.radial_poo[spec_start..spec_start + n_spec_over];
                    fft.forward(time_col, spec_col);
                }
            }
        }
        self.ks[idx].fill(Complex::new(0.0, 0.0));
        for r in 0..n_r {
            let in_start = r * n_spec_over;
            let out_start = r * n_spec;
            for i in 0..n_spec {
                self.ks[idx][out_start + i] = self.radial_poo[in_start + i] * scale_inv2;
            }
        }

        // ── Step 7: normalization — ks[idx] *= M (folds norm_radial + ωwin) ──────
        for i in 0..(n_spec * n_r) {
            self.ks[idx][i] *= self.radial_m[i];
        }
    }

    /// EnvGrid (c2c) counterpart of `rhs_radial` — Phase D.1 (docs/dev/BACKLOG.md),
    /// MATH.md §3.2's deferred EnvGrid-radial follow-up. Mirrors
    /// `rhs_mode_avg_env`'s c2c `to_time!`/`to_freq!` convention
    /// (half-spectrum zero-pad, `no/n` scale, 3/4 `Kerr_env` SVEA factor —
    /// `Nonlinear.jl:121` `KerrScalarEnv!`) applied per r-column, and reuses
    /// the resident `QdhtFfiHandle`'s `apply_cplx` instead of `apply_real`.
    ///
    /// All 2-D buffers are column-major `(n_time, n_r)` exactly like
    /// `rhs_radial` (see that function's doc for the layout invariant),
    /// except elements are `Complex<f64>` (`radial_eto_c`/`radial_pto_c`)
    /// rather than `f64` — `radial_eoo`/`radial_poo` (frequency-domain) are
    /// already `Complex<f64>` and are shared with the RealGrid path as-is.
    fn rhs_radial_env(&mut self, idx: usize, eomega: &[Complex<f64>]) {
        let n_r = self.n_r;
        let n = self.n_spec; // = n_time for EnvGrid (c2c: Nω = Nt)
        let no = self.n_time_over;
        let half = n / 2;

        // ── Step 1: to_time! per r-column (EnvGrid c2c, half-spectrum copy-both) ──
        let scale_fwd = no as f64 / n as f64;
        self.radial_eoo.fill(Complex::new(0.0, 0.0));
        for r in 0..n_r {
            let in_col = &eomega[r * n..(r + 1) * n];
            let out_col = &mut self.radial_eoo[r * no..(r + 1) * no];
            for i in 0..half {
                out_col[i] = in_col[i] * scale_fwd;
            }
            for i in 0..half {
                out_col[no - half + i] = in_col[n - half + i] * scale_fwd;
            }
        }
        if let Some(ref fft) = self.fft_c2c_over {
            // Same disjoint-column-slices argument as rhs_radial's Step 1
            // (docs/dev/BACKLOG.md S2 item 2) — `ComplexFft1d` is `Sync` for the
            // same reason.
            if self.n_threads > 1 {
                let n_threads = self.n_threads;
                let pool = self.thread_pool.get_or_insert_with(|| {
                    rayon::ThreadPoolBuilder::new().num_threads(n_threads).build()
                        .expect("failed to build rayon thread pool for native RHS (docs/dev/BACKLOG.md S2)")
                });
                let eoo = &mut self.radial_eoo;
                let eto_c = &mut self.radial_eto_c;
                pool.install(|| {
                    eoo.par_chunks_mut(no)
                        .zip(eto_c.par_chunks_mut(no))
                        .for_each(|(spec_col, time_col)| fft.inverse(spec_col, time_col));
                });
            } else {
                for r in 0..n_r {
                    let spec_start = r * no;
                    let time_start = r * no;
                    let spec_col = &mut self.radial_eoo[spec_start..spec_start + no];
                    let time_col = &mut self.radial_eto_c[time_start..time_start + no];
                    fft.inverse(spec_col, time_col);
                }
            }
            let inv_nto = self.fft_norm_over;
            for v in &mut self.radial_eto_c {
                *v *= inv_nto;
            }
        }

        // ── Step 2: QDHT ldiv! (k → r), complex ──────────────────────────────────
        let scale_inv = self.qdht_scale_inv;
        if let Some(ref mut qdht) = self.qdht {
            // Safety: `Complex<f64>` is `#[repr(C)]` (two contiguous `f64`s),
            // same idiom as `rhs_modal`'s `ks[idx]` reinterpretation above.
            let buf = unsafe {
                std::slice::from_raw_parts_mut(
                    self.radial_eto_c.as_mut_ptr() as *mut f64,
                    2 * self.radial_eto_c.len(),
                )
            };
            qdht.apply_cplx(buf, no, scale_inv);
        }

        // ── Step 2b: modified shot-noise injection (Et_nl = Eto + Et_noise) ──────
        if self.has_radial_noise {
            for i in 0..(no * n_r) {
                self.radial_eto_c[i] += self.radial_et_noise_cplx[i];
            }
        }

        // ── Step 3: Kerr_env: Pto += (3/4)*kerr_fac · |E|² · E, pointwise (t,r) ───
        let kf = Complex::new(0.75 * self.kerr_fac, 0.0);
        self.radial_pto_c.fill(Complex::new(0.0, 0.0));
        for i in 0..(no * n_r) {
            let e = self.radial_eto_c[i];
            self.radial_pto_c[i] += kf * e.norm_sqr() * e;
        }

        // ── Step 4: time-window apodization per r-column ─────────────────────────
        for r in 0..n_r {
            let start = r * no;
            for t in 0..no {
                self.radial_pto_c[start + t] *= self.towin[t];
            }
        }

        // ── Step 5: QDHT mul! (r → k), complex ───────────────────────────────────
        let scale_fwd_q = self.qdht_scale_fwd;
        if let Some(ref mut qdht) = self.qdht {
            let buf = unsafe {
                std::slice::from_raw_parts_mut(
                    self.radial_pto_c.as_mut_ptr() as *mut f64,
                    2 * self.radial_pto_c.len(),
                )
            };
            qdht.apply_cplx(buf, no, scale_fwd_q);
        }

        // ── Step 6: to_freq! per r-column (EnvGrid c2c) ──────────────────────────
        let scale_inv2 = n as f64 / no as f64;
        if let Some(ref fft) = self.fft_c2c_over {
            // Same disjoint-column-slices argument as Step 1 above.
            if self.n_threads > 1 {
                let n_threads = self.n_threads;
                let pool = self.thread_pool.get_or_insert_with(|| {
                    rayon::ThreadPoolBuilder::new().num_threads(n_threads).build()
                        .expect("failed to build rayon thread pool for native RHS (docs/dev/BACKLOG.md S2)")
                });
                let pto_c = &mut self.radial_pto_c;
                let poo = &mut self.radial_poo;
                pool.install(|| {
                    pto_c
                        .par_chunks_mut(no)
                        .zip(poo.par_chunks_mut(no))
                        .for_each(|(time_col, spec_col)| fft.forward(time_col, spec_col));
                });
            } else {
                for r in 0..n_r {
                    let time_start = r * no;
                    let spec_start = r * no;
                    let time_col = &mut self.radial_pto_c[time_start..time_start + no];
                    let spec_col = &mut self.radial_poo[spec_start..spec_start + no];
                    fft.forward(time_col, spec_col);
                }
            }
        }
        self.ks[idx].fill(Complex::new(0.0, 0.0));
        for r in 0..n_r {
            let in_start = r * no;
            let out_start = r * n;
            for i in 0..half {
                self.ks[idx][out_start + i] = self.radial_poo[in_start + i] * scale_inv2;
            }
            for i in 0..half {
                self.ks[idx][out_start + n - half + i] =
                    self.radial_poo[in_start + no - half + i] * scale_inv2;
            }
        }

        // ── Step 7: normalization — ks[idx] *= M (folds norm_radial + ωwin) ──────
        for i in 0..(n * n_r) {
            self.ks[idx][i] *= self.radial_m[i];
        }
    }

    /// Modal (`TransModal`) RHS, narrow scope — see MATH.md §3.3. Drives the
    /// resident `libcubature` (`pcubature_v`, 1-D radial integral — the
    /// `full=false` case Luna itself selects for `HE, n=1` mode collections,
    /// `Interface.needfull`) with `rhs_modal_pointcalc` as the C callback.
    /// The cubature output (`val`, length `2·n_spec·n_modes`) is exactly the
    /// packed real/imag `Prmω` array Julia's `pointcalc!` builds, so it is
    /// written directly into `ks[idx]` reinterpreted as `f64`.
    fn rhs_modal(&mut self, idx: usize, eomega: &[Complex<f64>]) {
        self.modal_emega.copy_from_slice(eomega);

        let n_spec = self.n_spec;
        let n_modes = self.n_modes;
        let fdim = 2 * n_spec * n_modes;
        let mut valbuf = vec![0.0f64; fdim];
        let mut errbuf = vec![0.0f64; fdim];

        // `self.cubature` is `take()`n out (not borrowed) before the FFI call:
        // the C library re-enters Rust via `modal_integrand_v`/`_full`, which
        // reconstructs a fresh `&mut NativeSim` from the raw `self` pointer —
        // holding any live `&`/`&mut` borrow of a `self` field across that
        // call (including a view into `self.ks[idx]`, hence the separate
        // `valbuf` written back afterward) would alias it.
        let cubature = self
            .cubature
            .take()
            .expect("rhs_modal called before native_set_modal_params");
        let rc = if self.modal_full {
            // Phase E.3: genuine 2-D `(r,θ)` cubature — `hcubature_v`, the
            // same h-adaptive routine Julia's `full=true` branch calls
            // (`Cubature.hcubature_v`, `NonlinearRHS.jl`'s `(t::TransModal)`).
            unsafe {
                cubature.hcubature_v_2d(
                    fdim,
                    modal_integrand_v_full,
                    self as *mut CpuNativeSim as *mut c_void,
                    [0.0, 0.0],
                    [self.modal_a, 2.0 * std::f64::consts::PI],
                    self.modal_maxevals,
                    self.modal_atol,
                    self.modal_rtol,
                    &mut valbuf,
                    &mut errbuf,
                )
            }
        } else {
            unsafe {
                cubature.pcubature_v(
                    fdim,
                    modal_integrand_v,
                    self as *mut CpuNativeSim as *mut c_void,
                    0.0,
                    self.modal_a,
                    self.modal_maxevals,
                    self.modal_atol,
                    self.modal_rtol,
                    &mut valbuf,
                    &mut errbuf,
                )
            }
        };
        self.cubature = Some(cubature);
        if rc != 0 {
            eprintln!("rhs_modal: cubature returned {}", rc);
        }

        let ks_f64: &mut [f64] =
            unsafe { std::slice::from_raw_parts_mut(self.ks[idx].as_mut_ptr() as *mut f64, fdim) };
        ks_f64.copy_from_slice(&valbuf);
    }

    /// Per-node integrand: mirrors `Erω_to_Prω!` + the mode back-projection +
    /// polar Jacobian in Julia's `pointcalc!` (`src/NonlinearRHS.jl:363-399`)
    /// at coordinate `(r,θ)`. For `full=false`, always called with `θ=0`
    /// fixed — the azimuthal dependence is then carried analytically in the
    /// mode-field formula (`mode_angle_xy`) rather than integrated
    /// numerically, and the Jacobian is `2πr` (the θ-integral done
    /// analytically). For `full=true` (Phase E.3), `θ` is a genuine
    /// cubature dimension and the Jacobian is just `r`. Writes the packed
    /// real/imag `Prmω[·,·]·jacobian` into `out` (length `2·n_spec·n_modes`).
    fn modal_pointcalc(ro: &ModalRO, sc: &mut ModalScratch, r: f64, theta: f64, out: &mut [f64]) {
        let n_modes = ro.n_modes;
        let npol = ro.npol;
        let n_spec = ro.n_spec;
        let n_spec_over = ro.n_spec_over;
        let n_time_over = ro.n_time_over;

        if r <= 0.0 || r >= ro.modal_a {
            out.fill(0.0);
            return;
        }

        // ── mode field synthesis at (r,θ): Exy = J_order(r·unm/a)/√N ·
        // (ax,ay) — see `mode_angle_xy` for the per-kind angular formula.
        for m in 0..n_modes {
            let x = r * ro.modal_unm[m] / ro.modal_a;
            let base = jn(ro.modal_order[m], x) * ro.modal_inv_sqrt_n[m];
            let (ax, ay) = mode_angle_xy(
                ro.modal_kind[m],
                ro.modal_order[m],
                ro.modal_phi[m],
                theta,
            );
            for (p, &sel) in ro.modal_pol_select.iter().enumerate() {
                sc.ems[m * npol + p] = base * if sel == 0 { ax } else { ay };
            }
        }

        // ── to_space!: Erω[iω,p] = Σ_m Emω[iω,m]·Ems[m,p] ────────────────────────
        for v in sc.erw.iter_mut() {
            *v = Complex::new(0.0, 0.0);
        }
        for p in 0..npol {
            for m in 0..n_modes {
                let coeff = sc.ems[m * npol + p];
                if coeff == 0.0 {
                    continue;
                }
                let em_col = &ro.modal_emega[m * n_spec..(m + 1) * n_spec];
                let er_col = &mut sc.erw[p * n_spec..(p + 1) * n_spec];
                for i in 0..n_spec {
                    er_col[i] += em_col[i] * coeff;
                }
            }
        }

        sc.erwo.fill(Complex::new(0.0, 0.0));
        if ro.is_real {
            // ── to_time! per polarisation column (RealGrid r2c, looped rank-1 plan) ──
            let scale_fwd = (n_spec_over - 1) as f64 / (n_spec - 1) as f64;
            for p in 0..npol {
                let in_col = &sc.erw[p * n_spec..(p + 1) * n_spec];
                let out_col = &mut sc.erwo[p * n_spec_over..(p + 1) * n_spec_over];
                for i in 0..n_spec {
                    out_col[i] = in_col[i] * scale_fwd;
                }
            }
            if let Some(fft) = ro.fft_r2c_over {
                for p in 0..npol {
                    let spec_start = p * n_spec_over;
                    let time_start = p * n_time_over;
                    let spec_col = &mut sc.erwo[spec_start..spec_start + n_spec_over];
                    let time_col = &mut sc.er[time_start..time_start + n_time_over];
                    fft.inverse(spec_col, time_col);
                }
                let inv_nto = ro.fft_norm_over;
                for v in &mut sc.er {
                    *v *= inv_nto;
                }
            }

            // ── Kerr (KerrScalar!/KerrVector!, src/Nonlinear.jl:81-93) ───────────────
            sc.pr.fill(0.0);
            let fac = ro.modal_kerr_fac;
            if npol == 1 {
                for i in 0..n_time_over {
                    let e = sc.er[i];
                    sc.pr[i] += fac * e * e * e;
                }
            } else {
                for i in 0..n_time_over {
                    let ex = sc.er[i];
                    let ey = sc.er[n_time_over + i];
                    let sq = ex * ex + ey * ey;
                    sc.pr[i] += fac * sq * ex;
                    sc.pr[n_time_over + i] += fac * sq * ey;
                }
            }

            // ── Raman polarisation (if enabled), npol=1 scalar field only ────────────
            // Phase D.4 (docs/dev/BACKLOG.md): same `apply_raman_real` ADE-solve formula,
            // applied per quadrature node. Each parallel worker owns its own
            // cloned `raman_solver` + Hilbert scratch in `sc` (S2 Phase 4 modal
            // threading), so no cross-node/cross-thread state leaks; `solve()`
            // resets oscillator state at entry, so a cloned solver is
            // bit-identical to the shared one (exactly like `apply_raman_radial`).
            // The Hilbert FFT plan is shared read-only (`ro.raman_hilbert_fft`,
            // Sync; `fftw_execute` is thread-safe against distinct buffers).
            // RealGrid-only (native.rs's inline ADE solve is a real-buffer
            // solver) — Julia rejects Raman+EnvGrid before this is reached.
            if ro.has_raman {
                let rho = ro.raman_density;
                if ro.raman_thg {
                    for i in 0..n_time_over {
                        let e = sc.er[i];
                        sc.raman_intensity[i] = e * e;
                    }
                } else if let Some(fft) = ro.raman_hilbert_fft {
                    let norm = ro.fft_norm_over;
                    hilbert_intensity(
                        fft,
                        &mut sc.raman_hilbert_a,
                        &mut sc.raman_hilbert_b,
                        &sc.er[..n_time_over],
                        &mut sc.raman_intensity[..n_time_over],
                        norm,
                    );
                }
                if let Some(ref mut solver) = sc.raman_solver {
                    solver.solve(
                        &sc.raman_intensity[..n_time_over],
                        &mut sc.raman_p[..n_time_over],
                    );
                }
                for i in 0..n_time_over {
                    sc.pr[i] += rho * sc.er[i] * sc.raman_p[i];
                }
            }

            // ── towin apodization per column (reuses the 1-D towin buffer) ───────────
            for p in 0..npol {
                let start = p * n_time_over;
                for t in 0..n_time_over {
                    sc.pr[start + t] *= ro.towin[t];
                }
            }

            // ── to_freq! per column ───────────────────────────────────────────────────
            let scale_inv2 = (n_spec - 1) as f64 / (n_spec_over - 1) as f64;
            if let Some(fft) = ro.fft_r2c_over {
                for p in 0..npol {
                    let time_start = p * n_time_over;
                    let spec_start = p * n_spec_over;
                    let time_col = &mut sc.pr[time_start..time_start + n_time_over];
                    let spec_col = &mut sc.prwo[spec_start..spec_start + n_spec_over];
                    fft.forward(time_col, spec_col);
                }
            }
            for p in 0..npol {
                let in_start = p * n_spec_over;
                let out_start = p * n_spec;
                for i in 0..n_spec {
                    sc.prw[out_start + i] = sc.prwo[in_start + i] * scale_inv2;
                }
            }
        } else {
            // ── EnvGrid (Phase E.4 item 5) — c2c FFT, half-spectrum copy-both
            // padding/truncation (same trick as `rhs_radial_env`, per
            // polarisation column instead of per-r column), and the envelope
            // Kerr formulas (`KerrScalarEnv!`/`KerrVectorEnv!`,
            // src/Nonlinear.jl:120-133). Raman is Julia-side ineligible for
            // EnvGrid modal (see RK45.jl), so no Raman branch here.
            let half = n_spec / 2;
            let scale_fwd = n_spec_over as f64 / n_spec as f64;
            for p in 0..npol {
                let in_col = &sc.erw[p * n_spec..(p + 1) * n_spec];
                let out_col = &mut sc.erwo[p * n_spec_over..(p + 1) * n_spec_over];
                for i in 0..half {
                    out_col[i] = in_col[i] * scale_fwd;
                }
                for i in 0..half {
                    out_col[n_spec_over - half + i] = in_col[n_spec - half + i] * scale_fwd;
                }
            }
            if let Some(fft) = ro.fft_c2c_over {
                for p in 0..npol {
                    let spec_start = p * n_spec_over;
                    let time_start = p * n_time_over;
                    let spec_col = &mut sc.erwo[spec_start..spec_start + n_spec_over];
                    let time_col = &mut sc.er_c[time_start..time_start + n_time_over];
                    fft.inverse(spec_col, time_col);
                }
                let inv_nto = ro.fft_norm_over;
                for v in &mut sc.er_c {
                    *v *= inv_nto;
                }
            }

            // ── Kerr_env (KerrScalarEnv!/KerrVectorEnv!, src/Nonlinear.jl:120-133) ──
            sc.pr_c.fill(Complex::new(0.0, 0.0));
            let fac = Complex::new(0.75 * ro.modal_kerr_fac, 0.0);
            if npol == 1 {
                for i in 0..n_time_over {
                    let e = sc.er_c[i];
                    sc.pr_c[i] += fac * e.norm_sqr() * e;
                }
            } else {
                let third = 1.0 / 3.0;
                for i in 0..n_time_over {
                    let ex = sc.er_c[i];
                    let ey = sc.er_c[n_time_over + i];
                    let ex2 = ex.norm_sqr();
                    let ey2 = ey.norm_sqr();
                    sc.pr_c[i] +=
                        fac * ((ex2 + 2.0 * third * ey2) * ex + third * ex.conj() * ey * ey);
                    sc.pr_c[n_time_over + i] +=
                        fac * ((ey2 + 2.0 * third * ex2) * ey + third * ey.conj() * ex * ex);
                }
            }

            // ── towin apodization per column ─────────────────────────────────────────
            for p in 0..npol {
                let start = p * n_time_over;
                for t in 0..n_time_over {
                    sc.pr_c[start + t] *= ro.towin[t];
                }
            }

            // ── to_freq! per column ───────────────────────────────────────────────────
            let scale_inv2 = n_spec as f64 / n_spec_over as f64;
            if let Some(fft) = ro.fft_c2c_over {
                for p in 0..npol {
                    let time_start = p * n_time_over;
                    let spec_start = p * n_spec_over;
                    let time_col = &mut sc.pr_c[time_start..time_start + n_time_over];
                    let spec_col = &mut sc.prwo[spec_start..spec_start + n_spec_over];
                    fft.forward(time_col, spec_col);
                }
            }
            for p in 0..npol {
                let in_start = p * n_spec_over;
                let out_start = p * n_spec;
                for i in 0..half {
                    sc.prw[out_start + i] = sc.prwo[in_start + i] * scale_inv2;
                }
                for i in 0..half {
                    sc.prw[out_start + n_spec - half + i] =
                        sc.prwo[in_start + n_spec_over - half + i] * scale_inv2;
                }
            }
        }

        // ── ωwin + norm! (precomputed modal_nlfac, per ω, same for every pol column) ──
        for p in 0..npol {
            let col = &mut sc.prw[p * n_spec..(p + 1) * n_spec];
            for i in 0..n_spec {
                col[i] *= ro.modal_nlfac[i];
            }
        }

        // ── back-projection: Prmω[iω,m] = Σ_p Prω[iω,p]·Ems[m,p] ─────────────────
        // `full=false`: Jacobian is `2πr` (θ-integral done analytically).
        // `full=true`: Jacobian is just `r` (θ genuinely integrated).
        let pre = if ro.modal_full {
            r
        } else {
            2.0 * std::f64::consts::PI * r
        };
        for m in 0..n_modes {
            for i in 0..n_spec {
                let mut acc = Complex::new(0.0, 0.0);
                for p in 0..npol {
                    acc += sc.prw[p * n_spec + i] * sc.ems[m * npol + p];
                }
                acc *= pre;
                let idx = i + n_spec * m;
                out[2 * idx] = acc.re;
                out[2 * idx + 1] = acc.im;
            }
        }
    }

    /// Grow `modal_scratch_pool` to at least `n_workers` entries (entry 0 is
    /// created in `native_set_modal_params`), and — when Raman is active —
    /// lazily size each worker's own Raman scratch and clone the resident
    /// `raman_solver` into it. The solver resets oscillator state at
    /// solve-entry, so a persistent per-worker clone is bit-identical to the
    /// shared one (see `apply_raman_radial`). docs/dev/BACKLOG.md S2 Phase 4.
    fn ensure_modal_pool(&mut self, n_workers: usize) {
        if self.modal_scratch_pool.is_empty() {
            self.modal_scratch_pool.push(ModalScratch::new(
                self.n_modes,
                self.npol,
                self.n_spec,
                self.n_spec_over,
                self.n_time_over,
            ));
        }
        while self.modal_scratch_pool.len() < n_workers {
            let sc = ModalScratch::new(
                self.n_modes,
                self.npol,
                self.n_spec,
                self.n_spec_over,
                self.n_time_over,
            );
            self.modal_scratch_pool.push(sc);
        }
        if self.has_raman {
            let nto = self.n_time_over;
            for i in 0..n_workers {
                let sc = &mut self.modal_scratch_pool[i];
                if sc.raman_intensity.len() != nto {
                    sc.raman_intensity = vec![0.0; nto];
                    sc.raman_p = vec![0.0; nto];
                    sc.raman_hilbert_a = vec![Complex::new(0.0, 0.0); nto];
                    sc.raman_hilbert_b = vec![Complex::new(0.0, 0.0); nto];
                }
            }
            for i in 0..n_workers {
                if self.modal_scratch_pool[i].raman_solver.is_none() {
                    let cloned = self.raman_solver.clone();
                    self.modal_scratch_pool[i].raman_solver = cloned;
                }
            }
        }
    }

    /// `full=false` (1-D pcubature) integrand batch: `xs[p]` = r, θ = 0.
    fn modal_eval_batch_1d(&mut self, xs: &[f64], fvals: &mut [f64], fdim: usize) {
        let coords: Vec<(f64, f64)> = xs.iter().map(|&r| (r, 0.0)).collect();
        self.modal_eval_pairs(&coords, fvals, fdim);
    }

    /// `full=true` (2-D hcubature) integrand batch: `xs` point-major,
    /// `xs[2p]` = r, `xs[2p+1]` = θ.
    fn modal_eval_batch_2d(&mut self, xs: &[f64], fvals: &mut [f64], fdim: usize) {
        let coords: Vec<(f64, f64)> = xs.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        self.modal_eval_pairs(&coords, fvals, fdim);
    }

    /// Evaluate `modal_pointcalc` at a batch of `(r,θ)` nodes, writing each
    /// node's packed result into `fvals[p*fdim..(p+1)*fdim]`. Sequential for
    /// `n_threads == 1` or a single node; otherwise partitions the nodes into
    /// `≤ n_threads` contiguous groups, each rayon worker owning a disjoint
    /// `ModalScratch`. Bit-identical to sequential (each node writes only its
    /// own `out` slice, scratch is re-initialised per node inside
    /// `modal_pointcalc`, cloned Raman solvers reset at solve-entry; no
    /// cross-node reduction). docs/dev/BACKLOG.md S2 Phase 4 (modal RHS).
    fn modal_eval_pairs(&mut self, coords: &[(f64, f64)], fvals: &mut [f64], fdim: usize) {
        let npt = coords.len();
        if npt == 0 {
            return;
        }
        let n_threads = self.n_threads;
        let n_workers = if n_threads > 1 && npt > 1 {
            n_threads.min(npt)
        } else {
            1
        };
        self.ensure_modal_pool(n_workers);

        // Disjoint field borrows via destructuring: `ro` holds immutable reads,
        // `modal_scratch_pool` the per-worker mutable scratch, `thread_pool` the
        // resident rayon pool — all distinct fields, so all three coexist.
        let Self {
            n_modes,
            npol,
            n_spec,
            n_spec_over,
            n_time_over,
            modal_a,
            modal_full,
            is_real,
            modal_unm,
            modal_order,
            modal_kind,
            modal_phi,
            modal_inv_sqrt_n,
            modal_pol_select,
            modal_emega,
            modal_nlfac,
            modal_kerr_fac,
            fft_r2c_over,
            fft_c2c_over,
            fft_norm_over,
            towin,
            has_raman,
            raman_density,
            raman_thg,
            raman_hilbert_fft,
            modal_scratch_pool,
            thread_pool,
            ..
        } = self;
        let ro = ModalRO {
            n_modes: *n_modes,
            npol: *npol,
            n_spec: *n_spec,
            n_spec_over: *n_spec_over,
            n_time_over: *n_time_over,
            modal_a: *modal_a,
            modal_full: *modal_full,
            is_real: *is_real,
            modal_unm: modal_unm.as_slice(),
            modal_order: modal_order.as_slice(),
            modal_kind: modal_kind.as_slice(),
            modal_phi: modal_phi.as_slice(),
            modal_inv_sqrt_n: modal_inv_sqrt_n.as_slice(),
            modal_pol_select: modal_pol_select.as_slice(),
            modal_emega: modal_emega.as_slice(),
            modal_nlfac: modal_nlfac.as_slice(),
            modal_kerr_fac: *modal_kerr_fac,
            fft_r2c_over: fft_r2c_over.as_ref(),
            fft_c2c_over: fft_c2c_over.as_ref(),
            fft_norm_over: *fft_norm_over,
            towin: towin.as_slice(),
            has_raman: *has_raman,
            raman_density: *raman_density,
            raman_thg: *raman_thg,
            raman_hilbert_fft: raman_hilbert_fft.as_ref(),
        };

        if n_workers == 1 {
            let sc = &mut modal_scratch_pool[0];
            for (p, &(r, theta)) in coords.iter().enumerate() {
                Self::modal_pointcalc(&ro, sc, r, theta, &mut fvals[p * fdim..(p + 1) * fdim]);
            }
            return;
        }

        let group = npt.div_ceil(n_workers);
        let n_chunks = npt.div_ceil(group);
        let pool = thread_pool.get_or_insert_with(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("failed to build rayon thread pool for native RHS (docs/dev/BACKLOG.md S2)")
        });
        pool.install(|| {
            coords
                .par_chunks(group)
                .zip(fvals.par_chunks_mut(group * fdim))
                .zip(modal_scratch_pool[..n_chunks].par_iter_mut())
                .for_each(|((coord_chunk, fval_chunk), sc)| {
                    for (i, &(r, theta)) in coord_chunk.iter().enumerate() {
                        Self::modal_pointcalc(
                            &ro,
                            sc,
                            r,
                            theta,
                            &mut fval_chunk[i * fdim..(i + 1) * fdim],
                        );
                    }
                });
        });
    }

    /// Free-space (`TransFree`) RHS: joint 3-D FFT (RealGrid, scalar Kerr —
    /// see MATH.md §3.4). Mirrors `TransFree.__call__`
    /// (src/NonlinearRHS.jl:818-838). Unlike `rhs_radial`, there is no
    /// per-column spatial step: Kerr and the final normalization multiply
    /// are plain flat elementwise ops over the whole `(t,y,x)`/`(ω,ky,kx)`
    /// volume; only the zero-pad/truncate and `towin` steps need a
    /// per-`(y,x)`-column loop, since those act along the `t`/`ω` axis.
    fn rhs_free(&mut self, idx: usize, eomega: &[Complex<f64>]) {
        let n_spec = self.n_spec;
        let n_spec_over = self.n_spec_over;
        let n_time_over = self.n_time_over;
        let n_cols = self.n_y * self.n_x;

        // ── Step 1: zero-pad ω → oversampled, per (y,x) column ───────────────────
        let scale_fwd = (n_spec_over - 1) as f64 / (n_spec - 1) as f64;
        self.free_eoo.fill(Complex::new(0.0, 0.0));
        for c in 0..n_cols {
            let in_col = &eomega[c * n_spec..(c + 1) * n_spec];
            let out_col = &mut self.free_eoo[c * n_spec_over..(c + 1) * n_spec_over];
            for i in 0..n_spec {
                out_col[i] = in_col[i] * scale_fwd;
            }
        }

        // ── Step 2: ONE joint 3-D inverse transform (ω,ky,kx) → (t,y,x) ──────────
        if let Some(ref fft) = self.fft_r2c_3d {
            fft.inverse(&mut self.free_eoo, &mut self.free_eto);
            let inv_nto = self.free_fft_norm_over;
            for v in &mut self.free_eto {
                *v *= inv_nto;
            }
        }

        // ── Step 3: Kerr (KerrScalar!), flat over the whole (t,y,x) volume ───────
        self.free_pto.fill(0.0);
        for i in 0..(n_time_over * n_cols) {
            let e = self.free_eto[i];
            self.free_pto[i] += self.kerr_fac * e * e * e;
        }

        // ── Step 3b/3c: plasma / Raman, per (y,x) column — Phase I item 6 ────────
        if self.has_plasma {
            self.apply_plasma_free();
        }
        if self.has_raman {
            self.apply_raman_free();
        }

        // ── Step 4: towin apodization per (y,x) column ───────────────────────────
        for c in 0..n_cols {
            let start = c * n_time_over;
            for t in 0..n_time_over {
                self.free_pto[start + t] *= self.towin[t];
            }
        }

        // ── Step 5: ONE joint 3-D forward transform (t,y,x) → (ω,ky,kx) ──────────
        if let Some(ref fft) = self.fft_r2c_3d {
            fft.forward(&mut self.free_pto, &mut self.free_poo);
        }

        // ── Step 6: truncate oversampled → base, per (y,x) column ────────────────
        let scale_inv2 = (n_spec - 1) as f64 / (n_spec_over - 1) as f64;
        self.ks[idx].fill(Complex::new(0.0, 0.0));
        for c in 0..n_cols {
            let in_start = c * n_spec_over;
            let out_start = c * n_spec;
            for i in 0..n_spec {
                self.ks[idx][out_start + i] = self.free_poo[in_start + i] * scale_inv2;
            }
        }

        // ── Step 7: normalization — ks[idx] *= M (folds norm_free + ωwin) ────────
        for i in 0..(n_spec * n_cols) {
            self.ks[idx][i] *= self.free_m[i];
        }
    }

    /// EnvGrid (c2c) counterpart of `rhs_free` — Phase D.3 (docs/dev/BACKLOG.md).
    /// Mirrors `rhs_radial_env`'s c2c `to_time!`/`to_freq!` convention
    /// (half-spectrum zero-pad, `no/n` scale, 3/4 `Kerr_env` SVEA factor)
    /// applied per `(y,x)` column for the zero-pad/truncate/window steps,
    /// but — like `rhs_free`'s real-grid Kerr step — the joint 3-D
    /// transform and the flat elementwise Kerr multiply act over the whole
    /// `(t,y,x)` volume at once, not per column.
    fn rhs_free_env(&mut self, idx: usize, eomega: &[Complex<f64>]) {
        let n = self.n_spec; // = n_time for EnvGrid (c2c: Nω = Nt)
        let no = self.n_time_over;
        let half = n / 2;
        let n_cols = self.n_y * self.n_x;

        // ── Step 1: to_time! per (y,x) column (EnvGrid c2c, half-spectrum copy-both) ──
        let scale_fwd = no as f64 / n as f64;
        self.free_eoo.fill(Complex::new(0.0, 0.0));
        for c in 0..n_cols {
            let in_col = &eomega[c * n..(c + 1) * n];
            let out_col = &mut self.free_eoo[c * no..(c + 1) * no];
            for i in 0..half {
                out_col[i] = in_col[i] * scale_fwd;
            }
            for i in 0..half {
                out_col[no - half + i] = in_col[n - half + i] * scale_fwd;
            }
        }

        // ── Step 2: ONE joint 3-D inverse transform (ω,ky,kx) → (t,y,x) ──────────
        if let Some(ref fft) = self.fft_c2c_3d {
            fft.inverse(&mut self.free_eoo, &mut self.free_eto_c);
            let inv_nto = self.free_fft_norm_over;
            for v in &mut self.free_eto_c {
                *v *= inv_nto;
            }
        }

        // ── Step 3: Kerr_env: Pto += (3/4)*kerr_fac · |E|² · E, flat over volume ──
        let kf = Complex::new(0.75 * self.kerr_fac, 0.0);
        self.free_pto_c.fill(Complex::new(0.0, 0.0));
        for i in 0..(no * n_cols) {
            let e = self.free_eto_c[i];
            self.free_pto_c[i] += kf * e.norm_sqr() * e;
        }

        // ── Step 4: towin apodization per (y,x) column ───────────────────────────
        for c in 0..n_cols {
            let start = c * no;
            for t in 0..no {
                self.free_pto_c[start + t] *= self.towin[t];
            }
        }

        // ── Step 5: ONE joint 3-D forward transform (t,y,x) → (ω,ky,kx) ──────────
        if let Some(ref fft) = self.fft_c2c_3d {
            fft.forward(&mut self.free_pto_c, &mut self.free_poo);
        }

        // ── Step 6: to_freq! per (y,x) column (EnvGrid c2c, truncate) ────────────
        let scale_inv2 = n as f64 / no as f64;
        self.ks[idx].fill(Complex::new(0.0, 0.0));
        for c in 0..n_cols {
            let in_start = c * no;
            let out_start = c * n;
            for i in 0..half {
                self.ks[idx][out_start + i] = self.free_poo[in_start + i] * scale_inv2;
            }
            for i in 0..half {
                self.ks[idx][out_start + n - half + i] =
                    self.free_poo[in_start + no - half + i] * scale_inv2;
            }
        }

        // ── Step 7: normalization — ks[idx] *= M (folds norm_free + ωwin) ────────
        for i in 0..(n * n_cols) {
            self.ks[idx][i] *= self.free_m[i];
        }
    }

    /// Recompute `self.linop` in place for propagation position `z`, for the
    /// z-dependent mode-averaged (Phase 7) case. No-op for Phases 1-6 (the
    /// constant-linop geometries) and a no-op if `z` matches the last call
    /// (mirrors Julia's `make_prop!`'s `lastt2` memoization — every
    /// forward/backward `prop!` pair in `native_step` shares one `z`). See
    /// MATH.md §3.5.
    fn ensure_linop_at(&mut self, z: f64) {
        if !self.is_zdep_mode_avg {
            return;
        }
        if self.zdep_last_z == z {
            return;
        }

        // Closed-form z -> pressure (TwoPointGradient/MultiPointGradient's
        // p(z), docs/dev/BACKLOG.md Phase F item 3), ported exactly (no interpolation
        // error) — flat-clamped outside the breakpoint range, reproducing
        // Capillary.gradient's own boundary behavior.
        let pressure = zdep_pressure_at(z, &self.zdep_grad_z, &self.zdep_grad_p);
        let dens = self.zdep_dens_lut.as_ref().unwrap().eval(pressure);
        const C: f64 = 299_792_458.0;

        // β1(z) closed form (docs/dev/native-port/BETA1_ANALYTIC.md): εco(ω0;z)-1
        // = γ0·dens(z), and nwg0/dnwg0 are z-independent (constant radius),
        // so β1 needs only these 4 precomputed constants plus the just-
        // computed scalar `dens`, not a per-z LUT lookup.
        let eco0 = 1.0 + self.zdep_gamma0 * dens;
        let deco0 = self.zdep_dgamma0 * dens;
        let (neff0, dneff0) = if self.zdep_model == 0 {
            // :full — neff0 = sqrt(εco0 − nwg0), dneff0 = (dεco0 − dnwg0)/(2·neff0)
            let neff0 = (Complex::new(eco0, 0.0) - self.zdep_nwg0).sqrt();
            let dneff0 = (Complex::new(deco0, 0.0) - self.zdep_dnwg0) / (2.0 * neff0);
            (neff0, dneff0)
        } else {
            // :reduced — neff0 = 1 + (εco0−1)/2 − nwg0, dneff0 = dεco0/2 − dnwg0
            let neff0 = Complex::new(1.0 + (eco0 - 1.0) / 2.0, 0.0) - self.zdep_nwg0;
            let dneff0 = Complex::new(deco0 / 2.0, 0.0) - self.zdep_dnwg0;
            (neff0, dneff0)
        };
        let beta1 = (neff0.re + self.zdep_omega0 * dneff0.re) / C;

        for i in 0..self.n {
            if !self.sidx[i] {
                self.linop[i] = Complex::new(0.0, 0.0);
                continue;
            }
            let eco = 1.0 + self.zdep_gamma[i] * dens;
            let nwg = Complex::new(self.zdep_nwg_re[i], self.zdep_nwg_im[i]);
            let mut neff = if self.zdep_model == 0 {
                // :full — neff = sqrt(complex(εco − nwg))
                (Complex::new(eco, 0.0) - nwg).sqrt()
            } else {
                // :reduced — neff = 1 + (εco−1)/2 − nwg
                Complex::new(1.0 + (eco - 1.0) / 2.0, 0.0) - nwg
            };
            if !self.zdep_loss_on {
                neff = Complex::new(neff.re, 0.0);
            }
            // conj_clamp(n, ω) = clamp(re(n), 1e-3, Inf) - i·clamp(im(n), 0, 3000·c/ω)
            let omega = self.zdep_omega[i];
            let re_c = neff.re.max(1e-3);
            let im_bound = 3000.0 * C / omega;
            let im_c = neff.im.clamp(0.0, im_bound);
            let nc = Complex::new(re_c, -im_c);

            // -i·w where w = ω/c·nc - ω·β1  (avoids relying on Complex::i())
            let w = nc * (omega / C) - Complex::new(omega * beta1, 0.0);
            self.linop[i] = Complex::new(w.im, -w.re);

            // norm_mode_average's β(ω,z) = ω/c·real(neff(ω,z)) (Modes.jl:136-138)
            // — UNCLAMPED, unlike the linop's nc.re — reusing `neff.re` from
            // above, computed before the conj_clamp above is applied.
            if i < self.beta.len() {
                self.beta[i] = omega / C * neff.re;
            }
            // beta[i] just changed — norm_pre_beta (rhs_mode_avg_{real,env}'s
            // precomputed Step 6 factor, S1 item 4) must be refreshed to stay
            // in sync; it is not recomputed per-RHS-call like the un-cached
            // original `pre[i]/beta[i]*sqrt_aeff`, only here, on real z-updates.
            if i < self.norm_pre_beta.len() && self.sidx[i] {
                self.norm_pre_beta[i] = self.pre[i] / self.beta[i] * self.sqrt_aeff;
            }
        }
        // TransModeAvg calls `t.densityfun(z)` fresh every RK stage
        // (NonlinearRHS.jl), so `kerr_fac = density(z)·ε₀·γ3` must be
        // rescaled here too, not just baked in once at construction.
        self.kerr_fac = self.zdep_kerr_fac_per_dens * dens;
        self.zdep_last_z = z;
        self.linop_version += 1; // invalidate `exp_cache` — `self.linop` just changed
    }

    /// Recompute `self.linop` and `self.free_m` in place for propagation
    /// position `z` — the free-space (`TransFree`) counterpart of
    /// `ensure_linop_at`, Phase D.5 (docs/dev/BACKLOG.md). No-op if `is_zdep_free` is
    /// false or `z` matches the last call (same memoization rationale).
    ///
    /// Unlike `ensure_linop_at` there is no waveguide term: `n(ω;z) =
    /// sqrt(1+γ(λ(ω))·ρ(z))` is the whole index, so `k²(ω;z) = n(ω;z)²·
    /// (ω/c)²` is computed once per ω and reused for both the linop fill
    /// (`LinearOps._fill_linop_xy!`'s formula) and the nonlinear norm fill
    /// (`NonlinearRHS.norm_free`'s formula) at every (ω,k⊥) pair — the two
    /// Julia functions duplicate the same `k²-k⊥²` computation with
    /// different post-processing, so this method does too, rather than
    /// factoring out a shared helper Julia itself doesn't have (keeping the
    /// two formulas visibly side-by-side made the transcription easier to
    /// verify against both Julia sources independently).
    fn ensure_free_norm_at(&mut self, z: f64) {
        if !self.is_zdep_free {
            return;
        }
        if self.zdep_free_last_z == z {
            return;
        }

        // Closed-form z -> pressure (TwoPointGradient.p(z)), same formula
        // and boundary clamp as `ensure_linop_at`.
        let zc = z.clamp(0.0, self.zdep_free_flength);
        let p0 = self.zdep_free_p0;
        let p1 = self.zdep_free_p1;
        let l = self.zdep_free_flength;
        let pressure = if zc >= l {
            p1
        } else if zc <= 0.0 {
            p0
        } else {
            (p0 * p0 + zc / l * (p1 * p1 - p0 * p0)).sqrt()
        };
        let dens = self.zdep_free_dens_lut.as_ref().unwrap().eval(pressure);
        const C: f64 = 299_792_458.0;
        const MU_0: f64 = 1.25663706212e-6;

        // β1(z) = (n0(z) + ω0·dn0/dω(z))/c, with n0(z)=sqrt(1+γ0·ρ(z)) and
        // dn0/dω(z) = dγ0·ρ(z)/(2·n0(z)) — see `LinearOps.ZDepLinopFree`'s
        // doc for the derivation (no waveguide term, unlike Phase 7).
        let n0 = (1.0 + self.zdep_free_gamma0 * dens).sqrt();
        let dn0 = self.zdep_free_dgamma0 * dens / (2.0 * n0);
        let beta1 = (n0 + self.zdep_free_omega0 * dn0) / C;

        let n_spec = self.n_spec;
        let n_cols = self.n_y * self.n_x;

        // Per-ω k²(ω;z), honoring the sidx-gated default exactly like
        // Julia's `k2 = zero(grid.ω); k2[grid.sidx] .= ...` (leaving k2=0,
        // not (n=1)²(ω/c)², outside sidx — γ alone being 0 there isn't
        // enough to reproduce this).
        let mut k2 = vec![0.0f64; n_spec];
        for iw in 0..n_spec {
            if self.zdep_free_sidx[iw] {
                let omega = self.zdep_free_omega[iw];
                k2[iw] = (1.0 + self.zdep_free_gamma[iw] * dens) * (omega / C).powi(2);
            }
        }

        for col in 0..n_cols {
            let kp2 = self.zdep_free_kperp2[col];
            for iw in 0..n_spec {
                let idx = iw + n_spec * col;
                let omega = self.zdep_free_omega[iw];
                let bsq = k2[iw] - kp2;

                // ── linop (LinearOps._fill_linop_xy!) ──────────────────────
                self.linop[idx] = if bsq < 0.0 {
                    let atten = (-bsq).sqrt().min(200.0);
                    Complex::new(-atten, beta1 * omega)
                } else {
                    Complex::new(0.0, beta1 * omega - bsq.sqrt())
                };

                // ── nonlinear norm (NonlinearRHS.norm_free) ─────────────────
                let normval = if omega == 0.0 || bsq <= 0.0 {
                    1.0
                } else {
                    bsq.sqrt() / (MU_0 * omega)
                };
                let owin = self.zdep_free_omegawin[iw];
                self.free_m[idx] = Complex::new(0.0, -owin * omega) / (2.0 * normval);
            }
        }

        // TransFree calls `t.densityfun(z)` fresh every RK stage
        // (NonlinearRHS.jl), so `kerr_fac` must be rescaled here too.
        self.kerr_fac = self.zdep_free_kerr_fac_per_dens * dens;
        self.zdep_free_last_z = z;
        self.linop_version += 1; // invalidate `exp_cache` — `self.linop` just changed
    }

    /// Recompute `self.linop` in place for propagation position `z` — the
    /// modal (`TransModal`, `full=false`) counterpart of `ensure_linop_at`,
    /// docs/dev/BACKLOG.md Phase E.2 (tapered/per-mode radius). No-op if
    /// `is_zdep_modal` is false or `z` matches the last call.
    ///
    /// Density is constant here (checked in `RK45.jl` before wiring this
    /// path), so unlike Phase 7/D.5 there is no `kerr_fac` rescale — the
    /// z-dependence is entirely in the per-mode waveguide term `nwg(ω;a(z))`,
    /// via the analytic separable form `nwg(ω,a) = unm²·(φ(ω)/a)²·
    /// (1-i·vn(ω)·φ(ω)/a)²` with `φ(ω)=c/ω` (see `zdep_modal_vn_re`/`_im`'s
    /// doc comment). `β1(z)` reuses the same closed-form chain-rule pattern
    /// as Phase 7/D.5, differentiating this expression w.r.t. ω at ω0 for
    /// the current (fixed) `a=a(z)` using the precomputed `v0`/`dv0`
    /// constants — no per-z BigFloat call, just algebra.
    fn ensure_modal_linop_at(&mut self, z: f64) {
        if !self.is_zdep_modal {
            return;
        }
        if self.zdep_modal_last_z == z {
            return;
        }

        const C: f64 = 299_792_458.0;
        let zc = z.clamp(0.0, self.zdep_modal_flength);
        let a = self.zdep_modal_a_lut.as_ref().unwrap().eval(zc);

        let n_spec = self.n_spec;
        let n_modes = self.n_modes;
        let ref_mode = self.zdep_modal_ref_mode;

        // Mode-field synthesis (`rhs_modal_pointcalc`) reads `self.modal_a`
        // (both the cubature integration bound and the `J_order` argument's
        // denominator) and `self.modal_inv_sqrt_n` — both must track the
        // current `a(z)`, not just the linop.
        self.modal_a = a;
        let scale = self.zdep_modal_a0 / a;
        for m in 0..n_modes {
            self.modal_inv_sqrt_n[m] = self.zdep_modal_inv_sqrt_n0[m] * scale;
        }

        // ── β1(z), from the reference mode only ─────────────────────────
        let beta1 = {
            let m = ref_mode;
            let u = self.modal_unm[m];
            let eco0 = self.zdep_modal_eco0[m];
            let deco0 = self.zdep_modal_deco0[m];
            let v0 = Complex::new(self.zdep_modal_v0_re[m], self.zdep_modal_v0_im[m]);
            let dv0 = Complex::new(self.zdep_modal_dv0_re[m], self.zdep_modal_dv0_im[m]);
            let ω0 = self.zdep_modal_omega0;

            let phi0 = C / ω0;
            let dphi0 = -C / (ω0 * ω0);
            let g0 = phi0 / a;
            let dg0 = dphi0 / a;
            let h0 = v0 * g0;
            let dh0 = dv0 * g0 + v0 * dg0;

            let one = Complex::new(1.0, 0.0);
            let i = Complex::new(0.0, 1.0);
            let f0 = g0 * g0;
            let df0 = 2.0 * g0 * dg0;
            let big_g0 = (one - i * h0) * (one - i * h0);
            let d_big_g0 = -2.0 * i * dh0 * (one - i * h0);

            let u2 = Complex::new(u * u, 0.0);
            let nwg0 = u2 * f0 * big_g0;
            let dnwg0 = u2 * (df0 * big_g0 + f0 * d_big_g0);

            let (neff0, dneff0) = if self.zdep_modal_model == 0 {
                let neff0 = (Complex::new(eco0, 0.0) - nwg0).sqrt();
                let dneff0 = (Complex::new(deco0, 0.0) - dnwg0) / (2.0 * neff0);
                (neff0, dneff0)
            } else {
                let neff0 = Complex::new(1.0 + (eco0 - 1.0) / 2.0, 0.0) - nwg0;
                let dneff0 = Complex::new(deco0 / 2.0, 0.0) - dnwg0;
                (neff0, dneff0)
            };
            (neff0.re + ω0 * dneff0.re) / C
        };

        // ── per-(mode,ω) linop fill ───────────────────────────────────────
        for m in 0..n_modes {
            let u = self.modal_unm[m];
            for iω in 0..n_spec {
                let idx = iω + n_spec * m;
                if !self.zdep_modal_sidx[iω] {
                    self.linop[idx] = Complex::new(0.0, 0.0);
                    continue;
                }
                let omega = self.zdep_modal_omega[iω];
                let eco = self.zdep_modal_eco[idx];
                let vn = Complex::new(self.zdep_modal_vn_re[idx], self.zdep_modal_vn_im[idx]);

                let phi = C / omega;
                let g = phi / a;
                let h = vn * g;
                let one = Complex::new(1.0, 0.0);
                let i = Complex::new(0.0, 1.0);
                let nwg = Complex::new(u * u, 0.0) * g * g * (one - i * h) * (one - i * h);

                let mut neff = if self.zdep_modal_model == 0 {
                    (Complex::new(eco, 0.0) - nwg).sqrt()
                } else {
                    Complex::new(1.0 + (eco - 1.0) / 2.0, 0.0) - nwg
                };
                if !self.zdep_modal_loss_on {
                    neff = Complex::new(neff.re, 0.0);
                }
                // conj_clamp(n, ω) = clamp(re(n), 1e-3, Inf) - i·clamp(im(n), 0, 3000·c/ω)
                let re_c = neff.re.max(1e-3);
                let im_bound = 3000.0 * C / omega;
                let im_c = neff.im.clamp(0.0, im_bound);
                let nc = Complex::new(re_c, -im_c);

                let w = nc * (omega / C) - Complex::new(omega * beta1, 0.0);
                self.linop[idx] = Complex::new(w.im, -w.re);
            }
        }

        self.zdep_modal_last_z = z;
        self.linop_version += 1; // invalidate `exp_cache` — `self.linop` just changed
    }
}

/// Per-worker mutable scratch for `CpuNativeSim::modal_pointcalc` — one
/// instance per rayon worker (entry 0 alone on the sequential path). Holds
/// every buffer a single node writes, so the read-only `ModalRO` view of the
/// sim can be shared `&` across threads while each worker mutates only its own
/// entry (docs/dev/BACKLOG.md S2 Phase 4 — modal RHS threading).
pub struct ModalScratch {
    /// Mode-field weights `(ax,ay)·J_order/√N` — `n_modes*npol`.
    ems: Vec<f64>,
    /// `to_space!` result `Erω[iω,p]` — `npol*n_spec`.
    erw: Vec<Complex<f64>>,
    /// Oversampled spectrum scratch — `npol*n_spec_over`.
    erwo: Vec<Complex<f64>>,
    /// RealGrid time-domain field — `npol*n_time_over`.
    er: Vec<f64>,
    /// RealGrid time-domain polarisation — `npol*n_time_over`.
    pr: Vec<f64>,
    /// Oversampled polarisation spectrum — `npol*n_spec_over`.
    prwo: Vec<Complex<f64>>,
    /// `to_freq!`'d polarisation `Prω[iω,p]` — `npol*n_spec`.
    prw: Vec<Complex<f64>>,
    /// EnvGrid (c2c) counterparts of `er`/`pr` — `npol*n_time_over` (Phase E.4).
    er_c: Vec<Complex<f64>>,
    pr_c: Vec<Complex<f64>>,
    /// Per-worker Raman scratch — separate from the shared `self.raman_*`
    /// buffers (used by mode-averaged/radial). Sized + solver cloned lazily by
    /// `ensure_modal_pool`, only when `has_raman`.
    raman_intensity: Vec<f64>,
    raman_p: Vec<f64>,
    raman_hilbert_a: Vec<Complex<f64>>,
    raman_hilbert_b: Vec<Complex<f64>>,
    raman_solver: Option<TimeDomainRamanSolver>,
}

impl ModalScratch {
    fn new(
        n_modes: usize,
        npol: usize,
        n_spec: usize,
        n_spec_over: usize,
        n_time_over: usize,
    ) -> Self {
        ModalScratch {
            ems: vec![0.0; n_modes * npol],
            erw: vec![Complex::new(0.0, 0.0); n_spec * npol],
            erwo: vec![Complex::new(0.0, 0.0); n_spec_over * npol],
            er: vec![0.0; n_time_over * npol],
            pr: vec![0.0; n_time_over * npol],
            prwo: vec![Complex::new(0.0, 0.0); n_spec_over * npol],
            prw: vec![Complex::new(0.0, 0.0); n_spec * npol],
            er_c: vec![Complex::new(0.0, 0.0); n_time_over * npol],
            pr_c: vec![Complex::new(0.0, 0.0); n_time_over * npol],
            raman_intensity: Vec::new(),
            raman_p: Vec::new(),
            raman_hilbert_a: Vec::new(),
            raman_hilbert_b: Vec::new(),
            raman_solver: None,
        }
    }
}

/// Read-only borrowed view of the parts of `CpuNativeSim` that
/// `modal_pointcalc` reads. Every field is `Copy`, `&[..]`, or `Option<&Plan>`
/// where the FFT-plan wrappers are `Sync` (fftw.rs) — so `ModalRO: Sync` and a
/// single `&ModalRO` can be shared across rayon workers.
struct ModalRO<'a> {
    n_modes: usize,
    npol: usize,
    n_spec: usize,
    n_spec_over: usize,
    n_time_over: usize,
    modal_a: f64,
    modal_full: bool,
    is_real: bool,
    modal_unm: &'a [f64],
    modal_order: &'a [i32],
    modal_kind: &'a [u8],
    modal_phi: &'a [f64],
    modal_inv_sqrt_n: &'a [f64],
    modal_pol_select: &'a [u8],
    modal_emega: &'a [Complex<f64>],
    modal_nlfac: &'a [Complex<f64>],
    modal_kerr_fac: f64,
    fft_r2c_over: Option<&'a RealFft1d>,
    fft_c2c_over: Option<&'a ComplexFft1d>,
    fft_norm_over: f64,
    towin: &'a [f64],
    has_raman: bool,
    raman_density: f64,
    raman_thg: bool,
    raman_hilbert_fft: Option<&'a ComplexFft1d>,
}

/// C `integrand_v` callback for `pcubature_v` — reconstructs `&mut NativeSim`
/// from `fdata` and hands the batch to `modal_eval_batch_1d`, which evaluates
/// `modal_pointcalc` per node (sequential, or rayon-parallel over the batch's
/// nodes when `n_threads > 1` — docs/dev/BACKLOG.md S2 Phase 4). Nodes are
/// evaluated one at a time per worker, not truly vectorized (mirrors Phase 3's
/// "loop the existing rank-1 plan" precedent, see MATH.md §3.3).
///
/// # Safety
/// `fdata` must be a valid `*mut NativeSim` (guaranteed: it is always
/// `rhs_modal`'s own `self` pointer, round-tripped through `libcubature`).
/// `x`/`fval` must be valid for `npt`/`npt*fdim` reads/writes respectively.
unsafe extern "C" fn modal_integrand_v(
    _ndim: c_uint,
    npt: size_t,
    x: *const c_double,
    fdata: *mut c_void,
    fdim: c_uint,
    fval: *mut c_double,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let sim = unsafe { &mut *(fdata as *mut CpuNativeSim) };
        let npt = npt;
        let fdim = fdim as usize;
        let xs = unsafe { std::slice::from_raw_parts(x, npt) };
        let fvals = unsafe { std::slice::from_raw_parts_mut(fval, npt * fdim) };
        sim.modal_eval_batch_1d(xs, fvals, fdim);
    }));
    match result {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// Per-mode angular field factor `(ax,ay)` at `(θ,ϕ)`, mirroring
/// `Capillary.field(m,(r,θ))/J_order(r·unm/a)` (Capillary.jl:271-283):
/// `:HE` (`kind=0`) → `(cosθ·sin(n(θ+ϕ)) - sinθ·cos(n(θ+ϕ)),
/// sinθ·sin(n(θ+ϕ)) + cosθ·cos(n(θ+ϕ)))` with `n=order+1`; `:TE` (`kind=1`)
/// → `(-sinθ,cosθ)`; `:TM` (`kind=2`) → `(cosθ,sinθ)`. At `θ=0` these reduce
/// exactly to E.1's `(sin(nϕ),cos(nϕ))`/`(0,1)`/`(1,0)` shortcuts — Phase
/// E.3 subsumes that special case into this single general formula.
fn mode_angle_xy(kind: u8, order: i32, phi: f64, theta: f64) -> (f64, f64) {
    match kind {
        1 => (-theta.sin(), theta.cos()),
        2 => (theta.cos(), theta.sin()),
        _ => {
            let n = (order + 1) as f64;
            let arg = n * (theta + phi);
            (
                theta.cos() * arg.sin() - theta.sin() * arg.cos(),
                theta.sin() * arg.sin() + theta.cos() * arg.cos(),
            )
        }
    }
}

/// C `integrand_v` callback for `hcubature_v` — the `full=true` (Phase E.3)
/// counterpart of `modal_integrand_v`. `x` is `2·npt` doubles, point-major
/// (`x[2·p]=r`, `x[2·p+1]=θ` for point `p`) per `cubature.h`'s convention.
///
/// # Safety
/// Same contract as `modal_integrand_v`, with `x` valid for `2*npt` reads.
unsafe extern "C" fn modal_integrand_v_full(
    _ndim: c_uint,
    npt: size_t,
    x: *const c_double,
    fdata: *mut c_void,
    fdim: c_uint,
    fval: *mut c_double,
) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let sim = unsafe { &mut *(fdata as *mut CpuNativeSim) };
        let npt = npt;
        let fdim = fdim as usize;
        let xs = unsafe { std::slice::from_raw_parts(x, 2 * npt) };
        let fvals = unsafe { std::slice::from_raw_parts_mut(fval, npt * fdim) };
        sim.modal_eval_batch_2d(xs, fvals, fdim);
    }));
    match result {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// Thin wrapper making a raw pointer `Send + Sync` for use inside a rayon
/// closure — docs/dev/BACKLOG.md S2 item 2 (`apply_plasma_radial`'s per-column
/// parallelization). Safe here because the pointee (the ionization-rate
/// LUT, `PptIonizationRate`/`AdkIonizationRate`) is: (a) owned by Julia and
/// valid for the lifetime of the enclosing `RustNativeStepper` — same
/// invariant `plasma_ion_ptr`/`plasma_adk_ptr` already document; (b) never
/// mutated after construction — only `.rate(...)`, a pure read, is called
/// through it — so concurrent reads from multiple rayon workers are
/// race-free. Raw pointers are neither `Send` nor `Sync` by default in
/// Rust (a conservative blanket default, not a specific hazard here), so
/// this wrapper exists purely to let the borrow/thread checker see what is
/// already true of the underlying data.
struct ReadOnlyPtr<T>(*const T);
// Manual impls (not `#[derive]`): `*const T` is `Copy` for any `T`, but
// `#[derive(Copy)]` on a generic struct naively adds a `T: Copy` bound,
// which `PptIonizationRate`/`AdkIonizationRate` don't need to satisfy.
impl<T> Clone for ReadOnlyPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for ReadOnlyPtr<T> {}
unsafe impl<T> Send for ReadOnlyPtr<T> {}
unsafe impl<T> Sync for ReadOnlyPtr<T> {}

/// Free-function twin of `plasma_rate` (no `&self`) for use inside a rayon
/// closure that also holds disjoint `&mut self.field` borrows — calling
/// the `&self` method there would conservatively conflict even though it
/// only touches `plasma_is_adk`/`plasma_adk_ptr`/`plasma_ion_ptr`. Same
/// logic as `plasma_rate`, verbatim.
fn plasma_rate_at(
    is_adk: bool,
    adk_ptr: ReadOnlyPtr<crate::ionization::AdkIonizationRate>,
    ion_ptr: ReadOnlyPtr<crate::ionization::PptIonizationRate>,
    e_abs: f64,
) -> f64 {
    if is_adk {
        let ion = unsafe { &*adk_ptr.0 };
        ion.rate(e_abs).unwrap_or(0.0)
    } else {
        let ion = unsafe { &*ion_ptr.0 };
        ion.rate(e_abs)
            .unwrap_or_else(|_| ion.rate(ion.e_max).unwrap_or(0.0))
    }
}

/// Cumulative trapezoidal integral: `dst[0]=0`, `dst[i] = dst[i-1] + (src[i-1]+src[i])/2·dt`.
/// Reproduces `Maths.cumtrapz!(out, y, δt)` (src/Maths.jl:323-328).
fn cumtrapz_slice_f64(src: &[f64], dst: &mut [f64], dt: f64) {
    let n = src.len();
    if n == 0 {
        return;
    }
    dst[0] = 0.0;
    for i in 1..n {
        dst[i] = dst[i - 1] + 0.5 * (src[i - 1] + src[i]) * dt;
    }
}

/// Backend-agnostic resident-stepper interface: `CpuNativeSim` and
/// `CudaNativeSim` both implement this, and `NativeSim` (below) holds one as
/// a boxed trait object so the `extern "C"` FFI layer (`init_native_sim`,
/// `init_cuda_native_sim`, `native_step`, …) can dispatch without knowing
/// which backend it has.
///
/// This trait is a Rust-internal abstraction, not itself part of the FFI
/// ABI — Julia never calls these methods directly, only through the
/// `extern "C"` free functions later in this file, which validate their own
/// arguments and document their own `# Safety` contracts before forwarding
/// here. Every method below nonetheless carries the same general contract
/// its FFI-facing caller already upholds: every raw-pointer parameter must
/// be non-null (unless explicitly nullable) and valid for reads (`*const`)
/// or writes (`*mut`) of exactly the element count implied by the
/// accompanying length parameter (e.g. `n`, `n_time`, `n_r`, `n_osc`) for
/// the duration of the call, with no concurrent aliasing access. Individual
/// methods note anything beyond this baseline.
#[allow(clippy::missing_safety_doc)]
pub trait NativeBackend {
    unsafe fn set_field(&mut self, data: *const c_double, n: size_t) -> i32;
    unsafe fn resync_field(&mut self, data: *const c_double, n: size_t) -> i32;
    unsafe fn get_field(&self, data: *mut c_double, n: size_t) -> i32;
    unsafe fn get_ks_stage(&self, idx: size_t, data: *mut c_double, n: size_t) -> i32;
    unsafe fn apply_prop(&mut self, y: *mut c_double, n: size_t, t1: f64, t2: f64) -> i32;
    unsafe fn debug_linop_at(&mut self, z: c_double, data: *mut c_double, n: size_t) -> i32;
    unsafe fn debug_beta1_at(
        &mut self,
        z: c_double,
        out_dens: *mut c_double,
        out_beta1: *mut c_double,
    ) -> i32;
    unsafe fn set_fftw_plans(
        &mut self,
        lib_path: *const c_char,
        n_time: size_t,
        n_time_over: size_t,
        is_real: c_int,
        flags: c_uint,
        wisdom_path: *const c_char,
    ) -> i32;
    unsafe fn set_mode_avg_params(
        &mut self,
        n_time: size_t,
        n_time_over: size_t,
        towin: *const c_double,
        owin: *const c_double,
        sidx: *const u8,
        pre_re: *const c_double,
        pre_im: *const c_double,
        beta: *const c_double,
        kerr_fac: c_double,
        nlscale: c_double,
        sqrt_aeff: c_double,
    ) -> i32;
    unsafe fn set_mode_avg_noise(&mut self, noise: *const c_double, n: size_t) -> i32;
    unsafe fn set_mode_avg_noise_cplx(
        &mut self,
        noise_re: *const c_double,
        noise_im: *const c_double,
        n: size_t,
    ) -> i32;
    unsafe fn set_zdep_mode_avg_params(
        &mut self,
        n_z: size_t,
        z_pts: *const c_double,
        p_pts: *const c_double,
        n_dspl: size_t,
        dspl_x: *const c_double,
        dspl_y: *const c_double,
        dspl_d: *const c_double,
        gamma: *const c_double,
        nwg_re: *const c_double,
        nwg_im: *const c_double,
        omega: *const c_double,
        model: c_uint,
        loss_on: c_uint,
        eps0_gamma3: c_double,
        omega0: c_double,
        gamma0: c_double,
        dgamma0: c_double,
        nwg0_re: c_double,
        nwg0_im: c_double,
        dnwg0_re: c_double,
        dnwg0_im: c_double,
    ) -> i32;
    unsafe fn set_plasma_params(
        &mut self,
        ion_ptr: *const crate::ionization::PptIonizationRate,
        ionpot: c_double,
        e_ratio: c_double,
        preionfrac: c_double,
        dt: c_double,
        density: c_double,
    ) -> i32;
    unsafe fn set_plasma_params_adk(
        &mut self,
        ion_ptr: *const crate::ionization::AdkIonizationRate,
        ionpot: c_double,
        e_ratio: c_double,
        preionfrac: c_double,
        dt: c_double,
        density: c_double,
    ) -> i32;
    unsafe fn set_radial_params(
        &mut self,
        n_time: size_t,
        n_time_over: size_t,
        n_r: size_t,
        t_matrix: *const c_double,
        scale_fwd: c_double,
        scale_inv: c_double,
        towin: *const c_double,
        kerr_fac: c_double,
        m_re: *const c_double,
        m_im: *const c_double,
    ) -> i32;
    unsafe fn set_radial_noise(&mut self, noise: *const c_double, n: size_t) -> i32;
    unsafe fn set_radial_noise_cplx(
        &mut self,
        noise_re: *const c_double,
        noise_im: *const c_double,
        n: size_t,
    ) -> i32;
    unsafe fn set_raman_params(
        &mut self,
        omega: *const c_double,
        gamma: *const c_double,
        coupling: *const c_double,
        n_osc: size_t,
        dt: c_double,
        density: c_double,
        thg: c_int,
    ) -> i32;
    unsafe fn set_raman_fft_params(
        &mut self,
        omega: *const c_double,
        amp: *const c_double,
        gauss_w: *const c_double,
        lorentz_w: *const c_double,
        n_osc: size_t,
        scale: c_double,
        dt: c_double,
        n_time: size_t,
        density: c_double,
    ) -> i32;
    unsafe fn set_modal_params(
        &mut self,
        n_time: size_t,
        n_time_over: size_t,
        n_modes: size_t,
        npol: size_t,
        a: c_double,
        unm: *const c_double,
        inv_sqrt_n: *const c_double,
        order: *const i32,
        kind: *const u8,
        phi: *const c_double,
        full: u8,
        pol_select: *const u8,
        towin: *const c_double,
        kerr_fac: c_double,
        nlfac_re: *const c_double,
        nlfac_im: *const c_double,
        lib_path: *const c_char,
        rtol: c_double,
        atol: c_double,
        maxevals: size_t,
    ) -> i32;
    unsafe fn set_free_params(
        &mut self,
        n_time: size_t,
        n_time_over: size_t,
        n_y: size_t,
        n_x: size_t,
        flags: c_uint,
        towin: *const c_double,
        kerr_fac: c_double,
        m_re: *const c_double,
        m_im: *const c_double,
    ) -> i32;
    unsafe fn set_free_zdep_params(
        &mut self,
        flength: c_double,
        p0: c_double,
        p1: c_double,
        n_dspl: size_t,
        dspl_x: *const c_double,
        dspl_y: *const c_double,
        dspl_d: *const c_double,
        gamma: *const c_double,
        omega: *const c_double,
        omegawin: *const c_double,
        kperp2: *const c_double,
        sidx: *const u8,
        eps0_gamma3: c_double,
        omega0: c_double,
        gamma0: c_double,
        dgamma0: c_double,
    ) -> i32;
    unsafe fn set_modal_zdep_params(
        &mut self,
        flength: c_double,
        a0: c_double,
        n_a: size_t,
        a_x: *const c_double,
        a_y: *const c_double,
        a_d: *const c_double,
        omega: *const c_double,
        sidx: *const u8,
        model: u8,
        loss_on: u8,
        eco: *const c_double,
        vn_re: *const c_double,
        vn_im: *const c_double,
        omega0: c_double,
        ref_mode: size_t,
        eco0: *const c_double,
        deco0: *const c_double,
        v0_re: *const c_double,
        v0_im: *const c_double,
        dv0_re: *const c_double,
        dv0_im: *const c_double,
    ) -> i32;
    /// Sets the rayon thread count used by later-phase parallel RHS seams
    /// (docs/dev/BACKLOG.md S2 item 1). `n <= 1` means "sequential" — every parallel
    /// branch this enables is gated behind `n_threads > 1`, so the default
    /// (1, matching pre-S2 behavior) never changes results. Builds a
    /// per-sim `rayon::ThreadPool` (not the process-global pool) so
    /// multiple sims/tests coexisting in one process never contend over a
    /// single global configuration.
    unsafe fn set_threads(&mut self, n: size_t) -> i32;
    /// docs/dev/BACKLOG.md S5.2: forces the QDHT BLAS-3 `dgemm` path (opt-in via
    /// `AMALTHEA_QDHT_BLAS`) to be skipped in favor of the row-parallel Rayon
    /// fallback, regardless of whether a BLAS library was loaded.
    ///
    /// **What this actually guards against:** the default native path is
    /// already run-to-run deterministic on one machine — every other
    /// parallel seam in the codebase (the QDHT fallback itself, the older
    /// per-kernel `AMALTHEA_USE_RUST_QDHT` batch loops in `ffi.rs`) is
    /// embarrassingly parallel (each output row/column computed
    /// independently, no cross-thread reduction), and FFTW native plans
    /// only ever use `FFTW_ESTIMATE`, so thread count and repeated runs
    /// don't change the result even with `deterministic=false`. The one
    /// real lever this flag has is `BLAS_API`: it's a process-global
    /// `OnceLock`, populated once (if ever) by the per-kernel
    /// `AMALTHEA_USE_RUST_QDHT`+`AMALTHEA_QDHT_BLAS` path, and once populated it
    /// silently makes *every later* native-path QDHT call in that process
    /// eligible for the BLAS-3 route too. `deterministic=true` is what
    /// makes that eligibility **invariant to whether some other part of
    /// the process happened to touch BLAS**, and to which BLAS
    /// implementation/thread-count is linked — it is not claiming a
    /// same-process, same-BLAS-state run is otherwise non-reproducible
    /// (at the problem sizes here, a single-threaded/sub-threshold BLAS
    /// call is itself almost certainly repeat-run stable). Does **not**
    /// guarantee bit-identical results across different machines/CPU
    /// targets (the crate is built with `target-cpu=native`, so a
    /// different build host takes a different SIMD/libm path).
    unsafe fn set_deterministic(&mut self, on: c_int) -> i32;
    unsafe fn step(
        &mut self,
        yn: *mut Complex<f64>,
        t_old: f64,
        t_new: f64,
        dtn: f64,
        rtol: f64,
        atol: f64,
        safety: f64,
        max_dt: f64,
        min_dt: f64,
        errlast_in: f64,
        locextrap: i32,
        result: *mut NativeStepResult,
    ) -> i32;
    /// Saves the process's current planner wisdom (accumulated across every
    /// plan created in this process, not just this `NativeSim` — FFTW's
    /// wisdom is process-global) to `path` — docs/dev/BACKLOG.md S1 item 1.
    /// Best-effort: returns 0 on success, 1 if unavailable (no `fftw_api`
    /// yet, symbol missing, or the write failed) — never a hard error,
    /// since wisdom is a planning-time speedup, not a correctness
    /// requirement. Call once, after all `set_*_params` calls (i.e. after
    /// every plan for this sim has been created), mirroring Julia's own
    /// load-before/save-after `Utils.loadFFTwisdom`/`saveFFTwisdom` pattern.
    unsafe fn wisdom_export(&mut self, path: *const c_char) -> i32;
}
pub struct NativeSim {
    pub backend: Box<dyn NativeBackend>,
}
impl NativeBackend for CpuNativeSim {
    unsafe fn set_field(&mut self, data: *const c_double, n: size_t) -> i32 {
        let sim = self;

        if data.is_null() || n != sim.n {
            return -1;
        }
        let src = unsafe { std::slice::from_raw_parts(data as *const Complex<f64>, n) };
        sim.field.copy_from_slice(src);

        if sim.is_free {
            let field = sim.field.clone();
            if sim.is_real {
                sim.rhs_free(0, &field);
            } else {
                sim.rhs_free_env(0, &field);
            }
        } else if sim.is_modal {
            let field = sim.field.clone();
            sim.rhs_modal(0, &field);
        } else if sim.is_radial {
            let field = sim.field.clone();
            if sim.is_real {
                sim.rhs_radial(0, &field);
            } else {
                sim.rhs_radial_env(0, &field);
            }
        } else if !sim.beta.is_empty() {
            let field = sim.field.clone();
            if sim.is_real {
                sim.rhs_mode_avg_real(0, &field);
            } else {
                sim.rhs_mode_avg_env(0, &field);
            }
        }
        0
    }
    unsafe fn resync_field(&mut self, data: *const c_double, n: size_t) -> i32 {
        let sim = self;

        if data.is_null() || n != sim.n {
            return -1;
        }
        let src = unsafe { std::slice::from_raw_parts(data as *const Complex<f64>, n) };
        sim.field.copy_from_slice(src);
        0
    }
    unsafe fn get_field(&self, data: *mut c_double, n: size_t) -> i32 {
        let sim = self;

        if data.is_null() || n != sim.n {
            return -1;
        }
        let dst = unsafe { std::slice::from_raw_parts_mut(data as *mut Complex<f64>, n) };
        dst.copy_from_slice(&sim.field);
        0
    }
    unsafe fn get_ks_stage(&self, idx: size_t, data: *mut c_double, n: size_t) -> i32 {
        let sim = self;

        if data.is_null() || idx >= 7 || n != sim.n {
            return -1;
        }
        let dst = unsafe { std::slice::from_raw_parts_mut(data as *mut Complex<f64>, n) };
        dst.copy_from_slice(&sim.ks[idx]);
        0
    }
    unsafe fn apply_prop(&mut self, y: *mut c_double, n: size_t, t1: f64, t2: f64) -> i32 {
        let sim = self;

        if y.is_null() || n != sim.n {
            return -1;
        }
        let slice = unsafe { std::slice::from_raw_parts_mut(y as *mut Complex<f64>, n) };
        sim.ensure_linop_at(t2);
        apply_prop_cached(
            slice,
            &sim.linop,
            t2 - t1,
            &mut sim.exp_cache,
            sim.linop_version,
        );
        0
    }
    unsafe fn debug_linop_at(&mut self, z: c_double, data: *mut c_double, n: size_t) -> i32 {
        let sim = self;

        if data.is_null() || n != sim.n {
            return -1;
        }
        sim.ensure_linop_at(z);
        let dst = unsafe { std::slice::from_raw_parts_mut(data as *mut Complex<f64>, n) };
        dst.copy_from_slice(&sim.linop);
        0
    }
    unsafe fn debug_beta1_at(
        &mut self,
        z: c_double,
        out_dens: *mut c_double,
        out_beta1: *mut c_double,
    ) -> i32 {
        let sim = self;

        if out_dens.is_null() || out_beta1.is_null() {
            return -1;
        }
        sim.ensure_linop_at(z);
        let pressure = zdep_pressure_at(z, &sim.zdep_grad_z, &sim.zdep_grad_p);
        let dens = sim.zdep_dens_lut.as_ref().unwrap().eval(pressure);
        let eco0 = 1.0 + sim.zdep_gamma0 * dens;
        let deco0 = sim.zdep_dgamma0 * dens;
        let (neff0, dneff0) = if sim.zdep_model == 0 {
            let neff0 = (Complex::new(eco0, 0.0) - sim.zdep_nwg0).sqrt();
            let dneff0 = (Complex::new(deco0, 0.0) - sim.zdep_dnwg0) / (2.0 * neff0);
            (neff0, dneff0)
        } else {
            let neff0 = Complex::new(1.0 + (eco0 - 1.0) / 2.0, 0.0) - sim.zdep_nwg0;
            let dneff0 = Complex::new(deco0 / 2.0, 0.0) - sim.zdep_dnwg0;
            (neff0, dneff0)
        };
        const C: f64 = 299_792_458.0;
        let beta1 = (neff0.re + sim.zdep_omega0 * dneff0.re) / C;
        unsafe {
            *out_dens = dens;
            *out_beta1 = beta1;
        }
        0
    }
    unsafe fn set_fftw_plans(
        &mut self,
        lib_path: *const c_char,
        n_time: size_t,
        n_time_over: size_t,
        is_real: c_int,
        flags: c_uint,
        wisdom_path: *const c_char,
    ) -> i32 {
        let sim = self;

        if lib_path.is_null() {
            return -1;
        }
        let path_str = unsafe { CStr::from_ptr(lib_path).to_str().unwrap_or("") };
        let api = match FftwApi::load(Some(path_str)) {
            Ok(api) => api,
            Err(e) => {
                eprintln!("native_set_fftw_plans: failed to load FFTW: {}", e);
                return -2;
            }
        };

        // S1 item 1 (docs/dev/BACKLOG.md): best-effort wisdom load, before any plans
        // below are created (only plans created *after* this call can benefit —
        // FFTW's own contract). `wisdom_path` is null unless the Julia caller
        // opts in; a missing/unreadable file is silently ignored either way.
        if !wisdom_path.is_null()
            && let Ok(wisdom_str) = unsafe { CStr::from_ptr(wisdom_path) }.to_str()
        {
            api.import_wisdom_from_filename(wisdom_str);
        }

        sim.is_real = is_real != 0;
        sim.fft_norm = 1.0 / (n_time as f64);
        sim.fft_norm_over = 1.0 / (n_time_over as f64);

        if sim.is_real {
            sim.fft_r2c = Some(RealFft1d::new(&api, n_time, flags));
            sim.fft_r2c_over = Some(RealFft1d::new(&api, n_time_over, flags));
        } else {
            sim.fft_c2c = Some(ComplexFft1d::new(&api, n_time, flags));
            sim.fft_c2c_over = Some(ComplexFft1d::new(&api, n_time_over, flags));
        }

        sim.fftw_api = Some(api);
        0
    }
    unsafe fn set_threads(&mut self, n: size_t) -> i32 {
        let sim = self;
        let n = n.max(1);
        if sim.n_threads != n {
            sim.n_threads = n;
            // Drop any existing pool so it's rebuilt lazily (correct size)
            // the next time a parallel seam actually runs.
            sim.thread_pool = None;
        }
        0
    }
    unsafe fn set_deterministic(&mut self, on: c_int) -> i32 {
        let sim = self;
        sim.deterministic = on != 0;
        if let Some(ref mut qdht) = sim.qdht {
            qdht.set_deterministic(sim.deterministic);
        }
        0
    }
    unsafe fn wisdom_export(&mut self, path: *const c_char) -> i32 {
        let sim = self;
        let api = match sim.fftw_api.as_ref() {
            Some(api) => api,
            None => return 1,
        };
        if path.is_null() {
            return 1;
        }
        let Ok(path_str) = unsafe { CStr::from_ptr(path) }.to_str() else {
            return 1;
        };
        if api.export_wisdom_to_filename(path_str) {
            0
        } else {
            1
        }
    }
    unsafe fn set_mode_avg_params(
        &mut self,
        n_time: size_t,
        n_time_over: size_t,
        towin: *const c_double,
        owin: *const c_double,
        sidx: *const u8,
        pre_re: *const c_double,
        pre_im: *const c_double,
        beta: *const c_double,
        kerr_fac: c_double,
        nlscale: c_double,
        sqrt_aeff: c_double,
    ) -> i32 {
        let s = self;

        s.n_time = n_time;
        s.n_time_over = n_time_over;
        s.n_spec = s.n;

        if s.is_real {
            // RealGrid: r2c — spectral length is Nt/2+1
            s.n_spec_over = n_time_over / 2 + 1;
            s.eto = vec![0.0; n_time_over];
            s.pto = vec![0.0; n_time_over];
            s.eoo = vec![Complex::new(0.0, 0.0); s.n_spec_over];
            s.poo = vec![Complex::new(0.0, 0.0); s.n_spec_over];
        } else {
            // EnvGrid: c2c — spectral length equals time length
            s.n_spec_over = n_time_over;
            s.eto_cplx = vec![Complex::new(0.0, 0.0); n_time_over];
            s.pto_cplx = vec![Complex::new(0.0, 0.0); n_time_over];
            s.eoo_cplx = vec![Complex::new(0.0, 0.0); n_time_over];
            s.poo_cplx = vec![Complex::new(0.0, 0.0); n_time_over];
        }

        if !towin.is_null() {
            s.towin = unsafe { std::slice::from_raw_parts(towin, n_time_over) }.to_vec();
        } else {
            s.towin = vec![1.0; n_time_over];
        }
        if !owin.is_null() {
            s.owin = unsafe { std::slice::from_raw_parts(owin, s.n_spec) }.to_vec();
        } else {
            s.owin = vec![1.0; s.n_spec];
        }
        if !sidx.is_null() {
            let sidx_sl = unsafe { std::slice::from_raw_parts(sidx, s.n_spec) };
            s.sidx = sidx_sl.iter().map(|&x| x != 0).collect();
        } else {
            s.sidx = vec![true; s.n_spec];
        }
        if !pre_re.is_null() && !pre_im.is_null() {
            let pre_re_sl = unsafe { std::slice::from_raw_parts(pre_re, s.n_spec) };
            let pre_im_sl = unsafe { std::slice::from_raw_parts(pre_im, s.n_spec) };
            s.pre = pre_re_sl
                .iter()
                .zip(pre_im_sl.iter())
                .map(|(&r, &i)| Complex::new(r, i))
                .collect();
        } else {
            s.pre = vec![Complex::new(0.0, 0.0); s.n_spec];
        }
        if !beta.is_null() {
            s.beta = unsafe { std::slice::from_raw_parts(beta, s.n_spec) }.to_vec();
        } else {
            s.beta = vec![1.0; s.n_spec]; // prevent division by zero if default is used
        }
        s.kerr_fac = kerr_fac;
        s.nlscale = nlscale;
        s.sqrt_aeff = sqrt_aeff;
        s.has_noise = false;
        s.et_noise = Vec::new();
        s.et_noise_cplx = Vec::new();

        // De-branch the sidx-gated norm/freq-window steps (docs/dev/BACKLOG.md S1 item
        // 4): fold sidx into owin, and precompute pre/beta*sqrt_aeff, both as
        // the identity (1+0i) outside sidx. pre/beta/sqrt_aeff/owin are fixed
        // for the whole solve (never mutated after this call), so this is a
        // one-time cost, not a per-RHS-call one.
        s.norm_pre_beta = (0..s.n_spec)
            .map(|i| {
                if s.sidx[i] {
                    s.pre[i] / s.beta[i] * s.sqrt_aeff
                } else {
                    Complex::new(1.0, 0.0)
                }
            })
            .collect();
        for i in 0..s.n_spec {
            if !s.sidx[i] {
                s.owin[i] = 1.0;
            }
        }
        0
    }
    unsafe fn set_mode_avg_noise(&mut self, noise: *const c_double, n: size_t) -> i32 {
        let s = self;
        if noise.is_null() || n == 0 {
            return -1;
        }
        if n != s.n_time_over {
            return -2;
        }
        s.et_noise = unsafe { std::slice::from_raw_parts(noise, n) }.to_vec();
        s.has_noise = true;
        0
    }
    unsafe fn set_mode_avg_noise_cplx(
        &mut self,
        noise_re: *const c_double,
        noise_im: *const c_double,
        n: size_t,
    ) -> i32 {
        let s = self;
        if noise_re.is_null() || noise_im.is_null() || n == 0 {
            return -1;
        }
        if n != s.n_time_over {
            return -2;
        }
        let re = unsafe { std::slice::from_raw_parts(noise_re, n) };
        let im = unsafe { std::slice::from_raw_parts(noise_im, n) };
        s.et_noise_cplx = re
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        s.has_noise = true;
        0
    }
    unsafe fn set_zdep_mode_avg_params(
        &mut self,
        n_z: size_t,
        z_pts: *const c_double,
        p_pts: *const c_double,
        n_dspl: size_t,
        dspl_x: *const c_double,
        dspl_y: *const c_double,
        dspl_d: *const c_double,
        gamma: *const c_double,
        nwg_re: *const c_double,
        nwg_im: *const c_double,
        omega: *const c_double,
        model: c_uint,
        loss_on: c_uint,
        eps0_gamma3: c_double,
        omega0: c_double,
        gamma0: c_double,
        dgamma0: c_double,
        nwg0_re: c_double,
        nwg0_im: c_double,
        dnwg0_re: c_double,
        dnwg0_im: c_double,
    ) -> i32 {
        let s = self;

        if s.sidx.len() != s.n {
            return -3;
        }
        if n_z < 2 {
            return -4;
        }
        if z_pts.is_null()
            || p_pts.is_null()
            || dspl_x.is_null()
            || dspl_y.is_null()
            || dspl_d.is_null()
            || gamma.is_null()
            || nwg_re.is_null()
            || nwg_im.is_null()
            || omega.is_null()
        {
            return -1;
        }

        let dspl_x_v = unsafe { std::slice::from_raw_parts(dspl_x, n_dspl) }.to_vec();
        let dspl_y_v = unsafe { std::slice::from_raw_parts(dspl_y, n_dspl) }.to_vec();
        let dspl_d_v = unsafe { std::slice::from_raw_parts(dspl_d, n_dspl) }.to_vec();

        s.zdep_grad_z = unsafe { std::slice::from_raw_parts(z_pts, n_z) }.to_vec();
        s.zdep_grad_p = unsafe { std::slice::from_raw_parts(p_pts, n_z) }.to_vec();
        s.zdep_dens_lut = Some(HermiteSpline::from_parts(dspl_x_v, dspl_y_v, dspl_d_v));
        s.zdep_gamma = unsafe { std::slice::from_raw_parts(gamma, s.n) }.to_vec();
        s.zdep_nwg_re = unsafe { std::slice::from_raw_parts(nwg_re, s.n) }.to_vec();
        s.zdep_nwg_im = unsafe { std::slice::from_raw_parts(nwg_im, s.n) }.to_vec();
        s.zdep_omega = unsafe { std::slice::from_raw_parts(omega, s.n) }.to_vec();
        s.zdep_model = if model == 0 { 0 } else { 1 };
        s.zdep_loss_on = loss_on != 0;
        s.zdep_kerr_fac_per_dens = eps0_gamma3;
        s.zdep_omega0 = omega0;
        s.zdep_gamma0 = gamma0;
        s.zdep_dgamma0 = dgamma0;
        s.zdep_nwg0 = Complex::new(nwg0_re, nwg0_im);
        s.zdep_dnwg0 = Complex::new(dnwg0_re, dnwg0_im);
        s.zdep_last_z = f64::NAN;
        s.is_zdep_mode_avg = true;
        0
    }
    unsafe fn set_plasma_params(
        &mut self,
        ion_ptr: *const crate::ionization::PptIonizationRate,
        ionpot: c_double,
        e_ratio: c_double,
        preionfrac: c_double,
        dt: c_double,
        density: c_double,
    ) -> i32 {
        let s = self;

        // Radial (Phase D.2): one independent plasma state per r-column, laid out
        // column-major `(n_time_over, n_r)` exactly like `radial_eto`/`radial_pto`
        // — `native_set_radial_params` must run first so `s.n_r` is already set.
        let n = if s.is_radial {
            s.n_time_over * s.n_r
        } else if s.is_free {
            s.n_time_over * s.n_y * s.n_x
        } else {
            s.n_time_over
        };
        if n == 0 {
            return -2;
        }
        s.plasma_ion_ptr = ion_ptr;
        s.plasma_adk_ptr = std::ptr::null();
        s.plasma_is_adk = false;
        s.plasma_ionpot = ionpot;
        s.plasma_e_ratio = e_ratio;
        s.plasma_preionfrac = preionfrac;
        s.plasma_dt = dt;
        s.plasma_density = density;
        s.plas_rate = vec![0.0; n];
        s.plas_fraction = vec![0.0; n];
        s.plas_phase = vec![0.0; n];
        s.plas_j = vec![0.0; n];
        s.plas_p = vec![0.0; n];
        s.has_plasma = true;
        0
    }
    unsafe fn set_plasma_params_adk(
        &mut self,
        ion_ptr: *const crate::ionization::AdkIonizationRate,
        ionpot: c_double,
        e_ratio: c_double,
        preionfrac: c_double,
        dt: c_double,
        density: c_double,
    ) -> i32 {
        let s = self;
        let n = if s.is_radial {
            s.n_time_over * s.n_r
        } else if s.is_free {
            s.n_time_over * s.n_y * s.n_x
        } else {
            s.n_time_over
        };
        if n == 0 {
            return -2;
        }
        s.plasma_ion_ptr = std::ptr::null();
        s.plasma_adk_ptr = ion_ptr;
        s.plasma_is_adk = true;
        s.plasma_ionpot = ionpot;
        s.plasma_e_ratio = e_ratio;
        s.plasma_preionfrac = preionfrac;
        s.plasma_dt = dt;
        s.plasma_density = density;
        s.plas_rate = vec![0.0; n];
        s.plas_fraction = vec![0.0; n];
        s.plas_phase = vec![0.0; n];
        s.plas_j = vec![0.0; n];
        s.plas_p = vec![0.0; n];
        s.has_plasma = true;
        0
    }
    unsafe fn set_radial_params(
        &mut self,
        n_time: size_t,
        n_time_over: size_t,
        n_r: size_t,
        t_matrix: *const c_double,
        scale_fwd: c_double,
        scale_inv: c_double,
        towin: *const c_double,
        kerr_fac: c_double,
        m_re: *const c_double,
        m_im: *const c_double,
    ) -> i32 {
        let s = self;

        if n_r == 0 || !s.n.is_multiple_of(n_r) {
            return -1;
        }
        if t_matrix.is_null() || m_re.is_null() || m_im.is_null() {
            return -1;
        }
        let n_spec = s.n / n_r;

        s.is_radial = true;
        s.n_r = n_r;
        s.n_time = n_time;
        s.n_time_over = n_time_over;
        s.n_spec = n_spec;
        s.n_spec_over = if s.is_real {
            n_time_over / 2 + 1
        } else {
            n_time_over
        };

        let t_mat_sl = unsafe { std::slice::from_raw_parts(t_matrix, n_r * n_r) };
        let mut qdht = QdhtFfiHandle::new(t_mat_sl, n_r, scale_fwd, scale_inv, n_time_over);
        qdht.set_deterministic(s.deterministic);
        s.qdht = Some(qdht);
        s.qdht_scale_fwd = scale_fwd;
        s.qdht_scale_inv = scale_inv;

        if !towin.is_null() {
            s.towin = unsafe { std::slice::from_raw_parts(towin, n_time_over) }.to_vec();
        } else {
            s.towin = vec![1.0; n_time_over];
        }
        s.kerr_fac = kerr_fac;

        let m_re_sl = unsafe { std::slice::from_raw_parts(m_re, n_spec * n_r) };
        let m_im_sl = unsafe { std::slice::from_raw_parts(m_im, n_spec * n_r) };
        s.radial_m = m_re_sl
            .iter()
            .zip(m_im_sl.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();

        s.radial_eoo = vec![Complex::new(0.0, 0.0); s.n_spec_over * n_r];
        s.radial_poo = vec![Complex::new(0.0, 0.0); s.n_spec_over * n_r];
        if s.is_real {
            s.radial_eto = vec![0.0; n_time_over * n_r];
            s.radial_pto = vec![0.0; n_time_over * n_r];
            s.radial_eto_c = Vec::new();
            s.radial_pto_c = Vec::new();
        } else {
            s.radial_eto = Vec::new();
            s.radial_pto = Vec::new();
            s.radial_eto_c = vec![Complex::new(0.0, 0.0); n_time_over * n_r];
            s.radial_pto_c = vec![Complex::new(0.0, 0.0); n_time_over * n_r];
        }
        s.has_radial_noise = false;
        s.radial_et_noise = Vec::new();
        s.radial_et_noise_cplx = Vec::new();

        0
    }
    unsafe fn set_radial_noise(&mut self, noise: *const c_double, n: size_t) -> i32 {
        let s = self;
        if noise.is_null() || n == 0 {
            return -1;
        }
        if n != s.n_time_over * s.n_r {
            return -2;
        }
        s.radial_et_noise = unsafe { std::slice::from_raw_parts(noise, n) }.to_vec();
        s.has_radial_noise = true;
        0
    }
    unsafe fn set_radial_noise_cplx(
        &mut self,
        noise_re: *const c_double,
        noise_im: *const c_double,
        n: size_t,
    ) -> i32 {
        let s = self;
        if noise_re.is_null() || noise_im.is_null() || n == 0 {
            return -1;
        }
        if n != s.n_time_over * s.n_r {
            return -2;
        }
        let re = unsafe { std::slice::from_raw_parts(noise_re, n) };
        let im = unsafe { std::slice::from_raw_parts(noise_im, n) };
        s.radial_et_noise_cplx = re
            .iter()
            .zip(im.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();
        s.has_radial_noise = true;
        0
    }
    unsafe fn set_raman_params(
        &mut self,
        omega: *const c_double,
        gamma: *const c_double,
        coupling: *const c_double,
        n_osc: size_t,
        dt: c_double,
        density: c_double,
        thg: c_int,
    ) -> i32 {
        let s = self;

        if s.n_time_over == 0 {
            return -2;
        }
        if omega.is_null() || gamma.is_null() || coupling.is_null() {
            return -1;
        }

        let omega_sl = unsafe { std::slice::from_raw_parts(omega, n_osc) };
        let gamma_sl = unsafe { std::slice::from_raw_parts(gamma, n_osc) };
        let coupling_sl = unsafe { std::slice::from_raw_parts(coupling, n_osc) };

        let oscillators: Vec<RamanOscillator> = (0..n_osc)
            .map(|i| RamanOscillator {
                omega: omega_sl[i],
                gamma: gamma_sl[i],
                coupling: coupling_sl[i],
            })
            .collect();

        // Radial (Phase D.4): one time-march per r-column, laid out column-major
        // `(n_time_over, n_r)` like `radial_eto`/`plas_*` — `solve()` resets its
        // internal oscillator state at entry (see `raman.rs`), so the *same*
        // solver instance can be called once per r-column with no cross-column
        // state leakage; only the scratch buffers need the extra width.
        let n = if s.is_radial {
            s.n_time_over * s.n_r
        } else if s.is_free {
            s.n_time_over * s.n_y * s.n_x
        } else {
            s.n_time_over
        };
        if n == 0 {
            return -2;
        }

        s.raman_solver = Some(TimeDomainRamanSolver::new(oscillators, dt));
        s.raman_density = density;
        s.raman_intensity = vec![0.0; n];
        s.raman_p = vec![0.0; n];
        s.has_raman = true;

        // Phase F.1: `thg=false` intensity is `1/2·|hilbert(E)|²` instead of
        // `E²` — needs its own resident c2c plan since RealGrid otherwise has
        // none (`fft_c2c`/`fft_c2c_over` are EnvGrid-only, see `set_fftw_plans`).
        // Built here (not `set_fftw_plans`) since it's only needed for this one
        // opt-in feature, and `fftw_api`/`n_time_over` are already available by
        // the time Raman is wired (always after `native_set_fftw_plans`).
        s.raman_thg = thg != 0;
        if !s.raman_thg {
            let api = match s.fftw_api.as_ref() {
                Some(api) => api,
                None => return -2,
            };
            s.raman_hilbert_fft = Some(ComplexFft1d::new(api, s.n_time_over, 1 << 6));
            s.raman_hilbert_a = vec![Complex::new(0.0, 0.0); s.n_time_over];
            s.raman_hilbert_b = vec![Complex::new(0.0, 0.0); s.n_time_over];
        }

        0
    }
    unsafe fn set_raman_fft_params(
        &mut self,
        omega: *const c_double,
        amp: *const c_double,
        gauss_w: *const c_double,
        lorentz_w: *const c_double,
        n_osc: size_t,
        scale: c_double,
        dt: c_double,
        n_time: size_t,
        density: c_double,
    ) -> i32 {
        let s = self;

        if s.n_time_over == 0 || n_time == 0 || n_time != s.n_time_over {
            return -2;
        }
        if omega.is_null()
            || amp.is_null()
            || gauss_w.is_null()
            || lorentz_w.is_null()
            || n_osc == 0
        {
            return -1;
        }
        let api = match s.fftw_api.as_ref() {
            Some(api) => api,
            None => return -2,
        };

        let omega_sl = unsafe { std::slice::from_raw_parts(omega, n_osc) };
        let amp_sl = unsafe { std::slice::from_raw_parts(amp, n_osc) };
        let gauss_sl = unsafe { std::slice::from_raw_parts(gauss_w, n_osc) };
        let lorentz_sl = unsafe { std::slice::from_raw_parts(lorentz_w, n_osc) };

        // Reproduces `(R::RamanRespIntermediateBroadening)(ht, ρ)`
        // (Raman.jl:97-105): a Gaussian-damped multi-line impulse response, zero
        // for t < 0 (causal, first half of the padded buffer) and zero-padded
        // in the second half (Nonlinear.jl:299's `h = zeros(length(t)*2)`).
        // `h` is real (Phase J.3: r2c/c2r halving, docs/dev/BACKLOG.md, MATH.md
        // §8.4) — previously packed into a `Complex<f64>` buffer for a c2c
        // plan; a `RealFft1d` gives the identical spectrum at ~half the cost.
        let n_over = 2 * n_time;
        let mut h = vec![0.0f64; n_over];
        for idx in 0..n_time {
            let t = idx as f64 * dt;
            let mut hv = 0.0;
            for i in 0..n_osc {
                hv += amp_sl[i]
                    * (-lorentz_sl[i] * t).exp()
                    * (-gauss_sl[i] * gauss_sl[i] * t * t / 4.0).exp()
                    * (omega_sl[i] * t).sin();
            }
            h[idx] = scale * hv;
        }

        let plan = RealFft1d::new(api, n_over, 1 << 6);
        let mut hw = vec![Complex::new(0.0, 0.0); plan.nspec()];
        plan.forward(&mut h, &mut hw);
        // Fold in Nonlinear.jl:415's `dt` and the ifft's `1/n_over` normalisation
        // once here, so the per-step convolution needs no further scaling.
        let post = dt / n_over as f64;
        for v in &mut hw {
            *v *= post;
        }

        s.raman_fft_plan = Some(plan);
        s.raman_fft_hw = hw;
        s.raman_fft_e2 = vec![0.0f64; n_over];
        s.raman_fft_ew = vec![Complex::new(0.0, 0.0); n_over / 2 + 1];
        s.raman_fft_density = density;
        s.has_raman_fft = true;

        0
    }
    unsafe fn set_modal_params(
        &mut self,
        n_time: size_t,
        n_time_over: size_t,
        n_modes: size_t,
        npol: size_t,
        a: c_double,
        unm: *const c_double,
        inv_sqrt_n: *const c_double,
        order: *const i32,
        kind: *const u8,
        phi: *const c_double,
        full: u8,
        pol_select: *const u8,
        towin: *const c_double,
        kerr_fac: c_double,
        nlfac_re: *const c_double,
        nlfac_im: *const c_double,
        lib_path: *const c_char,
        rtol: c_double,
        atol: c_double,
        maxevals: size_t,
    ) -> i32 {
        let s = self;

        if n_modes == 0 || npol == 0 || (npol != 1 && npol != 2) || !s.n.is_multiple_of(n_modes) {
            return -1;
        }
        if unm.is_null()
            || inv_sqrt_n.is_null()
            || order.is_null()
            || kind.is_null()
            || phi.is_null()
            || pol_select.is_null()
            || nlfac_re.is_null()
            || nlfac_im.is_null()
            || lib_path.is_null()
        {
            return -1;
        }
        let n_spec = s.n / n_modes;

        let path_str = unsafe { CStr::from_ptr(lib_path).to_str().unwrap_or("") };
        let cubature = match CubatureApi::load(path_str) {
            Ok(api) => api,
            Err(e) => {
                eprintln!("native_set_modal_params: failed to load libcubature: {}", e);
                return -2;
            }
        };

        s.is_modal = true;
        s.n_time = n_time;
        s.n_time_over = n_time_over;
        s.n_modes = n_modes;
        s.npol = npol;
        s.n_spec = n_spec;
        s.n_spec_over = if s.is_real {
            n_time_over / 2 + 1
        } else {
            n_time_over
        };
        s.modal_a = a;

        s.modal_unm = unsafe { std::slice::from_raw_parts(unm, n_modes) }.to_vec();
        s.modal_inv_sqrt_n = unsafe { std::slice::from_raw_parts(inv_sqrt_n, n_modes) }.to_vec();
        s.modal_order = unsafe { std::slice::from_raw_parts(order, n_modes) }.to_vec();
        s.modal_kind = unsafe { std::slice::from_raw_parts(kind, n_modes) }.to_vec();
        s.modal_phi = unsafe { std::slice::from_raw_parts(phi, n_modes) }.to_vec();
        s.modal_full = full != 0;
        s.modal_pol_select = unsafe { std::slice::from_raw_parts(pol_select, npol) }.to_vec();

        if !towin.is_null() {
            s.towin = unsafe { std::slice::from_raw_parts(towin, n_time_over) }.to_vec();
        } else {
            s.towin = vec![1.0; n_time_over];
        }
        s.modal_kerr_fac = kerr_fac;

        let re_sl = unsafe { std::slice::from_raw_parts(nlfac_re, n_spec) };
        let im_sl = unsafe { std::slice::from_raw_parts(nlfac_im, n_spec) };
        s.modal_nlfac = re_sl
            .iter()
            .zip(im_sl.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();

        s.cubature = Some(cubature);
        s.modal_rtol = rtol;
        s.modal_atol = atol;
        s.modal_maxevals = maxevals;

        // Per-worker scratch pool, entry 0 sized here (sole scratch on the
        // sequential path). `ensure_modal_pool` grows it to `n_threads` entries
        // lazily on the first threaded batch (S2 Phase 4 — modal RHS threading).
        s.modal_scratch_pool = vec![ModalScratch::new(
            n_modes,
            npol,
            n_spec,
            s.n_spec_over,
            n_time_over,
        )];
        s.modal_emega = vec![Complex::new(0.0, 0.0); s.n];

        0
    }
    unsafe fn set_free_params(
        &mut self,
        n_time: size_t,
        n_time_over: size_t,
        n_y: size_t,
        n_x: size_t,
        flags: c_uint,
        towin: *const c_double,
        kerr_fac: c_double,
        m_re: *const c_double,
        m_im: *const c_double,
    ) -> i32 {
        let s = self;

        if n_y == 0 || n_x == 0 {
            return -1;
        }
        if m_re.is_null() || m_im.is_null() {
            return -1;
        }
        let n_cols = n_y * n_x;
        if !s.n.is_multiple_of(n_cols) {
            return -1;
        }
        let n_spec = s.n / n_cols;

        let n_spec_over_tmp = if s.is_real {
            n_time_over / 2 + 1
        } else {
            n_time_over
        };

        s.is_free = true;
        s.n_time = n_time;
        s.n_time_over = n_time_over;
        s.n_spec = n_spec;
        s.n_spec_over = n_spec_over_tmp;
        s.n_y = n_y;
        s.n_x = n_x;
        s.free_fft_norm_over = 1.0 / (n_time_over * n_y * n_x) as f64;

        // Phase D.3: EnvGrid free-space uses a c2c 3-D plan (ComplexFft3d) and
        // complex time-domain buffers; RealGrid (Phase 6) keeps the r2c plan and
        // real time-domain buffers — same split as native_set_radial_params.
        if s.is_real {
            let fft3d = match s.fftw_api.as_ref() {
                Some(api) => RealFft3d::new(api, n_time_over, n_y, n_x, flags),
                None => return -2,
            };
            s.fft_r2c_3d = Some(fft3d);
            s.fft_c2c_3d = None;
            s.free_eto = vec![0.0; n_time_over * n_cols];
            s.free_pto = vec![0.0; n_time_over * n_cols];
            s.free_eto_c = Vec::new();
            s.free_pto_c = Vec::new();
        } else {
            let fft3d = match s.fftw_api.as_ref() {
                Some(api) => ComplexFft3d::new(api, n_time_over, n_y, n_x, flags),
                None => return -2,
            };
            s.fft_c2c_3d = Some(fft3d);
            s.fft_r2c_3d = None;
            s.free_eto = Vec::new();
            s.free_pto = Vec::new();
            s.free_eto_c = vec![Complex::new(0.0, 0.0); n_time_over * n_cols];
            s.free_pto_c = vec![Complex::new(0.0, 0.0); n_time_over * n_cols];
        }

        if !towin.is_null() {
            s.towin = unsafe { std::slice::from_raw_parts(towin, n_time_over) }.to_vec();
        } else {
            s.towin = vec![1.0; n_time_over];
        }
        s.kerr_fac = kerr_fac;

        let m_re_sl = unsafe { std::slice::from_raw_parts(m_re, n_spec * n_cols) };
        let m_im_sl = unsafe { std::slice::from_raw_parts(m_im, n_spec * n_cols) };
        s.free_m = m_re_sl
            .iter()
            .zip(m_im_sl.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect();

        s.free_eoo = vec![Complex::new(0.0, 0.0); s.n_spec_over * n_cols];
        s.free_poo = vec![Complex::new(0.0, 0.0); s.n_spec_over * n_cols];

        0
    }
    unsafe fn set_free_zdep_params(
        &mut self,
        flength: c_double,
        p0: c_double,
        p1: c_double,
        n_dspl: size_t,
        dspl_x: *const c_double,
        dspl_y: *const c_double,
        dspl_d: *const c_double,
        gamma: *const c_double,
        omega: *const c_double,
        omegawin: *const c_double,
        kperp2: *const c_double,
        sidx: *const u8,
        eps0_gamma3: c_double,
        omega0: c_double,
        gamma0: c_double,
        dgamma0: c_double,
    ) -> i32 {
        let s = self;

        if s.n_spec == 0 || s.n_y == 0 || s.n_x == 0 {
            return -3;
        }
        if dspl_x.is_null()
            || dspl_y.is_null()
            || dspl_d.is_null()
            || gamma.is_null()
            || omega.is_null()
            || omegawin.is_null()
            || kperp2.is_null()
            || sidx.is_null()
        {
            return -1;
        }

        let dspl_x_v = unsafe { std::slice::from_raw_parts(dspl_x, n_dspl) }.to_vec();
        let dspl_y_v = unsafe { std::slice::from_raw_parts(dspl_y, n_dspl) }.to_vec();
        let dspl_d_v = unsafe { std::slice::from_raw_parts(dspl_d, n_dspl) }.to_vec();

        s.zdep_free_flength = flength;
        s.zdep_free_p0 = p0;
        s.zdep_free_p1 = p1;
        s.zdep_free_dens_lut = Some(HermiteSpline::from_parts(dspl_x_v, dspl_y_v, dspl_d_v));
        s.zdep_free_gamma = unsafe { std::slice::from_raw_parts(gamma, s.n_spec) }.to_vec();
        s.zdep_free_omega = unsafe { std::slice::from_raw_parts(omega, s.n_spec) }.to_vec();
        s.zdep_free_omegawin = unsafe { std::slice::from_raw_parts(omegawin, s.n_spec) }.to_vec();
        s.zdep_free_kperp2 = unsafe { std::slice::from_raw_parts(kperp2, s.n_y * s.n_x) }.to_vec();
        s.zdep_free_sidx = unsafe { std::slice::from_raw_parts(sidx, s.n_spec) }
            .iter()
            .map(|&b| b != 0)
            .collect();
        s.zdep_free_kerr_fac_per_dens = eps0_gamma3;
        s.zdep_free_omega0 = omega0;
        s.zdep_free_gamma0 = gamma0;
        s.zdep_free_dgamma0 = dgamma0;
        s.zdep_free_last_z = f64::NAN;
        s.is_zdep_free = true;
        0
    }
    unsafe fn set_modal_zdep_params(
        &mut self,
        flength: c_double,
        a0: c_double,
        n_a: size_t,
        a_x: *const c_double,
        a_y: *const c_double,
        a_d: *const c_double,
        omega: *const c_double,
        sidx: *const u8,
        model: u8,
        loss_on: u8,
        eco: *const c_double,
        vn_re: *const c_double,
        vn_im: *const c_double,
        omega0: c_double,
        ref_mode: size_t,
        eco0: *const c_double,
        deco0: *const c_double,
        v0_re: *const c_double,
        v0_im: *const c_double,
        dv0_re: *const c_double,
        dv0_im: *const c_double,
    ) -> i32 {
        let s = self;

        if s.n_spec == 0 || s.n_modes == 0 {
            return -3;
        }
        if a_x.is_null()
            || a_y.is_null()
            || a_d.is_null()
            || omega.is_null()
            || sidx.is_null()
            || eco.is_null()
            || vn_re.is_null()
            || vn_im.is_null()
            || eco0.is_null()
            || deco0.is_null()
            || v0_re.is_null()
            || v0_im.is_null()
            || dv0_re.is_null()
            || dv0_im.is_null()
        {
            return -1;
        }
        let n_spec = s.n_spec;
        let n_modes = s.n_modes;

        let a_x_v = unsafe { std::slice::from_raw_parts(a_x, n_a) }.to_vec();
        let a_y_v = unsafe { std::slice::from_raw_parts(a_y, n_a) }.to_vec();
        let a_d_v = unsafe { std::slice::from_raw_parts(a_d, n_a) }.to_vec();

        s.zdep_modal_flength = flength;
        s.zdep_modal_a0 = a0;
        s.zdep_modal_inv_sqrt_n0 = s.modal_inv_sqrt_n.clone();
        s.zdep_modal_a_lut = Some(HermiteSpline::from_parts(a_x_v, a_y_v, a_d_v));
        s.zdep_modal_omega = unsafe { std::slice::from_raw_parts(omega, n_spec) }.to_vec();
        s.zdep_modal_sidx = unsafe { std::slice::from_raw_parts(sidx, n_spec) }
            .iter()
            .map(|&b| b != 0)
            .collect();
        s.zdep_modal_model = model;
        s.zdep_modal_loss_on = loss_on != 0;
        s.zdep_modal_eco = unsafe { std::slice::from_raw_parts(eco, n_spec * n_modes) }.to_vec();
        s.zdep_modal_vn_re =
            unsafe { std::slice::from_raw_parts(vn_re, n_spec * n_modes) }.to_vec();
        s.zdep_modal_vn_im =
            unsafe { std::slice::from_raw_parts(vn_im, n_spec * n_modes) }.to_vec();
        s.zdep_modal_omega0 = omega0;
        s.zdep_modal_ref_mode = ref_mode;
        s.zdep_modal_eco0 = unsafe { std::slice::from_raw_parts(eco0, n_modes) }.to_vec();
        s.zdep_modal_deco0 = unsafe { std::slice::from_raw_parts(deco0, n_modes) }.to_vec();
        s.zdep_modal_v0_re = unsafe { std::slice::from_raw_parts(v0_re, n_modes) }.to_vec();
        s.zdep_modal_v0_im = unsafe { std::slice::from_raw_parts(v0_im, n_modes) }.to_vec();
        s.zdep_modal_dv0_re = unsafe { std::slice::from_raw_parts(dv0_re, n_modes) }.to_vec();
        s.zdep_modal_dv0_im = unsafe { std::slice::from_raw_parts(dv0_im, n_modes) }.to_vec();
        s.zdep_modal_last_z = f64::NAN;
        s.is_zdep_modal = true;
        0
    }
    unsafe fn step(
        &mut self,
        yn: *mut Complex<f64>,
        t_old: f64,
        t_new: f64,
        dtn: f64,
        rtol: f64,
        atol: f64,
        safety: f64,
        max_dt: f64,
        min_dt: f64,
        errlast_in: f64,
        locextrap: i32,
        result: *mut NativeStepResult,
    ) -> i32 {
        let s = self;

        let n = s.n;

        // Evaluate!
        // y is in s.field. yn comes in via pointer, but we also maintain yn locally in Rust?
        // Wait, in `PreconStepper` y is `s.field`. yn is passed in.
        let yn_sl = unsafe { std::slice::from_raw_parts_mut(yn, n) };

        // s.yn .= s.y -> but wait, `s.field` is `y`.
        yn_sl.copy_from_slice(&s.field);

        // prop!(s.ks[1], s.t, s.tn)  — linop evaluated at the later time, t_new
        s.ensure_linop_at(t_new);
        s.ensure_free_norm_at(t_new);
        s.ensure_modal_linop_at(t_new);
        apply_prop_cached(
            &mut s.ks[0],
            &s.linop,
            t_new - t_old,
            &mut s.exp_cache,
            s.linop_version,
        );

        let dt = dtn;
        let t = t_new;

        for ii in 0..6 {
            // Fused single pass: previously this did one full ystage read-modify-write
            // sweep per nonzero DP_B[ii][j] coefficient (up to 6 sweeps for the last
            // stage). Collecting the (scale, k_j) terms once and accumulating each
            // element in one sweep preserves the exact same addition order (field[idx]
            // + scale0*k0[idx] + scale1*k1[idx] + ...) so results are bit-identical,
            // while touching the ystage buffer only once instead of once per term.
            let (ks_slice, _) = s.ks.split_at(ii + 1);
            let mut terms: [Option<(f64, &[Complex<f64>])>; 6] = [None; 6];
            let mut n_terms = 0usize;
            for (j, k_j) in ks_slice.iter().enumerate() {
                let bij = DP_B[ii][j];
                if bij != 0.0 {
                    terms[n_terms] = Some((dt * bij, k_j.as_slice()));
                    n_terms += 1;
                }
            }
            for idx in 0..n {
                let mut re = s.field[idx].re;
                let mut im = s.field[idx].im;
                for term in &terms[..n_terms] {
                    let (scale, k_arr) = term.unwrap();
                    re += scale * k_arr[idx].re;
                    im += scale * k_arr[idx].im;
                }
                s.ystage[idx] = Complex::new(re, im);
            }

            if s.is_free {
                let dt_prop = DP_NODES[ii] * dt;
                s.ensure_linop_at(t + dt_prop);
                s.ensure_free_norm_at(t + dt_prop);
                let mut ystage_prop = s.ystage.clone();
                apply_prop_cached(
                    &mut ystage_prop,
                    &s.linop,
                    dt_prop,
                    &mut s.exp_cache,
                    s.linop_version,
                );
                if s.is_real {
                    s.rhs_free(ii + 1, &ystage_prop);
                } else {
                    s.rhs_free_env(ii + 1, &ystage_prop);
                }
                apply_prop_cached(
                    &mut s.ks[ii + 1],
                    &s.linop,
                    -dt_prop,
                    &mut s.exp_cache,
                    s.linop_version,
                );
            } else if s.is_modal {
                let dt_prop = DP_NODES[ii] * dt;
                s.ensure_linop_at(t + dt_prop);
                s.ensure_modal_linop_at(t + dt_prop);
                let mut ystage_prop = s.ystage.clone();
                apply_prop_cached(
                    &mut ystage_prop,
                    &s.linop,
                    dt_prop,
                    &mut s.exp_cache,
                    s.linop_version,
                );
                s.rhs_modal(ii + 1, &ystage_prop);
                apply_prop_cached(
                    &mut s.ks[ii + 1],
                    &s.linop,
                    -dt_prop,
                    &mut s.exp_cache,
                    s.linop_version,
                );
            } else if s.is_radial {
                let dt_prop = DP_NODES[ii] * dt;
                s.ensure_linop_at(t + dt_prop);
                let mut ystage_prop = s.ystage.clone();
                apply_prop_cached(
                    &mut ystage_prop,
                    &s.linop,
                    dt_prop,
                    &mut s.exp_cache,
                    s.linop_version,
                );
                if s.is_real {
                    s.rhs_radial(ii + 1, &ystage_prop);
                } else {
                    s.rhs_radial_env(ii + 1, &ystage_prop);
                }
                apply_prop_cached(
                    &mut s.ks[ii + 1],
                    &s.linop,
                    -dt_prop,
                    &mut s.exp_cache,
                    s.linop_version,
                );
            } else if s.n_time_over > 0 && (s.fft_r2c_over.is_some() || s.fft_c2c_over.is_some()) {
                let dt_prop = DP_NODES[ii] * dt;
                s.ensure_linop_at(t + dt_prop);
                let mut ystage_prop = s.ystage.clone();
                apply_prop_cached(
                    &mut ystage_prop,
                    &s.linop,
                    dt_prop,
                    &mut s.exp_cache,
                    s.linop_version,
                );
                if s.is_real {
                    s.rhs_mode_avg_real(ii + 1, &ystage_prop);
                } else {
                    s.rhs_mode_avg_env(ii + 1, &ystage_prop);
                }
                apply_prop_cached(
                    &mut s.ks[ii + 1],
                    &s.linop,
                    -dt_prop,
                    &mut s.exp_cache,
                    s.linop_version,
                );
            } else {
                for k in 0..n {
                    s.ks[ii + 1][k] = Complex::new(0.0, 0.0);
                }
            }
        }

        if locextrap != 0 {
            let b0 = dt * DP_B5[0];
            let b1 = dt * DP_B5[1];
            let b2 = dt * DP_B5[2];
            let b3 = dt * DP_B5[3];
            let b4 = dt * DP_B5[4];
            let b5 = dt * DP_B5[5];
            let b6 = dt * DP_B5[6];

            let (ks_slice, _) = s.ks.split_at(7);
            let (ks0, ks1, ks2, ks3, ks4, ks5, ks6) = (
                &ks_slice[0],
                &ks_slice[1],
                &ks_slice[2],
                &ks_slice[3],
                &ks_slice[4],
                &ks_slice[5],
                &ks_slice[6],
            );

            for (((((yn, f), k0), k1), k2), (((k3, k4), k5), k6)) in yn_sl
                .iter_mut()
                .zip(&s.field)
                .zip(ks0)
                .zip(ks1)
                .zip(ks2)
                .zip(ks3.iter().zip(ks4).zip(ks5).zip(ks6))
            {
                yn.re = f.re
                    + b0 * k0.re
                    + b1 * k1.re
                    + b2 * k2.re
                    + b3 * k3.re
                    + b4 * k4.re
                    + b5 * k5.re
                    + b6 * k6.re;
                yn.im = f.im
                    + b0 * k0.im
                    + b1 * k1.im
                    + b2 * k2.im
                    + b3 * k3.im
                    + b4 * k4.im
                    + b5 * k5.im
                    + b6 * k6.im;
            }
        }

        // Error estimate
        let e0 = dt * DP_ERREST[0];
        let e1 = dt * DP_ERREST[1];
        let e2 = dt * DP_ERREST[2];
        let e3 = dt * DP_ERREST[3];
        let e4 = dt * DP_ERREST[4];
        let e5 = dt * DP_ERREST[5];
        let e6 = dt * DP_ERREST[6];

        let (ks_slice, _) = s.ks.split_at(7);
        let (ks0, ks1, ks2, ks3, ks4, ks5, ks6) = (
            &ks_slice[0],
            &ks_slice[1],
            &ks_slice[2],
            &ks_slice[3],
            &ks_slice[4],
            &ks_slice[5],
            &ks_slice[6],
        );

        for ((((y, k0), k1), k2), (((k3, k4), k5), k6)) in s
            .yerr
            .iter_mut()
            .zip(ks0)
            .zip(ks1)
            .zip(ks2)
            .zip(ks3.iter().zip(ks4).zip(ks5).zip(ks6))
        {
            y.re = e0 * k0.re
                + e1 * k1.re
                + e2 * k2.re
                + e3 * k3.re
                + e4 * k4.re
                + e5 * k5.re
                + e6 * k6.re;
            y.im = e0 * k0.im
                + e1 * k1.im
                + e2 * k2.im
                + e3 * k3.im
                + e4 * k4.im
                + e5 * k5.im
                + e6 * k6.im;
        }

        let err = weaknorm_c64(&s.yerr, &s.field, yn_sl, rtol, atol);
        let ok = err <= 1.0;

        let (dtn_new, errlast_new, ok_final) =
            stepcontrol_pi(ok, err, errlast_in, dt, safety, max_dt, min_dt);

        let tn_new;
        if ok_final {
            tn_new = t + dt;
            let (left, right) = s.ks.split_at_mut(6);
            left[0].copy_from_slice(&right[0]); // FSAL
            s.ensure_linop_at(tn_new);
            s.ensure_free_norm_at(tn_new);
            s.ensure_modal_linop_at(tn_new);
            apply_prop_cached(
                yn_sl,
                &s.linop,
                tn_new - t,
                &mut s.exp_cache,
                s.linop_version,
            );
            s.field.copy_from_slice(yn_sl);
        } else {
            yn_sl.copy_from_slice(&s.field);
            tn_new = t_new;
            s.ensure_linop_at(tn_new);
            s.ensure_free_norm_at(tn_new);
            s.ensure_modal_linop_at(tn_new);
            apply_prop_cached(
                yn_sl,
                &s.linop,
                tn_new - t,
                &mut s.exp_cache,
                s.linop_version,
            ); // no-op since t == tn_new
        }

        unsafe {
            *result = NativeStepResult {
                ok: ok_final as i32,
                dt,
                t,
                tn: tn_new,
                dtn: dtn_new,
                err,
                errlast: errlast_new,
            };
        }

        0
    }
}

/// GPU-resident stepper V1 (`CudaNativeSim`) covers only mode-averaged,
/// Kerr-only, RealGrid propagation (see `cuda_native.rs`'s `NativeBackend`
/// impl — every other geometry/physics method returns -1). Verified against
/// the Julia oracle on real CUDA hardware for that scope (docs/dev/BACKLOG.md's
/// "GPU-resident stepper" entry); everything outside it remains unimplemented,
/// not just untested. Refuses to initialize unless this env var is
/// explicitly set, so it can never be reached by accident via default
/// dispatch — Julia's own eligibility guard (`RK45._gpu_native_eligible`)
/// checks the same env var independently before ever calling this.
const CUDA_NATIVE_OPT_IN_VAR: &str = "AMALTHEA_USE_RUST_CUDA_NATIVE";

/// See `init_native_sim`'s doc — same seed-linop contract, GPU-backed.
///
/// # Safety
/// `linop` must be valid for `n` `ComplexF64` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_cuda_native_sim(linop: *const c_double, n: size_t) -> *mut NativeSim {
    use crate::cuda_native::CudaNativeSim;
    if linop.is_null() || n == 0 {
        return std::ptr::null_mut();
    }
    if std::env::var(CUDA_NATIVE_OPT_IN_VAR).as_deref() != Ok("1") {
        eprintln!(
            "Amalthea warning: GPU-resident stepper (CudaNativeSim) is experimental \
             (mode-averaged Kerr-only) — refusing to initialize. Set {}=1 to opt in (see \
             docs/dev/BACKLOG.md).",
            CUDA_NATIVE_OPT_IN_VAR
        );
        return std::ptr::null_mut();
    }
    eprintln!(
        "Amalthea warning: {}=1 — using the GPU-resident stepper (mode-averaged Kerr-only; \
         any other geometry/physics on this config is not implemented, not just unverified).",
        CUDA_NATIVE_OPT_IN_VAR
    );
    let linop_sl = unsafe { std::slice::from_raw_parts(linop as *const Complex<f64>, n) };
    match CudaNativeSim::new(n, linop_sl) {
        Ok(backend) => {
            let sim = NativeSim {
                backend: Box::new(backend),
            };
            Box::into_raw(Box::new(sim))
        }
        Err(e) => {
            eprintln!("Failed to initialize CudaNativeSim: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Allocate a `NativeSim` for a spectral grid of length `n`.
///
/// `linop` is the constant linear operator, `n` complex values laid out as
/// interleaved `(re, im)` `f64` pairs — i.e. Julia `ComplexF64`. It is copied
/// in; the caller's array need not outlive this call.
///
/// Returns null on null input, `n == 0`, or panic. Free with [`free_native_sim`].
///
/// # Safety
/// `linop` must be non-null and valid for `n` `ComplexF64` (= `2*n` `f64`) reads
/// for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_native_sim(linop: *const c_double, n: size_t) -> *mut NativeSim {
    if linop.is_null() || n == 0 {
        return std::ptr::null_mut();
    }
    // ComplexF64 is two contiguous f64 (re, im); reinterpret the f64* as Complex.
    let linop_sl = unsafe { std::slice::from_raw_parts(linop as *const Complex<f64>, n) };
    let result = std::panic::catch_unwind(|| CpuNativeSim::new(n, linop_sl));
    match result {
        Ok(h) => Box::into_raw(Box::new(NativeSim {
            backend: Box::new(h),
        })),
        Err(e) => {
            eprintln!("init_native_sim: panic: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

/// Free a `NativeSim`. Passing null is a no-op.
///
/// # Safety
/// `ptr` must be a valid, non-aliased pointer from [`init_native_sim`] that has
/// not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_native_sim(ptr: *mut NativeSim) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}

/// Copy `n` `ComplexF64` from Julia into the resident field buffer.
///
/// Returns 0 on success, -1 on null/length mismatch.
///
/// # Safety
/// `sim` must be valid; `data` must be valid for `n` `ComplexF64` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_field(sim: *mut NativeSim, data: *const c_double, n: size_t) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_field(data, n) }
}

/// Like `set_field`, but does NOT recompute the FSAL stage-0 RHS.
///
/// Used by `RK45.jl`'s post-`stepfun` resync (Phase 8): after an accepted
/// step, Julia's `stepfun` callback (`Luna.run`'s grid windowing —
/// `Eω .*= grid.ωwin`, then a time-domain `twin`) mutates the field in
/// place. Julia's own `PreconStepper` reference behaviour does *not*
/// re-evaluate the nonlinear RHS after windowing either: it keeps the
/// FSAL-carried last stage and merely re-expresses it in the new
/// interaction-picture frame via a *linear* re-propagation
/// (`evaluate!(s::PreconStepper)`'s `s.prop!(s.ks[1], s.t, s.tn)`).
/// `native_step` already performs the equivalent FSAL-carry + relinearize
/// internally (`ks[0] = ks[6]` then `apply_prop`, at the top of the next
/// call) — recomputing k0 fresh here (as `set_field` does, correctly, for
/// the brand-new-initial-condition case where no FSAL history exists yet)
/// would silently diverge from Julia's established approximation instead
/// of matching it. This function only updates the resident `field` buffer
/// that the *next* step reads its starting point from.
///
/// Returns 0 on success, -1 on null/length mismatch.
///
/// # Safety
/// `sim` must be valid; `data` must be valid for `n` `ComplexF64` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_resync_field(
    sim: *mut NativeSim,
    data: *const c_double,
    n: size_t,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.resync_field(data, n) }
}

/// Copy the resident field buffer back out to Julia (`n` `ComplexF64`).
///
/// Returns 0 on success, -1 on null/length mismatch.
///
/// # Safety
/// `sim` must be valid; `data` must be valid for `n` `ComplexF64` writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_field(sim: *const NativeSim, data: *mut c_double, n: size_t) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &*sim };
    unsafe { s.backend.get_field(data, n) }
}

/// Copy out RK stage `idx`'s `k` buffer (`n` `ComplexF64`), used by
/// `RK45.jl::interpolate` to reconstruct Julia's quartic dense output on
/// the native path (see `native_apply_prop`'s doc for the surrounding
/// interpolation scheme).
///
/// Returns 0 on success, -1 on null/length/index mismatch.
///
/// # Safety
/// `sim` must be valid; `data` must be valid for `n` `ComplexF64` writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_ks_stage(
    sim: *const NativeSim,
    idx: size_t,
    data: *mut c_double,
    n: size_t,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &*sim };
    unsafe { s.backend.get_ks_stage(idx, data, n) }
}

/// Apply the interaction-picture linear propagator `exp(linop(t2)*(t2-t1))`
/// to `y` in place (`n` `ComplexF64`), evaluating the (possibly
/// z-dependent) linop at `t2` — the later time, matching
/// `RK45.jl::make_prop!`'s non-constant-linop convention exactly.
///
/// Used by `RK45.jl::interpolate(s::RustNativeStepper, ti)` to port
/// Julia's `PreconStepper`'s quartic dense output (`interpC`, all 7 RK
/// stages) to the native path: the polynomial correction is computed
/// relative to the step's start time `s.t`, then re-expressed at the
/// query time `ti` via this propagator, exactly mirroring
/// `interpolate(s::PreconStepper, ti)`'s trailing `s.prop!(out, s.t, ti)`
/// call. Before this, the native path only had linear dense output between
/// accepted steps (see PORT_LOG Phase 8) — correct only at the endpoints.
///
/// Returns 0 on success, -1 on null/length mismatch.
///
/// # Safety
/// `sim` must be valid; `y` must be valid for `n` `ComplexF64` reads/writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_apply_prop(
    sim: *mut NativeSim,
    y: *mut c_double,
    n: size_t,
    t1: f64,
    t2: f64,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.apply_prop(y, n, t1, t2) }
}

/// Debug getter (mirrors `get_field`/`get_ks_stage`): force `ensure_linop_at(z)`
/// and copy out the resulting `self.linop`. Used only by Rust-vs-Julia
/// diagnostic scripts for Phase 7 (z-dependent linop) — not part of the hot
/// loop.
///
/// Returns 0 on success, -1 on null/length mismatch.
///
/// # Safety
/// `sim` must be valid; `data` must be valid for `n` `ComplexF64` writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_debug_linop_at(
    sim: *mut NativeSim,
    z: c_double,
    data: *mut c_double,
    n: size_t,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.debug_linop_at(z, data, n) }
}

/// Debug getter (Phase 7 diagnostics, used by `test_native_zdep_linop.jl`'s
/// unit test): force `ensure_linop_at(z)` and report `[dens, beta1]` at that
/// z, so the analytic β1(z) closed form (`docs/dev/native-port/BETA1_ANALYTIC.md`)
/// can be checked directly against a BigFloat ground truth, independent of
/// the full linop/RHS machinery.
///
/// Returns 0 on success, -1 on null/unsupported config.
///
/// # Safety
/// `sim` must be valid; `out_dens`/`out_beta1` must each be valid for one
/// `f64` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_debug_beta1_at(
    sim: *mut NativeSim,
    z: c_double,
    out_dens: *mut c_double,
    out_beta1: *mut c_double,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.debug_beta1_at(z, out_dens, out_beta1) }
}

/// Create this sim's resident FFTW plans (`n_time`-point real or complex
/// transform, oversampled to `n_time_over`), loading `libfftw3` from
/// `lib_path` and optionally pre-loading planner wisdom from `wisdom_path`.
///
/// Returns 0 on success, -1 on null/load failure.
///
/// # Safety
/// `sim` must be a valid, non-null pointer from `init_native_sim`/
/// `init_cuda_native_sim`. `lib_path` must be a valid, null-terminated C
/// string; `wisdom_path` must be either null (skip wisdom loading) or a
/// valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_fftw_plans(
    sim: *mut NativeSim,
    lib_path: *const c_char,
    n_time: size_t,
    n_time_over: size_t,
    is_real: c_int,
    flags: c_uint,
    wisdom_path: *const c_char,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe {
        s.backend
            .set_fftw_plans(lib_path, n_time, n_time_over, is_real, flags, wisdom_path)
    }
}

/// Sets the rayon thread count for later-phase parallel RHS seams —
/// docs/dev/BACKLOG.md S2 item 1. `n <= 1` is sequential (the default, bit-identical
/// to pre-S2 behavior). Wired from Julia's `Threads.nthreads()` at
/// `RustNativeStepper` construction.
///
/// # Safety
/// `sim` must be a valid, non-null pointer from `init_native_sim`/
/// `init_cuda_native_sim`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_threads(sim: *mut NativeSim, n: size_t) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_threads(n) }
}

/// Enables/disables deterministic mode — docs/dev/BACKLOG.md S5.2. `on != 0` skips
/// the QDHT BLAS-3 path (see `NativeBackend::set_deterministic`'s doc for
/// why that's the one lever this covers today). Call any time before
/// `native_set_radial_params` — the flag is also re-applied at radial-QDHT
/// construction time so call order doesn't matter.
///
/// # Safety
/// `sim` must be a valid, non-null pointer from `init_native_sim`/
/// `init_cuda_native_sim`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_deterministic(sim: *mut NativeSim, on: c_int) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_deterministic(on) }
}

/// Best-effort planner-wisdom save — docs/dev/BACKLOG.md S1 item 1. Call once, after
/// every `native_set_*_params` call for this sim has run (so every plan it
/// creates gets included). Returns 0 on success, 1 if unavailable (never a
/// hard error — see `NativeBackend::wisdom_export`'s doc).
///
/// # Safety
/// `sim` must be a valid, non-null pointer from `init_native_sim`/
/// `init_cuda_native_sim`. `path` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_export_fftw_wisdom(
    sim: *mut NativeSim,
    path: *const c_char,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.wisdom_export(path) }
}

/// Configure the mode-averaged (single spectral trace) RHS: windowing
/// (`towin`/`owin`), the frequency-window mask `sidx` (length `n_time_over`,
/// nonzero = passband), the precomputed Kerr prefactor (`pre_re`/`pre_im`),
/// the linear dispersion `beta`, and scalar Kerr/nonlinear-scale/effective-area
/// constants. Must be called after [`native_set_fftw_plans`] (the plans this
/// RHS uses must already exist).
///
/// Returns 0 on success, -1 on null/length mismatch.
///
/// # Safety
/// `sim` must be a valid, non-null pointer from `init_native_sim`/
/// `init_cuda_native_sim`. `towin`/`beta` must be valid for `n_time` `f64`
/// reads; `owin`/`sidx`/`pre_re`/`pre_im` must be valid for `n_time_over`
/// reads of their respective element type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_mode_avg_params(
    sim: *mut NativeSim,
    n_time: size_t,
    n_time_over: size_t,
    towin: *const c_double,
    owin: *const c_double,
    sidx: *const u8,
    pre_re: *const c_double,
    pre_im: *const c_double,
    beta: *const c_double,
    kerr_fac: c_double,
    nlscale: c_double,
    sqrt_aeff: c_double,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe {
        s.backend.set_mode_avg_params(
            n_time,
            n_time_over,
            towin,
            owin,
            sidx,
            pre_re,
            pre_im,
            beta,
            kerr_fac,
            nlscale,
            sqrt_aeff,
        )
    }
}

/// Wire the modified shot-noise time-domain field for the mode-averaged
/// RealGrid path (docs/dev/BACKLOG.md Phase I item 1). `noise` has length
/// `n_time_over` (the same grid `native_set_mode_avg_params` sized). Must be
/// called *after* `native_set_mode_avg_params`. Optional: if never called,
/// the RHS runs with no noise term (the pre-Phase-I behavior).
///
/// Returns 0 on success, -1 on null/empty args, -2 on a length mismatch.
///
/// # Safety
/// `sim` must be valid; `noise` must be valid for `n` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_mode_avg_noise(
    sim: *mut NativeSim,
    noise: *const c_double,
    n: size_t,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_mode_avg_noise(noise, n) }
}

/// EnvGrid (`ComplexF64`) counterpart of [`native_set_mode_avg_noise`] —
/// same injection point/formula, complex-valued noise (interleaved
/// re/im arrays, each length `n`).
///
/// # Safety
/// `sim` must be valid; `noise_re`/`noise_im` must each be valid for `n` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_mode_avg_noise_cplx(
    sim: *mut NativeSim,
    noise_re: *const c_double,
    noise_im: *const c_double,
    n: size_t,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_mode_avg_noise_cplx(noise_re, noise_im, n) }
}

/// Wire the z-dependent linop path (Phase 7 + docs/dev/BACKLOG.md Phase F item 3 —
/// mode-averaged, graded-core constant-radius `MarcatiliMode`, two-point OR
/// general multi-point piecewise pressure gradient, see MATH.md §3.5 and
/// `docs/dev/native-port/BETA1_ANALYTIC.md`). Must be called **after**
/// `native_set_mode_avg_params` (reuses `s.sidx`, sized there).
///
/// `dspl_x`/`dspl_y`/`dspl_d` are `PhysData.densityspline`'s own knots,
/// values, and first derivatives — **transferred**, not re-fit (see
/// `HermiteSpline`'s docstring for why re-fitting a *different* spline
/// through samples of `dspl` doesn't converge). `z_pts`/`p_pts` (length
/// `n_z`, `n_z>=2`) are the pressure-gradient breakpoints/pressures — `[0,L]`
/// / `[p0,p1]` for a two-point gradient, or `Capillary.MultiPointGradient`'s
/// general `Z`/`P` — needed to invert the closed-form z→pressure map at
/// runtime (`ensure_linop_at` selects the segment containing `z`).
/// `gamma`/`nwg_re`/`nwg_im`/`omega` are per-spectral-bin arrays of length
/// `sim.n` (0 outside `sidx` is fine — those bins are overwritten with 0
/// regardless, see `ensure_linop_at`).
///
/// `omega0`/`gamma0`/`dgamma0`/`nwg0_re`/`nwg0_im`/`dnwg0_re`/`dnwg0_im` are
/// the 4 z-independent constants (ω0 plus γ0, dγ0, nwg0, dnwg0) needed for
/// β1(z)'s closed form — computed once in Julia via `Maths.derivative` fed a
/// `BigFloat` argument (exact to far below `Float64` epsilon), **not** a
/// `(dens,β1)` LUT: an earlier version of this design sampled β1 into a
/// spline, which could only ever chase Julia's own adaptive-FD noise floor
/// (`Modes.dispersion`'s `central_fdm`) rather than the true derivative —
/// see BETA1_ANALYTIC.md for the derivation and postmortem.
///
/// `eps0_gamma3` is `PhysData.ε_0·γ3` (density factored out) —
/// `ensure_linop_at` rescales `sim.kerr_fac` by the freshly computed
/// `dens(z)` every call, and also overwrites `sim.beta[i]` with
/// `ω/c·real(neff(ω,z))` (Modes.jl's unclamped `β(ω,z)`, reusing the `neff`
/// already computed for the linop) — both are genuinely z-dependent for a
/// pressure gradient (`TransModeAvg` re-evaluates `densityfun(z)` and
/// `norm_mode_average`'s `βfun!(β,z)` fresh every RK stage in Julia) and
/// were the actual source of a ~9% full-solve mismatch before this was
/// wired up (see PORT_LOG's Phase 7 postmortem).
///
/// Returns 0 on success, -1 on null/empty args, -2 if `dspl` has fewer than
/// 2 samples, -3 if called before `native_set_mode_avg_params` sized
/// `sim.sidx`, -4 if `n_z < 2`.
///
/// # Safety
/// All pointer arguments must be non-null and valid for the lengths given.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_zdep_mode_avg_params(
    sim: *mut NativeSim,
    n_z: size_t,
    z_pts: *const c_double,
    p_pts: *const c_double,
    n_dspl: size_t,
    dspl_x: *const c_double,
    dspl_y: *const c_double,
    dspl_d: *const c_double,
    gamma: *const c_double,
    nwg_re: *const c_double,
    nwg_im: *const c_double,
    omega: *const c_double,
    model: c_uint,
    loss_on: c_uint,
    eps0_gamma3: c_double,
    omega0: c_double,
    gamma0: c_double,
    dgamma0: c_double,
    nwg0_re: c_double,
    nwg0_im: c_double,
    dnwg0_re: c_double,
    dnwg0_im: c_double,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe {
        s.backend.set_zdep_mode_avg_params(
            n_z,
            z_pts,
            p_pts,
            n_dspl,
            dspl_x,
            dspl_y,
            dspl_d,
            gamma,
            nwg_re,
            nwg_im,
            omega,
            model,
            loss_on,
            eps0_gamma3,
            omega0,
            gamma0,
            dgamma0,
            nwg0_re,
            nwg0_im,
            dnwg0_re,
            dnwg0_im,
        )
    }
}

/// Wire the PPT ionization handle for the plasma cumtrapz path (RealGrid only).
///
/// `ion_ptr` is a raw pointer to a `PptIonizationRate` already constructed by Julia
/// (via `init_ppt_ionization_lut`). Julia owns it; this call just borrows it for the
/// lifetime of the `NativeSim`. Must be called **after** `native_set_mode_avg_params`
/// (mode-averaged) or `native_set_radial_params` (radial, Phase D.2 — docs/dev/BACKLOG.md)
/// so `s.n_time_over`/`s.n_r` are already sized; the `plas_*` scratch buffers are
/// sized `n_time_over` for mode-averaged or `n_time_over*n_r` for radial based on
/// `s.is_radial`.
///
/// Returns 0 on success, -1 on null args, -2 if called before mode-avg params are set.
///
/// # Safety
/// `sim` must be valid; `ion_ptr` must be a non-null, valid `PptIonizationRate` pointer
/// that remains valid for the lifetime of the `NativeSim`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_plasma_params(
    sim: *mut NativeSim,
    ion_ptr: *const crate::ionization::PptIonizationRate,
    ionpot: c_double,
    e_ratio: c_double,
    preionfrac: c_double,
    dt: c_double,
    density: c_double,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe {
        s.backend
            .set_plasma_params(ion_ptr, ionpot, e_ratio, preionfrac, dt, density)
    }
}

/// ADK counterpart of [`native_set_plasma_params`] (docs/dev/BACKLOG.md Phase I
/// item 3) — `ion_ptr` is an `AdkIonizationRate` handle from
/// [`crate::ffi::init_adk_ionization`] instead of a PPT spline LUT.
///
/// # Safety
/// `sim` and `ion_ptr` must be valid; `ion_ptr` must remain valid for the
/// lifetime of the `NativeSim`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_plasma_params_adk(
    sim: *mut NativeSim,
    ion_ptr: *const crate::ionization::AdkIonizationRate,
    ionpot: c_double,
    e_ratio: c_double,
    preionfrac: c_double,
    dt: c_double,
    density: c_double,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe {
        s.backend
            .set_plasma_params_adk(ion_ptr, ionpot, e_ratio, preionfrac, dt, density)
    }
}

/// Wire the radial (`TransRadial`) RHS: resident QDHT + scalar Kerr.
///
/// Must be called **after** `native_set_fftw_plans`, which determines
/// `sim.is_real` and therefore which RHS variant (`rhs_radial` for RealGrid,
/// `rhs_radial_env` for EnvGrid — Phase D.1, MATH.md §3.2 follow-up) this
/// call wires the buffers for. Sets `sim.is_radial`, so this determines the
/// RHS branch `native_step` takes.
///
/// # Arguments
/// * `n_time`/`n_time_over` — grid.t / grid.to lengths.
/// * `n_r` — `Hankel.QDHT.N`; `sim.n` must be `n_spec * n_r` for some integer `n_spec`.
/// * `t_matrix` — Julia's `HT.T`, column-major `n_r × n_r` (read-only for this call).
/// * `scale_fwd`/`scale_inv` — `HT.scaleRK` / `1/HT.scaleRK`.
/// * `towin` — length `n_time_over`, or null for all-ones.
/// * `m_re`/`m_im` — precomputed normalization array (see MATH.md §3.2),
///   column-major `(n_spec, n_r)`, length `n_spec * n_r` each.
///
/// Returns 0 on success, -1 on null/shape-mismatched args.
///
/// # Safety
/// `sim` must be valid; `t_matrix` must be valid for `n_r*n_r` reads; `m_re`/`m_im`
/// must each be valid for `n_spec*n_r` reads (`n_spec = sim.n / n_r`); `towin`, if
/// non-null, must be valid for `n_time_over` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_radial_params(
    sim: *mut NativeSim,
    n_time: size_t,
    n_time_over: size_t,
    n_r: size_t,
    t_matrix: *const c_double,
    scale_fwd: c_double,
    scale_inv: c_double,
    towin: *const c_double,
    kerr_fac: c_double,
    m_re: *const c_double,
    m_im: *const c_double,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe {
        s.backend.set_radial_params(
            n_time,
            n_time_over,
            n_r,
            t_matrix,
            scale_fwd,
            scale_inv,
            towin,
            kerr_fac,
            m_re,
            m_im,
        )
    }
}

/// Wire the modified shot-noise time-domain field for the radial
/// (`TransRadial`) RealGrid path (docs/dev/BACKLOG.md Phase I item 1). `noise` has
/// length `n_time_over*n_r`, column-major `(n_time_over, n_r)` — the same
/// layout as `radial_eto`. Must be called *after* `native_set_radial_params`.
///
/// Returns 0 on success, -1 on null/empty args, -2 on a length mismatch.
///
/// # Safety
/// `sim` must be valid; `noise` must be valid for `n` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_radial_noise(
    sim: *mut NativeSim,
    noise: *const c_double,
    n: size_t,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_radial_noise(noise, n) }
}

/// EnvGrid (`ComplexF64`) counterpart of [`native_set_radial_noise`] —
/// same injection point/formula, complex-valued noise (interleaved re/im
/// arrays, each length `n`, column-major `(n_time_over, n_r)`).
///
/// # Safety
/// `sim` must be valid; `noise_re`/`noise_im` must each be valid for `n` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_radial_noise_cplx(
    sim: *mut NativeSim,
    noise_re: *const c_double,
    noise_im: *const c_double,
    n: size_t,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_radial_noise_cplx(noise_re, noise_im, n) }
}

/// Wire the Raman ADE solver as an additive term in `rhs_mode_avg_real`
/// or, since Phase D.4 (docs/dev/BACKLOG.md), `rhs_radial` (RealGrid, `thg=true`
/// only — see MATH.md §5.3).
///
/// Must be called **after** `native_set_mode_avg_params` or
/// `native_set_radial_params` (needs `n_time_over`/`n_r`/`is_radial` to size
/// the scratch buffers), before `set_field`.
///
/// # Arguments
/// * `omega`/`gamma`/`coupling` — per-oscillator SDO parameters (`Ω`,
///   `1/τ2ρ(1.0)`, `K`), same values `Interface._make_rust_raman_handle_from_response`
///   already extracts for the existing `AMALTHEA_USE_RUST_RAMAN` FFI wiring.
/// * `n_osc` — number of oscillators.
/// * `dt` — `grid.to[2] - grid.to[1]` (oversampled grid spacing).
/// * `density` — constant-medium density (raw, **not** folded with `ε₀·γ3`
///   like `kerr_fac` — Raman's accumulation formula needs it unscaled).
/// * `thg` — nonzero: intensity is `E²` (`sqr!`'s `thg=true` branch, the
///   original Phase D.4 scope). Zero (Phase F.1): intensity is
///   `1/2·|hilbert(E)|²` (`sqr!`'s `thg=false` branch) — builds a resident
///   c2c Hilbert-transform plan sized `n_time_over` (RealGrid has none
///   otherwise).
///
/// Returns 0 on success, -1 on null/zero-length args, -2 if called before
/// `native_set_mode_avg_params`.
///
/// # Safety
/// `sim` must be valid; `omega`/`gamma`/`coupling` must each be valid for
/// `n_osc` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_raman_params(
    sim: *mut NativeSim,
    omega: *const c_double,
    gamma: *const c_double,
    coupling: *const c_double,
    n_osc: size_t,
    dt: c_double,
    density: c_double,
    thg: c_int,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe {
        s.backend
            .set_raman_params(omega, gamma, coupling, n_osc, dt, density, thg)
    }
}

/// Wire the intermediate-broadening (`:SiO2`) resident FFT-convolution Raman
/// kernel as an additive term in `rhs_mode_avg_env` (Phase I item 2,
/// docs/dev/BACKLOG.md) — `RamanPolarEnv`/`RamanRespIntermediateBroadening` only
/// (`prop_gnlse(...; ramanmodel=:SiO2)`'s only reachable path; no finite
/// single-damped-oscillator decomposition exists for this response, so it
/// cannot reuse `native_set_raman_params`'s ADE solver).
///
/// Must be called **after** `native_set_fftw_plans` and
/// `native_set_mode_avg_params` (needs `fftw_api`/`n_time_over`), before
/// `set_field`.
///
/// # Arguments
/// * `omega`/`amp`/`gauss_w`/`lorentz_w` — per-component `ωi`/`Ai`/`Γi`/`γi`
///   (`Raman.jl:78-84`).
/// * `n_osc` — number of components.
/// * `scale` — `RamanRespIntermediateBroadening.scale`, already normalised
///   by Julia's `hquadrature` call at construction time (not recomputed
///   here).
/// * `dt` — `grid.to[2] - grid.to[1]` (oversampled grid spacing).
/// * `n_time` — oversampled grid length (`length(rr.t)`); must equal the
///   sim's own `n_time_over`.
/// * `density` — constant-medium density (raw, unscaled, like
///   `native_set_raman_params`'s `density`).
///
/// Returns 0 on success, -1 on null/zero-length args, -2 if called before
/// `native_set_fftw_plans`/`native_set_mode_avg_params` or `n_time` mismatches.
///
/// # Safety
/// `sim` must be valid; `omega`/`amp`/`gauss_w`/`lorentz_w` must each be
/// valid for `n_osc` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_raman_fft_params(
    sim: *mut NativeSim,
    omega: *const c_double,
    amp: *const c_double,
    gauss_w: *const c_double,
    lorentz_w: *const c_double,
    n_osc: size_t,
    scale: c_double,
    dt: c_double,
    n_time: size_t,
    density: c_double,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe {
        s.backend.set_raman_fft_params(
            omega, amp, gauss_w, lorentz_w, n_osc, scale, dt, n_time, density,
        )
    }
}

/// Wire the modal (`TransModal`) RHS: resident `libcubature` (`pcubature_v`,
/// `full=false` only) + analytic general-order Marcatili mode-field
/// synthesis (`:HE`/`:TE`/`:TM`, any mode order `n` — docs/dev/BACKLOG.md Phase E.1;
/// was `HE, n=1`-only through Phase 5/D.4). See MATH.md §3.3 for the design
/// and remaining scope restrictions (npol=1, `full=false`, constant radius).
///
/// Must be called **after** `native_set_fftw_plans` (`is_real=1` — RealGrid
/// only) and `native_set_mode_avg_params`-style buffer sizing is done here
/// directly (modal does not reuse `native_set_mode_avg_params`).
///
/// # Arguments
/// * `n_time`/`n_time_over` — `grid.t`/`grid.to` lengths.
/// * `n_modes` — number of modes; `sim.n` must equal `n_spec * n_modes`.
/// * `npol` — 1 or 2 (`Modes.ToSpace.npol`).
/// * `a` — shared physical core radius (constant across modes and z —
///   scope restriction).
/// * `unm`/`inv_sqrt_n` — per-mode arrays, length `n_modes` (`unm`, `1/√N(m)`
///   precomputed closed-form in Julia).
/// * `order` — per-mode Bessel order for `J_order(r·unm/a)`: `n-1` for
///   `:HE`, `1` for `:TE`/`:TM`. Length `n_modes`.
/// * `kind` — per-mode kind: 0=`:HE`, 1=`:TE`, 2=`:TM`.
/// * `phi` — per-mode `ϕ` (only meaningful for `:HE`). Length `n_modes` each.
/// * `full` — 0/1: `full=false` (radial-only, `pcubature_v`, θ fixed at 0)
///   or `full=true` (genuine 2-D `(r,θ)`, `hcubature_v` — Phase E.3).
/// * `pol_select` — length `npol`, 0 → x column, 1 → y column (see
///   `mode_angle_xy`).
/// * `towin` — length `n_time_over`, or null for all-ones.
/// * `kerr_fac` — `density · ε₀ · γ3` (same convention as every other phase).
/// * `nlfac_re`/`nlfac_im` — precomputed `ωwin[iω]·normfunc(ω[iω])`, length `n_spec` each.
/// * `lib_path` — path to the `libcubature` shared library
///   (`Cubature.Cubature_jll.libcubature` from Julia).
/// * `rtol`/`atol`/`maxevals` — passed straight through to `pcubature_v`/`hcubature_v`.
///
/// Returns 0 on success, -1 on null/shape-mismatched args, -2 if `libcubature`
/// failed to load.
///
/// # Safety
/// `sim` must be valid; `unm`/`inv_sqrt_n`/`order`/`kind`/`phi` must
/// each be valid for `n_modes` reads; `pol_select` for `npol` reads;
/// `nlfac_re`/`nlfac_im` for `n_spec` reads (`n_spec = sim.n / n_modes`);
/// `towin`, if non-null, for `n_time_over` reads; `lib_path` must be a valid
/// null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_modal_params(
    sim: *mut NativeSim,
    n_time: size_t,
    n_time_over: size_t,
    n_modes: size_t,
    npol: size_t,
    a: c_double,
    unm: *const c_double,
    inv_sqrt_n: *const c_double,
    order: *const i32,
    kind: *const u8,
    phi: *const c_double,
    full: u8,
    pol_select: *const u8,
    towin: *const c_double,
    kerr_fac: c_double,
    nlfac_re: *const c_double,
    nlfac_im: *const c_double,
    lib_path: *const c_char,
    rtol: c_double,
    atol: c_double,
    maxevals: size_t,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe {
        s.backend.set_modal_params(
            n_time,
            n_time_over,
            n_modes,
            npol,
            a,
            unm,
            inv_sqrt_n,
            order,
            kind,
            phi,
            full,
            pol_select,
            towin,
            kerr_fac,
            nlfac_re,
            nlfac_im,
            lib_path,
            rtol,
            atol,
            maxevals,
        )
    }
}

/// Wire the free-space (`TransFree`) RHS: resident joint 3-D FFTW plan +
/// scalar Kerr. See MATH.md §3.4 for the design (dimension order,
/// `1/(n_t·n_y·n_x)` normalization, no per-column spatial step).
///
/// Must be called **after** `native_set_fftw_plans` (`is_real=1` — Phase 6
/// gate is RealGrid-only; reuses the already-loaded FFTW library handle
/// rather than loading it again).
///
/// # Arguments
/// * `n_time`/`n_time_over` — `grid.t`/`grid.to` lengths.
/// * `n_y`/`n_x` — transverse grid sizes (`length(xygrid.y)`/`length(xygrid.x)`);
///   `sim.n` must equal `n_spec * n_y * n_x`.
/// * `flags` — FFTW planner flag, must match the one used for the 1-D plans
///   (planning-algorithm/summation-order parity, see `fftw.rs` header).
/// * `towin` — length `n_time_over`, or null for all-ones.
/// * `kerr_fac` — `density · ε₀ · γ3`.
/// * `m_re`/`m_im` — precomputed `ωwin[iω]·(-i·ω[iω])/(2·normfun(0.0)[iω,iky,ikx])`,
///   flattened column-major `(n_spec, n_y, n_x)`, length `n_spec*n_y*n_x` each.
///
/// Returns 0 on success, -1 on null/shape-mismatched args, -2 if
/// `native_set_fftw_plans` was not called first.
///
/// # Safety
/// `sim` must be valid; `m_re`/`m_im` must each be valid for `n_spec*n_y*n_x`
/// reads (`n_spec = sim.n / (n_y*n_x)`); `towin`, if non-null, valid for
/// `n_time_over` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_free_params(
    sim: *mut NativeSim,
    n_time: size_t,
    n_time_over: size_t,
    n_y: size_t,
    n_x: size_t,
    flags: c_uint,
    towin: *const c_double,
    kerr_fac: c_double,
    m_re: *const c_double,
    m_im: *const c_double,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe {
        s.backend.set_free_params(
            n_time,
            n_time_over,
            n_y,
            n_x,
            flags,
            towin,
            kerr_fac,
            m_re,
            m_im,
        )
    }
}

/// Wire the z-dependent linop + normfun for free-space (`TransFree`), a
/// two-point pressure-gradient gas cell — Phase D.5 (docs/dev/BACKLOG.md). Mirrors
/// `native_set_zdep_mode_avg_params` (see that function's doc for the
/// `dspl`-transfer rationale) but for `LinearOps.ZDepLinopFree`/
/// `NonlinearRHS.ZDepNormFree`'s free-space metadata: no waveguide term,
/// and `gamma`/`omega`/`omegawin` are `n_spec`-length (not `s.n`-length —
/// free-space's field spans `n_spec*n_y*n_x`, but the z-dependence only
/// varies per spectral bin, reused identically across every spatial column).
///
/// Must be called **after** `native_set_free_params` (needs `n_spec`/`n_y`/
/// `n_x` to size `kperp2` and validate array lengths), before `set_field`.
///
/// Returns 0 on success, -1 on null/mismatched args, -2 if `dspl` has fewer
/// than 2 knots, -3 if called before `native_set_free_params`.
///
/// # Safety
/// `sim` must be valid; every pointer argument must be valid for its stated length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_free_zdep_params(
    sim: *mut NativeSim,
    flength: c_double,
    p0: c_double,
    p1: c_double,
    n_dspl: size_t,
    dspl_x: *const c_double,
    dspl_y: *const c_double,
    dspl_d: *const c_double,
    gamma: *const c_double,
    omega: *const c_double,
    omegawin: *const c_double,
    kperp2: *const c_double,
    sidx: *const u8,
    eps0_gamma3: c_double,
    omega0: c_double,
    gamma0: c_double,
    dgamma0: c_double,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe {
        s.backend.set_free_zdep_params(
            flength,
            p0,
            p1,
            n_dspl,
            dspl_x,
            dspl_y,
            dspl_d,
            gamma,
            omega,
            omegawin,
            kperp2,
            sidx,
            eps0_gamma3,
            omega0,
            gamma0,
            dgamma0,
        )
    }
}

/// Wire the z-dependent modal linop — tapered/per-mode radius, docs/dev/BACKLOG.md
/// Phase E.2. Must be called **after** `native_set_modal_params` (needs
/// `n_spec`/`n_modes` to validate array lengths; reuses `modal_unm` for the
/// per-mode Bessel eigenvalue, already set there).
///
/// # Arguments
/// * `flength` — fibre length (`grid.zmax`), for clamping `a(z)` queries.
/// * `a0` — `a(0)`, for rescaling `modal_inv_sqrt_n` (`∝ 1/a(z)`, see
///   `zdep_modal_a0`'s doc).
/// * `n_a`/`a_x`/`a_y`/`a_d` — `Maths.CSpline(zgrid, a.(zgrid))`'s own
///   `(x,y,D)` knots (dense-sampled ground truth, not a re-fit of an
///   existing spline — see `zdep_modal_a_lut`'s doc).
/// * `omega`/`sidx` — `grid.ω`/`grid.sidx`, length `n_spec`.
/// * `model` — 0 → `:full`, 1 → `:reduced`. `loss_on` — 0/1.
/// * `eco`/`vn_re`/`vn_im` — per-(ω,mode) arrays, flattened column-major
///   `(n_spec, n_modes)`, length `n_spec*n_modes` each.
/// * `omega0`/`ref_mode` — β1(z) evaluation frequency and reference mode
///   index (0-based).
/// * `eco0`/`deco0`/`v0_re`/`v0_im`/`dv0_re`/`dv0_im` — per-mode BigFloat-
///   differentiated ω0 constants, length `n_modes` each.
///
/// Returns 0 on success, -1 on null/mismatched args, -2 if `n_a` has fewer
/// than 2 knots, -3 if called before `native_set_modal_params`.
///
/// # Safety
/// `sim` must be valid; every pointer argument must be valid for its stated length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_modal_zdep_params(
    sim: *mut NativeSim,
    flength: c_double,
    a0: c_double,
    n_a: size_t,
    a_x: *const c_double,
    a_y: *const c_double,
    a_d: *const c_double,
    omega: *const c_double,
    sidx: *const u8,
    model: u8,
    loss_on: u8,
    eco: *const c_double,
    vn_re: *const c_double,
    vn_im: *const c_double,
    omega0: c_double,
    ref_mode: size_t,
    eco0: *const c_double,
    deco0: *const c_double,
    v0_re: *const c_double,
    v0_im: *const c_double,
    dv0_re: *const c_double,
    dv0_im: *const c_double,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe {
        s.backend.set_modal_zdep_params(
            flength, a0, n_a, a_x, a_y, a_d, omega, sidx, model, loss_on, eco, vn_re, vn_im,
            omega0, ref_mode, eco0, deco0, v0_re, v0_im, dv0_re, dv0_im,
        )
    }
}

// ── Dormand-Prince constants (matching ffi.rs) ──────────────────────────────
pub const DP_B: [[f64; 6]; 6] = [
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
        0.0,
    ],
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];
pub const DP_B5: [f64; 7] = [
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
];
pub const DP_ERREST: [f64; 7] = [
    -71.0 / 57600.0,
    0.0,
    71.0 / 16695.0,
    -71.0 / 1920.0,
    17253.0 / 339200.0,
    -22.0 / 525.0,
    1.0 / 40.0,
];
pub const DP_NODES: [f64; 6] = [1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

#[repr(C)]
pub struct NativeStepResult {
    pub ok: i32,
    pub dt: f64,
    pub t: f64,
    pub tn: f64,
    pub dtn: f64,
    pub err: f64,
    pub errlast: f64,
}

/// Small most-recently-used cache of `exp(linop[i]*dt)` arrays — docs/dev/BACKLOG.md
/// S1 item 3. Keyed on the exact bit pattern of `dt` plus a `linop_version`
/// (bumped by `ensure_linop_at`/`ensure_free_norm_at`/`ensure_modal_linop_at`
/// whenever they actually rewrite `linop` in place), so a stale entry from
/// before a z-dependent linop update simply never matches again rather than
/// being read incorrectly — no need to eagerly evict on a version bump.
/// Capacity is small: one `step()` call needs at most 7 distinct `|dt|`
/// magnitudes (the fixed DP45 `DP_NODES` fractions), used both forward and
/// backward (the backward call passes `-dt`, a different key, so a full step
/// can populate up to 14 entries before repeats start) — `CAPACITY` covers
/// one step's worth without unbounded growth across a long propagation with
/// many distinct step sizes.
pub struct ExpCache {
    // (dt.to_bits(), linop_version, exp(linop*dt)), most-recently-used first.
    entries: Vec<(u64, u64, Vec<Complex<f64>>)>,
}

const EXP_CACHE_CAPACITY: usize = 16;

impl ExpCache {
    fn new() -> Self {
        ExpCache {
            entries: Vec::new(),
        }
    }

    /// Returns `exp(linop[i]*dt)`, computing and caching it on a miss.
    fn get(&mut self, dt: f64, linop: &[Complex<f64>], linop_version: u64) -> &[Complex<f64>] {
        let key = dt.to_bits();
        if let Some(pos) = self
            .entries
            .iter()
            .position(|(k, v, arr)| *k == key && *v == linop_version && arr.len() == linop.len())
        {
            if pos != 0 {
                let entry = self.entries.remove(pos);
                self.entries.insert(0, entry);
            }
            return &self.entries[0].2;
        }
        let arr: Vec<Complex<f64>> = linop.iter().map(|&l| (l * dt).exp()).collect();
        self.entries.insert(0, (key, linop_version, arr));
        if self.entries.len() > EXP_CACHE_CAPACITY {
            self.entries.pop();
        }
        &self.entries[0].2
    }
}

/// Cached counterpart of `apply_prop` — looks up (or computes and caches)
/// `exp(linop[i]*dt)` in `cache` instead of recomputing the transcendental
/// `exp` on every call. Numerically identical to `apply_prop` (same
/// `Complex::exp` values, just memoized) — this changes performance only,
/// not results, for any `dt`/`linop_version` pair that repeats.
fn apply_prop_cached(
    y: &mut [Complex<f64>],
    linop: &[Complex<f64>],
    dt: f64,
    cache: &mut ExpCache,
    linop_version: u64,
) {
    let factors = cache.get(dt, linop, linop_version);
    for i in 0..y.len() {
        y[i] *= factors[i];
    }
}

fn weaknorm_c64(
    y_err: &[Complex<f64>],
    y: &[Complex<f64>],
    yn: &[Complex<f64>],
    rtol: f64,
    atol: f64,
) -> f64 {
    let (mut sy, mut syn, mut syerr) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..y_err.len() {
        sy += y[i].norm_sqr();
        syn += yn[i].norm_sqr();
        syerr += y_err[i].norm_sqr();
    }
    let errwt = f64::max(f64::max(sy.sqrt(), syn.sqrt()), atol);
    syerr.sqrt() / rtol / errwt
}

pub(crate) fn stepcontrol_pi(
    ok: bool,
    err: f64,
    errlast: f64,
    dt: f64,
    safety: f64,
    max_dt: f64,
    min_dt: f64,
) -> (f64, f64, bool) {
    const BETA1: f64 = 3.0 / 5.0 / 5.0;
    const BETA2: f64 = -1.0 / 5.0 / 5.0;
    const EPS: f64 = 0.8;
    let mut dtn;
    let errlast_new;
    if ok {
        let el = if errlast == 0.0 { err } else { errlast };
        let fac = if err == 0.0 {
            1.5
        } else {
            safety * (EPS / err).powf(BETA1) * (EPS / el).powf(BETA2)
        };
        dtn = fac * dt;
        errlast_new = err;
    } else {
        dtn = if !err.is_finite() {
            dt / 2.0
        } else {
            dt * f64::max(0.1, safety * err.powf(-1.0 / 5.0))
        };
        errlast_new = errlast;
    }
    let mut ok_final = ok;
    if dtn > max_dt {
        dtn = max_dt;
    } else if dtn < min_dt {
        dtn = min_dt;
        ok_final = true;
    }
    (dtn, errlast_new, ok_final)
}

/// Advance the resident RK45 stepper one adaptive step, from `t_old` toward
/// `t_new` with proposed step `dtn`. `yn` holds the current field on entry
/// and the accepted/rejected result on return (`n` `ComplexF64`, `n`
/// implied by the sim's grid length set at construction); `result` receives
/// the step outcome (accepted?, actual `dt`/`t`/`tn`, error estimate, PI
/// controller state). This is the per-step FFI boundary the entire
/// resident stepper hot loop crosses — see `RK45.jl`'s `solve_precon`.
///
/// Returns 0 on success, -1 on null/length mismatch, -2 on internal panic.
///
/// # Safety
/// `sim` must be a valid, non-null pointer from `init_native_sim`/
/// `init_cuda_native_sim`. `yn` must be valid for `n` `ComplexF64`
/// reads/writes. `result` must be valid for one `NativeStepResult` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_step(
    sim: *mut NativeSim,
    yn: *mut Complex<f64>,
    t_old: f64,
    t_new: f64,
    dtn: f64,
    rtol: f64,
    atol: f64,
    safety: f64,
    max_dt: f64,
    min_dt: f64,
    errlast_in: f64,
    locextrap: i32,
    result: *mut NativeStepResult,
) -> i32 {
    if sim.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
    unsafe {
        s.backend.step(
            yn, t_old, t_new, dtn, rtol, atol, safety, max_dt, min_dt, errlast_in, locextrap,
            result,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn field_roundtrip_is_bit_exact() {
        let n = 8;
        let linop: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(i as f64, -(i as f64)))
            .collect();
        let sim = unsafe { init_native_sim(linop.as_ptr() as *const c_double, n) };
        assert!(!sim.is_null());

        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(1.5 * i as f64, 0.25 * i as f64))
            .collect();
        let rc = unsafe { set_field(sim, input.as_ptr() as *const c_double, n) };
        assert_eq!(rc, 0);

        let mut out = vec![Complex::new(0.0, 0.0); n];
        let rc = unsafe { get_field(sim, out.as_mut_ptr() as *mut c_double, n) };
        assert_eq!(rc, 0);

        // Bit-exact round-trip (Phase 0 gate, Rust side).
        assert_eq!(out, input);

        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn rejects_length_mismatch() {
        let n = 4;
        let linop = vec![Complex::new(0.0, 0.0); n];
        let sim = unsafe { init_native_sim(linop.as_ptr() as *const c_double, n) };
        let data = vec![Complex::new(0.0, 0.0); n + 1];
        let rc = unsafe { set_field(sim, data.as_ptr() as *const c_double, n + 1) };
        assert_eq!(rc, -1);
        unsafe { free_native_sim(sim) };
    }

    // ── FFI-surface robustness (docs/dev/BACKLOG.md Phase J item 7) ─────────
    //
    // The ~30 `native_set_*`/lifecycle FFI entry points were each hand-checked
    // for null/zero-length/bad-call-order behavior when written, but never
    // systematically exercised together. The tests below assert "no crash, no
    // UB" for every entry point under adversarial input.
    //
    // A first pass of this suite found that several setters dereferenced a
    // non-`sim` pointer argument unconditionally once their own shape/order
    // guard passed, with no null check on that argument specifically:
    // `native_set_radial_params`'s `t_matrix`/`m_re`/`m_im`,
    // `native_set_modal_params`'s `unm`/`inv_sqrt_n`/`order`/`kind`/`phi`/
    // `pol_select`/`nlfac_re`/`nlfac_im`/`lib_path`, `native_set_free_params`'s
    // `m_re`/`m_im`, the three `*_zdep_params` setters' spline/array pointers,
    // `native_set_raman_params`'s `omega`/`gamma`/`coupling`,
    // `native_set_fftw_plans`'s `lib_path`, and
    // `set_field`/`native_resync_field`/`get_field`/`get_ks_stage`/
    // `native_apply_prop`/`native_debug_linop_at`/`native_debug_beta1_at`'s
    // data/`y`/`out_*` buffers. None of these was reachable through any
    // documented Julia call site (Julia always passes real, GC-rooted,
    // correctly-sized buffers), but each was still a latent UB risk. Every
    // one of those pointers now has an explicit `is_null()` guard (returning
    // `-1`, matching every sibling setter's existing convention) placed
    // *before* its function's first `unsafe` dereference of that pointer, and
    // the tests below exercise those exact null-pointer combinations directly
    // instead of working around them.
    //
    // Deliberately still NOT exercised: `native_set_plasma_params(_adk)`'s
    // `ion_ptr` — that setter stores the pointer without ever dereferencing
    // it, so passing null there is not UB in the setter itself (see
    // `plasma_params_stores_null_handle_without_crashing_the_setter_itself`
    // below); the crash risk is downstream, in `native_step`, which Julia's
    // own caller (`src/RK45.jl`) already guards against by never passing a
    // null Rust handle in the first place.

    fn tiny_sim() -> *mut NativeSim {
        let linop = vec![Complex::new(0.0, 0.0); 4];
        unsafe { init_native_sim(linop.as_ptr() as *const c_double, 4) }
    }

    #[test]
    fn init_native_sim_rejects_null_linop_and_zero_n() {
        assert!(unsafe { init_native_sim(std::ptr::null(), 4) }.is_null());
        let linop = vec![Complex::new(0.0, 0.0); 4];
        assert!(unsafe { init_native_sim(linop.as_ptr() as *const c_double, 0) }.is_null());
    }

    #[test]
    fn free_native_sim_null_is_a_no_op() {
        unsafe { free_native_sim(std::ptr::null_mut()) };
    }

    #[test]
    fn null_sim_rejected_by_every_lifecycle_and_setter_entry_point() {
        // Every one of these extern "C" wrappers checks `sim.is_null()`
        // before touching any other argument, so it is safe to pass null/
        // zero for everything else too — the point of this test is that the
        // *whole surface* shares this invariant, not just a hand-picked few.
        let buf = vec![0.0f64; 64];
        let buf_u8 = vec![0u8; 64];
        let buf_i32 = vec![0i32; 64];
        let lib_path = std::ffi::CString::new("libfftw3.so").unwrap();
        let mut out = vec![0.0f64; 64];

        unsafe {
            assert_eq!(set_field(std::ptr::null_mut(), buf.as_ptr(), 4), -1);
            assert_eq!(native_resync_field(std::ptr::null_mut(), buf.as_ptr(), 4), -1);
            assert_eq!(get_field(std::ptr::null(), out.as_mut_ptr(), 4), -1);
            assert_eq!(get_ks_stage(std::ptr::null(), 0, out.as_mut_ptr(), 4), -1);
            assert_eq!(
                native_apply_prop(std::ptr::null_mut(), out.as_mut_ptr(), 4, 0.0, 1.0),
                -1
            );
            assert_eq!(
                native_debug_linop_at(std::ptr::null_mut(), 0.0, out.as_mut_ptr(), 4),
                -1
            );
            assert_eq!(
                native_debug_beta1_at(
                    std::ptr::null_mut(),
                    0.0,
                    out.as_mut_ptr(),
                    out.as_mut_ptr()
                ),
                -1
            );
            assert_eq!(
                native_set_fftw_plans(
                    std::ptr::null_mut(),
                    lib_path.as_ptr(),
                    8,
                    8,
                    1,
                    1 << 6,
                    std::ptr::null()
                ),
                -1
            );
            assert_eq!(native_set_threads(std::ptr::null_mut(), 4), -1);
            assert_eq!(native_set_deterministic(std::ptr::null_mut(), 1), -1);
            assert_eq!(
                native_export_fftw_wisdom(std::ptr::null_mut(), lib_path.as_ptr()),
                -1
            );
            assert_eq!(
                native_set_mode_avg_params(
                    std::ptr::null_mut(),
                    8,
                    8,
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf_u8.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    0.0,
                    0.0,
                    1.0
                ),
                -1
            );
            assert_eq!(
                native_set_mode_avg_noise(std::ptr::null_mut(), buf.as_ptr(), 4),
                -1
            );
            assert_eq!(
                native_set_mode_avg_noise_cplx(std::ptr::null_mut(), buf.as_ptr(), buf.as_ptr(), 4),
                -1
            );
            assert_eq!(
                native_set_zdep_mode_avg_params(
                    std::ptr::null_mut(),
                    2,
                    buf.as_ptr(),
                    buf.as_ptr(),
                    4,
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ),
                -1
            );
            assert_eq!(
                native_set_plasma_params(
                    std::ptr::null_mut(),
                    std::ptr::null::<crate::ionization::PptIonizationRate>(),
                    1.0,
                    1.0,
                    0.0,
                    1e-15,
                    1e25
                ),
                -1
            );
            assert_eq!(
                native_set_plasma_params_adk(
                    std::ptr::null_mut(),
                    std::ptr::null::<crate::ionization::AdkIonizationRate>(),
                    1.0,
                    1.0,
                    0.0,
                    1e-15,
                    1e25
                ),
                -1
            );
            assert_eq!(
                native_set_radial_params(
                    std::ptr::null_mut(),
                    8,
                    8,
                    2,
                    buf.as_ptr(),
                    1.0,
                    1.0,
                    buf.as_ptr(),
                    0.0,
                    buf.as_ptr(),
                    buf.as_ptr()
                ),
                -1
            );
            assert_eq!(
                native_set_radial_noise(std::ptr::null_mut(), buf.as_ptr(), 4),
                -1
            );
            assert_eq!(
                native_set_radial_noise_cplx(std::ptr::null_mut(), buf.as_ptr(), buf.as_ptr(), 4),
                -1
            );
            assert_eq!(
                native_set_raman_params(
                    std::ptr::null_mut(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    4,
                    1e-15,
                    1e25,
                    1
                ),
                -1
            );
            assert_eq!(
                native_set_raman_fft_params(
                    std::ptr::null_mut(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    4,
                    1.0,
                    1e-15,
                    8,
                    1e25
                ),
                -1
            );
            assert_eq!(
                native_set_modal_params(
                    std::ptr::null_mut(),
                    8,
                    8,
                    4,
                    1,
                    1.0,
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf_i32.as_ptr(),
                    buf_u8.as_ptr(),
                    buf.as_ptr(),
                    0,
                    buf_u8.as_ptr(),
                    buf.as_ptr(),
                    0.0,
                    buf.as_ptr(),
                    buf.as_ptr(),
                    lib_path.as_ptr(),
                    1e-6,
                    1e-9,
                    1000
                ),
                -1
            );
            assert_eq!(
                native_set_free_params(
                    std::ptr::null_mut(),
                    8,
                    8,
                    2,
                    2,
                    1 << 6,
                    buf.as_ptr(),
                    0.0,
                    buf.as_ptr(),
                    buf.as_ptr()
                ),
                -1
            );
            assert_eq!(
                native_set_free_zdep_params(
                    std::ptr::null_mut(),
                    1.0,
                    1.0,
                    2.0,
                    2,
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf_u8.as_ptr(),
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ),
                -1
            );
            assert_eq!(
                native_set_modal_zdep_params(
                    std::ptr::null_mut(),
                    1.0,
                    1.0,
                    2,
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf_u8.as_ptr(),
                    0,
                    0,
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    0.0,
                    0,
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr(),
                    buf.as_ptr()
                ),
                -1
            );
        }
    }

    #[test]
    fn field_lifecycle_rejects_length_mismatch_and_bad_index() {
        let sim = tiny_sim();
        assert!(!sim.is_null());
        let mut buf = vec![0.0f64; 32];
        unsafe {
            assert_eq!(set_field(sim, buf.as_ptr(), 3), -1);
            assert_eq!(native_resync_field(sim, buf.as_ptr(), 100), -1);
            assert_eq!(get_field(sim, buf.as_mut_ptr(), 0), -1);
            // idx must be < 7 (7 RK stages, 0-indexed).
            assert_eq!(get_ks_stage(sim, 7, buf.as_mut_ptr(), 4), -1);
            assert_eq!(get_ks_stage(sim, 0, buf.as_mut_ptr(), 999), -1);
            assert_eq!(native_apply_prop(sim, buf.as_mut_ptr(), 999, 0.0, 1.0), -1);
            assert_eq!(native_debug_linop_at(sim, 0.0, buf.as_mut_ptr(), 999), -1);

            // Null data/y buffer, with n matching sim.n == 4 (so the length
            // check alone wouldn't reject it) — exercises the null guard
            // specifically, not the length guard.
            assert_eq!(set_field(sim, std::ptr::null(), 4), -1);
            assert_eq!(native_resync_field(sim, std::ptr::null(), 4), -1);
            assert_eq!(get_field(sim, std::ptr::null_mut(), 4), -1);
            assert_eq!(get_ks_stage(sim, 0, std::ptr::null_mut(), 4), -1);
            assert_eq!(
                native_apply_prop(sim, std::ptr::null_mut(), 4, 0.0, 1.0),
                -1
            );
            assert_eq!(native_debug_linop_at(sim, 0.0, std::ptr::null_mut(), 4), -1);
            let mut one = 0.0f64;
            assert_eq!(
                native_debug_beta1_at(sim, 0.0, std::ptr::null_mut(), &mut one as *mut f64),
                -1
            );
            assert_eq!(
                native_debug_beta1_at(sim, 0.0, &mut one as *mut f64, std::ptr::null_mut()),
                -1
            );
        }
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn fftw_plans_rejects_unloadable_library_path() {
        // A deterministic, environment-independent adversarial input: no real
        // FFTW library needs to be present for this to exercise the load-
        // failure path (`Library::load`'s `dlopen`/`LoadLibraryW` returning
        // an error), unlike a success-path test would require.
        let sim = tiny_sim();
        let bad_path =
            std::ffi::CString::new("/definitely/not/a/real/path/libfftw3_missing.so").unwrap();
        let rc = unsafe {
            native_set_fftw_plans(sim, bad_path.as_ptr(), 8, 8, 1, 1 << 6, std::ptr::null())
        };
        assert_eq!(rc, -2);

        // Null lib_path -> -1, checked before ever reaching `CStr::from_ptr`.
        let rc = unsafe {
            native_set_fftw_plans(sim, std::ptr::null(), 8, 8, 1, 1 << 6, std::ptr::null())
        };
        assert_eq!(rc, -1);

        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn mode_avg_params_accepts_null_optional_arrays_as_defaults() {
        // Unlike most setters, `native_set_mode_avg_params` treats a null
        // towin/owin/sidx/pre/beta as "use the default" (all-ones/all-true/
        // zero) rather than an error — documented behavior, asserted here so
        // a future change can't silently flip it to `-1` unnoticed.
        let sim = tiny_sim();
        let rc = unsafe {
            native_set_mode_avg_params(
                sim,
                8,
                8,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0.0,
                0.0,
                1.0,
            )
        };
        assert_eq!(rc, 0);
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn mode_avg_noise_rejects_null_and_length_mismatch() {
        let sim = tiny_sim();
        let towin = vec![1.0f64; 8];
        let owin = vec![1.0f64; 4];
        let sidx = vec![1u8; 4];
        let pre = vec![0.0f64; 4];
        let beta = vec![1.0f64; 4];
        unsafe {
            let rc = native_set_mode_avg_params(
                sim,
                8,
                8,
                towin.as_ptr(),
                owin.as_ptr(),
                sidx.as_ptr(),
                pre.as_ptr(),
                pre.as_ptr(),
                beta.as_ptr(),
                0.0,
                0.0,
                1.0,
            );
            assert_eq!(rc, 0);

            assert_eq!(native_set_mode_avg_noise(sim, std::ptr::null(), 8), -1);
            assert_eq!(native_set_mode_avg_noise(sim, towin.as_ptr(), 0), -1);
            let wrong_len = vec![0.0f64; 3];
            assert_eq!(native_set_mode_avg_noise(sim, wrong_len.as_ptr(), 3), -2);

            assert_eq!(
                native_set_mode_avg_noise_cplx(sim, std::ptr::null(), towin.as_ptr(), 8),
                -1
            );
            assert_eq!(
                native_set_mode_avg_noise_cplx(sim, towin.as_ptr(), std::ptr::null(), 8),
                -1
            );
            assert_eq!(
                native_set_mode_avg_noise_cplx(sim, wrong_len.as_ptr(), wrong_len.as_ptr(), 3),
                -2
            );
        }
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn zdep_mode_avg_rejects_wrong_call_order_and_short_gradient() {
        let sim = tiny_sim(); // s.sidx starts empty; s.n == 4 -> length mismatch.
        let dummy = vec![0.0f64; 8];
        unsafe {
            let rc = native_set_zdep_mode_avg_params(
                sim,
                2,
                dummy.as_ptr(),
                dummy.as_ptr(),
                4,
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                0,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            );
            assert_eq!(rc, -3, "must be called after native_set_mode_avg_params");
        }

        let towin = vec![1.0f64; 8];
        let owin = vec![1.0f64; 4];
        let sidx = vec![1u8; 4];
        let pre = vec![0.0f64; 4];
        let beta = vec![1.0f64; 4];
        unsafe {
            let rc = native_set_mode_avg_params(
                sim,
                8,
                8,
                towin.as_ptr(),
                owin.as_ptr(),
                sidx.as_ptr(),
                pre.as_ptr(),
                pre.as_ptr(),
                beta.as_ptr(),
                0.0,
                0.0,
                1.0,
            );
            assert_eq!(rc, 0);

            let rc = native_set_zdep_mode_avg_params(
                sim,
                1, // n_z < 2
                dummy.as_ptr(),
                dummy.as_ptr(),
                4,
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                0,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            );
            assert_eq!(rc, -4);

            // n_z == 2 (valid) but z_pts is null -> -1, checked before any
            // spline/array pointer is dereferenced.
            let rc = native_set_zdep_mode_avg_params(
                sim,
                2,
                std::ptr::null(),
                dummy.as_ptr(),
                4,
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                0,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            );
            assert_eq!(rc, -1);
        }
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn plasma_params_rejects_zero_length_before_touching_the_handle_pointer() {
        let sim = tiny_sim();
        // Before any of native_set_mode_avg_params/native_set_radial_params/
        // native_set_free_params, s.n_time_over == 0 and is_radial/is_free
        // are both false, so plasma's derived buffer length is 0 -> -2,
        // regardless of the (here null) ionization handle pointer.
        unsafe {
            assert_eq!(
                native_set_plasma_params(
                    sim,
                    std::ptr::null::<crate::ionization::PptIonizationRate>(),
                    1.0,
                    1.0,
                    0.0,
                    1e-15,
                    1e25
                ),
                -2
            );
            assert_eq!(
                native_set_plasma_params_adk(
                    sim,
                    std::ptr::null::<crate::ionization::AdkIonizationRate>(),
                    1.0,
                    1.0,
                    0.0,
                    1e-15,
                    1e25
                ),
                -2
            );
        }
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn plasma_params_stores_null_handle_without_crashing_the_setter_itself() {
        // Found gap (documented in this module's header comment, not fixed
        // here): once n_time_over > 0, `native_set_plasma_params(_adk)`
        // stores whatever `ion_ptr` it's given without ever dereferencing
        // it, so a null handle is silently "accepted" (rc == 0) by the
        // setter — it would crash the first time `native_step` actually
        // evaluated the plasma response. `src/RK45.jl` never reaches this in
        // practice: it always gates on `!isnothing(irf.rust_handle) &&
        // irf.rust_handle.ptr != C_NULL` before calling. This test pins down
        // the setter's actual (permissive) behavior; it must not be extended
        // to call `native_step` afterwards.
        let sim = tiny_sim();
        let towin = vec![1.0f64; 8];
        let owin = vec![1.0f64; 4];
        let sidx = vec![1u8; 4];
        let pre = vec![0.0f64; 4];
        let beta = vec![1.0f64; 4];
        unsafe {
            let rc = native_set_mode_avg_params(
                sim,
                8,
                8,
                towin.as_ptr(),
                owin.as_ptr(),
                sidx.as_ptr(),
                pre.as_ptr(),
                pre.as_ptr(),
                beta.as_ptr(),
                0.0,
                0.0,
                1.0,
            );
            assert_eq!(rc, 0);
            let rc = native_set_plasma_params(
                sim,
                std::ptr::null::<crate::ionization::PptIonizationRate>(),
                1.0,
                1.0,
                0.0,
                1e-15,
                1e25,
            );
            assert_eq!(rc, 0);
        }
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn raman_params_rejects_wrong_call_order() {
        let sim = tiny_sim(); // n_time_over == 0 before any *_params setter runs.
        let dummy = vec![0.0f64; 4];
        unsafe {
            assert_eq!(
                native_set_raman_params(
                    sim,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    4,
                    1e-15,
                    1e25,
                    1
                ),
                -2
            );
        }
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn raman_params_rejects_null_oscillator_arrays() {
        let sim = tiny_sim();
        let towin = vec![1.0f64; 8];
        let owin = vec![1.0f64; 4];
        let sidx = vec![1u8; 4];
        let pre = vec![0.0f64; 4];
        let beta = vec![1.0f64; 4];
        let dummy = vec![0.0f64; 4];
        unsafe {
            let rc = native_set_mode_avg_params(
                sim,
                8,
                8,
                towin.as_ptr(),
                owin.as_ptr(),
                sidx.as_ptr(),
                pre.as_ptr(),
                pre.as_ptr(),
                beta.as_ptr(),
                0.0,
                0.0,
                1.0,
            );
            assert_eq!(rc, 0); // sets n_time_over = 8, so the wrong-order guard no longer fires

            assert_eq!(
                native_set_raman_params(
                    sim,
                    std::ptr::null(),
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    4,
                    1e-15,
                    1e25,
                    1
                ),
                -1
            );
            assert_eq!(
                native_set_raman_params(
                    sim,
                    dummy.as_ptr(),
                    std::ptr::null(),
                    dummy.as_ptr(),
                    4,
                    1e-15,
                    1e25,
                    1
                ),
                -1
            );
            assert_eq!(
                native_set_raman_params(
                    sim,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    std::ptr::null(),
                    4,
                    1e-15,
                    1e25,
                    1
                ),
                -1
            );
        }
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn raman_fft_params_rejects_null_arrays_zero_count_and_wrong_n_time() {
        let sim = tiny_sim();
        let towin = vec![1.0f64; 8];
        let owin = vec![1.0f64; 4];
        let sidx = vec![1u8; 4];
        let pre = vec![0.0f64; 4];
        let beta = vec![1.0f64; 4];
        let dummy = vec![0.0f64; 4];
        unsafe {
            let rc = native_set_mode_avg_params(
                sim,
                8,
                8,
                towin.as_ptr(),
                owin.as_ptr(),
                sidx.as_ptr(),
                pre.as_ptr(),
                pre.as_ptr(),
                beta.as_ptr(),
                0.0,
                0.0,
                1.0,
            );
            assert_eq!(rc, 0); // sets n_time_over = 8

            // n_time (3) != s.n_time_over (8) -> -2, checked first.
            assert_eq!(
                native_set_raman_fft_params(
                    sim,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    4,
                    1.0,
                    1e-15,
                    3,
                    1e25
                ),
                -2
            );
            // n_time matches, but omega is null -> -1.
            assert_eq!(
                native_set_raman_fft_params(
                    sim,
                    std::ptr::null(),
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    4,
                    1.0,
                    1e-15,
                    8,
                    1e25
                ),
                -1
            );
            // n_time matches, but n_osc == 0 -> -1.
            assert_eq!(
                native_set_raman_fft_params(
                    sim,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    0,
                    1.0,
                    1e-15,
                    8,
                    1e25
                ),
                -1
            );
            // Everything valid, but s.fftw_api was never set (no
            // native_set_fftw_plans call) -> -2.
            assert_eq!(
                native_set_raman_fft_params(
                    sim,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    4,
                    1.0,
                    1e-15,
                    8,
                    1e25
                ),
                -2
            );
        }
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn radial_params_rejects_zero_or_non_dividing_n_r() {
        let sim = tiny_sim(); // s.n == 4
        let dummy = vec![0.0f64; 64];
        unsafe {
            assert_eq!(
                native_set_radial_params(
                    sim,
                    8,
                    8,
                    0,
                    dummy.as_ptr(),
                    1.0,
                    1.0,
                    std::ptr::null(),
                    0.0,
                    dummy.as_ptr(),
                    dummy.as_ptr()
                ),
                -1
            );
            // n_r = 3 does not divide s.n = 4.
            assert_eq!(
                native_set_radial_params(
                    sim,
                    8,
                    8,
                    3,
                    dummy.as_ptr(),
                    1.0,
                    1.0,
                    std::ptr::null(),
                    0.0,
                    dummy.as_ptr(),
                    dummy.as_ptr()
                ),
                -1
            );
            // n_r = 2 (valid, divides s.n = 4), but t_matrix/m_re/m_im null.
            assert_eq!(
                native_set_radial_params(
                    sim,
                    8,
                    8,
                    2,
                    std::ptr::null(),
                    1.0,
                    1.0,
                    std::ptr::null(),
                    0.0,
                    dummy.as_ptr(),
                    dummy.as_ptr()
                ),
                -1
            );
            assert_eq!(
                native_set_radial_params(
                    sim,
                    8,
                    8,
                    2,
                    dummy.as_ptr(),
                    1.0,
                    1.0,
                    std::ptr::null(),
                    0.0,
                    std::ptr::null(),
                    dummy.as_ptr()
                ),
                -1
            );
            assert_eq!(
                native_set_radial_params(
                    sim,
                    8,
                    8,
                    2,
                    dummy.as_ptr(),
                    1.0,
                    1.0,
                    std::ptr::null(),
                    0.0,
                    dummy.as_ptr(),
                    std::ptr::null()
                ),
                -1
            );
        }
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn radial_noise_rejects_null_and_length_mismatch() {
        let sim = tiny_sim(); // s.n == 4
        let n_r = 2;
        let t_matrix = vec![0.0f64; n_r * n_r];
        let m = vec![0.0f64; 4]; // n_spec * n_r = (s.n/n_r)*n_r = s.n = 4
        unsafe {
            let rc = native_set_radial_params(
                sim,
                8,
                8,
                n_r,
                t_matrix.as_ptr(),
                1.0,
                1.0,
                std::ptr::null(),
                0.0,
                m.as_ptr(),
                m.as_ptr(),
            );
            assert_eq!(rc, 0); // n_time_over*n_r = 8*2 = 16

            assert_eq!(native_set_radial_noise(sim, std::ptr::null(), 8), -1);
            assert_eq!(native_set_radial_noise(sim, m.as_ptr(), 0), -1);
            let wrong = vec![0.0f64; 3];
            assert_eq!(native_set_radial_noise(sim, wrong.as_ptr(), 3), -2);

            assert_eq!(
                native_set_radial_noise_cplx(sim, std::ptr::null(), m.as_ptr(), 8),
                -1
            );
            assert_eq!(
                native_set_radial_noise_cplx(sim, m.as_ptr(), std::ptr::null(), 8),
                -1
            );
            assert_eq!(
                native_set_radial_noise_cplx(sim, wrong.as_ptr(), wrong.as_ptr(), 3),
                -2
            );
        }
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn modal_params_rejects_zero_or_invalid_npol_and_non_dividing_n_modes() {
        let sim = tiny_sim(); // s.n == 4
        let dummy = vec![0.0f64; 64];
        let dummy_u8 = vec![0u8; 64];
        let dummy_i32 = vec![0i32; 64];
        let lib_path = std::ffi::CString::new("libcubature.so").unwrap();
        unsafe {
            assert_eq!(
                native_set_modal_params(
                    sim,
                    8,
                    8,
                    0, // n_modes == 0
                    1,
                    1.0,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    dummy_i32.as_ptr(),
                    dummy_u8.as_ptr(),
                    dummy.as_ptr(),
                    0,
                    dummy_u8.as_ptr(),
                    std::ptr::null(),
                    0.0,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    lib_path.as_ptr(),
                    1e-6,
                    1e-9,
                    1000
                ),
                -1
            );
            assert_eq!(
                native_set_modal_params(
                    sim,
                    8,
                    8,
                    4,
                    3, // npol invalid (only 1 or 2 are valid)
                    1.0,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    dummy_i32.as_ptr(),
                    dummy_u8.as_ptr(),
                    dummy.as_ptr(),
                    0,
                    dummy_u8.as_ptr(),
                    std::ptr::null(),
                    0.0,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    lib_path.as_ptr(),
                    1e-6,
                    1e-9,
                    1000
                ),
                -1
            );
            assert_eq!(
                native_set_modal_params(
                    sim,
                    8,
                    8,
                    3, // n_modes = 3 does not divide s.n = 4
                    1,
                    1.0,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    dummy_i32.as_ptr(),
                    dummy_u8.as_ptr(),
                    dummy.as_ptr(),
                    0,
                    dummy_u8.as_ptr(),
                    std::ptr::null(),
                    0.0,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    lib_path.as_ptr(),
                    1e-6,
                    1e-9,
                    1000
                ),
                -1
            );
            // n_modes = 4 (valid, divides s.n = 4), npol = 1 (valid), but unm
            // is null -> -1, checked before lib_path is even loaded.
            assert_eq!(
                native_set_modal_params(
                    sim,
                    8,
                    8,
                    4,
                    1,
                    1.0,
                    std::ptr::null(),
                    dummy.as_ptr(),
                    dummy_i32.as_ptr(),
                    dummy_u8.as_ptr(),
                    dummy.as_ptr(),
                    0,
                    dummy_u8.as_ptr(),
                    std::ptr::null(),
                    0.0,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    lib_path.as_ptr(),
                    1e-6,
                    1e-9,
                    1000
                ),
                -1
            );
            // Everything else valid, but lib_path itself is null.
            assert_eq!(
                native_set_modal_params(
                    sim,
                    8,
                    8,
                    4,
                    1,
                    1.0,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    dummy_i32.as_ptr(),
                    dummy_u8.as_ptr(),
                    dummy.as_ptr(),
                    0,
                    dummy_u8.as_ptr(),
                    std::ptr::null(),
                    0.0,
                    dummy.as_ptr(),
                    dummy.as_ptr(),
                    std::ptr::null(),
                    1e-6,
                    1e-9,
                    1000
                ),
                -1
            );
        }
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn modal_zdep_rejects_wrong_call_order() {
        let sim = tiny_sim(); // s.n_spec == 0, s.n_modes == 0 before native_set_modal_params.
        let dummy = vec![0.0f64; 8];
        let dummy_u8 = vec![0u8; 8];
        unsafe {
            let rc = native_set_modal_zdep_params(
                sim,
                1.0,
                1.0,
                2,
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy_u8.as_ptr(),
                0,
                0,
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                0.0,
                0,
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
            );
            assert_eq!(rc, -3);
            // As with `free_zdep_rejects_wrong_call_order`: reaching past
            // `-3` here needs a prior successful `native_set_modal_params`
            // call, which needs a real, loadable `libcubature` — an
            // environment dependency avoided in this hermetic suite. The
            // null-pointer guard is verified by direct code inspection
            // instead (same pattern as the other `*_zdep_params` setters).
        }
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn free_params_rejects_zero_transverse_dims_and_non_dividing_size() {
        let sim = tiny_sim(); // s.n == 4
        let dummy = vec![0.0f64; 64];
        unsafe {
            assert_eq!(
                native_set_free_params(
                    sim,
                    8,
                    8,
                    0, // n_y == 0
                    2,
                    1 << 6,
                    std::ptr::null(),
                    0.0,
                    dummy.as_ptr(),
                    dummy.as_ptr()
                ),
                -1
            );
            assert_eq!(
                native_set_free_params(
                    sim,
                    8,
                    8,
                    2,
                    0, // n_x == 0
                    1 << 6,
                    std::ptr::null(),
                    0.0,
                    dummy.as_ptr(),
                    dummy.as_ptr()
                ),
                -1
            );
            assert_eq!(
                native_set_free_params(
                    sim,
                    8,
                    8,
                    3,
                    3, // n_y*n_x = 9 does not divide s.n = 4
                    1 << 6,
                    std::ptr::null(),
                    0.0,
                    dummy.as_ptr(),
                    dummy.as_ptr()
                ),
                -1
            );
            // n_y = n_x = 2 (valid, divides s.n = 4), but m_re/m_im are null.
            // This is checked before the `is_real`/`fftw_api` branch, so it
            // doesn't require a real FFTW library to be loadable.
            assert_eq!(
                native_set_free_params(
                    sim,
                    8,
                    8,
                    2,
                    2,
                    1 << 6,
                    std::ptr::null(),
                    0.0,
                    std::ptr::null(),
                    dummy.as_ptr()
                ),
                -1
            );
            assert_eq!(
                native_set_free_params(
                    sim,
                    8,
                    8,
                    2,
                    2,
                    1 << 6,
                    std::ptr::null(),
                    0.0,
                    dummy.as_ptr(),
                    std::ptr::null()
                ),
                -1
            );
        }
        unsafe { free_native_sim(sim) };
    }

    #[test]
    fn free_zdep_rejects_wrong_call_order() {
        let sim = tiny_sim(); // s.n_spec/n_y/n_x all still 0 before native_set_free_params.
        let dummy = vec![0.0f64; 8];
        let dummy_u8 = vec![0u8; 8];
        unsafe {
            let rc = native_set_free_zdep_params(
                sim,
                1.0,
                1.0,
                2.0,
                2,
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy.as_ptr(),
                dummy_u8.as_ptr(),
                0.0,
                0.0,
                0.0,
                0.0,
            );
            assert_eq!(rc, -3);
            // The null-pointer guard added alongside this call-order guard
            // (see this module's header comment) is not separately exercised
            // here: reaching past `-3` requires a prior successful
            // `native_set_free_params` call, which needs a real, loadable
            // FFTW library — an environment dependency this hermetic test
            // suite otherwise avoids (see `fftw_plans_rejects_unloadable_
            // library_path`'s comment). Verified instead by direct code
            // inspection: the guard is the same `is_null()` pattern already
            // exercised for `native_set_zdep_mode_avg_params` above.
        }
        unsafe { free_native_sim(sim) };
    }
}
