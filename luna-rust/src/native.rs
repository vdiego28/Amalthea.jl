//! Native resident-field simulation state for the exclusively-Rust backend.
//!
//! See `docs/native-port/ARCHITECTURE.md` (§3 `NativeSim`) and `MATH.md`.
//!
//! Phase 0 goal: a single opaque handle that owns, for the lifetime of one
//! `solve`, the spectral field and every scratch buffer the interaction-picture
//! Dormand-Prince loop touches — so the FFI boundary is crossed a handful of
//! times per `solve` instead of ~14× per step (no Julia callback in the hot
//! loop).
//!
//! ## Build status (keep in sync with `docs/native-port/PORT_LOG.md`)
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
//!       eligibility (same as the existing `LUNA_USE_RUST_RAMAN` FFI wiring).
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

use num_complex::Complex;
use libc::{c_char, c_double, c_int, c_uint, size_t, c_void};
use crate::fftw::{FftwApi, ComplexFft1d, RealFft1d, RealFft3d, ComplexFft3d};
use crate::ffi::QdhtFfiHandle;
use crate::spline::HermiteSpline;
use crate::raman::{TimeDomainRamanSolver, RamanOscillator};
use crate::cubature::CubatureApi;
use crate::diffraction::jn;
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
    pub plasma_ionpot: f64,
    pub plasma_e_ratio: f64,
    pub plasma_preionfrac: f64,
    pub plasma_dt: f64,
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
    /// — `n-1` for `:HE`, `1` for `:TE`/`:TM` (BACKLOG.md Phase E.1: general
    /// mode orders, generalizing the Phase 5 `kind=:HE,n=1`-only `J0`
    /// scope). Length `n_modes`.
    pub modal_order: Vec<i32>,
    /// Per-mode kind: 0=`:HE`, 1=`:TE`, 2=`:TM` — selects the angular
    /// formula in `mode_angle_xy` (BACKLOG.md Phase E.3: general θ,
    /// generalizing E.1/E.2's fixed-`θ=0` shortcut). Length `n_modes`.
    pub modal_kind: Vec<u8>,
    /// Per-mode rotation angle `ϕ` — only meaningful for `:HE` (`:TE`/`:TM`
    /// have no `ϕ` dependence, see `Capillary.field`). Length `n_modes`.
    pub modal_phi: Vec<f64>,
    /// `full=true`: genuine 2-D `(r,θ)` cubature (`hcubature_v`) instead of
    /// the `full=false` radial-only 1-D integral at fixed `θ=0`
    /// (`pcubature_v`) — BACKLOG.md Phase E.3.
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
    // Per-node scratch (one node at a time — looped, not batched, see doc
    // comment above). Sized once in `native_set_modal_params`.
    pub modal_ems: Vec<f64>,
    pub modal_erw: Vec<Complex<f64>>,
    pub modal_erwo: Vec<Complex<f64>>,
    pub modal_er: Vec<f64>,
    pub modal_pr: Vec<f64>,
    pub modal_prwo: Vec<Complex<f64>>,
    pub modal_prw: Vec<Complex<f64>>,
    /// EnvGrid (c2c) counterparts of `modal_er`/`modal_pr` — only allocated
    /// (non-empty) when `is_real == false` (Phase E.4 item 5), same split as
    /// `radial_eto_c`/`radial_pto_c` (Phase D.1).
    pub modal_er_c: Vec<Complex<f64>>,
    pub modal_pr_c: Vec<Complex<f64>>,
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
    /// EnvGrid (c2c) counterpart of `fft_r2c_3d` — Phase D.3 (BACKLOG.md).
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

    // ── Phase 7 (+ BACKLOG.md Phase F item 3): z-dependent linop — mode-
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
    /// is evaluated. Closed-form β1 constants (see `docs/native-port/
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
/// (`Capillary.TwoPointGradient`/`MultiPointGradient`, BACKLOG.md Phase F
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
            kerr_fac: 0.0,
            nlscale: 0.0,
            sqrt_aeff: 1.0,
            n_time: 0,
            n_time_over: 0,
            n_spec: n,
            n_spec_over: 0,
            et_noise: Vec::new(),
            has_noise: false,
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
            plasma_ionpot: 0.0,
            plasma_e_ratio: 0.0,
            plasma_preionfrac: 0.0,
            plasma_dt: 1.0,
            has_plasma: false,
            is_radial: false,
            radial_et_noise: Vec::new(),
            has_radial_noise: false,
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
            modal_ems: Vec::new(),
            modal_erw: Vec::new(),
            modal_erwo: Vec::new(),
            modal_er: Vec::new(),
            modal_pr: Vec::new(),
            modal_prwo: Vec::new(),
            modal_prw: Vec::new(),
            modal_er_c: Vec::new(),
            modal_pr_c: Vec::new(),
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

        // ── Step 6: norm! — apply pre[i]/β[i]*sqrt_aeff at sidx positions ────────
        for i in 0..self.n_spec {
            if self.sidx[i] {
                self.ks[idx][i] *= self.pre[i] / self.beta[i] * self.sqrt_aeff;
            }
        }

        // ── Step 7: freq-window apodization at sidx positions ────────────────────
        for i in 0..self.n_spec {
            if self.sidx[i] {
                self.ks[idx][i] *= self.owin[i];
            }
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
        // SAFETY: pointer set by `native_set_plasma_params`; Julia owns the handle
        // and keeps it alive for the lifetime of the stepper.
        let ion = unsafe { &*self.plasma_ion_ptr };

        // 1. ionization rate W(|E(t)|)
        for i in 0..n {
            let e_abs = self.eto[i].abs();
            self.plas_rate[i] = ion.rate(e_abs)
                .unwrap_or_else(|_| ion.rate(ion.e_max).unwrap_or(0.0));
        }

        // 2. cumtrapz(fraction, rate, dt) → raw integral ∫₀ᵗ W dτ
        cumtrapz_slice_f64(&self.plas_rate, &mut self.plas_fraction, dt);

        // 3. ρ(t) = preionfrac + 1 − exp(−∫W)
        for i in 0..n {
            self.plas_fraction[i] =
                self.plasma_preionfrac + 1.0 - (-self.plas_fraction[i]).exp();
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

        // 8. add P(t) to pto
        for i in 0..n {
            self.pto[i] += self.plas_p[i];
        }
    }

    /// Radial counterpart of `apply_plasma_real` — Phase D.2 (BACKLOG.md),
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
        // SAFETY: pointer set by `native_set_plasma_params`; Julia owns the handle
        // and keeps it alive for the lifetime of the stepper.
        let ion = unsafe { &*self.plasma_ion_ptr };

        for r in 0..n_r {
            let start = r * n_time_over;
            let end = start + n_time_over;

            // 1. ionization rate W(|E(t)|)
            for i in start..end {
                let e_abs = self.radial_eto[i].abs();
                self.plas_rate[i] = ion.rate(e_abs)
                    .unwrap_or_else(|_| ion.rate(ion.e_max).unwrap_or(0.0));
            }

            // 2. cumtrapz(fraction, rate, dt) → raw integral ∫₀ᵗ W dτ, per column
            cumtrapz_slice_f64(&self.plas_rate[start..end], &mut self.plas_fraction[start..end], dt);

            // 3. ρ(t) = preionfrac + 1 − exp(−∫W)
            for i in start..end {
                self.plas_fraction[i] =
                    self.plasma_preionfrac + 1.0 - (-self.plas_fraction[i]).exp();
            }

            // 4. phase[i] = ρ[i] * e_ratio * E[i]
            for i in start..end {
                self.plas_phase[i] = self.plas_fraction[i] * self.plasma_e_ratio * self.radial_eto[i];
            }

            // 5. cumtrapz(J, phase, dt) → free-electron current, per column
            cumtrapz_slice_f64(&self.plas_phase[start..end], &mut self.plas_j[start..end], dt);

            // 6. ionization loss current: J[i] += Ip * W[i] * (1−ρ[i]) / E[i]
            for i in start..end {
                let e = self.radial_eto[i];
                if e.abs() > 0.0 {
                    self.plas_j[i] +=
                        self.plasma_ionpot * self.plas_rate[i] * (1.0 - self.plas_fraction[i]) / e;
                }
            }

            // 7. cumtrapz(P, J, dt) → plasma polarisation, per column
            cumtrapz_slice_f64(&self.plas_j[start..end], &mut self.plas_p[start..end], dt);
        }

        // 8. add P(t,r) to radial_pto
        for i in 0..(n_time_over * n_r) {
            self.radial_pto[i] += self.plas_p[i];
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
                hilbert_intensity(fft, &mut self.raman_hilbert_a, &mut self.raman_hilbert_b,
                                   &self.eto[..n], &mut self.raman_intensity[..n], norm);
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

    /// Radial counterpart of `apply_raman_real` — Phase D.4 (BACKLOG.md).
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
                hilbert_intensity(fft, &mut self.raman_hilbert_a, &mut self.raman_hilbert_b,
                                   &self.radial_eto[start..end], &mut self.raman_intensity[start..end], norm);
            }
            if let Some(ref mut solver) = self.raman_solver {
                solver.solve(&self.raman_intensity[start..end], &mut self.raman_p[start..end]);
            }
            for i in start..end {
                self.radial_pto[i] += rho * self.radial_eto[i] * self.raman_p[i];
            }
        }
        self.raman_hilbert_fft = fft;
    }

    /// EnvGrid mode-averaged RHS: Kerr_env (|E|²·E) via c2c FFTW plans.
    /// Reproduces `TransModeAvg` for `ComplexF64` fields (src/NonlinearRHS.jl:531).
    fn rhs_mode_avg_env(&mut self, idx: usize, eomega: &[Complex<f64>]) {
        let n  = self.n_spec;       // = n_time for EnvGrid (c2c: Nω = Nt)
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
            for v in &mut self.eto_cplx { *v *= inv_nto; }
        }

        // ── Step 2: scale by 1/(nlscale·√Aeff) ──────────────────────────────────
        let inv_sc = 1.0 / (self.nlscale * self.sqrt_aeff);
        for v in &mut self.eto_cplx { *v *= inv_sc; }

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
            for i in 0..no {
                self.raman_intensity[i] = 0.5 * self.eto_cplx[i].norm_sqr();
            }
            if let Some(ref mut solver) = self.raman_solver {
                solver.solve(&self.raman_intensity[..no], &mut self.raman_p[..no]);
            }
            let rho = self.raman_density;
            for i in 0..no {
                self.pto_cplx[i] += self.eto_cplx[i] * (rho * self.raman_p[i]);
            }
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

        // ── Step 6: norm! and freq-window ────────────────────────────────────────
        for i in 0..n {
            if self.sidx[i] {
                self.ks[idx][i] *= self.pre[i] / self.beta[i] * self.sqrt_aeff;
            }
        }
        for i in 0..n {
            if self.sidx[i] {
                self.ks[idx][i] *= self.owin[i];
            }
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
            for r in 0..n_r {
                let spec_start = r * n_spec_over;
                let time_start = r * n_time_over;
                let spec_col = &mut self.radial_eoo[spec_start..spec_start + n_spec_over];
                let time_col = &mut self.radial_eto[time_start..time_start + n_time_over];
                fft.inverse(spec_col, time_col);
            }
            let inv_nto = self.fft_norm_over;
            for v in &mut self.radial_eto { *v *= inv_nto; }
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
            for r in 0..n_r {
                let time_start = r * n_time_over;
                let spec_start = r * n_spec_over;
                let time_col = &mut self.radial_pto[time_start..time_start + n_time_over];
                let spec_col = &mut self.radial_poo[spec_start..spec_start + n_spec_over];
                fft.forward(time_col, spec_col);
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

    /// EnvGrid (c2c) counterpart of `rhs_radial` — Phase D.1 (BACKLOG.md),
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
        let n = self.n_spec;   // = n_time for EnvGrid (c2c: Nω = Nt)
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
            for r in 0..n_r {
                let spec_start = r * no;
                let time_start = r * no;
                let spec_col = &mut self.radial_eoo[spec_start..spec_start + no];
                let time_col = &mut self.radial_eto_c[time_start..time_start + no];
                fft.inverse(spec_col, time_col);
            }
            let inv_nto = self.fft_norm_over;
            for v in &mut self.radial_eto_c { *v *= inv_nto; }
        }

        // ── Step 2: QDHT ldiv! (k → r), complex ──────────────────────────────────
        let scale_inv = self.qdht_scale_inv;
        if let Some(ref mut qdht) = self.qdht {
            // Safety: `Complex<f64>` is `#[repr(C)]` (two contiguous `f64`s),
            // same idiom as `rhs_modal`'s `ks[idx]` reinterpretation above.
            let buf = unsafe {
                std::slice::from_raw_parts_mut(
                    self.radial_eto_c.as_mut_ptr() as *mut f64,
                    2 * self.radial_eto_c.len())
            };
            qdht.apply_cplx(buf, no, scale_inv);
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
                    2 * self.radial_pto_c.len())
            };
            qdht.apply_cplx(buf, no, scale_fwd_q);
        }

        // ── Step 6: to_freq! per r-column (EnvGrid c2c) ──────────────────────────
        let scale_inv2 = n as f64 / no as f64;
        if let Some(ref fft) = self.fft_c2c_over {
            for r in 0..n_r {
                let time_start = r * no;
                let spec_start = r * no;
                let time_col = &mut self.radial_pto_c[time_start..time_start + no];
                let spec_col = &mut self.radial_poo[spec_start..spec_start + no];
                fft.forward(time_col, spec_col);
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
        let cubature = self.cubature.take().expect("rhs_modal called before native_set_modal_params");
        let rc = if self.modal_full {
            // Phase E.3: genuine 2-D `(r,θ)` cubature — `hcubature_v`, the
            // same h-adaptive routine Julia's `full=true` branch calls
            // (`Cubature.hcubature_v`, `NonlinearRHS.jl`'s `(t::TransModal)`).
            unsafe {
                cubature.hcubature_v_2d(
                    fdim,
                    modal_integrand_v_full,
                    self as *mut CpuNativeSim as *mut c_void,
                    [0.0, 0.0], [self.modal_a, 2.0 * std::f64::consts::PI],
                    self.modal_maxevals,
                    self.modal_atol, self.modal_rtol,
                    &mut valbuf, &mut errbuf,
                )
            }
        } else {
            unsafe {
                cubature.pcubature_v(
                    fdim,
                    modal_integrand_v,
                    self as *mut CpuNativeSim as *mut c_void,
                    0.0, self.modal_a,
                    self.modal_maxevals,
                    self.modal_atol, self.modal_rtol,
                    &mut valbuf, &mut errbuf,
                )
            }
        };
        self.cubature = Some(cubature);
        if rc != 0 {
            eprintln!("rhs_modal: cubature returned {}", rc);
        }

        let ks_f64: &mut [f64] = unsafe {
            std::slice::from_raw_parts_mut(self.ks[idx].as_mut_ptr() as *mut f64, fdim)
        };
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
    fn rhs_modal_pointcalc(&mut self, r: f64, theta: f64, out: &mut [f64]) {
        let n_modes = self.n_modes;
        let npol = self.npol;
        let n_spec = self.n_spec;
        let n_spec_over = self.n_spec_over;
        let n_time_over = self.n_time_over;

        if r <= 0.0 || r >= self.modal_a {
            out.fill(0.0);
            return;
        }

        // ── mode field synthesis at (r,θ): Exy = J_order(r·unm/a)/√N ·
        // (ax,ay) — see `mode_angle_xy` for the per-kind angular formula.
        for m in 0..n_modes {
            let x = r * self.modal_unm[m] / self.modal_a;
            let base = jn(self.modal_order[m], x) * self.modal_inv_sqrt_n[m];
            let (ax, ay) = mode_angle_xy(self.modal_kind[m], self.modal_order[m], self.modal_phi[m], theta);
            for (p, &sel) in self.modal_pol_select.iter().enumerate() {
                self.modal_ems[m * npol + p] = base * if sel == 0 { ax } else { ay };
            }
        }

        // ── to_space!: Erω[iω,p] = Σ_m Emω[iω,m]·Ems[m,p] ────────────────────────
        for v in self.modal_erw.iter_mut() { *v = Complex::new(0.0, 0.0); }
        for p in 0..npol {
            for m in 0..n_modes {
                let coeff = self.modal_ems[m * npol + p];
                if coeff == 0.0 { continue; }
                let em_col = &self.modal_emega[m * n_spec..(m + 1) * n_spec];
                let er_col = &mut self.modal_erw[p * n_spec..(p + 1) * n_spec];
                for i in 0..n_spec {
                    er_col[i] += em_col[i] * coeff;
                }
            }
        }

        self.modal_erwo.fill(Complex::new(0.0, 0.0));
        if self.is_real {
            // ── to_time! per polarisation column (RealGrid r2c, looped rank-1 plan) ──
            let scale_fwd = (n_spec_over - 1) as f64 / (n_spec - 1) as f64;
            for p in 0..npol {
                let in_col = &self.modal_erw[p * n_spec..(p + 1) * n_spec];
                let out_col = &mut self.modal_erwo[p * n_spec_over..(p + 1) * n_spec_over];
                for i in 0..n_spec { out_col[i] = in_col[i] * scale_fwd; }
            }
            if let Some(ref fft) = self.fft_r2c_over {
                for p in 0..npol {
                    let spec_start = p * n_spec_over;
                    let time_start = p * n_time_over;
                    let spec_col = &mut self.modal_erwo[spec_start..spec_start + n_spec_over];
                    let time_col = &mut self.modal_er[time_start..time_start + n_time_over];
                    fft.inverse(spec_col, time_col);
                }
                let inv_nto = self.fft_norm_over;
                for v in &mut self.modal_er { *v *= inv_nto; }
            }

            // ── Kerr (KerrScalar!/KerrVector!, src/Nonlinear.jl:81-93) ───────────────
            self.modal_pr.fill(0.0);
            let fac = self.modal_kerr_fac;
            if npol == 1 {
                for i in 0..n_time_over {
                    let e = self.modal_er[i];
                    self.modal_pr[i] += fac * e * e * e;
                }
            } else {
                for i in 0..n_time_over {
                    let ex = self.modal_er[i];
                    let ey = self.modal_er[n_time_over + i];
                    let sq = ex * ex + ey * ey;
                    self.modal_pr[i] += fac * sq * ex;
                    self.modal_pr[n_time_over + i] += fac * sq * ey;
                }
            }

            // ── Raman polarisation (if enabled), npol=1 scalar field only ────────────
            // Phase D.4 (BACKLOG.md): same `apply_raman_real` ADE-solve formula,
            // applied per quadrature node — `rhs_modal_pointcalc` is called
            // sequentially per node (see `modal_integrand_v`'s doc), so the
            // single resident `raman_solver` (which resets its own state at
            // solve-entry) can be reused across nodes with no cross-node
            // leakage, exactly like `apply_raman_radial`'s per-r-column reuse.
            // RealGrid-only (native.rs's inline ADE solve is a real-buffer
            // solver) — Julia rejects Raman+EnvGrid before this is reached.
            if self.has_raman {
                let rho = self.raman_density;
                if self.raman_thg {
                    for i in 0..n_time_over {
                        let e = self.modal_er[i];
                        self.raman_intensity[i] = e * e;
                    }
                } else {
                    let fft = self.raman_hilbert_fft.take();
                    if let Some(ref fft) = fft {
                        let norm = self.fft_norm_over;
                        hilbert_intensity(fft, &mut self.raman_hilbert_a, &mut self.raman_hilbert_b,
                                           &self.modal_er[..n_time_over], &mut self.raman_intensity[..n_time_over], norm);
                    }
                    self.raman_hilbert_fft = fft;
                }
                if let Some(ref mut solver) = self.raman_solver {
                    solver.solve(&self.raman_intensity[..n_time_over], &mut self.raman_p[..n_time_over]);
                }
                for i in 0..n_time_over {
                    self.modal_pr[i] += rho * self.modal_er[i] * self.raman_p[i];
                }
            }

            // ── towin apodization per column (reuses the 1-D towin buffer) ───────────
            for p in 0..npol {
                let start = p * n_time_over;
                for t in 0..n_time_over {
                    self.modal_pr[start + t] *= self.towin[t];
                }
            }

            // ── to_freq! per column ───────────────────────────────────────────────────
            let scale_inv2 = (n_spec - 1) as f64 / (n_spec_over - 1) as f64;
            if let Some(ref fft) = self.fft_r2c_over {
                for p in 0..npol {
                    let time_start = p * n_time_over;
                    let spec_start = p * n_spec_over;
                    let time_col = &mut self.modal_pr[time_start..time_start + n_time_over];
                    let spec_col = &mut self.modal_prwo[spec_start..spec_start + n_spec_over];
                    fft.forward(time_col, spec_col);
                }
            }
            for p in 0..npol {
                let in_start = p * n_spec_over;
                let out_start = p * n_spec;
                for i in 0..n_spec {
                    self.modal_prw[out_start + i] = self.modal_prwo[in_start + i] * scale_inv2;
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
                let in_col = &self.modal_erw[p * n_spec..(p + 1) * n_spec];
                let out_col = &mut self.modal_erwo[p * n_spec_over..(p + 1) * n_spec_over];
                for i in 0..half { out_col[i] = in_col[i] * scale_fwd; }
                for i in 0..half { out_col[n_spec_over - half + i] = in_col[n_spec - half + i] * scale_fwd; }
            }
            if let Some(ref fft) = self.fft_c2c_over {
                for p in 0..npol {
                    let spec_start = p * n_spec_over;
                    let time_start = p * n_time_over;
                    let spec_col = &mut self.modal_erwo[spec_start..spec_start + n_spec_over];
                    let time_col = &mut self.modal_er_c[time_start..time_start + n_time_over];
                    fft.inverse(spec_col, time_col);
                }
                let inv_nto = self.fft_norm_over;
                for v in &mut self.modal_er_c { *v *= inv_nto; }
            }

            // ── Kerr_env (KerrScalarEnv!/KerrVectorEnv!, src/Nonlinear.jl:120-133) ──
            self.modal_pr_c.fill(Complex::new(0.0, 0.0));
            let fac = Complex::new(0.75 * self.modal_kerr_fac, 0.0);
            if npol == 1 {
                for i in 0..n_time_over {
                    let e = self.modal_er_c[i];
                    self.modal_pr_c[i] += fac * e.norm_sqr() * e;
                }
            } else {
                let third = 1.0 / 3.0;
                for i in 0..n_time_over {
                    let ex = self.modal_er_c[i];
                    let ey = self.modal_er_c[n_time_over + i];
                    let ex2 = ex.norm_sqr();
                    let ey2 = ey.norm_sqr();
                    self.modal_pr_c[i] += fac * ((ex2 + 2.0 * third * ey2) * ex + third * ex.conj() * ey * ey);
                    self.modal_pr_c[n_time_over + i] += fac * ((ey2 + 2.0 * third * ex2) * ey + third * ey.conj() * ex * ex);
                }
            }

            // ── towin apodization per column ─────────────────────────────────────────
            for p in 0..npol {
                let start = p * n_time_over;
                for t in 0..n_time_over {
                    self.modal_pr_c[start + t] *= self.towin[t];
                }
            }

            // ── to_freq! per column ───────────────────────────────────────────────────
            let scale_inv2 = n_spec as f64 / n_spec_over as f64;
            if let Some(ref fft) = self.fft_c2c_over {
                for p in 0..npol {
                    let time_start = p * n_time_over;
                    let spec_start = p * n_spec_over;
                    let time_col = &mut self.modal_pr_c[time_start..time_start + n_time_over];
                    let spec_col = &mut self.modal_prwo[spec_start..spec_start + n_spec_over];
                    fft.forward(time_col, spec_col);
                }
            }
            for p in 0..npol {
                let in_start = p * n_spec_over;
                let out_start = p * n_spec;
                for i in 0..half {
                    self.modal_prw[out_start + i] = self.modal_prwo[in_start + i] * scale_inv2;
                }
                for i in 0..half {
                    self.modal_prw[out_start + n_spec - half + i] =
                        self.modal_prwo[in_start + n_spec_over - half + i] * scale_inv2;
                }
            }
        }

        // ── ωwin + norm! (precomputed modal_nlfac, per ω, same for every pol column) ──
        for p in 0..npol {
            let col = &mut self.modal_prw[p * n_spec..(p + 1) * n_spec];
            for i in 0..n_spec {
                col[i] *= self.modal_nlfac[i];
            }
        }

        // ── back-projection: Prmω[iω,m] = Σ_p Prω[iω,p]·Ems[m,p] ─────────────────
        // `full=false`: Jacobian is `2πr` (θ-integral done analytically).
        // `full=true`: Jacobian is just `r` (θ genuinely integrated).
        let pre = if self.modal_full { r } else { 2.0 * std::f64::consts::PI * r };
        for m in 0..n_modes {
            for i in 0..n_spec {
                let mut acc = Complex::new(0.0, 0.0);
                for p in 0..npol {
                    acc += self.modal_prw[p * n_spec + i] * self.modal_ems[m * npol + p];
                }
                acc *= pre;
                let idx = i + n_spec * m;
                out[2 * idx] = acc.re;
                out[2 * idx + 1] = acc.im;
            }
        }
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
            for i in 0..n_spec { out_col[i] = in_col[i] * scale_fwd; }
        }

        // ── Step 2: ONE joint 3-D inverse transform (ω,ky,kx) → (t,y,x) ──────────
        if let Some(ref fft) = self.fft_r2c_3d {
            fft.inverse(&mut self.free_eoo, &mut self.free_eto);
            let inv_nto = self.free_fft_norm_over;
            for v in &mut self.free_eto { *v *= inv_nto; }
        }

        // ── Step 3: Kerr (KerrScalar!), flat over the whole (t,y,x) volume ───────
        self.free_pto.fill(0.0);
        for i in 0..(n_time_over * n_cols) {
            let e = self.free_eto[i];
            self.free_pto[i] += self.kerr_fac * e * e * e;
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

    /// EnvGrid (c2c) counterpart of `rhs_free` — Phase D.3 (BACKLOG.md).
    /// Mirrors `rhs_radial_env`'s c2c `to_time!`/`to_freq!` convention
    /// (half-spectrum zero-pad, `no/n` scale, 3/4 `Kerr_env` SVEA factor)
    /// applied per `(y,x)` column for the zero-pad/truncate/window steps,
    /// but — like `rhs_free`'s real-grid Kerr step — the joint 3-D
    /// transform and the flat elementwise Kerr multiply act over the whole
    /// `(t,y,x)` volume at once, not per column.
    fn rhs_free_env(&mut self, idx: usize, eomega: &[Complex<f64>]) {
        let n = self.n_spec;    // = n_time for EnvGrid (c2c: Nω = Nt)
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
            for v in &mut self.free_eto_c { *v *= inv_nto; }
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
        if !self.is_zdep_mode_avg { return; }
        if self.zdep_last_z == z { return; }

        // Closed-form z -> pressure (TwoPointGradient/MultiPointGradient's
        // p(z), BACKLOG.md Phase F item 3), ported exactly (no interpolation
        // error) — flat-clamped outside the breakpoint range, reproducing
        // Capillary.gradient's own boundary behavior.
        let pressure = zdep_pressure_at(z, &self.zdep_grad_z, &self.zdep_grad_p);
        let dens = self.zdep_dens_lut.as_ref().unwrap().eval(pressure);
        const C: f64 = 299_792_458.0;

        // β1(z) closed form (docs/native-port/BETA1_ANALYTIC.md): εco(ω0;z)-1
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
        }
        // TransModeAvg calls `t.densityfun(z)` fresh every RK stage
        // (NonlinearRHS.jl), so `kerr_fac = density(z)·ε₀·γ3` must be
        // rescaled here too, not just baked in once at construction.
        self.kerr_fac = self.zdep_kerr_fac_per_dens * dens;
        self.zdep_last_z = z;
    }

    /// Recompute `self.linop` and `self.free_m` in place for propagation
    /// position `z` — the free-space (`TransFree`) counterpart of
    /// `ensure_linop_at`, Phase D.5 (BACKLOG.md). No-op if `is_zdep_free` is
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
        if !self.is_zdep_free { return; }
        if self.zdep_free_last_z == z { return; }

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
    }

    /// Recompute `self.linop` in place for propagation position `z` — the
    /// modal (`TransModal`, `full=false`) counterpart of `ensure_linop_at`,
    /// BACKLOG.md Phase E.2 (tapered/per-mode radius). No-op if
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
        if !self.is_zdep_modal { return; }
        if self.zdep_modal_last_z == z { return; }

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
    }
}

/// C `integrand_v` callback for `pcubature_v` — reconstructs `&mut NativeSim`
/// from `fdata` and loops `rhs_modal_pointcalc` over the batch of nodes
/// (evaluated one at a time, not truly vectorized — mirrors Phase 3's
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
        let npt = npt as usize;
        let fdim = fdim as usize;
        let xs = unsafe { std::slice::from_raw_parts(x, npt) };
        let fvals = unsafe { std::slice::from_raw_parts_mut(fval, npt * fdim) };
        for p in 0..npt {
            let out = &mut fvals[p * fdim..(p + 1) * fdim];
            sim.rhs_modal_pointcalc(xs[p], 0.0, out);
        }
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
            (theta.cos() * arg.sin() - theta.sin() * arg.cos(),
             theta.sin() * arg.sin() + theta.cos() * arg.cos())
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
        let npt = npt as usize;
        let fdim = fdim as usize;
        let xs = unsafe { std::slice::from_raw_parts(x, 2 * npt) };
        let fvals = unsafe { std::slice::from_raw_parts_mut(fval, npt * fdim) };
        for p in 0..npt {
            let out = &mut fvals[p * fdim..(p + 1) * fdim];
            sim.rhs_modal_pointcalc(xs[2 * p], xs[2 * p + 1], out);
        }
    }));
    match result {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

/// Cumulative trapezoidal integral: `dst[0]=0`, `dst[i] = dst[i-1] + (src[i-1]+src[i])/2·dt`.
/// Reproduces `Maths.cumtrapz!(out, y, δt)` (src/Maths.jl:323-328).
fn cumtrapz_slice_f64(src: &[f64], dst: &mut [f64], dt: f64) {
    let n = src.len();
    if n == 0 { return; }
    dst[0] = 0.0;
    for i in 1..n {
        dst[i] = dst[i - 1] + 0.5 * (src[i - 1] + src[i]) * dt;
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
pub trait NativeBackend {
    unsafe fn set_field(&mut self, data: *const c_double, n: size_t) -> i32;
    unsafe fn resync_field(&mut self, data: *const c_double, n: size_t) -> i32;
    unsafe fn get_field(&self, data: *mut c_double, n: size_t) -> i32;
    unsafe fn get_ks_stage(&self, idx: size_t, data: *mut c_double, n: size_t) -> i32;
    unsafe fn apply_prop(&mut self, y: *mut c_double, n: size_t, t1: f64, t2: f64) -> i32;
    unsafe fn debug_linop_at(&mut self, z: c_double, data: *mut c_double, n: size_t) -> i32;
    unsafe fn debug_beta1_at(&mut self, z: c_double, out_dens: *mut c_double, out_beta1: *mut c_double) -> i32;
    unsafe fn set_fftw_plans(&mut self, lib_path: *const c_char, n_time: size_t, n_time_over: size_t, is_real: c_int, flags: c_uint) -> i32;
    unsafe fn set_mode_avg_params(&mut self, n_time: size_t, n_time_over: size_t, towin: *const c_double, owin: *const c_double, sidx: *const u8, pre_re: *const c_double, pre_im: *const c_double, beta: *const c_double, kerr_fac: c_double, nlscale: c_double, sqrt_aeff: c_double) -> i32;
    unsafe fn set_mode_avg_noise(&mut self, noise: *const c_double, n: size_t) -> i32;
    unsafe fn set_zdep_mode_avg_params(&mut self, n_z: size_t, z_pts: *const c_double, p_pts: *const c_double, n_dspl: size_t, dspl_x: *const c_double, dspl_y: *const c_double, dspl_d: *const c_double, gamma: *const c_double, nwg_re: *const c_double, nwg_im: *const c_double, omega: *const c_double, model: c_uint, loss_on: c_uint, eps0_gamma3: c_double, omega0: c_double, gamma0: c_double, dgamma0: c_double, nwg0_re: c_double, nwg0_im: c_double, dnwg0_re: c_double, dnwg0_im: c_double) -> i32;
    unsafe fn set_plasma_params(&mut self, ion_ptr: *const crate::ionization::PptIonizationRate, ionpot: c_double, e_ratio: c_double, preionfrac: c_double, dt: c_double) -> i32;
    unsafe fn set_radial_params(&mut self, n_time: size_t, n_time_over: size_t, n_r: size_t, t_matrix: *const c_double, scale_fwd: c_double, scale_inv: c_double, towin: *const c_double, kerr_fac: c_double, m_re: *const c_double, m_im: *const c_double) -> i32;
    unsafe fn set_radial_noise(&mut self, noise: *const c_double, n: size_t) -> i32;
    unsafe fn set_raman_params(&mut self, omega: *const c_double, gamma: *const c_double, coupling: *const c_double, n_osc: size_t, dt: c_double, density: c_double, thg: c_int) -> i32;
    unsafe fn set_modal_params(&mut self, n_time: size_t, n_time_over: size_t, n_modes: size_t, npol: size_t, a: c_double, unm: *const c_double, inv_sqrt_n: *const c_double, order: *const i32, kind: *const u8, phi: *const c_double, full: u8, pol_select: *const u8, towin: *const c_double, kerr_fac: c_double, nlfac_re: *const c_double, nlfac_im: *const c_double, lib_path: *const c_char, rtol: c_double, atol: c_double, maxevals: size_t) -> i32;
    unsafe fn set_free_params(&mut self, n_time: size_t, n_time_over: size_t, n_y: size_t, n_x: size_t, flags: c_uint, towin: *const c_double, kerr_fac: c_double, m_re: *const c_double, m_im: *const c_double) -> i32;
    unsafe fn set_free_zdep_params(&mut self, flength: c_double, p0: c_double, p1: c_double, n_dspl: size_t, dspl_x: *const c_double, dspl_y: *const c_double, dspl_d: *const c_double, gamma: *const c_double, omega: *const c_double, omegawin: *const c_double, kperp2: *const c_double, sidx: *const u8, eps0_gamma3: c_double, omega0: c_double, gamma0: c_double, dgamma0: c_double) -> i32;
    unsafe fn set_modal_zdep_params(&mut self, flength: c_double, a0: c_double, n_a: size_t, a_x: *const c_double, a_y: *const c_double, a_d: *const c_double, omega: *const c_double, sidx: *const u8, model: u8, loss_on: u8, eco: *const c_double, vn_re: *const c_double, vn_im: *const c_double, omega0: c_double, ref_mode: size_t, eco0: *const c_double, deco0: *const c_double, v0_re: *const c_double, v0_im: *const c_double, dv0_re: *const c_double, dv0_im: *const c_double) -> i32;
    unsafe fn step(&mut self, yn: *mut Complex<f64>, t_old: f64, t_new: f64, dtn: f64, rtol: f64, atol: f64, safety: f64, max_dt: f64, min_dt: f64, errlast_in: f64, locextrap: i32, result: *mut NativeStepResult) -> i32;
}
pub struct NativeSim { pub backend: Box<dyn NativeBackend> }
impl NativeBackend for CpuNativeSim {
    unsafe fn set_field(&mut self, data: *const c_double, n: size_t) -> i32 {        let sim = self;

    if n != sim.n {
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
    unsafe fn resync_field(&mut self, data: *const c_double, n: size_t) -> i32 {        let sim = self;

    if n != sim.n {
        return -1;
    }
    let src = unsafe { std::slice::from_raw_parts(data as *const Complex<f64>, n) };
    sim.field.copy_from_slice(src);
    0
}
    unsafe fn get_field(&self, data: *mut c_double, n: size_t) -> i32 {        let sim = self;

    if n != sim.n {
        return -1;
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(data as *mut Complex<f64>, n) };
    dst.copy_from_slice(&sim.field);
    0
}
    unsafe fn get_ks_stage(&self, idx: size_t, data: *mut c_double, n: size_t) -> i32 {        let sim = self;

    if idx >= 7 || n != sim.n {
        return -1;
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(data as *mut Complex<f64>, n) };
    dst.copy_from_slice(&sim.ks[idx]);
    0
}
    unsafe fn apply_prop(&mut self, y: *mut c_double, n: size_t, t1: f64, t2: f64) -> i32 {        let sim = self;

    if n != sim.n {
        return -1;
    }
    let slice = unsafe { std::slice::from_raw_parts_mut(y as *mut Complex<f64>, n) };
    sim.ensure_linop_at(t2);
    apply_prop(slice, &sim.linop, t2 - t1);
    0
}
    unsafe fn debug_linop_at(&mut self, z: c_double, data: *mut c_double, n: size_t) -> i32 {        let sim = self;

    if n != sim.n { return -1; }
    sim.ensure_linop_at(z);
    let dst = unsafe { std::slice::from_raw_parts_mut(data as *mut Complex<f64>, n) };
    dst.copy_from_slice(&sim.linop);
    0
}
    unsafe fn debug_beta1_at(&mut self, z: c_double, out_dens: *mut c_double, out_beta1: *mut c_double) -> i32 {        let sim = self;

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
    unsafe fn set_fftw_plans(&mut self, lib_path: *const c_char, n_time: size_t, n_time_over: size_t, is_real: c_int, flags: c_uint) -> i32 {        let sim = self;

    
    let path_str = unsafe { CStr::from_ptr(lib_path).to_str().unwrap_or("") };
    let api = match FftwApi::load(Some(path_str)) {
        Ok(api) => api,
        Err(e) => {
            eprintln!("native_set_fftw_plans: failed to load FFTW: {}", e);
            return -2;
        }
    };
    
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
    unsafe fn set_mode_avg_params(&mut self, n_time: size_t, n_time_over: size_t, towin: *const c_double, owin: *const c_double, sidx: *const u8, pre_re: *const c_double, pre_im: *const c_double, beta: *const c_double, kerr_fac: c_double, nlscale: c_double, sqrt_aeff: c_double) -> i32 {        let s = self;

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
        s.pre = pre_re_sl.iter().zip(pre_im_sl.iter())
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
    0
}
    unsafe fn set_mode_avg_noise(&mut self, noise: *const c_double, n: size_t) -> i32 {
        let s = self;
        if noise.is_null() || n == 0 { return -1; }
        if n != s.n_time_over { return -2; }
        s.et_noise = unsafe { std::slice::from_raw_parts(noise, n) }.to_vec();
        s.has_noise = true;
        0
    }
    unsafe fn set_zdep_mode_avg_params(&mut self, n_z: size_t, z_pts: *const c_double, p_pts: *const c_double, n_dspl: size_t, dspl_x: *const c_double, dspl_y: *const c_double, dspl_d: *const c_double, gamma: *const c_double, nwg_re: *const c_double, nwg_im: *const c_double, omega: *const c_double, model: c_uint, loss_on: c_uint, eps0_gamma3: c_double, omega0: c_double, gamma0: c_double, dgamma0: c_double, nwg0_re: c_double, nwg0_im: c_double, dnwg0_re: c_double, dnwg0_im: c_double) -> i32 {        let s = self;

    if s.sidx.len() != s.n { return -3; }
    if n_z < 2 { return -4; }

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
    unsafe fn set_plasma_params(&mut self, ion_ptr: *const crate::ionization::PptIonizationRate, ionpot: c_double, e_ratio: c_double, preionfrac: c_double, dt: c_double) -> i32 {        let s = self;

    // Radial (Phase D.2): one independent plasma state per r-column, laid out
    // column-major `(n_time_over, n_r)` exactly like `radial_eto`/`radial_pto`
    // — `native_set_radial_params` must run first so `s.n_r` is already set.
    let n = if s.is_radial { s.n_time_over * s.n_r } else { s.n_time_over };
    if n == 0 { return -2; }
    s.plasma_ion_ptr   = ion_ptr;
    s.plasma_ionpot    = ionpot;
    s.plasma_e_ratio   = e_ratio;
    s.plasma_preionfrac = preionfrac;
    s.plasma_dt        = dt;
    s.plas_rate     = vec![0.0; n];
    s.plas_fraction = vec![0.0; n];
    s.plas_phase    = vec![0.0; n];
    s.plas_j        = vec![0.0; n];
    s.plas_p        = vec![0.0; n];
    s.has_plasma    = true;
    0
}
    unsafe fn set_radial_params(&mut self, n_time: size_t, n_time_over: size_t, n_r: size_t, t_matrix: *const c_double, scale_fwd: c_double, scale_inv: c_double, towin: *const c_double, kerr_fac: c_double, m_re: *const c_double, m_im: *const c_double) -> i32 {        let s = self;

    if n_r == 0 || s.n % n_r != 0 { return -1; }
    let n_spec = s.n / n_r;

    s.is_radial = true;
    s.n_r = n_r;
    s.n_time = n_time;
    s.n_time_over = n_time_over;
    s.n_spec = n_spec;
    s.n_spec_over = if s.is_real { n_time_over / 2 + 1 } else { n_time_over };

    let t_mat_sl = unsafe { std::slice::from_raw_parts(t_matrix, n_r * n_r) };
    s.qdht = Some(QdhtFfiHandle::new(t_mat_sl, n_r, scale_fwd, scale_inv, n_time_over));
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
    s.radial_m = m_re_sl.iter().zip(m_im_sl.iter())
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

    0
}
    unsafe fn set_radial_noise(&mut self, noise: *const c_double, n: size_t) -> i32 {
        let s = self;
        if noise.is_null() || n == 0 { return -1; }
        if n != s.n_time_over * s.n_r { return -2; }
        s.radial_et_noise = unsafe { std::slice::from_raw_parts(noise, n) }.to_vec();
        s.has_radial_noise = true;
        0
    }
    unsafe fn set_raman_params(&mut self, omega: *const c_double, gamma: *const c_double, coupling: *const c_double, n_osc: size_t, dt: c_double, density: c_double, thg: c_int) -> i32 {        let s = self;

    if s.n_time_over == 0 { return -2; }

    let omega_sl = unsafe { std::slice::from_raw_parts(omega, n_osc) };
    let gamma_sl = unsafe { std::slice::from_raw_parts(gamma, n_osc) };
    let coupling_sl = unsafe { std::slice::from_raw_parts(coupling, n_osc) };

    let oscillators: Vec<RamanOscillator> = (0..n_osc)
        .map(|i| RamanOscillator { omega: omega_sl[i], gamma: gamma_sl[i], coupling: coupling_sl[i] })
        .collect();

    // Radial (Phase D.4): one time-march per r-column, laid out column-major
    // `(n_time_over, n_r)` like `radial_eto`/`plas_*` — `solve()` resets its
    // internal oscillator state at entry (see `raman.rs`), so the *same*
    // solver instance can be called once per r-column with no cross-column
    // state leakage; only the scratch buffers need the extra width.
    let n = if s.is_radial { s.n_time_over * s.n_r } else { s.n_time_over };
    if n == 0 { return -2; }

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
    unsafe fn set_modal_params(&mut self, n_time: size_t, n_time_over: size_t, n_modes: size_t, npol: size_t, a: c_double, unm: *const c_double, inv_sqrt_n: *const c_double, order: *const i32, kind: *const u8, phi: *const c_double, full: u8, pol_select: *const u8, towin: *const c_double, kerr_fac: c_double, nlfac_re: *const c_double, nlfac_im: *const c_double, lib_path: *const c_char, rtol: c_double, atol: c_double, maxevals: size_t) -> i32 {        let s = self;

    if n_modes == 0 || npol == 0 || (npol != 1 && npol != 2) || s.n % n_modes != 0 {
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
    s.n_spec_over = if s.is_real { n_time_over / 2 + 1 } else { n_time_over };
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
    s.modal_nlfac = re_sl.iter().zip(im_sl.iter())
        .map(|(&r, &i)| Complex::new(r, i))
        .collect();

    s.cubature = Some(cubature);
    s.modal_rtol = rtol;
    s.modal_atol = atol;
    s.modal_maxevals = maxevals;

    s.modal_ems = vec![0.0; n_modes * npol];
    s.modal_erw = vec![Complex::new(0.0, 0.0); n_spec * npol];
    s.modal_erwo = vec![Complex::new(0.0, 0.0); s.n_spec_over * npol];
    s.modal_er = vec![0.0; n_time_over * npol];
    s.modal_pr = vec![0.0; n_time_over * npol];
    s.modal_prwo = vec![Complex::new(0.0, 0.0); s.n_spec_over * npol];
    s.modal_prw = vec![Complex::new(0.0, 0.0); n_spec * npol];
    s.modal_er_c = vec![Complex::new(0.0, 0.0); n_time_over * npol];
    s.modal_pr_c = vec![Complex::new(0.0, 0.0); n_time_over * npol];
    s.modal_emega = vec![Complex::new(0.0, 0.0); s.n];

    0
}
    unsafe fn set_free_params(&mut self, n_time: size_t, n_time_over: size_t, n_y: size_t, n_x: size_t, flags: c_uint, towin: *const c_double, kerr_fac: c_double, m_re: *const c_double, m_im: *const c_double) -> i32 {        let s = self;

    if n_y == 0 || n_x == 0 { return -1; }
    let n_cols = n_y * n_x;
    if s.n % n_cols != 0 { return -1; }
    let n_spec = s.n / n_cols;

    let n_spec_over_tmp = if s.is_real { n_time_over / 2 + 1 } else { n_time_over };

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
    s.free_m = m_re_sl.iter().zip(m_im_sl.iter())
        .map(|(&r, &i)| Complex::new(r, i))
        .collect();

    s.free_eoo = vec![Complex::new(0.0, 0.0); s.n_spec_over * n_cols];
    s.free_poo = vec![Complex::new(0.0, 0.0); s.n_spec_over * n_cols];

    0
}
    unsafe fn set_free_zdep_params(&mut self, flength: c_double, p0: c_double, p1: c_double, n_dspl: size_t, dspl_x: *const c_double, dspl_y: *const c_double, dspl_d: *const c_double, gamma: *const c_double, omega: *const c_double, omegawin: *const c_double, kperp2: *const c_double, sidx: *const u8, eps0_gamma3: c_double, omega0: c_double, gamma0: c_double, dgamma0: c_double) -> i32 {        let s = self;

    if s.n_spec == 0 || s.n_y == 0 || s.n_x == 0 { return -3; }

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
    s.zdep_free_sidx = unsafe { std::slice::from_raw_parts(sidx, s.n_spec) }.iter().map(|&b| b != 0).collect();
    s.zdep_free_kerr_fac_per_dens = eps0_gamma3;
    s.zdep_free_omega0 = omega0;
    s.zdep_free_gamma0 = gamma0;
    s.zdep_free_dgamma0 = dgamma0;
    s.zdep_free_last_z = f64::NAN;
    s.is_zdep_free = true;
    0
}
    unsafe fn set_modal_zdep_params(&mut self, flength: c_double, a0: c_double, n_a: size_t, a_x: *const c_double, a_y: *const c_double, a_d: *const c_double, omega: *const c_double, sidx: *const u8, model: u8, loss_on: u8, eco: *const c_double, vn_re: *const c_double, vn_im: *const c_double, omega0: c_double, ref_mode: size_t, eco0: *const c_double, deco0: *const c_double, v0_re: *const c_double, v0_im: *const c_double, dv0_re: *const c_double, dv0_im: *const c_double) -> i32 {        let s = self;

    if s.n_spec == 0 || s.n_modes == 0 { return -3; }
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
    s.zdep_modal_sidx = unsafe { std::slice::from_raw_parts(sidx, n_spec) }.iter().map(|&b| b != 0).collect();
    s.zdep_modal_model = model;
    s.zdep_modal_loss_on = loss_on != 0;
    s.zdep_modal_eco = unsafe { std::slice::from_raw_parts(eco, n_spec * n_modes) }.to_vec();
    s.zdep_modal_vn_re = unsafe { std::slice::from_raw_parts(vn_re, n_spec * n_modes) }.to_vec();
    s.zdep_modal_vn_im = unsafe { std::slice::from_raw_parts(vn_im, n_spec * n_modes) }.to_vec();
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
    unsafe fn step(&mut self, yn: *mut Complex<f64>, t_old: f64, t_new: f64, dtn: f64, rtol: f64, atol: f64, safety: f64, max_dt: f64, min_dt: f64, errlast_in: f64, locextrap: i32, result: *mut NativeStepResult) -> i32 {        let s = self;

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
    apply_prop(&mut s.ks[0], &s.linop, t_new - t_old);
    
    let dt = dtn;
    let t = t_new;
    
    for ii in 0..6 {
        s.ystage.copy_from_slice(&s.field);
        let (ks_slice, _) = s.ks.split_at(ii + 1);
        for (j, k_j) in ks_slice.iter().enumerate() {
            let bij = DP_B[ii][j];
            if bij != 0.0 {
                let scale = dt * bij;
                for (y, k) in s.ystage.iter_mut().zip(k_j.iter()) {
                    y.re += scale * k.re;
                    y.im += scale * k.im;
                }
            }
        }
        
        if s.is_free {
            let dt_prop = DP_NODES[ii] * dt;
            s.ensure_linop_at(t + dt_prop);
            s.ensure_free_norm_at(t + dt_prop);
            let mut ystage_prop = s.ystage.clone();
            apply_prop(&mut ystage_prop, &s.linop, dt_prop);
            if s.is_real {
                s.rhs_free(ii + 1, &ystage_prop);
            } else {
                s.rhs_free_env(ii + 1, &ystage_prop);
            }
            apply_prop(&mut s.ks[ii + 1], &s.linop, -dt_prop);
        } else if s.is_modal {
            let dt_prop = DP_NODES[ii] * dt;
            s.ensure_linop_at(t + dt_prop);
            s.ensure_modal_linop_at(t + dt_prop);
            let mut ystage_prop = s.ystage.clone();
            apply_prop(&mut ystage_prop, &s.linop, dt_prop);
            s.rhs_modal(ii + 1, &ystage_prop);
            apply_prop(&mut s.ks[ii + 1], &s.linop, -dt_prop);
        } else if s.is_radial {
            let dt_prop = DP_NODES[ii] * dt;
            s.ensure_linop_at(t + dt_prop);
            let mut ystage_prop = s.ystage.clone();
            apply_prop(&mut ystage_prop, &s.linop, dt_prop);
            if s.is_real {
                s.rhs_radial(ii + 1, &ystage_prop);
            } else {
                s.rhs_radial_env(ii + 1, &ystage_prop);
            }
            apply_prop(&mut s.ks[ii + 1], &s.linop, -dt_prop);
        } else if s.n_time_over > 0 && (s.fft_r2c_over.is_some() || s.fft_c2c_over.is_some()) {
            let dt_prop = DP_NODES[ii] * dt;
            s.ensure_linop_at(t + dt_prop);
            let mut ystage_prop = s.ystage.clone();
            apply_prop(&mut ystage_prop, &s.linop, dt_prop);
            if s.is_real {
                s.rhs_mode_avg_real(ii + 1, &ystage_prop);
            } else {
                s.rhs_mode_avg_env(ii + 1, &ystage_prop);
            }
            apply_prop(&mut s.ks[ii + 1], &s.linop, -dt_prop);
        } else {
            for k in 0..n {
                s.ks[ii+1][k] = Complex::new(0.0, 0.0);
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
        let (ks0, ks1, ks2, ks3, ks4, ks5, ks6) = (&ks_slice[0], &ks_slice[1], &ks_slice[2], &ks_slice[3], &ks_slice[4], &ks_slice[5], &ks_slice[6]);
        
        for (((((yn, f), k0), k1), k2), (((k3, k4), k5), k6)) in yn_sl.iter_mut().zip(&s.field)
            .zip(ks0).zip(ks1).zip(ks2)
            .zip(ks3.iter().zip(ks4).zip(ks5).zip(ks6)) {
            yn.re = f.re + b0 * k0.re + b1 * k1.re + b2 * k2.re + b3 * k3.re + b4 * k4.re + b5 * k5.re + b6 * k6.re;
            yn.im = f.im + b0 * k0.im + b1 * k1.im + b2 * k2.im + b3 * k3.im + b4 * k4.im + b5 * k5.im + b6 * k6.im;
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
    let (ks0, ks1, ks2, ks3, ks4, ks5, ks6) = (&ks_slice[0], &ks_slice[1], &ks_slice[2], &ks_slice[3], &ks_slice[4], &ks_slice[5], &ks_slice[6]);
    
    for ((((y, k0), k1), k2), (((k3, k4), k5), k6)) in s.yerr.iter_mut()
        .zip(ks0).zip(ks1).zip(ks2)
        .zip(ks3.iter().zip(ks4).zip(ks5).zip(ks6)) {
        y.re = e0 * k0.re + e1 * k1.re + e2 * k2.re + e3 * k3.re + e4 * k4.re + e5 * k5.re + e6 * k6.re;
        y.im = e0 * k0.im + e1 * k1.im + e2 * k2.im + e3 * k3.im + e4 * k4.im + e5 * k5.im + e6 * k6.im;
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
        apply_prop(yn_sl, &s.linop, tn_new - t);
        s.field.copy_from_slice(yn_sl);
    } else {
        yn_sl.copy_from_slice(&s.field);
        tn_new = t_new;
        s.ensure_linop_at(tn_new);
        s.ensure_free_norm_at(tn_new);
        s.ensure_modal_linop_at(tn_new);
        apply_prop(yn_sl, &s.linop, tn_new - t); // no-op since t == tn_new
    }
    
    unsafe {
        *result = NativeStepResult {
            ok: ok_final as i32, dt, t, tn: tn_new, dtn: dtn_new, err, errlast: errlast_new,
        };
    }
    
    0
}
}

/// GPU-resident stepper V1 (`CudaNativeSim`) covers only mode-averaged,
/// Kerr-only, RealGrid propagation (see `cuda_native.rs`'s `NativeBackend`
/// impl — every other geometry/physics method returns -1). Verified against
/// the Julia oracle on real CUDA hardware for that scope (BACKLOG.md's
/// "GPU-resident stepper" entry); everything outside it remains unimplemented,
/// not just untested. Refuses to initialize unless this env var is
/// explicitly set, so it can never be reached by accident via default
/// dispatch — Julia's own eligibility guard (`RK45._gpu_native_eligible`)
/// checks the same env var independently before ever calling this.
const CUDA_NATIVE_OPT_IN_VAR: &str = "LUNA_USE_RUST_CUDA_NATIVE";

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
            "Luna-Rust warning: GPU-resident stepper (CudaNativeSim) is experimental \
             (mode-averaged Kerr-only) — refusing to initialize. Set {}=1 to opt in (see \
             BACKLOG.md).",
            CUDA_NATIVE_OPT_IN_VAR
        );
        return std::ptr::null_mut();
    }
    eprintln!(
        "Luna-Rust warning: {}=1 — using the GPU-resident stepper (mode-averaged Kerr-only; \
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_native_sim(
    linop: *const c_double,
    n: size_t,
) -> *mut NativeSim {
    if linop.is_null() || n == 0 {
        return std::ptr::null_mut();
    }
    // ComplexF64 is two contiguous f64 (re, im); reinterpret the f64* as Complex.
    let linop_sl = unsafe {
        std::slice::from_raw_parts(linop as *const Complex<f64>, n)
    };
    let result = std::panic::catch_unwind(|| CpuNativeSim::new(n, linop_sl));
    match result {
        Ok(h) => Box::into_raw(Box::new(NativeSim { backend: Box::new(h) })),
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
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

/// Copy `n` `ComplexF64` from Julia into the resident field buffer.
///
/// Returns 0 on success, -1 on null/length mismatch.
///
/// # Safety
/// `sim` must be valid; `data` must be valid for `n` `ComplexF64` reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_field(
    sim: *mut NativeSim,
    data: *const c_double,
    n: size_t,
) -> i32 {
    if sim.is_null() { return -1; }
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
    if sim.is_null() { return -1; }
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
pub unsafe extern "C" fn get_field(
    sim: *const NativeSim,
    data: *mut c_double,
    n: size_t,
) -> i32 {
    if sim.is_null() { return -1; }
    let s = unsafe { & *sim };
    unsafe { s.backend.get_field(data, n) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_ks_stage(
    sim: *const NativeSim,
    idx: size_t,
    data: *mut c_double,
    n: size_t,
) -> i32 {
    if sim.is_null() { return -1; }
    let s = unsafe { & *sim };
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
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.apply_prop(y, n, t1, t2) }
}


/// Debug getter (mirrors `get_field`/`get_ks_stage`): force `ensure_linop_at(z)`
/// and copy out the resulting `self.linop`. Used only by Rust-vs-Julia
/// diagnostic scripts for Phase 7 (z-dependent linop) — not part of the hot
/// loop.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_debug_linop_at(
    sim: *mut NativeSim,
    z: c_double,
    data: *mut c_double,
    n: size_t,
) -> i32 {
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.debug_linop_at(z, data, n) }
}

/// Debug getter (Phase 7 diagnostics, used by `test_native_zdep_linop.jl`'s
/// unit test): force `ensure_linop_at(z)` and report `[dens, beta1]` at that
/// z, so the analytic β1(z) closed form (`docs/native-port/BETA1_ANALYTIC.md`)
/// can be checked directly against a BigFloat ground truth, independent of
/// the full linop/RHS machinery.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_debug_beta1_at(
    sim: *mut NativeSim,
    z: c_double,
    out_dens: *mut c_double,
    out_beta1: *mut c_double,
) -> i32 {
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.debug_beta1_at(z, out_dens, out_beta1) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn native_set_fftw_plans(
    sim: *mut NativeSim,
    lib_path: *const c_char,
    n_time: size_t,
    n_time_over: size_t,
    is_real: c_int,
    flags: c_uint,
) -> i32 {
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_fftw_plans(lib_path, n_time, n_time_over, is_real, flags) }
}

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
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_mode_avg_params(n_time, n_time_over, towin, owin, sidx, pre_re, pre_im, beta, kerr_fac, nlscale, sqrt_aeff) }
}

/// Wire the modified shot-noise time-domain field for the mode-averaged
/// RealGrid path (BACKLOG.md Phase I item 1). `noise` has length
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
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_mode_avg_noise(noise, n) }
}

/// Wire the z-dependent linop path (Phase 7 + BACKLOG.md Phase F item 3 —
/// mode-averaged, graded-core constant-radius `MarcatiliMode`, two-point OR
/// general multi-point piecewise pressure gradient, see MATH.md §3.5 and
/// `docs/native-port/BETA1_ANALYTIC.md`). Must be called **after**
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
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_zdep_mode_avg_params(n_z, z_pts, p_pts, n_dspl, dspl_x, dspl_y, dspl_d, gamma, nwg_re, nwg_im, omega, model, loss_on, eps0_gamma3, omega0, gamma0, dgamma0, nwg0_re, nwg0_im, dnwg0_re, dnwg0_im) }
}

/// Wire the PPT ionization handle for the plasma cumtrapz path (RealGrid only).
///
/// `ion_ptr` is a raw pointer to a `PptIonizationRate` already constructed by Julia
/// (via `init_ppt_ionization_lut`). Julia owns it; this call just borrows it for the
/// lifetime of the `NativeSim`. Must be called **after** `native_set_mode_avg_params`
/// (mode-averaged) or `native_set_radial_params` (radial, Phase D.2 — BACKLOG.md)
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
) -> i32 {
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_plasma_params(ion_ptr, ionpot, e_ratio, preionfrac, dt) }
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
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_radial_params(n_time, n_time_over, n_r, t_matrix, scale_fwd, scale_inv, towin, kerr_fac, m_re, m_im) }
}

/// Wire the modified shot-noise time-domain field for the radial
/// (`TransRadial`) RealGrid path (BACKLOG.md Phase I item 1). `noise` has
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
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_radial_noise(noise, n) }
}

/// Wire the Raman ADE solver as an additive term in `rhs_mode_avg_real`
/// or, since Phase D.4 (BACKLOG.md), `rhs_radial` (RealGrid, `thg=true`
/// only — see MATH.md §5.3).
///
/// Must be called **after** `native_set_mode_avg_params` or
/// `native_set_radial_params` (needs `n_time_over`/`n_r`/`is_radial` to size
/// the scratch buffers), before `set_field`.
///
/// # Arguments
/// * `omega`/`gamma`/`coupling` — per-oscillator SDO parameters (`Ω`,
///   `1/τ2ρ(1.0)`, `K`), same values `Interface._make_rust_raman_handle_from_response`
///   already extracts for the existing `LUNA_USE_RUST_RAMAN` FFI wiring.
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
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_raman_params(omega, gamma, coupling, n_osc, dt, density, thg) }
}

/// Wire the modal (`TransModal`) RHS: resident `libcubature` (`pcubature_v`,
/// `full=false` only) + analytic general-order Marcatili mode-field
/// synthesis (`:HE`/`:TE`/`:TM`, any mode order `n` — BACKLOG.md Phase E.1;
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
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_modal_params(n_time, n_time_over, n_modes, npol, a, unm, inv_sqrt_n, order, kind, phi, full, pol_select, towin, kerr_fac, nlfac_re, nlfac_im, lib_path, rtol, atol, maxevals) }
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
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_free_params(n_time, n_time_over, n_y, n_x, flags, towin, kerr_fac, m_re, m_im) }
}

/// Wire the z-dependent linop + normfun for free-space (`TransFree`), a
/// two-point pressure-gradient gas cell — Phase D.5 (BACKLOG.md). Mirrors
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
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_free_zdep_params(flength, p0, p1, n_dspl, dspl_x, dspl_y, dspl_d, gamma, omega, omegawin, kperp2, sidx, eps0_gamma3, omega0, gamma0, dgamma0) }
}

/// Wire the z-dependent modal linop — tapered/per-mode radius, BACKLOG.md
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
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.set_modal_zdep_params(flength, a0, n_a, a_x, a_y, a_d, omega, sidx, model, loss_on, eco, vn_re, vn_im, omega0, ref_mode, eco0, deco0, v0_re, v0_im, dv0_re, dv0_im) }
}

// ── Dormand-Prince constants (matching ffi.rs) ──────────────────────────────
pub const DP_B: [[f64; 6]; 6] = [
    [1.0/5.0,          0.0,              0.0,              0.0,             0.0,              0.0],
    [3.0/40.0,         9.0/40.0,         0.0,              0.0,             0.0,              0.0],
    [44.0/45.0,       -56.0/15.0,        32.0/9.0,         0.0,             0.0,              0.0],
    [19372.0/6561.0,  -25360.0/2187.0,   64448.0/6561.0,  -212.0/729.0,    0.0,              0.0],
    [9017.0/3168.0,   -355.0/33.0,       46732.0/5247.0,   49.0/176.0,    -5103.0/18656.0,   0.0],
    [35.0/384.0,       0.0,              500.0/1113.0,     125.0/192.0,   -2187.0/6784.0,    11.0/84.0],
];
pub const DP_B5: [f64; 7]     = [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0,
                              -92097.0/339200.0, 187.0/2100.0, 1.0/40.0];
pub const DP_ERREST: [f64; 7] = [-71.0/57600.0, 0.0, 71.0/16695.0, -71.0/1920.0,
                               17253.0/339200.0, -22.0/525.0, 1.0/40.0];
pub const DP_NODES: [f64; 6]  = [1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0];

#[repr(C)]
pub struct NativeStepResult {
    pub ok:      i32,
    pub dt:      f64,
    pub t:       f64,
    pub tn:      f64,
    pub dtn:     f64,
    pub err:     f64,
    pub errlast: f64,
}

fn apply_prop(y: &mut [Complex<f64>], linop: &[Complex<f64>], dt: f64) {
    for i in 0..y.len() {
        y[i] *= (linop[i] * dt).exp();
    }
}

fn weaknorm_c64(y_err: &[Complex<f64>], y: &[Complex<f64>], yn: &[Complex<f64>],
                rtol: f64, atol: f64) -> f64 {
    let (mut sy, mut syn, mut syerr) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..y_err.len() {
        sy    += y[i].norm_sqr();
        syn   += yn[i].norm_sqr();
        syerr += y_err[i].norm_sqr();
    }
    let errwt = f64::max(f64::max(sy.sqrt(), syn.sqrt()), atol);
    syerr.sqrt() / rtol / errwt
}

pub(crate) fn stepcontrol_pi(ok: bool, err: f64, errlast: f64, dt: f64,
                  safety: f64, max_dt: f64, min_dt: f64)
    -> (f64, f64, bool)
{
    const BETA1: f64 = 3.0 / 5.0 / 5.0;
    const BETA2: f64 = -1.0 / 5.0 / 5.0;
    const EPS:   f64 = 0.8;
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
    if sim.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    unsafe { s.backend.step(yn, t_old, t_new, dtn, rtol, atol, safety, max_dt, min_dt, errlast_in, locextrap, result) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn field_roundtrip_is_bit_exact() {
        let n = 8;
        let linop: Vec<Complex<f64>> =
            (0..n).map(|i| Complex::new(i as f64, -(i as f64))).collect();
        let sim = unsafe {
            init_native_sim(linop.as_ptr() as *const c_double, n)
        };
        assert!(!sim.is_null());

        let input: Vec<Complex<f64>> =
            (0..n).map(|i| Complex::new(1.5 * i as f64, 0.25 * i as f64)).collect();
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
}
