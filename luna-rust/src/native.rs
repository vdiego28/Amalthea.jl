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
//!       Scope: RealGrid, `MarcatiliMode{<:Number,...}` (constant radius)
//!       with `kind=:HE, n=1` only (needs only `besselj(0,·)`/`besselj(1,·)`,
//!       already in `diffraction.rs`), Kerr-only, `shotnoise=false`. See
//!       MATH.md §3.3 for the full design and deferred-scope list.

use num_complex::Complex;
use libc::{c_char, c_double, c_int, c_uint, size_t, c_void};
use crate::fftw::{FftwApi, ComplexFft1d, RealFft1d, RealFft3d};
use crate::ffi::QdhtFfiHandle;
use crate::raman::{TimeDomainRamanSolver, RamanOscillator};
use crate::cubature::CubatureApi;
use crate::diffraction::j0;
use std::ffi::CStr;

/// Resident simulation state. One per `solve`. Opaque to Julia.
///
/// Buffers are sized to `n` (the spectral length `Nω`) at construction and are
/// never reallocated during the solve — the hot loop is allocation-free.
pub struct NativeSim {
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

    // ── Phase 4: Raman (RealGrid, thg=true, additive term in rhs_mode_avg_real) ──
    pub has_raman: bool,
    pub raman_solver: Option<TimeDomainRamanSolver>,
    /// Constant-medium density (unscaled, unlike `kerr_fac` which folds in ε₀·γ3).
    pub raman_density: f64,
    pub raman_intensity: Vec<f64>,
    pub raman_p: Vec<f64>,

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
    /// Per-mode rotation angle `ϕ`, length `n_modes`.
    pub modal_phi: Vec<f64>,
    /// Per-polarisation-column selector: 0 → `sin(ϕ)` (x), 1 → `cos(ϕ)` (y).
    /// Length `npol`.
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
}

impl NativeSim {
    fn new(n: usize, linop: &[Complex<f64>]) -> Self {
        let z = || vec![Complex::new(0.0, 0.0); n];
        NativeSim {
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
            n_r: 0,
            qdht: None,
            qdht_scale_fwd: 1.0,
            qdht_scale_inv: 1.0,
            radial_m: Vec::new(),
            radial_eto: Vec::new(),
            radial_pto: Vec::new(),
            radial_eoo: Vec::new(),
            radial_poo: Vec::new(),
            has_raman: false,
            raman_solver: None,
            raman_density: 0.0,
            raman_intensity: Vec::new(),
            raman_p: Vec::new(),
            is_modal: false,
            n_modes: 0,
            npol: 0,
            modal_a: 0.0,
            modal_unm: Vec::new(),
            modal_inv_sqrt_n: Vec::new(),
            modal_phi: Vec::new(),
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
            modal_emega: Vec::new(),
            is_free: false,
            n_y: 0,
            n_x: 0,
            fft_r2c_3d: None,
            free_m: Vec::new(),
            free_fft_norm_over: 1.0,
            free_eto: Vec::new(),
            free_pto: Vec::new(),
            free_eoo: Vec::new(),
            free_poo: Vec::new(),
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

    /// Raman polarisation via the resident time-domain ADE solver (`thg=true`:
    /// intensity = E², no Hilbert transform), added to `self.pto` in-place.
    /// Reproduces `(R::RamanPolar)(out, Et, ρ)` for `RamanPolarField`
    /// (src/Nonlinear.jl:357-431) — see MATH.md §5.3.
    fn apply_raman_real(&mut self) {
        let n = self.n_time_over;
        for i in 0..n {
            let e = self.eto[i];
            self.raman_intensity[i] = e * e;
        }
        if let Some(ref mut solver) = self.raman_solver {
            solver.solve(&self.raman_intensity, &mut self.raman_p);
        }
        let rho = self.raman_density;
        for i in 0..n {
            self.pto[i] += rho * self.eto[i] * self.raman_p[i];
        }
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

        // ── Step 3: Kerr (KerrScalar!): Pto += kerr_fac · Eto³, pointwise (t,r) ───
        self.radial_pto.fill(0.0);
        for i in 0..(n_time_over * n_r) {
            let e = self.radial_eto[i];
            self.radial_pto[i] += self.kerr_fac * e * e * e;
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
        // the C library re-enters Rust via `modal_integrand_v`, which
        // reconstructs a fresh `&mut NativeSim` from the raw `self` pointer —
        // holding any live `&`/`&mut` borrow of a `self` field across that
        // call (including a view into `self.ks[idx]`, hence the separate
        // `valbuf` written back afterward) would alias it.
        let cubature = self.cubature.take().expect("rhs_modal called before native_set_modal_params");
        let rc = unsafe {
            cubature.pcubature_v(
                fdim,
                modal_integrand_v,
                self as *mut NativeSim as *mut c_void,
                0.0, self.modal_a,
                self.modal_maxevals,
                self.modal_atol, self.modal_rtol,
                &mut valbuf, &mut errbuf,
            )
        };
        self.cubature = Some(cubature);
        if rc != 0 {
            eprintln!("rhs_modal: pcubature_v returned {}", rc);
        }

        let ks_f64: &mut [f64] = unsafe {
            std::slice::from_raw_parts_mut(self.ks[idx].as_mut_ptr() as *mut f64, fdim)
        };
        ks_f64.copy_from_slice(&valbuf);
    }

    /// Per-node integrand: mirrors `Erω_to_Prω!` + the mode back-projection +
    /// polar Jacobian in Julia's `pointcalc!` (`src/NonlinearRHS.jl:363-399`)
    /// for a single radial coordinate `r`, `θ=0` fixed (the `full=false`
    /// branch always evaluates at `θ=0` — the azimuthal dependence is
    /// carried analytically in the mode-field formula, not integrated
    /// numerically). Writes the packed real/imag `Prmω[·,·]·2πr` into `out`
    /// (length `2·n_spec·n_modes`).
    fn rhs_modal_pointcalc(&mut self, r: f64, out: &mut [f64]) {
        let n_modes = self.n_modes;
        let npol = self.npol;
        let n_spec = self.n_spec;
        let n_spec_over = self.n_spec_over;
        let n_time_over = self.n_time_over;

        if r <= 0.0 || r >= self.modal_a {
            out.fill(0.0);
            return;
        }

        // ── mode field synthesis at (r, θ=0): Exy = J0(r·unm/a)/√N · (sin ϕ, cos ϕ) ──
        for m in 0..n_modes {
            let x = r * self.modal_unm[m] / self.modal_a;
            let base = j0(x) * self.modal_inv_sqrt_n[m];
            let phi = self.modal_phi[m];
            for (p, &sel) in self.modal_pol_select.iter().enumerate() {
                self.modal_ems[m * npol + p] = base * if sel == 0 { phi.sin() } else { phi.cos() };
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

        // ── to_time! per polarisation column (RealGrid r2c, looped rank-1 plan) ──
        let scale_fwd = (n_spec_over - 1) as f64 / (n_spec - 1) as f64;
        self.modal_erwo.fill(Complex::new(0.0, 0.0));
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

        // ── ωwin + norm! (precomputed modal_nlfac, per ω, same for every pol column) ──
        for p in 0..npol {
            let col = &mut self.modal_prw[p * n_spec..(p + 1) * n_spec];
            for i in 0..n_spec {
                col[i] *= self.modal_nlfac[i];
            }
        }

        // ── back-projection: Prmω[iω,m] = Σ_p Prω[iω,p]·Ems[m,p] ─────────────────
        let pre = 2.0 * std::f64::consts::PI * r;
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
        let sim = unsafe { &mut *(fdata as *mut NativeSim) };
        let npt = npt as usize;
        let fdim = fdim as usize;
        let xs = unsafe { std::slice::from_raw_parts(x, npt) };
        let fvals = unsafe { std::slice::from_raw_parts_mut(fval, npt * fdim) };
        for p in 0..npt {
            let out = &mut fvals[p * fdim..(p + 1) * fdim];
            sim.rhs_modal_pointcalc(xs[p], out);
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
    let result = std::panic::catch_unwind(|| NativeSim::new(n, linop_sl));
    match result {
        Ok(h) => Box::into_raw(Box::new(h)),
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
    if sim.is_null() || data.is_null() {
        return -1;
    }
    let sim = unsafe { &mut *sim };
    if n != sim.n {
        return -1;
    }
    let src = unsafe { std::slice::from_raw_parts(data as *const Complex<f64>, n) };
    sim.field.copy_from_slice(src);
    
    if sim.is_free {
        let field = sim.field.clone();
        sim.rhs_free(0, &field);
    } else if sim.is_modal {
        let field = sim.field.clone();
        sim.rhs_modal(0, &field);
    } else if sim.is_radial {
        let field = sim.field.clone();
        sim.rhs_radial(0, &field);
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
    if sim.is_null() || data.is_null() {
        return -1;
    }
    let sim = unsafe { &*sim };
    if n != sim.n {
        return -1;
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(data as *mut Complex<f64>, n) };
    dst.copy_from_slice(&sim.field);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_ks_stage(
    sim: *const NativeSim,
    idx: size_t,
    data: *mut c_double,
    n: size_t,
) -> i32 {
    if sim.is_null() || data.is_null() {
        return -1;
    }
    let sim = unsafe { &*sim };
    if idx >= 7 || n != sim.n {
        return -1;
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(data as *mut Complex<f64>, n) };
    dst.copy_from_slice(&sim.ks[idx]);
    0
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
    if sim.is_null() || lib_path.is_null() { return -1; }
    let sim = unsafe { &mut *sim };
    
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
    0
}

/// Wire the PPT ionization handle for the plasma cumtrapz path (RealGrid only).
///
/// `ion_ptr` is a raw pointer to a `PptIonizationRate` already constructed by Julia
/// (via `init_ppt_ionization_lut`). Julia owns it; this call just borrows it for the
/// lifetime of the `NativeSim`. Must be called **after** `native_set_mode_avg_params`.
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
    if sim.is_null() || ion_ptr.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    let n = s.n_time_over;
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

/// Wire the radial (`TransRadial`) RHS: resident QDHT + scalar Kerr.
///
/// Must be called **after** `native_set_fftw_plans` (which must be called with
/// `is_real=1` — Phase 3's first gate is RealGrid-only). Sets `sim.is_radial`,
/// so this determines the RHS branch `native_step` takes.
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
    if sim.is_null() || t_matrix.is_null() || m_re.is_null() || m_im.is_null() { return -1; }
    let s = unsafe { &mut *sim };
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

    s.radial_eto = vec![0.0; n_time_over * n_r];
    s.radial_pto = vec![0.0; n_time_over * n_r];
    s.radial_eoo = vec![Complex::new(0.0, 0.0); s.n_spec_over * n_r];
    s.radial_poo = vec![Complex::new(0.0, 0.0); s.n_spec_over * n_r];

    0
}

/// Wire the Raman ADE solver as an additive term in `rhs_mode_avg_real`
/// (RealGrid, `thg=true` only — see MATH.md §5.3).
///
/// Must be called **after** `native_set_mode_avg_params` (needs `n_time_over`
/// to size the scratch buffers), before `set_field`.
///
/// # Arguments
/// * `omega`/`gamma`/`coupling` — per-oscillator SDO parameters (`Ω`,
///   `1/τ2ρ(1.0)`, `K`), same values `Interface._make_rust_raman_handle_from_response`
///   already extracts for the existing `LUNA_USE_RUST_RAMAN` FFI wiring.
/// * `n_osc` — number of oscillators.
/// * `dt` — `grid.to[2] - grid.to[1]` (oversampled grid spacing).
/// * `density` — constant-medium density (raw, **not** folded with `ε₀·γ3`
///   like `kerr_fac` — Raman's accumulation formula needs it unscaled).
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
) -> i32 {
    if sim.is_null() || omega.is_null() || gamma.is_null() || coupling.is_null() || n_osc == 0 {
        return -1;
    }
    let s = unsafe { &mut *sim };
    if s.n_time_over == 0 { return -2; }

    let omega_sl = unsafe { std::slice::from_raw_parts(omega, n_osc) };
    let gamma_sl = unsafe { std::slice::from_raw_parts(gamma, n_osc) };
    let coupling_sl = unsafe { std::slice::from_raw_parts(coupling, n_osc) };

    let oscillators: Vec<RamanOscillator> = (0..n_osc)
        .map(|i| RamanOscillator { omega: omega_sl[i], gamma: gamma_sl[i], coupling: coupling_sl[i] })
        .collect();

    s.raman_solver = Some(TimeDomainRamanSolver::new(oscillators, dt));
    s.raman_density = density;
    s.raman_intensity = vec![0.0; s.n_time_over];
    s.raman_p = vec![0.0; s.n_time_over];
    s.has_raman = true;

    0
}

/// Wire the modal (`TransModal`) RHS: resident `libcubature` (`pcubature_v`,
/// `full=false` only) + analytic `HE, n=1` Marcatili mode-field synthesis.
/// See MATH.md §3.3 for the full design and scope restrictions.
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
/// * `unm`/`inv_sqrt_n`/`phi` — per-mode arrays, length `n_modes` (`unm`,
///   `1/√N(m)` precomputed closed-form, `ϕ`).
/// * `pol_select` — length `npol`, 0 → `sin ϕ` (x column), 1 → `cos ϕ` (y column).
/// * `towin` — length `n_time_over`, or null for all-ones.
/// * `kerr_fac` — `density · ε₀ · γ3` (same convention as every other phase).
/// * `nlfac_re`/`nlfac_im` — precomputed `ωwin[iω]·normfunc(ω[iω])`, length `n_spec` each.
/// * `lib_path` — path to the `libcubature` shared library
///   (`Cubature.Cubature_jll.libcubature` from Julia).
/// * `rtol`/`atol`/`maxevals` — passed straight through to `pcubature_v`.
///
/// Returns 0 on success, -1 on null/shape-mismatched args, -2 if `libcubature`
/// failed to load.
///
/// # Safety
/// `sim` must be valid; `unm`/`inv_sqrt_n`/`phi` must each be valid for
/// `n_modes` reads; `pol_select` for `npol` reads; `nlfac_re`/`nlfac_im` for
/// `n_spec` reads (`n_spec = sim.n / n_modes`); `towin`, if non-null, for
/// `n_time_over` reads; `lib_path` must be a valid null-terminated C string.
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
    phi: *const c_double,
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
    if sim.is_null() || unm.is_null() || inv_sqrt_n.is_null() || phi.is_null()
        || pol_select.is_null() || nlfac_re.is_null() || nlfac_im.is_null() || lib_path.is_null() {
        return -1;
    }
    let s = unsafe { &mut *sim };
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
    s.modal_phi = unsafe { std::slice::from_raw_parts(phi, n_modes) }.to_vec();
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
    s.modal_emega = vec![Complex::new(0.0, 0.0); s.n];

    0
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
    if sim.is_null() || m_re.is_null() || m_im.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    if n_y == 0 || n_x == 0 { return -1; }
    let n_cols = n_y * n_x;
    if s.n % n_cols != 0 { return -1; }
    let n_spec = s.n / n_cols;

    let n_spec_over_tmp = if s.is_real { n_time_over / 2 + 1 } else { n_time_over };

    let fft3d = match s.fftw_api.as_ref() {
        Some(api) => RealFft3d::new(api, n_time_over, n_y, n_x, flags),
        None => return -2,
    };

    s.is_free = true;
    s.n_time = n_time;
    s.n_time_over = n_time_over;
    s.n_spec = n_spec;
    s.n_spec_over = n_spec_over_tmp;
    s.n_y = n_y;
    s.n_x = n_x;
    s.fft_r2c_3d = Some(fft3d);
    s.free_fft_norm_over = 1.0 / (n_time_over * n_y * n_x) as f64;

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

    s.free_eto = vec![0.0; n_time_over * n_cols];
    s.free_pto = vec![0.0; n_time_over * n_cols];
    s.free_eoo = vec![Complex::new(0.0, 0.0); s.n_spec_over * n_cols];
    s.free_poo = vec![Complex::new(0.0, 0.0); s.n_spec_over * n_cols];

    0
}

// ── Dormand-Prince constants (matching ffi.rs) ──────────────────────────────
const DP_B: [[f64; 6]; 6] = [
    [1.0/5.0,          0.0,              0.0,              0.0,             0.0,              0.0],
    [3.0/40.0,         9.0/40.0,         0.0,              0.0,             0.0,              0.0],
    [44.0/45.0,       -56.0/15.0,        32.0/9.0,         0.0,             0.0,              0.0],
    [19372.0/6561.0,  -25360.0/2187.0,   64448.0/6561.0,  -212.0/729.0,    0.0,              0.0],
    [9017.0/3168.0,   -355.0/33.0,       46732.0/5247.0,   49.0/176.0,    -5103.0/18656.0,   0.0],
    [35.0/384.0,       0.0,              500.0/1113.0,     125.0/192.0,   -2187.0/6784.0,    11.0/84.0],
];
const DP_B5: [f64; 7]     = [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0,
                              -92097.0/339200.0, 187.0/2100.0, 1.0/40.0];
const DP_ERREST: [f64; 7] = [-71.0/57600.0, 0.0, 71.0/16695.0, -71.0/1920.0,
                               17253.0/339200.0, -22.0/525.0, 1.0/40.0];
const DP_NODES: [f64; 6]  = [1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0];

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

fn stepcontrol_pi(ok: bool, err: f64, errlast: f64, dt: f64,
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
    if sim.is_null() || yn.is_null() || result.is_null() { return -1; }
    let s = unsafe { &mut *sim };
    let n = s.n;
    
    // Evaluate!
    // y is in s.field. yn comes in via pointer, but we also maintain yn locally in Rust?
    // Wait, in `PreconStepper` y is `s.field`. yn is passed in.
    let yn_sl = unsafe { std::slice::from_raw_parts_mut(yn, n) };
    
    // s.yn .= s.y -> but wait, `s.field` is `y`.
    yn_sl.copy_from_slice(&s.field);
    
    // prop!(s.ks[1], s.t, s.tn)
    apply_prop(&mut s.ks[0], &s.linop, t_new - t_old);
    
    let dt = dtn;
    let t = t_new;
    
    for ii in 0..6 {
        s.ystage.copy_from_slice(&s.field);
        for jj in 0..=ii {
            let bij = DP_B[ii][jj];
            if bij != 0.0 {
                let scale = dt * bij;
                for k in 0..n {
                    s.ystage[k].re += scale * s.ks[jj][k].re;
                    s.ystage[k].im += scale * s.ks[jj][k].im;
                }
            }
        }
        
        if s.is_free {
            let dt_prop = DP_NODES[ii] * dt;
            let mut ystage_prop = s.ystage.clone();
            apply_prop(&mut ystage_prop, &s.linop, dt_prop);
            s.rhs_free(ii + 1, &ystage_prop);
            apply_prop(&mut s.ks[ii + 1], &s.linop, -dt_prop);
        } else if s.is_modal {
            let dt_prop = DP_NODES[ii] * dt;
            let mut ystage_prop = s.ystage.clone();
            apply_prop(&mut ystage_prop, &s.linop, dt_prop);
            s.rhs_modal(ii + 1, &ystage_prop);
            apply_prop(&mut s.ks[ii + 1], &s.linop, -dt_prop);
        } else if s.is_radial {
            let dt_prop = DP_NODES[ii] * dt;
            let mut ystage_prop = s.ystage.clone();
            apply_prop(&mut ystage_prop, &s.linop, dt_prop);
            s.rhs_radial(ii + 1, &ystage_prop);
            apply_prop(&mut s.ks[ii + 1], &s.linop, -dt_prop);
        } else if s.n_time_over > 0 && (s.fft_r2c_over.is_some() || s.fft_c2c_over.is_some()) {
            let dt_prop = DP_NODES[ii] * dt;
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
        yn_sl.copy_from_slice(&s.field);
        for jj in 0..7 {
            let b = DP_B5[jj];
            if b != 0.0 {
                let scale = dt * b;
                for k in 0..n {
                    yn_sl[k].re += scale * s.ks[jj][k].re;
                    yn_sl[k].im += scale * s.ks[jj][k].im;
                }
            }
        }
    }
    
    // Error estimate
    for k in 0..n { s.yerr[k] = Complex::new(0.0, 0.0); }
    for ii in 0..7 {
        let e = DP_ERREST[ii];
        if e != 0.0 {
            let scale = dt * e;
            for k in 0..n {
                s.yerr[k].re += scale * s.ks[ii][k].re;
                s.yerr[k].im += scale * s.ks[ii][k].im;
            }
        }
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
        apply_prop(yn_sl, &s.linop, tn_new - t);
        s.field.copy_from_slice(yn_sl);
    } else {
        yn_sl.copy_from_slice(&s.field);
        tn_new = t_new;
        apply_prop(yn_sl, &s.linop, tn_new - t); // no-op since t == tn_new
    }
    
    unsafe {
        *result = NativeStepResult {
            ok: ok_final as i32, dt, t, tn: tn_new, dtn: dtn_new, err, errlast: errlast_new,
        };
    }
    
    0
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
