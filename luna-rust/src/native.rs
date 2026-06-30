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
//! - [ ] Phase 2 — Plasma + EnvGrid Kerr (`rhs_*_env`, cumtrapz ×3, current
//!       assembly; `Kerr_env`/THG). Replaces `PlasmaCumtrapz` (Nonlinear.jl:161).

use num_complex::Complex;
use libc::{c_char, c_double, c_int, c_uint, size_t};
use crate::fftw::{FftwApi, ComplexFft1d, RealFft1d};
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
    
    if !sim.beta.is_empty() {
        let field = sim.field.clone();
        sim.rhs_mode_avg_real(0, &field);
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
    s.n_spec_over = n_time_over / 2 + 1;

    s.eto = vec![0.0; n_time_over];
    s.pto = vec![0.0; n_time_over];
    s.eoo = vec![Complex::new(0.0, 0.0); s.n_spec_over];
    s.poo = vec![Complex::new(0.0, 0.0); s.n_spec_over];

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
        
        if s.n_time_over > 0 && s.fft_r2c_over.is_some() {
            // RealGrid mode-averaged preconditioned RHS:
            // 1. ystage_prop = ystage * exp(linop * nodes[ii] * dt)
            let dt_prop = DP_NODES[ii] * dt;
            let mut ystage_prop = s.ystage.clone();
            apply_prop(&mut ystage_prop, &s.linop, dt_prop);
            
            // 2. ks[ii+1] = f!(ystage_prop)
            s.rhs_mode_avg_real(ii + 1, &ystage_prop);
            
            // 3. ks[ii+1] *= exp(linop * -nodes[ii] * dt)
            apply_prop(&mut s.ks[ii + 1], &s.linop, -dt_prop);
        } else {
            // Phase 0 / fallback
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
