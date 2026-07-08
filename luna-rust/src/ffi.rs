use num_complex::Complex;
use libc::{c_double, c_int, c_uint, size_t, c_void};
use crate::ionization::{PptIonizationRate, AdkIonizationRate};
use crate::raman::{RamanOscillator, TimeDomainRamanSolver};
use crate::dispersion::{ZeisbergerNeff, MarcatiliNeff};

// ─── QDHT batch transform FFI ─────────────────────────────────────────────────

/// Batch Hankel transform handle storing Julia's pre-computed T matrix.
///
/// Julia's `Hankel.QDHT` struct stores a real symmetric `T[n_r, n_r]` matrix (column-major)
/// and a scalar `scaleRK`. The forward transform `mul!(Y, Q, A)` computes
/// `Y = T × A * scaleRK` (applied along the r-axis of the 2-D `(n_time, n_r)` array);
/// the inverse `ldiv!(Y, Q, A)` uses `1/scaleRK` instead.
///
/// We store `T` transposed to row-major so that per-row dot products are cache-friendly,
/// and pre-allocate scratch space to avoid per-call heap allocation.
pub struct QdhtFfiHandle {
    t_row_major: Vec<f64>,  // T[r, s] stored row-major: index = r*n_r + s
    n_r: usize,
    scale_fwd: f64,         // scaleRK for mul!  (forward, r→k)
    scale_inv: f64,         // 1/scaleRK for ldiv! (inverse, k→r)
    scratch: Vec<f64>,      // 4 × n_r × capacity working space (a_re, a_im, b_re, b_im)
    capacity: usize,        // scratch capacity in n_time units
}

impl QdhtFfiHandle {
    /// Construct from Julia's column-major T matrix and scale factors.
    pub fn new(t_col_major: &[f64], n_r: usize, scale_fwd: f64, scale_inv: f64,
               n_time_hint: usize) -> Self {
        assert_eq!(t_col_major.len(), n_r * n_r);
        let mut t_row_major = vec![0.0f64; n_r * n_r];
        for r in 0..n_r {
            for s in 0..n_r {
                // Julia col-major: T[r, s] = t_col_major[r + n_r * s]
                t_row_major[r * n_r + s] = t_col_major[r + n_r * s];
            }
        }
        let capacity = n_time_hint.max(1);
        let scratch = vec![0.0f64; 4 * n_r * capacity];
        QdhtFfiHandle { t_row_major, n_r, scale_fwd, scale_inv, scratch, capacity }
    }

    fn ensure_capacity(&mut self, n_time: usize) {
        if n_time > self.capacity {
            self.capacity = n_time;
            self.scratch.resize(4 * self.n_r * self.capacity, 0.0);
        }
    }

    /// Apply batch real transform in-place.
    ///
    /// `data` is a Julia column-major `(n_time, n_r)` Float64 array:
    /// `data[t + n_time * r]` = A[t, r].  After the call, A ← scale × T × A
    /// (where multiplication is along the r axis, independently for each time t).
    pub fn apply_real(&mut self, data: &mut [f64], n_time: usize, scale: f64) {
        let n_r = self.n_r;
        let block = n_r * n_time;
        self.ensure_capacity(n_time);

        if let Ok(api) = crate::blas::get_blas_api() {
            // Use BLAS-3 dgemm directly on the col-major array
            self.scratch[..block].copy_from_slice(&data[..block]);
            unsafe {
                (api.cblas_dgemm)(
                    crate::blas::CBLAS_COL_MAJOR,
                    crate::blas::CBLAS_NO_TRANS,
                    crate::blas::CBLAS_NO_TRANS,
                    n_time as libc::c_int,
                    n_r as libc::c_int,
                    n_r as libc::c_int,
                    scale,
                    self.scratch.as_ptr(),
                    n_time as libc::c_int,
                    self.t_row_major.as_ptr(),
                    n_r as libc::c_int,
                    0.0,
                    data.as_mut_ptr(),
                    n_time as libc::c_int,
                );
            }
        } else {
            // Fallback to Rayon
            for s in 0..n_r {
                let col_start = n_time * s;
                for t in 0..n_time {
                    self.scratch[t * n_r + s] = data[t + col_start];
                }
            }
            {
                let t_rm = self.t_row_major.as_slice();
                let (a_rm, b_rm) = self.scratch[..2 * block].split_at_mut(block);
                use rayon::prelude::*;
                b_rm.par_chunks_mut(n_r).zip(a_rm.par_chunks(n_r)).for_each(|(b_row, a_row)| {
                    for r in 0..n_r {
                        let t_row = &t_rm[r * n_r .. (r + 1) * n_r];
                        let mut sum = 0.0f64;
                        for s in 0..n_r {
                            sum += t_row[s] * a_row[s];
                        }
                        b_row[r] = scale * sum;
                    }
                });
            }
            for r in 0..n_r {
                let col_start = n_time * r;
                for t in 0..n_time {
                    data[t + col_start] = self.scratch[block + t * n_r + r];
                }
            }
        }
    }

    /// Apply batch complex transform in-place on interleaved data.
    ///
    /// `data` is a Julia column-major `(n_time, n_r)` ComplexF64 array viewed as
    /// `f64` (2 × n_time × n_r elements):
    ///   Re(A[t, r]) = `data[2*t     + 2*n_time*r]`
    ///   Im(A[t, r]) = `data[2*t + 1 + 2*n_time*r]`
    pub fn apply_cplx(&mut self, data: &mut [f64], n_time: usize, scale: f64) {
        let n_r = self.n_r;
        let block2 = 2 * n_r * n_time; // total f64 elements
        self.ensure_capacity(n_time);

        if let Ok(api) = crate::blas::get_blas_api() {
            // Use BLAS-3 dgemm directly on the col-major complex array
            // Interleaved complex arrays act exactly as a 2*n_time x n_r real array.
            self.scratch[..block2].copy_from_slice(&data[..block2]);
            unsafe {
                (api.cblas_dgemm)(
                    crate::blas::CBLAS_COL_MAJOR,
                    crate::blas::CBLAS_NO_TRANS,
                    crate::blas::CBLAS_NO_TRANS,
                    (2 * n_time) as libc::c_int,
                    n_r as libc::c_int,
                    n_r as libc::c_int,
                    scale,
                    self.scratch.as_ptr(),
                    (2 * n_time) as libc::c_int,
                    self.t_row_major.as_ptr(),
                    n_r as libc::c_int,
                    0.0,
                    data.as_mut_ptr(),
                    (2 * n_time) as libc::c_int,
                );
            }
        } else {
            // Fallback to Rayon
            let block = n_r * n_time;
            for s in 0..n_r {
                let col_f = 2 * n_time * s;
                for t in 0..n_time {
                    self.scratch[t * n_r + s]         = data[2 * t     + col_f];
                    self.scratch[block + t * n_r + s] = data[2 * t + 1 + col_f];
                }
            }
            {
                let t_rm = self.t_row_major.as_slice();
                let (a_rm, temp) = self.scratch[..4 * block].split_at_mut(2 * block);
                let (a_rm_re, a_rm_im) = a_rm.split_at_mut(block);
                let (b_rm_re, b_rm_im) = temp.split_at_mut(block);
                use rayon::prelude::*;
                b_rm_re.par_chunks_mut(n_r)
                    .zip(b_rm_im.par_chunks_mut(n_r))
                    .zip(a_rm_re.par_chunks(n_r))
                    .zip(a_rm_im.par_chunks(n_r))
                    .for_each(|(((b_row_re, b_row_im), a_row_re), a_row_im)| {
                    for r in 0..n_r {
                        let t_row = &t_rm[r * n_r .. (r + 1) * n_r];
                        let mut sum_re = 0.0f64;
                        let mut sum_im = 0.0f64;
                        for s in 0..n_r {
                            let ts = t_row[s];
                            sum_re += ts * a_row_re[s];
                            sum_im += ts * a_row_im[s];
                        }
                        b_row_re[r] = scale * sum_re;
                        b_row_im[r] = scale * sum_im;
                    }
                });
            }
            for r in 0..n_r {
                let col_f = 2 * n_time * r;
                for t in 0..n_time {
                    data[2 * t     + col_f] = self.scratch[2 * block + t * n_r + r];
                    data[2 * t + 1 + col_f] = self.scratch[3 * block + t * n_r + r];
                }
            }
        }
    }
}

/// Initialise a QDHT batch-transform handle from Julia's pre-computed T matrix.
///
/// # Arguments
/// * `t_matrix`     – Julia column-major `T[n_r, n_r]` matrix (read-only)
/// * `n_r`          – number of radial grid points
/// * `scale_fwd`    – `scaleRK` from `Hankel.QDHT` (used by `mul!`)
/// * `scale_inv`    – `1/scaleRK` from `Hankel.QDHT` (used by `ldiv!`)
/// * `n_time_hint`  – expected `n_time`; scratch buffer pre-allocated to this size
///
/// Returns a heap-allocated `*mut QdhtFfiHandle`, or null on error (null `t_matrix`,
/// `n_r == 0`, or internal panic).  Free with [`free_qdht_ffi`].
///
/// # Safety
/// `t_matrix` must be non-null and valid for `n_r * n_r` reads for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_qdht_ffi(
    t_matrix:    *const c_double,
    n_r:         size_t,
    scale_fwd:   c_double,
    scale_inv:   c_double,
    n_time_hint: size_t,
) -> *mut QdhtFfiHandle {
    if t_matrix.is_null() || n_r == 0 {
        return std::ptr::null_mut();
    }
    let t_sl = unsafe { std::slice::from_raw_parts(t_matrix, n_r * n_r) };
    let result = std::panic::catch_unwind(|| {
        QdhtFfiHandle::new(t_sl, n_r, scale_fwd, scale_inv, n_time_hint)
    });
    match result {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(e) => {
            eprintln!("init_qdht_ffi: panic: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

/// Free a `QdhtFfiHandle`. Passing null is a no-op.
///
/// # Safety
/// `ptr` must be a valid, non-aliased pointer from [`init_qdht_ffi`] that has not been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_qdht_ffi(ptr: *mut QdhtFfiHandle) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

/// Batch forward Hankel transform (`mul!`) for a real-valued Julia array.
///
/// Applies `A[:,r] ← scaleRK × T × A[:,r]` in-place on the Julia column-major
/// `(n_time, n_r)` Float64 array pointed to by `data`.
///
/// # Returns
/// * `0`  success
/// * `-1` null pointer or `n_r` mismatch
/// * `-2` internal panic
///
/// # Safety
/// `data` must point to a live, aligned `f64[n_time * n_r]` array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdht_ffi_mul_real(
    ptr:    *mut QdhtFfiHandle,
    data:   *mut c_double,
    n_time: size_t,
    n_r:    size_t,
) -> c_int {
    if ptr.is_null() || data.is_null() { return -1; }
    let handle = unsafe { &mut *ptr };
    if n_r != handle.n_r { return -1; }
    let data_sl = unsafe { std::slice::from_raw_parts_mut(data, n_time * n_r) };
    let scale = handle.scale_fwd;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handle.apply_real(data_sl, n_time, scale);
    }));
    match result {
        Ok(()) => 0,
        Err(e) => { eprintln!("qdht_ffi_mul_real: panic: {:?}", e); -2 }
    }
}

/// Batch inverse Hankel transform (`ldiv!`) for a real-valued Julia array.
///
/// Applies `A[:,r] ← (1/scaleRK) × T × A[:,r]` in-place.
///
/// # Safety
/// Same as [`qdht_ffi_mul_real`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdht_ffi_ldiv_real(
    ptr:    *mut QdhtFfiHandle,
    data:   *mut c_double,
    n_time: size_t,
    n_r:    size_t,
) -> c_int {
    if ptr.is_null() || data.is_null() { return -1; }
    let handle = unsafe { &mut *ptr };
    if n_r != handle.n_r { return -1; }
    let data_sl = unsafe { std::slice::from_raw_parts_mut(data, n_time * n_r) };
    let scale = handle.scale_inv;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handle.apply_real(data_sl, n_time, scale);
    }));
    match result {
        Ok(()) => 0,
        Err(e) => { eprintln!("qdht_ffi_ldiv_real: panic: {:?}", e); -2 }
    }
}

/// Batch forward Hankel transform (`mul!`) for an interleaved complex Julia array.
///
/// `data` is a Julia column-major `(n_time, n_r)` ComplexF64 array accessed as `f64`:
///   `data[2*t     + 2*n_time*r]` = Re(A[t, r])
///   `data[2*t + 1 + 2*n_time*r]` = Im(A[t, r])
///
/// # Returns
/// `0` success, `-1` null / mismatch, `-2` panic.
///
/// # Safety
/// `data` must point to a live `f64[2 * n_time * n_r]` buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdht_ffi_mul_cplx(
    ptr:    *mut QdhtFfiHandle,
    data:   *mut c_double,
    n_time: size_t,
    n_r:    size_t,
) -> c_int {
    if ptr.is_null() || data.is_null() { return -1; }
    let handle = unsafe { &mut *ptr };
    if n_r != handle.n_r { return -1; }
    let data_sl = unsafe { std::slice::from_raw_parts_mut(data, 2 * n_time * n_r) };
    let scale = handle.scale_fwd;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handle.apply_cplx(data_sl, n_time, scale);
    }));
    match result {
        Ok(()) => 0,
        Err(e) => { eprintln!("qdht_ffi_mul_cplx: panic: {:?}", e); -2 }
    }
}

/// Batch inverse Hankel transform (`ldiv!`) for an interleaved complex Julia array.
///
/// # Safety
/// Same as [`qdht_ffi_mul_cplx`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qdht_ffi_ldiv_cplx(
    ptr:    *mut QdhtFfiHandle,
    data:   *mut c_double,
    n_time: size_t,
    n_r:    size_t,
) -> c_int {
    if ptr.is_null() || data.is_null() { return -1; }
    let handle = unsafe { &mut *ptr };
    if n_r != handle.n_r { return -1; }
    let data_sl = unsafe { std::slice::from_raw_parts_mut(data, 2 * n_time * n_r) };
    let scale = handle.scale_inv;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handle.apply_cplx(data_sl, n_time, scale);
    }));
    match result {
        Ok(()) => 0,
        Err(e) => { eprintln!("qdht_ffi_ldiv_cplx: panic: {:?}", e); -2 }
    }
}

/// In-place scales a complex double-precision array by a real factor.
/// This verifies zero-copy FFI communication between Julia and Rust.
///
/// # Safety
/// The caller must ensure that `data` points to a valid, aligned array of at least
/// `len` complex numbers (each layout-compatible with `Complex<f64>`, i.e., two contiguous `f64` values),
/// and that it is not null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn process_field_inplace(
    data: *mut c_double,
    len: size_t,
    factor: c_double,
) {
    if data.is_null() {
        return;
    }
    
    // Safety: Complex<f64> is represented as two f64 values (re, im),
    // which matches the binary layout of [f64; 2]. Therefore, we can cast
    // a *mut f64 to *mut Complex<f64> directly.
    let slice = unsafe {
        std::slice::from_raw_parts_mut(data as *mut Complex<f64>, len)
    };
    for val in slice.iter_mut() {
        val.re *= factor;
        val.im *= factor;
    }
}

// ─── PPT ionization LUT lifecycle ────────────────────────────────────────────

/// Create a `PptIonizationRate` from pre-sampled (E, rate) knot data.
///
/// # Arguments
/// * `e_min`        – lower cutoff field strength (V/m); fields below this return 0.
/// * `e_max`        – upper bound field strength (V/m); fields above this return an error.
/// * `n_points`     – number of knot points in `sample_e` / `sample_rate`.
/// * `sample_e`     – pointer to a contiguous `f64[n_points]` of field values (must be
///                    strictly increasing, in V/m).
/// * `sample_rate`  – pointer to a contiguous `f64[n_points]` of ionization rates (s⁻¹);
///                    zero-or-negative entries are mapped to the clamp value internally.
///
/// Returns a heap-allocated `*mut PptIonizationRate` (opaque to Julia).
/// The caller is responsible for freeing it with [`free_ppt_ionization_lut`].
/// Returns `null` on any error (null `sample_e`/`sample_rate`, or `n_points < 3`).
///
/// # Safety
/// `sample_e` and `sample_rate` must be non-null, valid for `n_points` reads,
/// aligned to `f64` (8 bytes), and live for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_ppt_ionization_lut(
    e_min: c_double,
    e_max: c_double,
    n_points: size_t,
    sample_e: *const c_double,
    sample_rate: *const c_double,
) -> *mut PptIonizationRate {
    if sample_e.is_null() || sample_rate.is_null() || n_points < 3 {
        return std::ptr::null_mut();
    }
    let e_slice    = unsafe { std::slice::from_raw_parts(sample_e,    n_points) };
    let rate_slice = unsafe { std::slice::from_raw_parts(sample_rate, n_points) };

    // Catch panics (e.g. non-monotone grid) so we never unwind across the FFI boundary.
    let result = std::panic::catch_unwind(|| {
        PptIonizationRate::from_samples(e_min, e_max, e_slice, rate_slice)
    });
    match result {
        Ok(lut) => Box::into_raw(Box::new(lut)),
        Err(e) => {
            eprintln!("init_ppt_ionization_lut: panic: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

/// Free a `PptIonizationRate` handle previously returned by [`init_ppt_ionization_lut`].
///
/// # Safety
/// `ptr` must be a valid, non-aliased pointer returned by [`init_ppt_ionization_lut`]
/// that has not yet been freed.  Passing `null` is safe (no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_ppt_ionization_lut(ptr: *mut PptIonizationRate) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

/// Create an `AdkIonizationRate` (BACKLOG.md Phase I item 3) from Julia's
/// own precomputed `IonRateADK` constants (`src/Ionisation.jl:131-174`) —
/// closed-form, no LUT/spline fitting at all.
///
/// Returns a heap-allocated `*mut AdkIonizationRate`. Free with
/// [`free_adk_ionization`].
///
/// # Safety
/// No pointer arguments; always safe to call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_adk_ionization(
    occupancy: c_double,
    omega_p: c_double,
    cn_sq: c_double,
    nstar: c_double,
    omega_t_prefac: c_double,
    thr: c_double,
    avfac: c_double,
) -> *mut AdkIonizationRate {
    Box::into_raw(Box::new(AdkIonizationRate {
        occupancy, omega_p, cn_sq, nstar, omega_t_prefac, thr, avfac,
    }))
}

/// Free an `AdkIonizationRate` handle previously returned by [`init_adk_ionization`].
///
/// # Safety
/// `ptr` must be a valid, non-aliased pointer returned by [`init_adk_ionization`]
/// that has not yet been freed. Passing `null` is safe (no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_adk_ionization(ptr: *mut AdkIonizationRate) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

// ─── Time-domain Raman ADE solver lifecycle ──────────────────────────────────

/// Create a `TimeDomainRamanSolver` from arrays of oscillator parameters.
///
/// # Arguments
/// * `omega`    – pointer to `f64[n_osc]` of oscillator frequencies Ω (rad/s)
/// * `gamma`    – pointer to `f64[n_osc]` of damping rates Γ = 1/τ₂ (rad/s)
/// * `coupling` – pointer to `f64[n_osc]` of coupling constants K
/// * `n_osc`    – number of oscillators (must be ≥ 1)
/// * `dt`       – time step (s); used to precompute matrix-exponential step coefficients
///
/// Returns a heap-allocated `*mut TimeDomainRamanSolver`.
/// Returns `null` on error (null pointers, n_osc==0, or a panic during coefficient computation).
/// Free with [`free_raman_solver`].
///
/// # Safety
/// `omega`, `gamma`, `coupling` must be non-null, valid for `n_osc` reads, aligned to `f64`,
/// and live for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_raman_solver(
    omega: *const c_double,
    gamma: *const c_double,
    coupling: *const c_double,
    n_osc: size_t,
    dt: c_double,
) -> *mut TimeDomainRamanSolver {
    if omega.is_null() || gamma.is_null() || coupling.is_null() || n_osc == 0 {
        return std::ptr::null_mut();
    }
    let omega_sl    = unsafe { std::slice::from_raw_parts(omega,    n_osc) };
    let gamma_sl    = unsafe { std::slice::from_raw_parts(gamma,    n_osc) };
    let coupling_sl = unsafe { std::slice::from_raw_parts(coupling, n_osc) };

    let result = std::panic::catch_unwind(|| {
        let oscillators: Vec<RamanOscillator> = (0..n_osc)
            .map(|i| RamanOscillator {
                omega:    omega_sl[i],
                gamma:    gamma_sl[i],
                coupling: coupling_sl[i],
            })
            .collect();
        TimeDomainRamanSolver::new(oscillators, dt)
    });
    match result {
        Ok(solver) => Box::into_raw(Box::new(solver)),
        Err(e) => {
            eprintln!("init_raman_solver: panic: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

/// Free a `TimeDomainRamanSolver` handle previously returned by [`init_raman_solver`].
///
/// # Safety
/// `ptr` must be a valid, non-aliased pointer from [`init_raman_solver`] that has not been freed.
/// Passing `null` is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_raman_solver(ptr: *mut TimeDomainRamanSolver) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

/// Solve the Raman ADE for a time-domain intensity array.
///
/// Drives the precomputed `TimeDomainRamanSolver` with the given intensity array and writes
/// the Raman polarisation response into `polarization`.  Internal oscillator state is reset
/// to zero at the start of each call (the call is stateless from Julia's perspective).
///
/// # Arguments
/// * `ptr`          – solver handle from [`init_raman_solver`] (must not be null)
/// * `intensity`    – pointer to `f64[n_t]` of intensity values (E² for carrier field,
///                    |E|²/2 for envelope)
/// * `polarization` – pointer to `f64[n_t]` output buffer; filled with Raman polarization
/// * `n_t`          – number of time samples
///
/// # Returns
/// * `0`  on success
/// * `-1` if any pointer is null
/// * `-2` on internal panic
///
/// # Safety
/// `intensity` must be readable and `polarization` writable for `n_t` `f64` values.
/// All pointers must be non-null, aligned, and valid for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn raman_solve(
    ptr: *mut TimeDomainRamanSolver,
    intensity: *const c_double,
    polarization: *mut c_double,
    n_t: size_t,
) -> c_int {
    if ptr.is_null() || intensity.is_null() || polarization.is_null() {
        return -1;
    }
    let solver       = unsafe { &mut *ptr };
    let intensity_sl = unsafe { std::slice::from_raw_parts(intensity,       n_t) };
    let polar_sl     = unsafe { std::slice::from_raw_parts_mut(polarization, n_t) };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        solver.solve(intensity_sl, polar_sl);
    }));
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("raman_solve: panic: {:?}", e);
            -2
        }
    }
}

// ─── Zeisberger neff dispersion lifecycle ────────────────────────────────────

/// Create a `ZeisbergerNeff` geometry handle from mode parameters.
///
/// The handle contains only the *geometric* information (Bessel root, kind, wall
/// thickness, loss); the refractive indices `nco`/`ncl` are supplied at each
/// call to [`zeisberger_neff_vector`] by Julia (using its multi-term Sellmeier).
///
/// # Arguments
/// * `unm`           – Bessel root `u_nm` (e.g. 2.4048 for HE11)
/// * `m_az`          – azimuthal order `m` (used in C-coefficient for HE/EH modes)
/// * `kind`          – 0=HE, 1=EH, 2=TE, 3=TM
/// * `wallthickness` – anti-resonant strut wall thickness (m)
/// * `loss_on`       – 0 → `Val{false}` (force real neff); non-zero → include imaginary term
/// * `loss_scale`    – scale factor applied to D·σ⁴ imaginary term (1.0 for `Val{true}`)
///
/// Returns a heap-allocated `*mut ZeisbergerNeff` (opaque to Julia).
/// Free with [`free_zeisberger_neff`].  Returns `null` on any panic.
///
/// # Safety
/// All arguments are plain scalars; this function is always safe to call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_zeisberger_neff(
    unm:           c_double,
    m_az:          c_double,
    kind:          c_uint,
    wallthickness: c_double,
    loss_on:       c_uint,
    loss_scale:    c_double,
) -> *mut ZeisbergerNeff {
    let result = std::panic::catch_unwind(|| {
        ZeisbergerNeff::new(
            unm,
            m_az,
            kind as u8,
            wallthickness,
            loss_on != 0,
            loss_scale,
        )
    });
    match result {
        Ok(handle) => Box::into_raw(Box::new(handle)),
        Err(e) => {
            eprintln!("init_zeisberger_neff: panic: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

/// Free a `ZeisbergerNeff` handle previously returned by [`init_zeisberger_neff`].
///
/// Passing `null` is a no-op.
///
/// # Safety
/// `ptr` must be a valid, non-aliased pointer from [`init_zeisberger_neff`]
/// that has not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_zeisberger_neff(ptr: *mut ZeisbergerNeff) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

/// Evaluate complex effective index over a frequency grid.
///
/// Julia calls this once per propagation step (z-value), passing the per-ω
/// core and cladding refractive indices from its own Sellmeier model. Rust
/// applies the full Zeisberger eq.(15) geometry and writes the complex
/// `neff(ω)` into the output arrays.
///
/// # Arguments
/// * `ptr`          – handle from [`init_zeisberger_neff`] (must not be null)
/// * `omegas`       – `f64[n]` angular frequencies (rad/s)
/// * `nco_re`       – `f64[n]` real parts of core refractive index
/// * `nco_im`       – `f64[n]` imaginary parts of core refractive index
/// * `ncl_re`       – `f64[n]` real parts of cladding refractive index
/// * `ncl_im`       – `f64[n]` imaginary parts of cladding refractive index
/// * `radius`       – core radius (m) at the current z position
/// * `neff_re_out`  – `f64[n]` output: real parts of neff
/// * `neff_im_out`  – `f64[n]` output: imaginary parts of neff
/// * `n`            – number of frequency points
///
/// # Returns
/// * `0`  on success
/// * `-1` if any pointer is null
/// * `-2` on internal panic
///
/// # Safety
/// All array pointers must be non-null, aligned, valid for `n` elements, and live
/// for the duration of this call.  `neff_re_out` and `neff_im_out` are written.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zeisberger_neff_vector(
    ptr:         *const ZeisbergerNeff,
    omegas:      *const c_double,
    nco_re:      *const c_double,
    nco_im:      *const c_double,
    ncl_re:      *const c_double,
    ncl_im:      *const c_double,
    radius:      c_double,
    neff_re_out: *mut c_double,
    neff_im_out: *mut c_double,
    n:           size_t,
) -> c_int {
    if ptr.is_null() || omegas.is_null() || nco_re.is_null() || nco_im.is_null()
        || ncl_re.is_null() || ncl_im.is_null() || neff_re_out.is_null() || neff_im_out.is_null()
    {
        return -1;
    }
    let handle      = unsafe { &*ptr };
    let omegas_sl   = unsafe { std::slice::from_raw_parts(omegas,      n) };
    let nco_re_sl   = unsafe { std::slice::from_raw_parts(nco_re,      n) };
    let nco_im_sl   = unsafe { std::slice::from_raw_parts(nco_im,      n) };
    let ncl_re_sl   = unsafe { std::slice::from_raw_parts(ncl_re,      n) };
    let ncl_im_sl   = unsafe { std::slice::from_raw_parts(ncl_im,      n) };
    let neff_re_sl  = unsafe { std::slice::from_raw_parts_mut(neff_re_out, n) };
    let neff_im_sl  = unsafe { std::slice::from_raw_parts_mut(neff_im_out, n) };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handle.neff_vector(
            omegas_sl, nco_re_sl, nco_im_sl, ncl_re_sl, ncl_im_sl,
            radius, neff_re_sl, neff_im_sl,
        );
    }));
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("zeisberger_neff_vector: panic: {:?}", e);
            -2
        }
    }
}

/// Initialise a [`MarcatiliNeff`] handle for hollow-capillary mode dispersion.
///
/// The `nwg_re`/`nwg_im` arrays hold the precomputed waveguide factor `nwg(ω)`
/// for each positive-frequency grid point (length `n`).  They are computed once
/// per propagation setup in Julia (using its multi-term Sellmeier for the cladding
/// material) and then stored in the handle.  Per propagation step only the
/// gas refractive index `nco(ω; z)` changes and is passed to
/// [`marcatili_neff_vector`].
///
/// # Arguments
/// * `nwg_re`    – real part of precomputed waveguide factor (`f64[n]`, read-only)
/// * `nwg_im`    – imaginary part of precomputed waveguide factor (`f64[n]`, read-only)
/// * `n`         – number of frequency points
/// * `model`     – `0` → `:full` model `sqrt(εco − nwg)`;  `1` → `:reduced`
/// * `loss_on`   – `0` → force `Im(neff) = 0`; non-zero → return full complex neff
///
/// # Returns
/// Heap-allocated `*mut MarcatiliNeff`, or `null` on error (null inputs, `n == 0`, panic).
/// Free with [`free_marcatili_neff`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_marcatili_neff(
    nwg_re:  *const c_double,
    nwg_im:  *const c_double,
    n:       size_t,
    model:   c_uint,
    loss_on: c_uint,
) -> *mut MarcatiliNeff {
    if nwg_re.is_null() || nwg_im.is_null() || n == 0 {
        return std::ptr::null_mut();
    }
    let re_sl = unsafe { std::slice::from_raw_parts(nwg_re, n) };
    let im_sl = unsafe { std::slice::from_raw_parts(nwg_im, n) };
    let result = std::panic::catch_unwind(|| {
        MarcatiliNeff::new(re_sl.to_vec(), im_sl.to_vec(), model as u8, loss_on != 0)
    });
    match result {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(e) => {
            eprintln!("init_marcatili_neff: panic: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

/// Free a [`MarcatiliNeff`] handle.  A null pointer is a safe no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_marcatili_neff(ptr: *mut MarcatiliNeff) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

/// Batch-evaluate Marcatili neff for the current propagation step.
///
/// # Arguments
/// * `ptr`          – handle from [`init_marcatili_neff`] (must not be null)
/// * `nco_re`       – real part of gas refractive index `nco(ω; z)` (`f64[n]`)
/// * `nco_im`       – imaginary part of `nco(ω; z)` (`f64[n]`)
/// * `neff_re_out`  – real part of `neff` output (`f64[n]`, overwritten)
/// * `neff_im_out`  – imaginary part of `neff` output (`f64[n]`, overwritten)
/// * `n`            – number of frequency points (must match handle)
///
/// # Returns
/// * `0`  on success.
/// * `-1` if any pointer is null.
/// * `-2` on internal panic.
///
/// # Safety
/// All pointers must be non-null, aligned, and valid for `n` `f64` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn marcatili_neff_vector(
    ptr:         *const MarcatiliNeff,
    nco_re:      *const c_double,
    nco_im:      *const c_double,
    neff_re_out: *mut c_double,
    neff_im_out: *mut c_double,
    n:           size_t,
) -> c_int {
    if ptr.is_null() || nco_re.is_null() || nco_im.is_null()
       || neff_re_out.is_null() || neff_im_out.is_null()
    {
        return -1;
    }
    let handle       = unsafe { &*ptr };
    let nco_re_sl    = unsafe { std::slice::from_raw_parts(nco_re, n) };
    let nco_im_sl    = unsafe { std::slice::from_raw_parts(nco_im, n) };
    let neff_re_sl   = unsafe { std::slice::from_raw_parts_mut(neff_re_out, n) };
    let neff_im_sl   = unsafe { std::slice::from_raw_parts_mut(neff_im_out, n) };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handle.neff_vector(nco_re_sl, nco_im_sl, neff_re_sl, neff_im_sl);
    }));
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("marcatili_neff_vector: panic: {:?}", e);
            -2
        }
    }
}

/// Evaluate ionization rates for a batch of field-strength values.
///
/// # Arguments
/// * `ptr`    – handle from [`init_ppt_ionization_lut`] (must not be null).
/// * `fields` – pointer to `f64[len]` of electric field strengths (V/m; may be signed).
/// * `rates`  – pointer to `f64[len]` output buffer; overwritten on success.
/// * `len`    – number of elements.
///
/// # Returns
/// * `0`  on success.
/// * `-1` if `ptr`, `fields`, or `rates` is null.
/// * `-2` if a field value is above the table's `e_max` (out-of-range error).
///
/// # Safety
/// `fields` must be readable and `rates` writable for `len` `f64` elements.
/// All pointers must be non-null, aligned, and valid for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ppt_ionization_rate_vector(
    ptr: *const PptIonizationRate,
    fields: *const c_double,
    rates: *mut c_double,
    len: size_t,
) -> c_int {
    if ptr.is_null() || fields.is_null() || rates.is_null() {
        return -1;
    }
    let lut       = unsafe { &*ptr };
    let field_sl  = unsafe { std::slice::from_raw_parts(fields, len) };
    let rates_sl  = unsafe { std::slice::from_raw_parts_mut(rates, len) };

    match lut.rate_vector(field_sl, rates_sl) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("ppt_ionization_rate_vector: {}", e);
            -2
        }
    }
}

// ─── Interaction-picture PreconStepper FFI ────────────────────────────────────
//
// Mirrors Julia's RK45.step!(s::PreconStepper) exactly:
//   evaluate! → 5th-order yn → weaknorm → Lund PI → FSAL copy → prop!_maybe
//
// Callbacks use the Julia `unsafe_pointer_to_objref(userdata)` pattern:
//   fbar_fn(t1, t2, y_in, k_out, n, userdata)  -- interaction-picture RHS
//   prop_fn(t1, t2, y_inout, n, userdata)        -- forward propagator

/// Interaction-picture RHS: `k_out ← fbar!(y_in, t1, t2)`.
type FbarFn = unsafe extern "C" fn(
    t1:       f64,
    t2:       f64,
    y_in:     *const Complex<f64>,
    k_out:    *mut Complex<f64>,
    n:        usize,
    userdata: *mut c_void,
);

/// Forward propagator (in-place): `y_inout ← exp(L*(t2-t1)) * y_inout`.
type PropFn = unsafe extern "C" fn(
    t1:       f64,
    t2:       f64,
    y_inout:  *mut Complex<f64>,
    n:        usize,
    userdata: *mut c_void,
);

/// Result struct returned by [`precon_step_ffi`].  Must match Julia's `PreconStepResult`.
#[repr(C)]
pub struct PreconStepResult {
    pub ok:      i32,
    pub dt:      f64,
    pub t:       f64,
    pub tn:      f64,
    pub dtn:     f64,
    pub err:     f64,
    pub errlast: f64,
}

/// Internal scratch buffers for the interaction-picture Dormand-Prince step.
pub struct PreconStepFfiHandle {
    y_stage: Vec<Complex<f64>>,
    y_err:   Vec<Complex<f64>>,
    _n:       usize,
}

// ── Dormand-Prince 5(4) Butcher tableau (matches Julia src/dopri.jl) ──────────

const DP_B: [[f64; 6]; 6] = [
    [1.0/5.0,          0.0,              0.0,              0.0,             0.0,              0.0],
    [3.0/40.0,         9.0/40.0,         0.0,              0.0,             0.0,              0.0],
    [44.0/45.0,       -56.0/15.0,        32.0/9.0,         0.0,             0.0,              0.0],
    [19372.0/6561.0,  -25360.0/2187.0,   64448.0/6561.0,  -212.0/729.0,    0.0,              0.0],
    [9017.0/3168.0,   -355.0/33.0,       46732.0/5247.0,   49.0/176.0,    -5103.0/18656.0,   0.0],
    [35.0/384.0,       0.0,              500.0/1113.0,     125.0/192.0,   -2187.0/6784.0,    11.0/84.0],
];
const DP_NODES: [f64; 6]  = [1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0];
const DP_B5: [f64; 7]     = [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0,
                              -92097.0/339200.0, 187.0/2100.0, 1.0/40.0];
// errest = b5 - b4
const DP_ERREST: [f64; 7] = [-71.0/57600.0, 0.0, 71.0/16695.0, -71.0/1920.0,
                               17253.0/339200.0, -22.0/525.0, 1.0/40.0];

// ── Helper: Julia-compatible weaknorm ─────────────────────────────────────────

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

// ── Helper: Lund PI step controller (Julia's stepcontrolPI! + steplims!) ──────

fn stepcontrol_pi(ok: bool, err: f64, errlast: f64, dt: f64,
                  safety: f64, max_dt: f64, min_dt: f64)
    -> (f64, f64, bool)
{
    const BETA1: f64 = 3.0 / 5.0 / 5.0;
    const BETA2: f64 = -1.0 / 5.0 / 5.0;
    const EPS:   f64 = 0.8;
    let (mut dtn, errlast_new);
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

// ── FFI exports ───────────────────────────────────────────────────────────────

/// Allocate a `PreconStepFfiHandle` for arrays of `n` complex elements.
///
/// Returns null on `n == 0`.  Free with [`free_precon_step_ffi`].
///
/// # Safety
/// Always safe to call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_precon_step_ffi(n: usize) -> *mut PreconStepFfiHandle {
    if n == 0 { return std::ptr::null_mut(); }
    Box::into_raw(Box::new(PreconStepFfiHandle {
        y_stage: vec![Complex::new(0.0, 0.0); n],
        y_err:   vec![Complex::new(0.0, 0.0); n],
        _n: n,
    }))
}

/// Free a `PreconStepFfiHandle`.  Null is a no-op.
///
/// # Safety
/// `ptr` must be a valid, non-aliased pointer from [`init_precon_step_ffi`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_precon_step_ffi(ptr: *mut PreconStepFfiHandle) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

/// Execute one interaction-picture Dormand-Prince 5(4) adaptive step.
///
/// Mirrors Julia's `RK45.step!(s::PreconStepper)` exactly.
///
/// # Arguments
/// * `ptr`         – handle from [`init_precon_step_ffi`]
/// * `y`           – Julia's `s.y` (length-`n` `Complex<f64>`); overwritten with `s.yn`
/// * `yn`          – Julia's `s.yn`; holds 5th-order solution on return
/// * `k_ptrs`      – 7-element array of `*mut Complex<f64>` (Julia's `s.ks[1..7]`)
/// * `n`           – array length (number of complex elements)
/// * `t_old`       – `s.t` before this call
/// * `t_new`       – `s.tn` before this call (k1 propagated from `t_old` to `t_new`)
/// * `dtn`         – proposed step size
/// * `rtol`, `atol`, `safety`, `max_dt`, `min_dt` – stepper tolerances
/// * `errlast_in`  – `s.errlast` from previous accepted step (0 on first step)
/// * `locextrap`   – non-zero → use 5th-order `yn` (always true in Luna)
/// * `fbar_fn`, `prop_fn`, `userdata` – Julia callbacks
/// * `result`      – output struct (filled on success)
///
/// # Returns
/// `0` success · `-1` null-pointer · `-2` internal panic.
///
/// # Safety
/// All pointer arguments must be non-null and valid. `k_ptrs[0..7]` must each
/// point to a live `Complex<f64>[n]` buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn precon_step_ffi(
    ptr:         *mut PreconStepFfiHandle,
    y:           *mut Complex<f64>,
    yn:          *mut Complex<f64>,
    k_ptrs:      *const *mut Complex<f64>,
    n:           usize,
    t_old:       f64,
    t_new:       f64,
    dtn:         f64,
    rtol:        f64,
    atol:        f64,
    safety:      f64,
    max_dt:      f64,
    min_dt:      f64,
    errlast_in:  f64,
    locextrap:   i32,
    fbar_fn:     FbarFn,
    prop_fn:     PropFn,
    userdata:    *mut c_void,
    result:      *mut PreconStepResult,
) -> i32 {
    if ptr.is_null() || y.is_null() || yn.is_null() || k_ptrs.is_null() || result.is_null() {
        return -1;
    }
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        precon_step_inner(ptr, y, yn, k_ptrs, n, t_old, t_new, dtn,
                          rtol, atol, safety, max_dt, min_dt, errlast_in,
                          locextrap, fbar_fn, prop_fn, userdata, result)
    }));
    match res {
        Ok(rc) => rc,
        Err(e) => { eprintln!("precon_step_ffi: panic: {:?}", e); -2 }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn precon_step_inner(
    ptr:         *mut PreconStepFfiHandle,
    y:           *mut Complex<f64>,
    yn:          *mut Complex<f64>,
    k_ptrs:      *const *mut Complex<f64>,
    n:           usize,
    t_old:       f64,
    t_new:       f64,
    dtn:         f64,
    rtol:        f64,
    atol:        f64,
    safety:      f64,
    max_dt:      f64,
    min_dt:      f64,
    errlast_in:  f64,
    locextrap:   i32,
    fbar_fn:     FbarFn,
    prop_fn:     PropFn,
    userdata:    *mut c_void,
    result:      *mut PreconStepResult,
) -> i32 {
    unsafe {
        let h = &mut *ptr;

        // Collect the 7 k pointers
        let ks: [*mut Complex<f64>; 7] = [
            *k_ptrs.add(0), *k_ptrs.add(1), *k_ptrs.add(2), *k_ptrs.add(3),
            *k_ptrs.add(4), *k_ptrs.add(5), *k_ptrs.add(6),
        ];

        // ── evaluate! ─────────────────────────────────────────────────────────────
        // s.y .= s.yn
        std::ptr::copy_nonoverlapping(yn, y, n);

        // s.prop!(s.ks[1], s.t, s.tn)  — FSAL: propagate k1 from t_old to t_new
        prop_fn(t_old, t_new, ks[0], n, userdata);

        let dt = dtn;
        let t  = t_new;     // s.t = s.tn after evaluate!

        for ii in 0..6usize {
            // s.yn .= s.y  (use y_stage as scratch)
            std::ptr::copy_nonoverlapping(y, h.y_stage.as_mut_ptr(), n);

            // s.yn .+= dt * B[ii][jj] * ks[jj]  for jj in 0..=ii
            // Use real-scalar × component form to match Julia's `(real) .* complex` broadcast.
            for jj in 0..=ii {
                let bij = DP_B[ii][jj];
                if bij != 0.0 {
                    let scale = dt * bij;
                    for k in 0..n {
                        let ks_k = *ks[jj].add(k);
                        let p = h.y_stage.as_mut_ptr().add(k);
                        (*p).re += scale * ks_k.re;
                        (*p).im += scale * ks_k.im;
                    }
                }
            }

            // s.fbar!(s.ks[ii+1], s.yn, s.t, s.t + nodes[ii]*dt)
            fbar_fn(t, t + DP_NODES[ii] * dt, h.y_stage.as_ptr(), ks[ii + 1], n, userdata);
        }

        // ── 5th-order yn (locextrap) ──────────────────────────────────────────────
        if locextrap != 0 {
            std::ptr::copy_nonoverlapping(y, yn, n);
            for jj in 0..7usize {
                let b = DP_B5[jj];
                if b != 0.0 {
                    let scale = dt * b;
                    for k in 0..n {
                        let ks_k = *ks[jj].add(k);
                        (*yn.add(k)).re += scale * ks_k.re;
                        (*yn.add(k)).im += scale * ks_k.im;
                    }
                }
            }
        }

        // ── error estimate ────────────────────────────────────────────────────────
        for i in 0..n { h.y_err[i] = Complex::new(0.0, 0.0); }
        for ii in 0..7usize {
            let e = DP_ERREST[ii];
            if e != 0.0 {
                let scale = dt * e;
                for k in 0..n {
                    let ks_k = *ks[ii].add(k);
                    h.y_err[k].re += scale * ks_k.re;
                    h.y_err[k].im += scale * ks_k.im;
                }
            }
        }

        let y_sl  = std::slice::from_raw_parts(y,  n);
        let yn_sl = std::slice::from_raw_parts(yn, n);
        let err   = weaknorm_c64(&h.y_err, y_sl, yn_sl, rtol, atol);
        let ok    = err <= 1.0;

        // ── Lund PI step control ──────────────────────────────────────────────────
        let (dtn_new, errlast_new, ok_final) =
            stepcontrol_pi(ok, err, errlast_in, dt, safety, max_dt, min_dt);

        // ── FSAL copy + prop!_maybe ───────────────────────────────────────────────
        let tn_new;
        if ok_final {
            tn_new = t + dt;
            std::ptr::copy_nonoverlapping(ks[6], ks[0], n);    // k1 ← k7 (FSAL)
            prop_fn(t, tn_new, yn, n, userdata);                // propagate yn forward
        } else {
            std::ptr::copy_nonoverlapping(y, yn, n);            // restore yn = y
            tn_new = t_new;
            prop_fn(t, tn_new, yn, n, userdata);                // no-op (t == tn_new)
        }

        *result = PreconStepResult {
            ok: ok_final as i32, dt, t, tn: tn_new, dtn: dtn_new, err, errlast: errlast_new,
        };
        0
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_blas_path(path_ptr: *const libc::c_char) -> c_int {
    if path_ptr.is_null() {
        return -1;
    }
    let c_str = unsafe { std::ffi::CStr::from_ptr(path_ptr) };
    if let Ok(s) = c_str.to_str() {
        match crate::blas::init_blas_api(std::path::Path::new(s)) {
            Ok(_) => 0,
            Err(e) => {
                eprintln!("Luna-Rust BLAS loader error: {}", e);
                -1
            }
        }
    } else {
        -1
    }
}
