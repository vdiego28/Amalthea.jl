use num_complex::Complex;
use libc::{c_double, c_int, size_t};
use crate::ionization::PptIonizationRate;

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
