use num_complex::Complex;
use libc::{c_double, size_t};

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
