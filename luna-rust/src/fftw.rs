//! Runtime FFTW3 binding for the native backend.
//!
//! We bind the **same** double-precision FFTW3 C library that Julia's `FFTW.jl`
//! uses (path passed in from `FFTW.FFTW_jll.libfftw3`), so the native transforms
//! are bit-parity with the Julia oracle — see `docs/native-port/ARCHITECTURE.md`
//! §4.1 and `MATH.md` §4. The library is located + `dlopen`ed at runtime exactly
//! like `io.rs` does for libhdf5 (no link-time dependency).
//!
//! Phase 0b scope: load the library, build forward/inverse plans (c2c for
//! EnvGrid `fft`/`ifft`; r2c/c2r for RealGrid `rfft`/`irfft`), and prove an
//! FFT→IFFT round-trip. Normalization is applied **explicitly** by the caller
//! (FFTW is unnormalized), matching Julia's `copy_scale!` convention — never
//! folded into the plan.
//!
//! ## Parity note for later phases (do not lose)
//! The planner *flag* (`FFTW_ESTIMATE` vs `FFTW_MEASURE`/`FFTW_PATIENT`) selects
//! the algorithm and therefore the summation order. To get bit-parity at
//! integration (Phase 1+), the flag here must match the one `FFTW.jl` used for
//! the run under test — Luna's package default is `:patient`, but its **test**
//! suite uses `:estimate` (see `CLAUDE.md`). The flag is a parameter for exactly
//! this reason. `FFTW_UNALIGNED` is always set so the new-array execute variants
//! are safe on arbitrary Rust `Vec` buffers.

use std::ffi::CString;
#[cfg(unix)]
use std::ffi::CStr;
use std::path::Path;
use std::sync::Mutex;
use libc::{c_int, c_uint, c_void};
use num_complex::Complex;

/// FFTW plan **creation** is not thread-safe (only `fftw_execute*` is); concurrent
/// `fftw_plan_*` calls race and crash. We serialize all planning behind this lock,
/// exactly as `FFTW.jl` holds a global planner lock. Execution is unguarded.
static PLANNER_LOCK: Mutex<()> = Mutex::new(());

// ── FFTW constants ──────────────────────────────────────────────────────────
pub const FFTW_FORWARD: c_int = -1;
pub const FFTW_BACKWARD: c_int = 1;
pub const FFTW_MEASURE: c_uint = 0;
pub const FFTW_DESTROY_INPUT: c_uint = 1 << 0;
pub const FFTW_UNALIGNED: c_uint = 1 << 1;
pub const FFTW_PRESERVE_INPUT: c_uint = 1 << 4;
pub const FFTW_PATIENT: c_uint = 1 << 5;
pub const FFTW_ESTIMATE: c_uint = 1 << 6;

/// Opaque FFTW plan pointer.
type FftwPlan = *mut c_void;
/// FFTW's `fftw_complex` is two contiguous f64 — identical layout to
/// `num_complex::Complex<f64>` and Julia `ComplexF64`.
type FftwComplex = [f64; 2];

// ── minimal runtime loader (mirrors io.rs::Library) ─────────────────────────
struct Library {
    handle: *mut c_void,
}

impl Library {
    unsafe fn load(path: &Path) -> Result<Self, String> {
        #[cfg(unix)]
        {
            let path_str = path.to_string_lossy();
            let c_path = CString::new(path_str.as_ref()).map_err(|e| e.to_string())?;
            let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
            if handle.is_null() {
                let err = unsafe { libc::dlerror() };
                let msg = if err.is_null() {
                    "Unknown dlopen error".to_string()
                } else {
                    unsafe { CStr::from_ptr(err).to_string_lossy().into_owned() }
                };
                return Err(msg);
            }
            Ok(Self { handle })
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
            wide.push(0);
            unsafe extern "system" {
                fn LoadLibraryW(lpLibFileName: *const u16) -> *mut c_void;
                fn GetLastError() -> u32;
            }
            let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
            if handle.is_null() {
                return Err(format!("Windows error code: {}", unsafe { GetLastError() }));
            }
            Ok(Self { handle })
        }
    }

    unsafe fn sym(&self, name: &str) -> Option<*mut c_void> {
        let c_name = CString::new(name).ok()?;
        #[cfg(unix)]
        {
            let p = unsafe { libc::dlsym(self.handle, c_name.as_ptr()) };
            if p.is_null() { None } else { Some(p) }
        }
        #[cfg(windows)]
        {
            unsafe extern "system" {
                fn GetProcAddress(h: *mut c_void, name: *const libc::c_char) -> *mut c_void;
            }
            let p = unsafe { GetProcAddress(self.handle, c_name.as_ptr()) };
            if p.is_null() { None } else { Some(p) }
        }
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            #[cfg(unix)]
            { libc::dlclose(self.handle); }
            #[cfg(windows)]
            {
                unsafe extern "system" { fn FreeLibrary(h: *mut c_void) -> c_int; }
                FreeLibrary(self.handle);
            }
        }
    }
}

/// Candidate library names to try when no explicit path is given.
fn default_names() -> &'static [&'static str] {
    #[cfg(target_os = "macos")]
    { &["libfftw3.dylib", "libfftw3.3.dylib"] }
    #[cfg(target_os = "windows")]
    { &["libfftw3-3.dll", "fftw3.dll"] }
    #[cfg(all(unix, not(target_os = "macos")))]
    { &["libfftw3.so.3", "libfftw3.so"] }
}

// ── bound entry points ──────────────────────────────────────────────────────
type PlanDft1d =
    unsafe extern "C" fn(c_int, *mut FftwComplex, *mut FftwComplex, c_int, c_uint) -> FftwPlan;
type PlanR2c1d =
    unsafe extern "C" fn(c_int, *mut f64, *mut FftwComplex, c_uint) -> FftwPlan;
type PlanC2r1d =
    unsafe extern "C" fn(c_int, *mut FftwComplex, *mut f64, c_uint) -> FftwPlan;
type ExecDft = unsafe extern "C" fn(FftwPlan, *mut FftwComplex, *mut FftwComplex);
type ExecR2c = unsafe extern "C" fn(FftwPlan, *mut f64, *mut FftwComplex);
type ExecC2r = unsafe extern "C" fn(FftwPlan, *mut FftwComplex, *mut f64);
type DestroyPlan = unsafe extern "C" fn(FftwPlan);

/// Loaded FFTW API + the live library handle (kept alive while the API exists).
pub struct FftwApi {
    _lib: Library,
    plan_dft_1d: PlanDft1d,
    plan_r2c_1d: PlanR2c1d,
    plan_c2r_1d: PlanC2r1d,
    exec_dft: ExecDft,
    exec_r2c: ExecR2c,
    exec_c2r: ExecC2r,
    destroy_plan: DestroyPlan,
}

impl FftwApi {
    /// Load FFTW from an explicit path (e.g. `FFTW.FFTW_jll.libfftw3`), or — if
    /// `path` is `None` — try the platform default names on the loader path.
    pub fn load(path: Option<&str>) -> Result<Self, String> {
        let lib = unsafe {
            if let Some(p) = path {
                Library::load(Path::new(p))?
            } else {
                let mut last = String::from("no FFTW library name succeeded");
                let mut found = None;
                for name in default_names() {
                    match Library::load(Path::new(name)) {
                        Ok(l) => { found = Some(l); break; }
                        Err(e) => last = e,
                    }
                }
                found.ok_or(last)?
            }
        };

        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                let p = unsafe { lib.sym($name) }
                    .ok_or_else(|| format!("FFTW symbol not found: {}", $name))?;
                unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
            }};
        }

        let plan_dft_1d = sym!("fftw_plan_dft_1d", PlanDft1d);
        let plan_r2c_1d = sym!("fftw_plan_dft_r2c_1d", PlanR2c1d);
        let plan_c2r_1d = sym!("fftw_plan_dft_c2r_1d", PlanC2r1d);
        let exec_dft = sym!("fftw_execute_dft", ExecDft);
        let exec_r2c = sym!("fftw_execute_dft_r2c", ExecR2c);
        let exec_c2r = sym!("fftw_execute_dft_c2r", ExecC2r);
        let destroy_plan = sym!("fftw_destroy_plan", DestroyPlan);

        Ok(FftwApi {
            _lib: lib,
            plan_dft_1d,
            plan_r2c_1d,
            plan_c2r_1d,
            exec_dft,
            exec_r2c,
            exec_c2r,
            destroy_plan,
        })
    }
}

/// A complex↔complex 1-D plan pair (EnvGrid `fft`/`ifft`), length `n`.
///
/// FFTW transforms are **unnormalized**: `ifft(fft(x)) == n*x`. The caller
/// applies the `1/n` (or Luna's `copy_scale!`) factor — never the plan.
pub struct ComplexFft1d {
    n: usize,
    fwd: FftwPlan,
    inv: FftwPlan,
    destroy_plan: DestroyPlan,
    exec_dft: ExecDft,
}

impl ComplexFft1d {
    pub fn new(api: &FftwApi, n: usize, flags: c_uint) -> Self {
        // Scratch arrays only define the plan's array kind/size; with
        // FFTW_UNALIGNED the plan is reusable on any buffer via the new-array
        // execute variant. FFTW_ESTIMATE does not clobber these during planning.
        let mut a = vec![[0.0f64; 2]; n];
        let mut b = vec![[0.0f64; 2]; n];
        let f = flags | FFTW_UNALIGNED;
        let _guard = PLANNER_LOCK.lock().unwrap();
        let fwd = unsafe {
            (api.plan_dft_1d)(n as c_int, a.as_mut_ptr(), b.as_mut_ptr(), FFTW_FORWARD, f)
        };
        let inv = unsafe {
            (api.plan_dft_1d)(n as c_int, a.as_mut_ptr(), b.as_mut_ptr(), FFTW_BACKWARD, f)
        };
        ComplexFft1d { n, fwd, inv, destroy_plan: api.destroy_plan, exec_dft: api.exec_dft }
    }

    /// Forward transform `out = fft(inp)` (unnormalized).
    pub fn forward(&self, inp: &mut [Complex<f64>], out: &mut [Complex<f64>]) {
        assert_eq!(inp.len(), self.n);
        assert_eq!(out.len(), self.n);
        unsafe {
            (self.exec_dft)(self.fwd, inp.as_mut_ptr() as *mut FftwComplex,
                            out.as_mut_ptr() as *mut FftwComplex);
        }
    }

    /// Inverse transform `out = ifft_unnormalized(inp)` (caller divides by `n`).
    pub fn inverse(&self, inp: &mut [Complex<f64>], out: &mut [Complex<f64>]) {
        assert_eq!(inp.len(), self.n);
        assert_eq!(out.len(), self.n);
        unsafe {
            (self.exec_dft)(self.inv, inp.as_mut_ptr() as *mut FftwComplex,
                            out.as_mut_ptr() as *mut FftwComplex);
        }
    }
}

impl Drop for ComplexFft1d {
    fn drop(&mut self) {
        let _guard = PLANNER_LOCK.lock().unwrap();
        unsafe { (self.destroy_plan)(self.fwd); (self.destroy_plan)(self.inv); }
    }
}

/// A real↔complex 1-D plan pair (RealGrid `rfft`/`irfft`).
/// Time length `n` (real), spectral length `n/2+1` (complex).
pub struct RealFft1d {
    n: usize,
    nspec: usize,
    r2c: FftwPlan,
    c2r: FftwPlan,
    destroy_plan: DestroyPlan,
    exec_r2c: ExecR2c,
    exec_c2r: ExecC2r,
}

impl RealFft1d {
    pub fn new(api: &FftwApi, n: usize, flags: c_uint) -> Self {
        let nspec = n / 2 + 1;
        let mut tbuf = vec![0.0f64; n];
        let mut sbuf = vec![[0.0f64; 2]; nspec];
        let f = flags | FFTW_UNALIGNED;
        let _guard = PLANNER_LOCK.lock().unwrap();
        let r2c = unsafe {
            (api.plan_r2c_1d)(n as c_int, tbuf.as_mut_ptr(), sbuf.as_mut_ptr(), f)
        };
        // c2r destroys its input by default; PRESERVE_INPUT keeps the spectrum
        // intact (1-D c2r supports it). Matches the safe out-of-place pattern.
        let c2r = unsafe {
            (api.plan_c2r_1d)(n as c_int, sbuf.as_mut_ptr(), tbuf.as_mut_ptr(),
                              f | FFTW_PRESERVE_INPUT)
        };
        RealFft1d { n, nspec, r2c, c2r, destroy_plan: api.destroy_plan,
                    exec_r2c: api.exec_r2c, exec_c2r: api.exec_c2r }
    }

    pub fn nspec(&self) -> usize { self.nspec }

    /// `spec = rfft(time)` (unnormalized), `spec.len() == n/2+1`.
    pub fn forward(&self, time: &mut [f64], spec: &mut [Complex<f64>]) {
        assert_eq!(time.len(), self.n);
        assert_eq!(spec.len(), self.nspec);
        unsafe {
            (self.exec_r2c)(self.r2c, time.as_mut_ptr(),
                            spec.as_mut_ptr() as *mut FftwComplex);
        }
    }

    /// `time = irfft_unnormalized(spec)` (caller divides by `n`).
    pub fn inverse(&self, spec: &mut [Complex<f64>], time: &mut [f64]) {
        assert_eq!(spec.len(), self.nspec);
        assert_eq!(time.len(), self.n);
        unsafe {
            (self.exec_c2r)(self.c2r, spec.as_mut_ptr() as *mut FftwComplex,
                            time.as_mut_ptr());
        }
    }
}

impl Drop for RealFft1d {
    fn drop(&mut self) {
        let _guard = PLANNER_LOCK.lock().unwrap();
        unsafe { (self.destroy_plan)(self.r2c); (self.destroy_plan)(self.c2r); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    /// Try to load FFTW for tests; return None to "skip" (the Rust analog of the
    /// Julia skip-guard) when no FFTW is installed on the test host.
    fn try_api() -> Option<FftwApi> {
        let path = std::env::var("LUNA_FFTW_LIB").ok();
        FftwApi::load(path.as_deref()).ok()
    }

    #[test]
    fn c2c_roundtrip() {
        let api = match try_api() {
            Some(a) => a,
            None => { eprintln!("skip c2c_roundtrip: no FFTW found"); return; }
        };
        let n = 16;
        let plan = ComplexFft1d::new(&api, n, FFTW_ESTIMATE);
        let mut x: Vec<Complex<f64>> =
            (0..n).map(|i| Complex::new((i as f64).sin(), (0.3 * i as f64).cos())).collect();
        let orig = x.clone();
        let mut spec = vec![Complex::new(0.0, 0.0); n];
        let mut back = vec![Complex::new(0.0, 0.0); n];
        plan.forward(&mut x, &mut spec);
        plan.inverse(&mut spec, &mut back);
        // ifft(fft(x)) = n*x ; normalize and compare.
        for i in 0..n {
            let r = back[i] / n as f64;
            assert!((r - orig[i]).norm() < 1e-12, "c2c roundtrip mismatch at {i}");
        }
    }

    #[test]
    fn r2c_roundtrip() {
        let api = match try_api() {
            Some(a) => a,
            None => { eprintln!("skip r2c_roundtrip: no FFTW found"); return; }
        };
        let n = 32;
        let plan = RealFft1d::new(&api, n, FFTW_ESTIMATE);
        let mut t: Vec<f64> = (0..n).map(|i| (0.5 * i as f64).sin()).collect();
        let orig = t.clone();
        let mut spec = vec![Complex::new(0.0, 0.0); plan.nspec()];
        let mut back = vec![0.0f64; n];
        plan.forward(&mut t, &mut spec);
        plan.inverse(&mut spec, &mut back);
        for i in 0..n {
            assert!((back[i] / n as f64 - orig[i]).abs() < 1e-12,
                    "r2c roundtrip mismatch at {i}");
        }
    }
}
