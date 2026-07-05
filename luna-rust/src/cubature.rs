//! Runtime binding for the C `libcubature` library (`Cubature_jll`) — the
//! *same* adaptive cubature routines Julia's `Cubature.jl` wraps via `ccall`
//! (confirmed: `Cubature.jl` is a thin wrapper, not a pure-Julia
//! reimplementation — see `docs/native-port/MATH.md` §3.3). Loaded exactly
//! like `fftw.rs` loads libfftw3: `dlopen`ed at runtime from a path passed in
//! by Julia (`Cubature.Cubature_jll.libcubature`), no link-time dependency.
//!
//! Binding the same binary makes adaptive node placement **bit-identical**
//! to the Julia oracle by construction, instead of merely close — avoiding
//! the adaptive-path-divergence class of bug documented in `TESTING.md` §3
//! (a reimplemented cubature routine's region-subdivision decisions are just
//! as FP-summation-order-sensitive as the RK45 step controller, but with no
//! `max_dt=min_dt` escape hatch to pin node placement).
//!
//! Phase 5 scope bound only `pcubature_v` (1-D p-adaptive, vectorized) — the
//! radial-only (`full=false`) modal integral. Phase E.3 (BACKLOG.md) adds
//! `hcubature_v` (h-adaptive, arbitrary `ndim`) for `full=true`'s genuine
//! 2-D `(r,θ)` integral — the same routine Julia's `Cubature.hcubature_v`
//! calls, sharing `pcubature_v`'s C prototype (`cubature.h` gives every
//! `{h,p}cubature{,_v}` variant an identical signature, differing only in
//! subdivision strategy), so no new FFI type is needed.

use std::ffi::CString;
#[cfg(unix)]
use std::ffi::CStr;
use std::path::Path;
use libc::{c_double, c_int, c_uint, c_void, size_t};

pub const ERROR_NORM_L2: c_int = 2;

/// C `integrand_v` callback signature (`cubature.h`):
/// `int (*)(unsigned ndim, size_t npt, const double *x, void *fdata, unsigned fdim, double *fval)`.
pub type IntegrandV = unsafe extern "C" fn(c_uint, size_t, *const c_double, *mut c_void, c_uint, *mut c_double) -> c_int;

type PcubatureVFn = unsafe extern "C" fn(
    c_uint, IntegrandV, *mut c_void,
    c_uint, *const c_double, *const c_double,
    size_t, c_double, c_double, c_int,
    *mut c_double, *mut c_double,
) -> c_int;

// ── minimal runtime loader (mirrors fftw.rs::Library / io.rs::Library) ──────
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

/// Resident handle on the dlopen'd `libcubature`. `NativeSim` owns one.
pub struct CubatureApi {
    _lib: Library,
    pcubature_v: PcubatureVFn,
    hcubature_v: PcubatureVFn,
}

// Safety: `pcubature_v` is a plain function pointer into a shared library;
// the library itself is never mutated after load. Matches `FftwApi`'s
// Send/Sync story (see fftw.rs) — the native stepper is single-threaded.
unsafe impl Send for CubatureApi {}
unsafe impl Sync for CubatureApi {}

impl CubatureApi {
    pub fn load(path: &str) -> Result<Self, String> {
        let lib = unsafe { Library::load(Path::new(path))? };
        let pcubature_v = unsafe { lib.sym("pcubature_v") }
            .ok_or("symbol pcubature_v not found in libcubature")?;
        let hcubature_v = unsafe { lib.sym("hcubature_v") }
            .ok_or("symbol hcubature_v not found in libcubature")?;
        Ok(Self {
            _lib: lib,
            pcubature_v: unsafe { std::mem::transmute::<*mut c_void, PcubatureVFn>(pcubature_v) },
            hcubature_v: unsafe { std::mem::transmute::<*mut c_void, PcubatureVFn>(hcubature_v) },
        })
    }

    /// 1-D p-adaptive vectorized cubature over `[xmin, xmax]`.
    ///
    /// `f` is called with batches of nodes; `fdata` is passed through
    /// unchanged (the native RHS passes a `*mut NativeSim`). `val`/`err` must
    /// each be `fdim` long. Returns the C library's status code (0 = success).
    ///
    /// # Safety
    /// `val`/`err` must be valid for `fdim` `f64` writes; `f`/`fdata` must
    /// satisfy the `integrand_v` contract (any panic inside `f` must be
    /// caught — unwinding across the C call boundary is undefined behaviour).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn pcubature_v(
        &self,
        fdim: usize,
        f: IntegrandV,
        fdata: *mut c_void,
        xmin: f64,
        xmax: f64,
        max_eval: usize,
        req_abs_error: f64,
        req_rel_error: f64,
        val: &mut [f64],
        err: &mut [f64],
    ) -> i32 {
        debug_assert_eq!(val.len(), fdim);
        debug_assert_eq!(err.len(), fdim);
        let xmin_arr = [xmin];
        let xmax_arr = [xmax];
        unsafe {
            (self.pcubature_v)(
                fdim as c_uint, f, fdata,
                1, xmin_arr.as_ptr(), xmax_arr.as_ptr(),
                max_eval as size_t, req_abs_error, req_rel_error, ERROR_NORM_L2,
                val.as_mut_ptr(), err.as_mut_ptr(),
            )
        }
    }

    /// 2-D h-adaptive vectorized cubature over `[xmin[0],xmax[0]] ×
    /// [xmin[1],xmax[1]]` — Phase E.3 (BACKLOG.md), the `full=true` modal
    /// integral. Same contract as `pcubature_v` otherwise.
    ///
    /// # Safety
    /// Same as `pcubature_v`, plus: the `x` buffer `f` receives is `2·npt`
    /// doubles, point-major (`x[2·p]`/`x[2·p+1]` are point `p`'s two
    /// coordinates), per `cubature.h`'s convention.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn hcubature_v_2d(
        &self,
        fdim: usize,
        f: IntegrandV,
        fdata: *mut c_void,
        xmin: [f64; 2],
        xmax: [f64; 2],
        max_eval: usize,
        req_abs_error: f64,
        req_rel_error: f64,
        val: &mut [f64],
        err: &mut [f64],
    ) -> i32 {
        debug_assert_eq!(val.len(), fdim);
        debug_assert_eq!(err.len(), fdim);
        unsafe {
            (self.hcubature_v)(
                fdim as c_uint, f, fdata,
                2, xmin.as_ptr(), xmax.as_ptr(),
                max_eval as size_t, req_abs_error, req_rel_error, ERROR_NORM_L2,
                val.as_mut_ptr(), err.as_mut_ptr(),
            )
        }
    }
}
