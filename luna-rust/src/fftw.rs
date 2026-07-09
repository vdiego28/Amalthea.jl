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
type PlanR2c3d =
    unsafe extern "C" fn(c_int, c_int, c_int, *mut f64, *mut FftwComplex, c_uint) -> FftwPlan;
type PlanC2r3d =
    unsafe extern "C" fn(c_int, c_int, c_int, *mut FftwComplex, *mut f64, c_uint) -> FftwPlan;
type PlanDft3d =
    unsafe extern "C" fn(c_int, c_int, c_int, *mut FftwComplex, *mut FftwComplex, c_int, c_uint) -> FftwPlan;
type ExecDft = unsafe extern "C" fn(FftwPlan, *mut FftwComplex, *mut FftwComplex);
type ExecR2c = unsafe extern "C" fn(FftwPlan, *mut f64, *mut FftwComplex);
type ExecC2r = unsafe extern "C" fn(FftwPlan, *mut FftwComplex, *mut f64);
type DestroyPlan = unsafe extern "C" fn(FftwPlan);
// `int fftw_import_wisdom_from_filename(const char*)` / `int
// fftw_export_wisdom_to_filename(const char*)` — both return nonzero on
// success (FFTW's own convention, unlike most of this file's 0=success FFI).
type ImportWisdom = unsafe extern "C" fn(*const libc::c_char) -> c_int;
type ExportWisdom = unsafe extern "C" fn(*const libc::c_char) -> c_int;

/// Loaded FFTW API + the live library handle (kept alive while the API exists).
pub struct FftwApi {
    _lib: Library,
    plan_dft_1d: PlanDft1d,
    plan_r2c_1d: PlanR2c1d,
    plan_c2r_1d: PlanC2r1d,
    plan_r2c_3d: PlanR2c3d,
    plan_c2r_3d: PlanC2r3d,
    plan_dft_3d: PlanDft3d,
    exec_dft: ExecDft,
    exec_r2c: ExecR2c,
    exec_c2r: ExecC2r,
    destroy_plan: DestroyPlan,
    import_wisdom: Option<ImportWisdom>,
    export_wisdom: Option<ExportWisdom>,
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
        let plan_r2c_3d = sym!("fftw_plan_dft_r2c_3d", PlanR2c3d);
        let plan_c2r_3d = sym!("fftw_plan_dft_c2r_3d", PlanC2r3d);
        let plan_dft_3d = sym!("fftw_plan_dft_3d", PlanDft3d);
        let exec_dft = sym!("fftw_execute_dft", ExecDft);
        let exec_r2c = sym!("fftw_execute_dft_r2c", ExecR2c);
        let exec_c2r = sym!("fftw_execute_dft_c2r", ExecC2r);
        let destroy_plan = sym!("fftw_destroy_plan", DestroyPlan);
        // Optional (S1 item 1, BACKLOG.md): every standard FFTW3 build
        // exports these, but lookup failure here should never block plan
        // creation (the whole feature is a planning-time speedup, not a
        // correctness dependency) — `None` just means `import`/`export`
        // silently no-op below.
        let import_wisdom = unsafe { lib.sym("fftw_import_wisdom_from_filename") }
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, ImportWisdom>(p) });
        let export_wisdom = unsafe { lib.sym("fftw_export_wisdom_to_filename") }
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, ExportWisdom>(p) });

        Ok(FftwApi {
            _lib: lib,
            plan_dft_1d,
            plan_r2c_1d,
            plan_c2r_1d,
            plan_r2c_3d,
            plan_c2r_3d,
            plan_dft_3d,
            exec_dft,
            exec_r2c,
            exec_c2r,
            destroy_plan,
            import_wisdom,
            export_wisdom,
        })
    }

    /// Load previously-saved planner wisdom from `path` — BACKLOG.md S1
    /// item 1. Best-effort: a missing file, a symbol-lookup miss at `load`
    /// time, or a malformed wisdom file all just mean "no wisdom available
    /// yet" (returns `false`), never an error — planning still works from
    /// scratch, just without the speedup. Call before creating any plans
    /// (`fftw_import_wisdom_from_filename` only affects plans created after
    /// it runs).
    pub fn import_wisdom_from_filename(&self, path: &str) -> bool {
        let f = match self.import_wisdom {
            Some(f) => f,
            None => return false,
        };
        let Ok(c_path) = CString::new(path) else { return false };
        let _guard = PLANNER_LOCK.lock().unwrap();
        unsafe { f(c_path.as_ptr()) != 0 }
    }

    /// Save the process's current planner wisdom (accumulated across every
    /// plan created so far, not just this call's caller) to `path` —
    /// BACKLOG.md S1 item 1. Best-effort, same as `import_wisdom_from_filename`.
    pub fn export_wisdom_to_filename(&self, path: &str) -> bool {
        let f = match self.export_wisdom {
            Some(f) => f,
            None => return false,
        };
        let Ok(c_path) = CString::new(path) else { return false };
        let _guard = PLANNER_LOCK.lock().unwrap();
        unsafe { f(c_path.as_ptr()) != 0 }
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

/// A real↔complex 3-D plan pair (`TransFree`'s `plan_rfft(x, (1,2,3))` —
/// RealGrid free-space, transform spans all three axes: time + 2 transverse).
///
/// Buffers are Julia column-major `(n_t, n_y, n_x)` (`n_t` fastest-varying).
/// FFTW's basic-interface dimension list is given slowest→fastest, so the
/// constructor passes `(n_x, n_y, n_t)` — **reversed** — to align FFTW's
/// fastest dimension with Julia's `n_t` axis; verified against
/// `FFTW.rfft(x,(1,2,3))`/`irfft` on a fixed array (`fftw.rs` unit test)
/// before this was trusted, not assumed from the row/column-major rule alone.
/// The conjugate-symmetric halving lands on `n_t` (→ `n_t/2+1`), matching
/// Julia's `size(rfft(x,(1,2,3))) == (n_t÷2+1, n_y, n_x)`.
///
/// **Multi-dim c2r destroys its input** (unlike 1-D c2r, `PRESERVE_INPUT` is
/// not supported for rank>1 c2r in FFTW) — callers must copy the spectrum
/// into scratch before calling `inverse`, exactly like every other native
/// RHS in this port already does before its inverse transform.
pub struct RealFft3d {
    n_t: usize,
    n_y: usize,
    n_x: usize,
    nspec: usize,
    r2c: FftwPlan,
    c2r: FftwPlan,
    destroy_plan: DestroyPlan,
    exec_r2c: ExecR2c,
    exec_c2r: ExecC2r,
}

impl RealFft3d {
    pub fn new(api: &FftwApi, n_t: usize, n_y: usize, n_x: usize, flags: c_uint) -> Self {
        let nspec = n_t / 2 + 1;
        let mut tbuf = vec![0.0f64; n_t * n_y * n_x];
        let mut sbuf = vec![[0.0f64; 2]; nspec * n_y * n_x];
        let f = flags | FFTW_UNALIGNED;
        let _guard = PLANNER_LOCK.lock().unwrap();
        let r2c = unsafe {
            (api.plan_r2c_3d)(n_x as c_int, n_y as c_int, n_t as c_int,
                              tbuf.as_mut_ptr(), sbuf.as_mut_ptr(), f)
        };
        let c2r = unsafe {
            (api.plan_c2r_3d)(n_x as c_int, n_y as c_int, n_t as c_int,
                              sbuf.as_mut_ptr(), tbuf.as_mut_ptr(), f)
        };
        RealFft3d { n_t, n_y, n_x, nspec, r2c, c2r,
                    destroy_plan: api.destroy_plan, exec_r2c: api.exec_r2c, exec_c2r: api.exec_c2r }
    }

    pub fn nspec(&self) -> usize { self.nspec }

    /// `spec = rfft(time, (1,2,3))` (unnormalized), column-major `(nspec, n_y, n_x)`.
    pub fn forward(&self, time: &mut [f64], spec: &mut [Complex<f64>]) {
        assert_eq!(time.len(), self.n_t * self.n_y * self.n_x);
        assert_eq!(spec.len(), self.nspec * self.n_y * self.n_x);
        unsafe {
            (self.exec_r2c)(self.r2c, time.as_mut_ptr(), spec.as_mut_ptr() as *mut FftwComplex);
        }
    }

    /// `time = irfft_unnormalized(spec, (1,2,3))` (caller divides by
    /// `n_t*n_y*n_x`). **Destroys `spec`** — copy first if the caller still
    /// needs it (see struct doc).
    pub fn inverse(&self, spec: &mut [Complex<f64>], time: &mut [f64]) {
        assert_eq!(spec.len(), self.nspec * self.n_y * self.n_x);
        assert_eq!(time.len(), self.n_t * self.n_y * self.n_x);
        unsafe {
            (self.exec_c2r)(self.c2r, spec.as_mut_ptr() as *mut FftwComplex, time.as_mut_ptr());
        }
    }
}

impl Drop for RealFft3d {
    fn drop(&mut self) {
        let _guard = PLANNER_LOCK.lock().unwrap();
        unsafe { (self.destroy_plan)(self.r2c); (self.destroy_plan)(self.c2r); }
    }
}

/// A complex↔complex 3-D plan pair (`TransFree`'s `plan_fft(x, (1,2,3))` —
/// EnvGrid free-space, Phase D.3). Same buffer/dimension-order convention as
/// [`RealFft3d`]: Julia column-major `(n_t, n_y, n_x)`, FFTW dims passed
/// reversed as `(n_x, n_y, n_t)`. Unlike `RealFft3d`'s r2c/c2r pair, both
/// directions here are full-length (no conjugate-symmetric halving) and a
/// single `fftw_plan_dft_3d` per direction (FFTW_FORWARD/FFTW_BACKWARD)
/// suffices — c2c multi-dim plans support `FFTW_PRESERVE_INPUT` with
/// `FFTW_ESTIMATE`/`FFTW_MEASURE` (unlike c2r), but callers still treat the
/// input as scratch to match every other native RHS's copy-before-inverse
/// convention in this port.
pub struct ComplexFft3d {
    n_t: usize,
    n_y: usize,
    n_x: usize,
    fwd: FftwPlan,
    inv: FftwPlan,
    destroy_plan: DestroyPlan,
    exec_dft: ExecDft,
}

impl ComplexFft3d {
    pub fn new(api: &FftwApi, n_t: usize, n_y: usize, n_x: usize, flags: c_uint) -> Self {
        let ntot = n_t * n_y * n_x;
        let mut a = vec![[0.0f64; 2]; ntot];
        let mut b = vec![[0.0f64; 2]; ntot];
        let f = flags | FFTW_UNALIGNED;
        let _guard = PLANNER_LOCK.lock().unwrap();
        let fwd = unsafe {
            (api.plan_dft_3d)(n_x as c_int, n_y as c_int, n_t as c_int,
                              a.as_mut_ptr(), b.as_mut_ptr(), FFTW_FORWARD, f)
        };
        let inv = unsafe {
            (api.plan_dft_3d)(n_x as c_int, n_y as c_int, n_t as c_int,
                              a.as_mut_ptr(), b.as_mut_ptr(), FFTW_BACKWARD, f)
        };
        ComplexFft3d { n_t, n_y, n_x, fwd, inv, destroy_plan: api.destroy_plan, exec_dft: api.exec_dft }
    }

    /// `out = fft(inp, (1,2,3))` (unnormalized).
    pub fn forward(&self, inp: &mut [Complex<f64>], out: &mut [Complex<f64>]) {
        let ntot = self.n_t * self.n_y * self.n_x;
        assert_eq!(inp.len(), ntot);
        assert_eq!(out.len(), ntot);
        unsafe {
            (self.exec_dft)(self.fwd, inp.as_mut_ptr() as *mut FftwComplex,
                            out.as_mut_ptr() as *mut FftwComplex);
        }
    }

    /// `out = ifft_unnormalized(inp, (1,2,3))` (caller divides by `n_t*n_y*n_x`).
    pub fn inverse(&self, inp: &mut [Complex<f64>], out: &mut [Complex<f64>]) {
        let ntot = self.n_t * self.n_y * self.n_x;
        assert_eq!(inp.len(), ntot);
        assert_eq!(out.len(), ntot);
        unsafe {
            (self.exec_dft)(self.inv, inp.as_mut_ptr() as *mut FftwComplex,
                            out.as_mut_ptr() as *mut FftwComplex);
        }
    }
}

impl Drop for ComplexFft3d {
    fn drop(&mut self) {
        let _guard = PLANNER_LOCK.lock().unwrap();
        unsafe { (self.destroy_plan)(self.fwd); (self.destroy_plan)(self.inv); }
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

    /// Cross-validates `RealFft3d`'s dimension order (reversed `(n_x,n_y,n_t)`
    /// passed to FFTW for Julia's column-major `(n_t,n_y,n_x)`) against a
    /// literal reference computed independently in Julia:
    /// `FFTW.rfft(reshape(Float64.(1:24), 4,3,2), (1,2,3))`. A pure Rust
    /// round-trip test (forward+inverse self-consistency) cannot catch a
    /// dimension-order bug — forward/inverse would still round-trip
    /// correctly even if transposed relative to Julia's convention — so this
    /// compares actual spectral values, not just round-trip agreement. See
    /// MATH.md §3.4.
    #[test]
    fn r2c_3d_matches_julia_reference() {
        let api = match try_api() {
            Some(a) => a,
            None => { eprintln!("skip r2c_3d_matches_julia_reference: no FFTW found"); return; }
        };
        let (n_t, n_y, n_x) = (4usize, 3usize, 2usize);
        let plan = RealFft3d::new(&api, n_t, n_y, n_x, FFTW_ESTIMATE);

        // Column-major (n_t,n_y,n_x): reshape(Float64.(1:24), 4,3,2) in Julia.
        let mut x: Vec<f64> = (1..=24).map(|v| v as f64).collect();
        let nspec = plan.nspec();
        let mut spec = vec![Complex::new(0.0, 0.0); nspec * n_y * n_x];
        plan.forward(&mut x, &mut spec);

        // Julia: FFTW.rfft(x, (1,2,3))[i,j,k], column-major (nspec,n_y,n_x).
        let expected: [(usize, usize, usize, f64, f64); 6] = [
            (0, 0, 0, 300.0, 0.0),
            (1, 0, 0, -12.0, 12.0),
            (2, 0, 0, -12.0, 0.0),
            (0, 1, 0, -48.0, 27.712812921102035),
            (0, 2, 0, -48.0, -27.712812921102035),
            (0, 0, 1, -144.0, 0.0),
        ];
        for (i, j, k, re, im) in expected {
            let idx = i + nspec * (j + n_y * k);
            let got = spec[idx];
            assert!((got.re - re).abs() < 1e-9 && (got.im - im).abs() < 1e-9,
                "r2c_3d mismatch at ({i},{j},{k}): got {got}, expected {re}+{im}i");
        }
        // Everything else in this particular input is exactly zero.
        for k in 0..n_x {
            for j in 0..n_y {
                for i in 0..nspec {
                    if expected.iter().any(|&(ei, ej, ek, ..)| ei == i && ej == j && ek == k) {
                        continue;
                    }
                    let idx = i + nspec * (j + n_y * k);
                    assert!(spec[idx].norm() < 1e-9, "expected ~0 at ({i},{j},{k}), got {}", spec[idx]);
                }
            }
        }

        // Round-trip: irfft(rfft(x)) == n_t*n_y*n_x * x. c2r destroys `spec`,
        // so this also exercises that spec is scratch, not reusable after.
        let mut back = vec![0.0f64; n_t * n_y * n_x];
        plan.inverse(&mut spec, &mut back);
        let norm = (n_t * n_y * n_x) as f64;
        for i in 0..back.len() {
            assert!((back[i] / norm - x[i]).abs() < 1e-9, "r2c_3d roundtrip mismatch at {i}");
        }
    }

    /// Cross-validates `ComplexFft3d`'s dimension order against
    /// `FFTW.fft(reshape(ComplexF64.(1:24), 4,3,2), (1,2,3))` (Phase D.3,
    /// BACKLOG.md) — the complex counterpart of `r2c_3d_matches_julia_reference`.
    /// Full-length spectrum (no conjugate-symmetric halving), so all 24
    /// entries are checked against the Julia reference, not just 6.
    #[test]
    fn c2c_3d_matches_julia_reference() {
        let api = match try_api() {
            Some(a) => a,
            None => { eprintln!("skip c2c_3d_matches_julia_reference: no FFTW found"); return; }
        };
        let (n_t, n_y, n_x) = (4usize, 3usize, 2usize);
        let plan = ComplexFft3d::new(&api, n_t, n_y, n_x, FFTW_ESTIMATE);

        let mut x: Vec<Complex<f64>> = (1..=24).map(|v| Complex::new(v as f64, 0.0)).collect();
        let orig = x.clone();
        let ntot = n_t * n_y * n_x;
        let mut spec = vec![Complex::new(0.0, 0.0); ntot];
        plan.forward(&mut x, &mut spec);

        // Julia: FFTW.fft(x, (1,2,3))[i,j,k], column-major (n_t,n_y,n_x).
        let expected: [(usize, usize, usize, f64, f64); 6] = [
            (0, 0, 0, 300.0, 0.0),
            (1, 0, 0, -12.0, 12.0),
            (2, 0, 0, -12.0, 0.0),
            (0, 1, 0, -48.0, 27.712812921102035),
            (0, 2, 0, -48.0, -27.712812921102035),
            (0, 0, 1, -144.0, 0.0),
        ];
        for (i, j, k, re, im) in expected {
            let idx = i + n_t * (j + n_y * k);
            let got = spec[idx];
            assert!((got.re - re).abs() < 1e-9 && (got.im - im).abs() < 1e-9,
                "c2c_3d mismatch at ({i},{j},{k}): got {got}, expected {re}+{im}i");
        }

        let mut back = vec![Complex::new(0.0, 0.0); ntot];
        plan.inverse(&mut spec, &mut back);
        let norm = ntot as f64;
        for i in 0..ntot {
            assert!((back[i] / norm - orig[i]).norm() < 1e-9, "c2c_3d roundtrip mismatch at {i}");
        }
    }
}
