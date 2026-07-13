use libc::{c_double, c_void};
#[cfg(unix)]
use std::ffi::CStr;
use std::ffi::CString;
use std::path::Path;
use std::sync::OnceLock;

/// docs/dev/BACKLOG.md S1 item 5: `libblastrampoline`'s CBLAS-name entry point
/// (`cblas_dgemm`) is a dispatch stub that requires Julia's own ILP64
/// Fortran-symbol registration to route anywhere — calling it directly
/// errors ("no BLAS/LAPACK library loaded for cblas_dgemm()") instead of
/// computing anything. The fix is to bind the plain Fortran BLAS entry
/// point instead, exactly the way Julia's own `LinearAlgebra.BLAS.gemm!`
/// does it (`stdlib/LinearAlgebra/src/blas.jl`): `@blasfunc(dgemm_)`
/// resolves to `dgemm_64_` because Julia is built with `USE_BLAS64=true`
/// (`BlasInt = Int64`); every scalar is passed *by reference* (a Fortran
/// calling-convention requirement, not a choice), and two trailing `size_t`
/// "hidden string length" arguments follow the argument list — one per
/// `CHARACTER` argument (`transa`, `transb`), each hardcoded to `1` since
/// both are always single ASCII chars ('N'/'T'/'C'). This is the same
/// `libblastrampoline` → OpenBLAS64 symbol Julia's own BLAS calls resolve
/// to, so it is guaranteed to exist and behave identically to what Julia
/// itself would compute.
///
/// SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
type Dgemm64Fn = unsafe extern "C" fn(
    transa: *const u8,
    transb: *const u8,
    m: *const i64,
    n: *const i64,
    k: *const i64,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const i64,
    b: *const c_double,
    ldb: *const i64,
    beta: *const c_double,
    c: *mut c_double,
    ldc: *const i64,
    len_transa: usize,
    len_transb: usize,
);

struct Library {
    handle: *mut c_void,
}

unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    unsafe fn load(path: &Path) -> Result<Self, String> {
        #[cfg(unix)]
        {
            let path_str = path.to_string_lossy();
            let c_path = CString::new(path_str.as_ref()).map_err(|e| e.to_string())?;
            let handle =
                unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
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
            use windows_sys::Win32::System::LibraryLoader::LoadLibraryW;
            let mut wide_path: Vec<u16> = path.as_os_str().encode_wide().collect();
            wide_path.push(0);
            let handle = unsafe { LoadLibraryW(wide_path.as_ptr()) };
            if handle.is_null() {
                return Err(format!("LoadLibraryW failed to load {:?}", path));
            }
            Ok(Self {
                handle: handle as *mut c_void,
            })
        }
    }

    unsafe fn get(&self, symbol: &str) -> Result<*mut c_void, String> {
        let c_sym = CString::new(symbol).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        {
            let sym = unsafe { libc::dlsym(self.handle, c_sym.as_ptr()) };
            if sym.is_null() {
                return Err(format!("Symbol {} not found", symbol));
            }
            Ok(sym)
        }
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
            let sym = unsafe { GetProcAddress(self.handle as _, c_sym.as_ptr() as *const u8) };
            if sym.is_none() {
                return Err(format!("Symbol {} not found", symbol));
            }
            Ok(sym.unwrap() as *mut c_void)
        }
    }
}

pub struct BlasApi {
    _lib: Library,
    dgemm_64: Dgemm64Fn,
}

impl BlasApi {
    /// Column-major `C := alpha*op(A)*op(B) + beta*C`, `trans ∈ {b'N', b'T', b'C'}`.
    /// `A` is `m×k` (or `k×m` if transposed), `B` is `k×n` (or `n×k`), `C` is `m×n`;
    /// `lda`/`ldb`/`ldc` are the respective leading dimensions (column strides).
    /// Callers must ensure the slices are large enough for `op(A)`/`op(B)`/`C`'s
    /// shapes given the leading dimensions passed in.
    #[allow(clippy::too_many_arguments)]
    pub fn dgemm(
        &self,
        trans_a: u8,
        trans_b: u8,
        m: i64,
        n: i64,
        k: i64,
        alpha: f64,
        a: &[f64],
        lda: i64,
        b: &[f64],
        ldb: i64,
        beta: f64,
        c: &mut [f64],
        ldc: i64,
    ) {
        unsafe {
            (self.dgemm_64)(
                &trans_a,
                &trans_b,
                &m,
                &n,
                &k,
                &alpha,
                a.as_ptr(),
                &lda,
                b.as_ptr(),
                &ldb,
                &beta,
                c.as_mut_ptr(),
                &ldc,
                1,
                1,
            );
        }
    }
}

static BLAS_API: OnceLock<BlasApi> = OnceLock::new();

/// Initialize the BLAS API by loading the library at the given path.
/// Usually called with the path to `libblastrampoline` provided by Julia.
pub fn init_blas_api(path: &Path) -> Result<(), String> {
    if BLAS_API.get().is_some() {
        return Ok(());
    }

    let lib = unsafe { Library::load(path)? };
    let dgemm_64_ptr = unsafe { lib.get("dgemm_64_")? };

    let api = BlasApi {
        _lib: lib,
        dgemm_64: unsafe { std::mem::transmute(dgemm_64_ptr) },
    };

    BLAS_API
        .set(api)
        .map_err(|_| "BLAS API already initialized".to_string())?;
    Ok(())
}

/// Get a reference to the global BLAS API if initialized.
pub fn get_blas_api() -> Result<&'static BlasApi, String> {
    BLAS_API
        .get()
        .ok_or_else(|| "BLAS API not initialized".to_string())
}

// Deliberately no standalone `#[cfg(test)]` unit test in this file: a bare
// `cargo test` process that `dlopen`s `libblastrampoline` directly segfaults
// on the very first `dgemm_64_` call, because the trampoline's forwarding
// table is only populated by Julia's own runtime startup (`LinearAlgebra`
// registers the default OpenBLAS64 provider) — dlopen-ing the shared object
// outside a running Julia process gets an *unconfigured* trampoline whose
// symbol resolves to a jump through a null/garbage slot. This was confirmed
// empirically while developing this fix (a naive `#[test]` doing exactly
// that crashed `cargo test` with SIGSEGV). The only place `dgemm_64_` can be
// validly exercised is inside a live Julia process where the trampoline is
// already configured — which is also exactly how this code is actually
// used in production (`NonlinearRHS._init_rust_qdht_blas` dlopens the
// *already-loaded* `libblastrampoline` via `Libdl.dlpath(Libdl.dlopen(...))`
// from within Julia, reusing its live, configured instance). The real gate
// is `test/test_qdht_rust.jl` run with `AMALTHEA_USE_RUST_QDHT=1
// AMALTHEA_QDHT_BLAS=1` (see docs/dev/BACKLOG.md S1 item 5).
