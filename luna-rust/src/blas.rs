use std::ffi::CString;
#[cfg(unix)]
use std::ffi::CStr;
use std::path::Path;
use std::sync::OnceLock;
use libc::{c_int, c_double, c_void};

pub const CBLAS_ROW_MAJOR: c_int = 101;
pub const CBLAS_COL_MAJOR: c_int = 102;
pub const CBLAS_NO_TRANS: c_int = 111;
pub const CBLAS_TRANS: c_int = 112;
pub const CBLAS_CONJ_TRANS: c_int = 113;

type CblasDgemmFn = unsafe extern "C" fn(
    layout: c_int,
    trans_a: c_int,
    trans_b: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: c_double,
    a: *const c_double,
    lda: c_int,
    b: *const c_double,
    ldb: c_int,
    beta: c_double,
    c: *mut c_double,
    ldc: c_int,
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
            use windows_sys::Win32::System::LibraryLoader::LoadLibraryW;
            let mut wide_path: Vec<u16> = path.as_os_str().encode_wide().collect();
            wide_path.push(0);
            let handle = unsafe { LoadLibraryW(wide_path.as_ptr()) };
            if handle.is_null() {
                return Err(format!("LoadLibraryW failed to load {:?}", path));
            }
            Ok(Self { handle: handle as *mut c_void })
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
    pub cblas_dgemm: CblasDgemmFn,
}

static BLAS_API: OnceLock<BlasApi> = OnceLock::new();

/// Initialize the BLAS API by loading the library at the given path.
/// Usually called with the path to `libblastrampoline` provided by Julia.
pub fn init_blas_api(path: &Path) -> Result<(), String> {
    if BLAS_API.get().is_some() {
        return Ok(());
    }

    let lib = unsafe { Library::load(path)? };
    let cblas_dgemm_ptr = unsafe { lib.get("cblas_dgemm")? };

    let api = BlasApi {
        _lib: lib,
        cblas_dgemm: unsafe { std::mem::transmute(cblas_dgemm_ptr) },
    };

    BLAS_API.set(api).map_err(|_| "BLAS API already initialized".to_string())?;
    Ok(())
}

/// Get a reference to the global BLAS API if initialized.
pub fn get_blas_api() -> Result<&'static BlasApi, String> {
    BLAS_API.get().ok_or_else(|| "BLAS API not initialized".to_string())
}
