use num_complex::Complex;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// HDF5 Constants
pub const H5F_ACC_RDONLY: libc::c_uint = 0x0000;
pub const H5F_ACC_RDWR: libc::c_uint = 0x0001;
pub const H5F_ACC_TRUNC: libc::c_uint = 0x0002;
pub const H5P_DEFAULT: i64 = 0;
pub const H5S_ALL: i64 = 0;
pub const H5S_UNLIMITED: u64 = u64::MAX;
pub const H5T_COMPOUND: libc::c_int = 3;

// Dynamic API structure loaded at runtime
#[allow(non_snake_case)]
pub struct Hdf5Api {
    pub H5open: unsafe extern "C" fn() -> libc::c_int,
    pub H5close: unsafe extern "C" fn() -> libc::c_int,
    pub H5Fopen: unsafe extern "C" fn(*const libc::c_char, libc::c_uint, i64) -> i64,
    pub H5Fcreate: unsafe extern "C" fn(*const libc::c_char, libc::c_uint, i64, i64) -> i64,
    pub H5Fclose: unsafe extern "C" fn(i64) -> libc::c_int,
    pub H5Gcreate2: unsafe extern "C" fn(i64, *const libc::c_char, i64, i64, i64) -> i64,
    pub H5Gopen2: unsafe extern "C" fn(i64, *const libc::c_char, i64) -> i64,
    pub H5Gclose: unsafe extern "C" fn(i64) -> libc::c_int,
    pub H5Dcreate2: unsafe extern "C" fn(i64, *const libc::c_char, i64, i64, i64, i64, i64) -> i64,
    pub H5Dopen2: unsafe extern "C" fn(i64, *const libc::c_char, i64) -> i64,
    pub H5Dwrite: unsafe extern "C" fn(i64, i64, i64, i64, i64, *const libc::c_void) -> libc::c_int,
    pub H5Dread: unsafe extern "C" fn(i64, i64, i64, i64, i64, *mut libc::c_void) -> libc::c_int,
    pub H5Dclose: unsafe extern "C" fn(i64) -> libc::c_int,
    pub H5Screate_simple: unsafe extern "C" fn(libc::c_int, *const u64, *const u64) -> i64,
    pub H5Sclose: unsafe extern "C" fn(i64) -> libc::c_int,
    pub H5Tcopy: unsafe extern "C" fn(i64) -> i64,
    pub H5Tclose: unsafe extern "C" fn(i64) -> libc::c_int,
    pub H5Tcreate: unsafe extern "C" fn(libc::c_int, libc::size_t) -> i64,
    pub H5Tinsert: unsafe extern "C" fn(i64, *const libc::c_char, libc::size_t, i64) -> libc::c_int,
    pub H5Dset_extent: unsafe extern "C" fn(i64, *const u64) -> libc::c_int,
    pub H5Lexists: unsafe extern "C" fn(i64, *const libc::c_char, i64) -> libc::c_int,

    // Datatype cache
    pub h5t_native_double: i64,
    pub h5t_native_int: i64,
    pub h5t_complex: i64,
}

struct Library {
    handle: *mut std::ffi::c_void,
}

impl Library {
    unsafe fn load(path: &Path) -> Result<Self, String> {
        #[cfg(unix)]
        {
            // Moved inside the unix block so Windows doesn't complain about it
            let path_str = path.to_string_lossy();
            let c_path = CString::new(path_str.as_ref()).map_err(|e| e.to_string())?;
            let handle =
                unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
            if handle.is_null() {
                let err = unsafe { libc::dlerror() };
                let err_msg = if err.is_null() {
                    "Unknown dlopen error".to_string()
                } else {
                    // Added std::ffi:: here since we removed the global import
                    unsafe { std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned() }
                };
                return Err(err_msg);
            }
            Ok(Self { handle })
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            let path_os = path.as_os_str();
            let mut wide: Vec<u16> = path_os.encode_wide().collect();
            wide.push(0);
            unsafe extern "system" {
                fn LoadLibraryW(lpLibFileName: *const u16) -> *mut std::ffi::c_void;
                fn GetLastError() -> u32;
            }
            let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
            if handle.is_null() {
                let err_code = unsafe { GetLastError() };
                return Err(format!("Windows error code: {}", err_code));
            }
            Ok(Self { handle })
        }
    }

    unsafe fn get_symbol(&self, name: &str) -> Option<*mut std::ffi::c_void> {
        let c_name = CString::new(name).ok()?;
        #[cfg(unix)]
        {
            let ptr = unsafe { libc::dlsym(self.handle, c_name.as_ptr()) };
            if ptr.is_null() { None } else { Some(ptr) }
        }
        #[cfg(windows)]
        {
            unsafe extern "system" {
                fn GetProcAddress(
                    hModule: *mut std::ffi::c_void,
                    lpProcName: *const libc::c_char,
                ) -> *mut std::ffi::c_void;
            }
            let ptr = unsafe { GetProcAddress(self.handle, c_name.as_ptr()) };
            if ptr.is_null() { None } else { Some(ptr) }
        }
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            #[cfg(unix)]
            {
                libc::dlclose(self.handle);
            }
            #[cfg(windows)]
            {
                unsafe extern "system" {
                    fn FreeLibrary(hModule: *mut std::ffi::c_void) -> libc::c_int;
                }
                FreeLibrary(self.handle);
            }
        }
    }
}

fn find_hdf5_lib_path() -> Option<PathBuf> {
    // 1. Check user override environment variable
    if let Ok(val) = std::env::var("LUNA_HDF5_LIB") {
        let p = PathBuf::from(val);
        if p.exists() {
            return Some(p);
        }
    }

    let (_ext, names) = if cfg!(target_os = "macos") {
        (
            "dylib",
            vec![
                "libhdf5.dylib",
                "libhdf5.310.dylib",
                "libhdf5.200.dylib",
                "libhdf5.103.dylib",
                "libhdf5.100.dylib",
                "libhdf5.10.dylib",
            ],
        )
    } else if cfg!(windows) {
        ("dll", vec!["hdf5.dll", "libhdf5.dll", "libhdf5-0.dll"])
    } else {
        (
            "so",
            vec![
                "libhdf5.so",
                "libhdf5.so.310",
                "libhdf5.so.200",
                "libhdf5.so.103",
                "libhdf5.so.100",
                "libhdf5.so.10",
            ],
        )
    };

    // 2. Try loading standard names from default system linker search paths
    for name in &names {
        unsafe {
            #[cfg(unix)]
            {
                if let Ok(c_name) = CString::new(*name) {
                    let handle = libc::dlopen(c_name.as_ptr(), libc::RTLD_LAZY);
                    if !handle.is_null() {
                        libc::dlclose(handle);
                        return Some(PathBuf::from(*name));
                    }
                }
            }
            #[cfg(windows)]
            {
                use std::os::windows::ffi::OsStrExt;
                let path_os = std::ffi::OsStr::new(*name);
                let mut wide: Vec<u16> = path_os.encode_wide().collect();
                wide.push(0);
                unsafe extern "system" {
                    fn LoadLibraryW(lpLibFileName: *const u16) -> *mut std::ffi::c_void;
                    fn FreeLibrary(hModule: *mut std::ffi::c_void) -> libc::c_int;
                }
                let handle = LoadLibraryW(wide.as_ptr());
                if !handle.is_null() {
                    FreeLibrary(handle);
                    return Some(PathBuf::from(*name));
                }
            }
        }
    }

    // 3. Search in ~/.julia/artifacts or Windows equivalent
    let home_var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    if let Some(home) = std::env::var_os(home_var) {
        let artifacts_path = Path::new(&home).join(".julia/artifacts");
        if artifacts_path.exists()
            && let Ok(entries) = std::fs::read_dir(artifacts_path)
        {
            for entry in entries.flatten() {
                let sub_path = if cfg!(windows) { "bin" } else { "lib" };
                for name in &names {
                    let lib_path = entry.path().join(sub_path).join(name);
                    if lib_path.exists() {
                        return Some(lib_path);
                    }
                }
            }
        }
    }
    None
}

static HDF5_API: OnceLock<Result<Hdf5Api, String>> = OnceLock::new();

#[allow(non_snake_case)]
pub fn get_hdf5_api() -> Result<&'static Hdf5Api, String> {
    HDF5_API
        .get_or_init(|| {
            let path = find_hdf5_lib_path().ok_or_else(|| {
                let err_name = if cfg!(target_os = "macos") {
                    "libhdf5.dylib"
                } else if cfg!(windows) {
                    "hdf5.dll"
                } else {
                    "libhdf5.so"
                };
                format!(
                    "Could not locate {} in standard search paths or Julia's artifact cache.",
                    err_name
                )
            })?;

            unsafe {
                let lib = Library::load(&path).map_err(|err_msg| {
                    format!("Failed to load HDF5 library at {:?}: {}", path, err_msg)
                })?;

                macro_rules! load_sym {
                    ($sym:expr, $ty:ty) => {{
                        let ptr = lib
                            .get_symbol($sym)
                            .ok_or_else(|| format!("Failed to locate HDF5 symbol: {}", $sym))?;
                        std::mem::transmute::<*mut std::ffi::c_void, $ty>(ptr)
                    }};
                }

                let H5open = load_sym!("H5open", unsafe extern "C" fn() -> libc::c_int);
                let H5close = load_sym!("H5close", unsafe extern "C" fn() -> libc::c_int);
                let H5Fopen = load_sym!(
                    "H5Fopen",
                    unsafe extern "C" fn(*const libc::c_char, libc::c_uint, i64) -> i64
                );
                let H5Fcreate = load_sym!(
                    "H5Fcreate",
                    unsafe extern "C" fn(*const libc::c_char, libc::c_uint, i64, i64) -> i64
                );
                let H5Fclose = load_sym!("H5Fclose", unsafe extern "C" fn(i64) -> libc::c_int);
                let H5Gcreate2 = load_sym!(
                    "H5Gcreate2",
                    unsafe extern "C" fn(i64, *const libc::c_char, i64, i64, i64) -> i64
                );
                let H5Gopen2 = load_sym!(
                    "H5Gopen2",
                    unsafe extern "C" fn(i64, *const libc::c_char, i64) -> i64
                );
                let H5Gclose = load_sym!("H5Gclose", unsafe extern "C" fn(i64) -> libc::c_int);
                let H5Dcreate2 = load_sym!(
                    "H5Dcreate2",
                    unsafe extern "C" fn(i64, *const libc::c_char, i64, i64, i64, i64, i64) -> i64
                );
                let H5Dopen2 = load_sym!(
                    "H5Dopen2",
                    unsafe extern "C" fn(i64, *const libc::c_char, i64) -> i64
                );
                let H5Dwrite = load_sym!(
                    "H5Dwrite",
                    unsafe extern "C" fn(
                        i64,
                        i64,
                        i64,
                        i64,
                        i64,
                        *const libc::c_void,
                    ) -> libc::c_int
                );
                let H5Dread = load_sym!(
                    "H5Dread",
                    unsafe extern "C" fn(i64, i64, i64, i64, i64, *mut libc::c_void) -> libc::c_int
                );
                let H5Dclose = load_sym!("H5Dclose", unsafe extern "C" fn(i64) -> libc::c_int);
                let H5Screate_simple = load_sym!(
                    "H5Screate_simple",
                    unsafe extern "C" fn(libc::c_int, *const u64, *const u64) -> i64
                );
                let H5Sclose = load_sym!("H5Sclose", unsafe extern "C" fn(i64) -> libc::c_int);
                let H5Tcopy = load_sym!("H5Tcopy", unsafe extern "C" fn(i64) -> i64);
                let H5Tclose = load_sym!("H5Tclose", unsafe extern "C" fn(i64) -> libc::c_int);
                let H5Tcreate = load_sym!(
                    "H5Tcreate",
                    unsafe extern "C" fn(libc::c_int, libc::size_t) -> i64
                );
                let H5Tinsert = load_sym!(
                    "H5Tinsert",
                    unsafe extern "C" fn(
                        i64,
                        *const libc::c_char,
                        libc::size_t,
                        i64,
                    ) -> libc::c_int
                );
                let H5Dset_extent = load_sym!(
                    "H5Dset_extent",
                    unsafe extern "C" fn(i64, *const u64) -> libc::c_int
                );
                let H5Lexists = load_sym!(
                    "H5Lexists",
                    unsafe extern "C" fn(i64, *const libc::c_char, i64) -> libc::c_int
                );

                let h5t_native_double_ptr = load_sym!("H5T_NATIVE_DOUBLE_g", *const i64);
                let h5t_native_int_ptr = load_sym!("H5T_NATIVE_INT_g", *const i64);

                let h5t_native_double = *h5t_native_double_ptr;
                let h5t_native_int = *h5t_native_int_ptr;

                let h5t_complex = H5Tcreate(H5T_COMPOUND, 16);
                let r_name = CString::new("r").unwrap();
                let i_name = CString::new("i").unwrap();
                H5Tinsert(h5t_complex, r_name.as_ptr(), 0, h5t_native_double);
                H5Tinsert(h5t_complex, i_name.as_ptr(), 8, h5t_native_double);

                H5open();

                std::mem::forget(lib);

                Ok(Hdf5Api {
                    H5open,
                    H5close,
                    H5Fopen,
                    H5Fcreate,
                    H5Fclose,
                    H5Gcreate2,
                    H5Gopen2,
                    H5Gclose,
                    H5Dcreate2,
                    H5Dopen2,
                    H5Dwrite,
                    H5Dread,
                    H5Dclose,
                    H5Screate_simple,
                    H5Sclose,
                    H5Tcopy,
                    H5Tclose,
                    H5Tcreate,
                    H5Tinsert,
                    H5Dset_extent,
                    H5Lexists,
                    h5t_native_double,
                    h5t_native_int,
                    h5t_complex,
                })
            }
        })
        .as_ref()
        .map_err(|e| e.clone())
}

pub struct Hdf5Writer {
    pub file_id: i64,
    api: &'static Hdf5Api,
}

impl Hdf5Writer {
    pub fn open_or_create(fpath: &str) -> Result<Self, String> {
        let api = get_hdf5_api()?;
        let c_fpath = CString::new(fpath).map_err(|_| "Invalid filepath".to_string())?;
        unsafe {
            let file_id = (api.H5Fopen)(c_fpath.as_ptr(), H5F_ACC_RDWR, H5P_DEFAULT);
            if file_id < 0 {
                let file_id =
                    (api.H5Fcreate)(c_fpath.as_ptr(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                if file_id < 0 {
                    return Err(format!("Failed to open or create HDF5 file: {}", fpath));
                }
                return Ok(Self { file_id, api });
            }
            Ok(Self { file_id, api })
        }
    }

    pub fn open_existing(fpath: &str) -> Result<Self, String> {
        let api = get_hdf5_api()?;
        let c_fpath = CString::new(fpath).map_err(|_| "Invalid filepath".to_string())?;
        unsafe {
            let file_id = (api.H5Fopen)(c_fpath.as_ptr(), H5F_ACC_RDWR, H5P_DEFAULT);
            if file_id < 0 {
                return Err(format!("Failed to open existing HDF5 file: {}", fpath));
            }
            Ok(Self { file_id, api })
        }
    }

    pub fn create_group(&self, name: &str) -> Result<i64, String> {
        let c_name = CString::new(name).map_err(|_| "Invalid group name".to_string())?;
        unsafe {
            let exists = (self.api.H5Lexists)(self.file_id, c_name.as_ptr(), H5P_DEFAULT);
            if exists > 0 {
                let group_id = (self.api.H5Gopen2)(self.file_id, c_name.as_ptr(), H5P_DEFAULT);
                if group_id >= 0 {
                    return Ok(group_id);
                }
            }
            let group_id = (self.api.H5Gcreate2)(
                self.file_id,
                c_name.as_ptr(),
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT,
            );
            if group_id < 0 {
                return Err(format!("Failed to create group: {}", name));
            }
            Ok(group_id)
        }
    }

    pub fn create_dataset_2d(
        &self,
        loc_id: i64,
        name: &str,
        dtype: i64,
        dims: &[u64],
        maxdims: &[u64],
    ) -> Result<i64, String> {
        let c_name = CString::new(name).map_err(|_| "Invalid dataset name".to_string())?;
        unsafe {
            let exists = (self.api.H5Lexists)(loc_id, c_name.as_ptr(), H5P_DEFAULT);
            if exists > 0 {
                let dset_id = (self.api.H5Dopen2)(loc_id, c_name.as_ptr(), H5P_DEFAULT);
                if dset_id >= 0 {
                    return Ok(dset_id);
                }
            }
            let space_id = (self.api.H5Screate_simple)(
                dims.len() as libc::c_int,
                dims.as_ptr(),
                maxdims.as_ptr(),
            );
            if space_id < 0 {
                return Err("Failed to create dataspace".to_string());
            }
            let dset_id = (self.api.H5Dcreate2)(
                loc_id,
                c_name.as_ptr(),
                dtype,
                space_id,
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT,
            );
            (self.api.H5Sclose)(space_id);
            if dset_id < 0 {
                return Err(format!("Failed to create dataset: {}", name));
            }
            Ok(dset_id)
        }
    }

    pub fn open_dataset_2d(&self, loc_id: i64, name: &str) -> Result<i64, String> {
        let c_name = CString::new(name).map_err(|_| "Invalid dataset name".to_string())?;
        unsafe {
            let exists = (self.api.H5Lexists)(loc_id, c_name.as_ptr(), H5P_DEFAULT);
            if exists <= 0 {
                return Err(format!("Failed to open dataset: {}", name));
            }
            let dset_id = (self.api.H5Dopen2)(loc_id, c_name.as_ptr(), H5P_DEFAULT);
            if dset_id < 0 {
                return Err(format!("Failed to open dataset: {}", name));
            }
            Ok(dset_id)
        }
    }

    pub fn with_existing_int_dataset_2d<T, F>(
        &self,
        name: &str,
        data: &mut [i32],
        mut op: F,
    ) -> Result<T, String>
    where
        F: FnMut(&Self, i64, &mut [i32]) -> Result<T, String>,
    {
        let dset_id = self.open_dataset_2d(self.file_id, name)?;
        let result = op(self, dset_id, data);
        self.close_dataset(dset_id);
        result
    }

    pub fn write_dataset_f64(&self, dset_id: i64, data: &[f64]) -> Result<(), String> {
        unsafe {
            let res = (self.api.H5Dwrite)(
                dset_id,
                self.api.h5t_native_double,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                data.as_ptr() as *const libc::c_void,
            );
            if res < 0 {
                return Err("Failed to write double dataset".to_string());
            }
            Ok(())
        }
    }

    pub fn write_dataset_complex(&self, dset_id: i64, data: &[Complex<f64>]) -> Result<(), String> {
        unsafe {
            let res = (self.api.H5Dwrite)(
                dset_id,
                self.api.h5t_complex,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                data.as_ptr() as *const libc::c_void,
            );
            if res < 0 {
                return Err("Failed to write complex dataset".to_string());
            }
            Ok(())
        }
    }

    pub fn write_dataset_int(&self, dset_id: i64, data: &[i32]) -> Result<(), String> {
        unsafe {
            let res = (self.api.H5Dwrite)(
                dset_id,
                self.api.h5t_native_int,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                data.as_ptr() as *const libc::c_void,
            );
            if res < 0 {
                return Err("Failed to write int dataset".to_string());
            }
            Ok(())
        }
    }

    pub fn read_dataset_int(&self, dset_id: i64, data: &mut [i32]) -> Result<(), String> {
        unsafe {
            let res = (self.api.H5Dread)(
                dset_id,
                self.api.h5t_native_int,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                data.as_mut_ptr() as *mut libc::c_void,
            );
            if res < 0 {
                return Err("Failed to read int dataset".to_string());
            }
            Ok(())
        }
    }

    pub fn set_dataset_extent(&self, dset_id: i64, size: &[u64]) -> Result<(), String> {
        unsafe {
            let res = (self.api.H5Dset_extent)(dset_id, size.as_ptr());
            if res < 0 {
                return Err("Failed to resize dataset".to_string());
            }
            Ok(())
        }
    }

    pub fn close_dataset(&self, dset_id: i64) {
        unsafe {
            (self.api.H5Dclose)(dset_id);
        }
    }

    pub fn close_group(&self, group_id: i64) {
        unsafe {
            (self.api.H5Gclose)(group_id);
        }
    }
}

impl Drop for Hdf5Writer {
    fn drop(&mut self) {
        unsafe {
            (self.api.H5Fclose)(self.file_id);
        }
    }
}
