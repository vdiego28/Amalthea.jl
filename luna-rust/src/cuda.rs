#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::ffi::CString;
use std::sync::OnceLock;

// CUDA types
pub type CUresult = libc::c_int;
pub type CUdevice = libc::c_int;
pub type CUcontext = *mut libc::c_void;
pub type CUmodule = *mut libc::c_void;
pub type CUfunction = *mut libc::c_void;
pub type CUdeviceptr = u64;
pub type CUstream = *mut libc::c_void;

pub type cublasStatus_t = libc::c_int;
pub type cublasHandle_t = *mut libc::c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
}

pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

// Embed the PTX compiled by build.rs
pub const KERNELS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

struct Library {
    handle: *mut std::ffi::c_void,
}

impl Library {
    unsafe fn load(names: &[&str]) -> Result<Self, String> {
        #[cfg(unix)]
        {
            let mut last_err = String::new();
            for name in names {
                let c_name = CString::new(*name).unwrap();
                let handle = unsafe { libc::dlopen(c_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
                if !handle.is_null() {
                    return Ok(Self { handle });
                }
                let err = unsafe { libc::dlerror() };
                if !err.is_null() {
                    last_err = unsafe { std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned() };
                }
            }
            Err(format!("dlopen failed for {:?}. Last error: {}", names, last_err))
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            unsafe extern "system" {
                fn LoadLibraryW(lpLibFileName: *const u16) -> *mut std::ffi::c_void;
                fn GetLastError() -> u32;
            }
            let mut last_err = 0;
            for name in names {
                let path_os = std::ffi::OsStr::new(name);
                let mut wide: Vec<u16> = path_os.encode_wide().collect();
                wide.push(0);
                let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
                if !handle.is_null() {
                    return Ok(Self { handle });
                }
                last_err = unsafe { GetLastError() };
            }
            Err(format!("LoadLibraryW failed for {:?}. Last error: {}", names, last_err))
        }
    }

    unsafe fn get_symbol(&self, name: &str) -> Result<*mut std::ffi::c_void, String> {
        let c_name = CString::new(name).unwrap();
        #[cfg(unix)]
        {
            let ptr = unsafe { libc::dlsym(self.handle, c_name.as_ptr()) };
            if ptr.is_null() {
                return Err(format!("Symbol not found: {}", name));
            }
            Ok(ptr)
        }
        #[cfg(windows)]
        {
            unsafe extern "system" {
                fn GetProcAddress(hModule: *mut std::ffi::c_void, lpProcName: *const libc::c_char) -> *mut std::ffi::c_void;
            }
            let ptr = unsafe { GetProcAddress(self.handle, c_name.as_ptr()) };
            if ptr.is_null() {
                return Err(format!("Symbol not found: {}", name));
            }
            Ok(ptr)
        }
    }
}

unsafe impl Send for Library {}
unsafe impl Sync for Library {}

pub struct CudaDriverApi {
    _lib: Library,
    pub cuInit: unsafe extern "C" fn(flags: libc::c_uint) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(device: *mut CUdevice, ordinal: libc::c_int) -> CUresult,
    pub cuCtxCreate_v2: unsafe extern "C" fn(pctx: *mut CUcontext, flags: libc::c_uint, dev: CUdevice) -> CUresult,
    pub cuCtxDestroy_v2: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    pub cuModuleLoadData: unsafe extern "C" fn(module: *mut CUmodule, image: *const libc::c_void) -> CUresult,
    pub cuModuleUnload: unsafe extern "C" fn(module: CUmodule) -> CUresult,
    pub cuModuleGetFunction: unsafe extern "C" fn(hfunc: *mut CUfunction, hmod: CUmodule, name: *const libc::c_char) -> CUresult,
    pub cuLaunchKernel: unsafe extern "C" fn(
        f: CUfunction,
        gridDimX: libc::c_uint,
        gridDimY: libc::c_uint,
        gridDimZ: libc::c_uint,
        blockDimX: libc::c_uint,
        blockDimY: libc::c_uint,
        blockDimZ: libc::c_uint,
        sharedMemBytes: libc::c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut libc::c_void,
        extra: *mut *mut libc::c_void,
    ) -> CUresult,
    pub cuMemAlloc_v2: unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: libc::size_t) -> CUresult,
    pub cuMemFree_v2: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
    pub cuMemcpyHtoD_v2: unsafe extern "C" fn(dstDevice: CUdeviceptr, srcHost: *const libc::c_void, ByteCount: libc::size_t) -> CUresult,
    pub cuMemcpyDtoH_v2: unsafe extern "C" fn(dstHost: *mut libc::c_void, srcDevice: CUdeviceptr, ByteCount: libc::size_t) -> CUresult,
    pub cuMemcpyDtoD_v2: unsafe extern "C" fn(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: libc::size_t) -> CUresult,
    pub cuCtxGetCurrent: unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult,
    pub cuCtxSetCurrent: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    pub cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
    pub cuGetErrorString: unsafe extern "C" fn(error: CUresult, pStr: *mut *const libc::c_char) -> CUresult,
}

#[allow(non_camel_case_types)]
pub type cufftHandle = libc::c_int;
#[allow(non_camel_case_types)]
pub type cufftResult = libc::c_int;
#[allow(non_camel_case_types)]
pub type cufftType = libc::c_int;
pub const CUFFT_D2Z: cufftType = 106;
pub const CUFFT_Z2D: cufftType = 108;
pub const CUFFT_Z2Z: cufftType = 105;

pub struct CufftApi {
    _lib: Library,
    pub cufftPlan1d: unsafe extern "C" fn(plan: *mut cufftHandle, nx: libc::c_int, type_: cufftType, batch: libc::c_int) -> cufftResult,
    pub cufftDestroy: unsafe extern "C" fn(plan: cufftHandle) -> cufftResult,
    pub cufftExecD2Z: unsafe extern "C" fn(plan: cufftHandle, idata: *mut f64, odata: *mut libc::c_void) -> cufftResult,
    pub cufftExecZ2D: unsafe extern "C" fn(plan: cufftHandle, idata: *mut libc::c_void, odata: *mut f64) -> cufftResult,
    pub cufftExecZ2Z: unsafe extern "C" fn(plan: cufftHandle, idata: *mut libc::c_void, odata: *mut libc::c_void, direction: libc::c_int) -> cufftResult,
}

pub fn get_cufft_api() -> Result<&'static CufftApi, String> {
    static API: OnceLock<Result<CufftApi, String>> = OnceLock::new();
    API.get_or_init(|| unsafe {
        let names = &[
            "libcufft.so.11",
            "libcufft.so.10",
            "libcufft.so",
            "cufft64_11.dll",
            "cufft64_10.dll",
            "cufft.dll",
        ];
        let lib = Library::load(names)?;
        Ok(CufftApi {
            cufftPlan1d: std::mem::transmute(lib.get_symbol("cufftPlan1d")?),
            cufftDestroy: std::mem::transmute(lib.get_symbol("cufftDestroy")?),
            cufftExecD2Z: std::mem::transmute(lib.get_symbol("cufftExecD2Z")?),
            cufftExecZ2D: std::mem::transmute(lib.get_symbol("cufftExecZ2D")?),
            cufftExecZ2Z: std::mem::transmute(lib.get_symbol("cufftExecZ2Z")?),
            _lib: lib,
        })
    }).as_ref().map_err(|e| e.clone())
}

pub struct CublasApi {
    _lib: Library,
    pub cublasCreate_v2: unsafe extern "C" fn(handle: *mut cublasHandle_t) -> cublasStatus_t,
    pub cublasDestroy_v2: unsafe extern "C" fn(handle: cublasHandle_t) -> cublasStatus_t,
    pub cublasDgemv_v2: unsafe extern "C" fn(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: libc::c_int,
        n: libc::c_int,
        alpha: *const f64,
        A: *const f64,
        lda: libc::c_int,
        x: *const f64,
        incx: libc::c_int,
        beta: *const f64,
        y: *mut f64,
        incy: libc::c_int,
    ) -> cublasStatus_t,
}

pub fn get_driver_api() -> Result<&'static CudaDriverApi, String> {
    static API: OnceLock<Result<CudaDriverApi, String>> = OnceLock::new();
    API.get_or_init(|| unsafe {
        let names = &["libcuda.so.1", "libcuda.so", "nvcuda.dll"];
        let lib = Library::load(names)?;
        Ok(CudaDriverApi {
            cuInit: std::mem::transmute(lib.get_symbol("cuInit")?),
            cuDeviceGet: std::mem::transmute(lib.get_symbol("cuDeviceGet")?),
            cuCtxCreate_v2: std::mem::transmute(lib.get_symbol("cuCtxCreate_v2")?),
            cuCtxDestroy_v2: std::mem::transmute(lib.get_symbol("cuCtxDestroy_v2")?),
            cuModuleLoadData: std::mem::transmute(lib.get_symbol("cuModuleLoadData")?),
            cuModuleUnload: std::mem::transmute(lib.get_symbol("cuModuleUnload")?),
            cuModuleGetFunction: std::mem::transmute(lib.get_symbol("cuModuleGetFunction")?),
            cuLaunchKernel: std::mem::transmute(lib.get_symbol("cuLaunchKernel")?),
            cuMemAlloc_v2: std::mem::transmute(lib.get_symbol("cuMemAlloc_v2")?),
            cuMemFree_v2: std::mem::transmute(lib.get_symbol("cuMemFree_v2")?),
            cuMemcpyHtoD_v2: std::mem::transmute(lib.get_symbol("cuMemcpyHtoD_v2")?),
            cuMemcpyDtoH_v2: std::mem::transmute(lib.get_symbol("cuMemcpyDtoH_v2")?),
            cuMemcpyDtoD_v2: std::mem::transmute(lib.get_symbol("cuMemcpyDtoD_v2")?),
            cuCtxGetCurrent: std::mem::transmute(lib.get_symbol("cuCtxGetCurrent")?),
            cuCtxSetCurrent: std::mem::transmute(lib.get_symbol("cuCtxSetCurrent")?),
            cuCtxSynchronize: std::mem::transmute(lib.get_symbol("cuCtxSynchronize")?),
            cuGetErrorString: std::mem::transmute(lib.get_symbol("cuGetErrorString")?),
            _lib: lib,
        })
    }).as_ref().map_err(|e| e.clone())
}

pub fn get_cublas_api() -> Result<&'static CublasApi, String> {
    static API: OnceLock<Result<CublasApi, String>> = OnceLock::new();
    API.get_or_init(|| unsafe {
        let names = &[
            "libcublas.so.12",
            "libcublas.so.11",
            "libcublas.so",
            "cublas64_12.dll",
            "cublas64_11.dll",
            "cublas.dll",
        ];
        let lib = Library::load(names)?;
        Ok(CublasApi {
            cublasCreate_v2: std::mem::transmute(lib.get_symbol("cublasCreate_v2")?),
            cublasDestroy_v2: std::mem::transmute(lib.get_symbol("cublasDestroy_v2")?),
            cublasDgemv_v2: std::mem::transmute(lib.get_symbol("cublasDgemv_v2")?),
            _lib: lib,
        })
    }).as_ref().map_err(|e| e.clone())
}

// Global GPU Context container
pub struct GpuContext {
    pub device: CUdevice,
    pub context: CUcontext,
    pub cublas_handle: cublasHandle_t,
    pub module: CUmodule,
    pub raman_fn: CUfunction,
    pub ppt_fn: CUfunction,
    pub apply_prop_fn: CUfunction,
    pub rk45_accumulate_stage_fn: CUfunction,
    pub rk45_accumulate_error_fn: CUfunction,
    pub weaknorm_elem_fn: CUfunction,
    pub weaknorm_reduce_fn: CUfunction,
    pub rhs_mode_avg_real_fn: CUfunction,
    pub rhs_mode_avg_env_fn: CUfunction,
    pub apply_time_window_fn: CUfunction,
    pub plasma_fraction_fn: CUfunction,
    pub plasma_phase_fn: CUfunction,
    pub plasma_current_fn: CUfunction,
    pub plasma_polarization_fn: CUfunction,
}

unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl Drop for GpuContext {
    fn drop(&mut self) {
        if let Ok(driver) = get_driver_api() {
            unsafe {
                let cublas = get_cublas_api();
                if let Ok(cb) = cublas {
                    (cb.cublas_destroy_func())(self.cublas_handle);
                }
                (driver.cuModuleUnload)(self.module);
                (driver.cuCtxDestroy_v2)(self.context);
            }
        }
    }
}

// Implement accessor so we don't have to call function pointer fields directly with weird syntax
impl CublasApi {
    pub fn cublas_destroy_func(&self) -> unsafe extern "C" fn(cublasHandle_t) -> cublasStatus_t {
        self.cublasDestroy_v2
    }
}

pub static GPU_CONTEXT: OnceLock<Result<GpuContext, String>> = OnceLock::new();

pub fn get_gpu_context() -> Option<&'static GpuContext> {
    GPU_CONTEXT.get()?.as_ref().ok()
}

pub fn init_gpu_context() -> Result<&'static GpuContext, String> {
    GPU_CONTEXT.get_or_init(|| {
        let driver = get_driver_api()?;
        let cublas = get_cublas_api()?;
        
        unsafe {
            let mut res = (driver.cuInit)(0);
            if res != 0 {
                return Err(format!("cuInit failed: {}", res));
            }
            
            let mut device = 0;
            res = (driver.cuDeviceGet)(&mut device, 0);
            if res != 0 {
                return Err(format!("cuDeviceGet failed: {}", res));
            }
            
            let mut context = std::ptr::null_mut();
            res = (driver.cuCtxCreate_v2)(&mut context, 0, device);
            if res != 0 {
                return Err(format!("cuCtxCreate_v2 failed: {}", res));
            }
            
            // Activate the context on the main thread for initialization
            res = (driver.cuCtxSetCurrent)(context);
            if res != 0 {
                (driver.cuCtxDestroy_v2)(context);
                return Err(format!("cuCtxSetCurrent failed during initialization: {}", res));
            }
            
            let mut cublas_handle = std::ptr::null_mut();
            let c_res = (cublas.cublasCreate_v2)(&mut cublas_handle);
            if c_res != CUBLAS_STATUS_SUCCESS {
                (driver.cuCtxDestroy_v2)(context);
                return Err(format!("cublasCreate_v2 failed: {}", c_res));
            }
            
            // JIT load the kernels PTX
            let ptx_c = CString::new(KERNELS_PTX).map_err(|e| e.to_string())?;
            let mut module = std::ptr::null_mut();
            res = (driver.cuModuleLoadData)(&mut module, ptx_c.as_ptr() as *const libc::c_void);
            if res != 0 {
                (cublas.cublasDestroy_v2)(cublas_handle);
                (driver.cuCtxDestroy_v2)(context);
                return Err(format!("cuModuleLoadData failed (PTX may be invalid or not compiled): {}", res));
            }
            
            let mut raman_fn = std::ptr::null_mut();
            let fn_name_raman = CString::new("raman_ade_kernel").unwrap();
            res = (driver.cuModuleGetFunction)(&mut raman_fn, module, fn_name_raman.as_ptr());
            if res != 0 {
                (driver.cuModuleUnload)(module);
                (cublas.cublasDestroy_v2)(cublas_handle);
                (driver.cuCtxDestroy_v2)(context);
                return Err(format!("cuModuleGetFunction for raman_ade_kernel failed: {}", res));
            }
            
            let mut ppt_fn = std::ptr::null_mut();
            let fn_name_ppt = CString::new("ppt_ionization_kernel").unwrap();
            res = (driver.cuModuleGetFunction)(&mut ppt_fn, module, fn_name_ppt.as_ptr());
            if res != 0 { return Err("cuModuleGetFunction failed".to_string()); }

            let mut apply_prop_fn = std::ptr::null_mut();
            res = (driver.cuModuleGetFunction)(&mut apply_prop_fn, module, CString::new("apply_prop_kernel").unwrap().as_ptr());
            if res != 0 { return Err("cuModuleGetFunction apply_prop_kernel failed".to_string()); }
            
            let mut rk45_accumulate_stage_fn = std::ptr::null_mut();
            res = (driver.cuModuleGetFunction)(&mut rk45_accumulate_stage_fn, module, CString::new("rk45_accumulate_stage_kernel").unwrap().as_ptr());
            if res != 0 { return Err("cuModuleGetFunction rk45_accumulate_stage_kernel failed".to_string()); }

            let mut rk45_accumulate_error_fn = std::ptr::null_mut();
            res = (driver.cuModuleGetFunction)(&mut rk45_accumulate_error_fn, module, CString::new("rk45_accumulate_error_kernel").unwrap().as_ptr());
            if res != 0 { return Err("cuModuleGetFunction rk45_accumulate_error_kernel failed".to_string()); }

            let mut weaknorm_elem_fn = std::ptr::null_mut();
            res = (driver.cuModuleGetFunction)(&mut weaknorm_elem_fn, module, CString::new("weaknorm_elem_kernel").unwrap().as_ptr());
            if res != 0 { return Err("cuModuleGetFunction weaknorm_elem_kernel failed".to_string()); }

            let mut weaknorm_reduce_fn = std::ptr::null_mut();
            res = (driver.cuModuleGetFunction)(&mut weaknorm_reduce_fn, module, CString::new("weaknorm_reduce_kernel").unwrap().as_ptr());
            if res != 0 { return Err("cuModuleGetFunction weaknorm_reduce_kernel failed".to_string()); }

            let mut rhs_mode_avg_real_fn = std::ptr::null_mut();
            res = (driver.cuModuleGetFunction)(&mut rhs_mode_avg_real_fn, module, CString::new("rhs_mode_avg_real_kernel").unwrap().as_ptr());
            if res != 0 { return Err("cuModuleGetFunction rhs_mode_avg_real_kernel failed".to_string()); }

            let mut rhs_mode_avg_env_fn = std::ptr::null_mut();
            res = (driver.cuModuleGetFunction)(&mut rhs_mode_avg_env_fn, module, CString::new("rhs_mode_avg_env_kernel").unwrap().as_ptr());
            if res != 0 { return Err("cuModuleGetFunction rhs_mode_avg_env_kernel failed".to_string()); }

            let mut apply_time_window_fn = std::ptr::null_mut();
            res = (driver.cuModuleGetFunction)(&mut apply_time_window_fn, module, CString::new("apply_time_window_kernel").unwrap().as_ptr());
            if res != 0 { return Err("cuModuleGetFunction apply_time_window_kernel failed".to_string()); }

            let mut plasma_fraction_fn = std::ptr::null_mut();
            res = (driver.cuModuleGetFunction)(&mut plasma_fraction_fn, module, CString::new("plasma_fraction_kernel").unwrap().as_ptr());
            if res != 0 { return Err("cuModuleGetFunction plasma_fraction_kernel failed".to_string()); }

            let mut plasma_phase_fn = std::ptr::null_mut();
            res = (driver.cuModuleGetFunction)(&mut plasma_phase_fn, module, CString::new("plasma_phase_kernel").unwrap().as_ptr());
            if res != 0 { return Err("cuModuleGetFunction plasma_phase_kernel failed".to_string()); }

            let mut plasma_current_fn = std::ptr::null_mut();
            res = (driver.cuModuleGetFunction)(&mut plasma_current_fn, module, CString::new("plasma_current_kernel").unwrap().as_ptr());
            if res != 0 { return Err("cuModuleGetFunction plasma_current_kernel failed".to_string()); }

            let mut plasma_polarization_fn = std::ptr::null_mut();
            res = (driver.cuModuleGetFunction)(&mut plasma_polarization_fn, module, CString::new("plasma_polarization_kernel").unwrap().as_ptr());
            if res != 0 { return Err("cuModuleGetFunction plasma_polarization_kernel failed".to_string()); }

            Ok(GpuContext {
                device, context, cublas_handle, module, raman_fn, ppt_fn,
                apply_prop_fn, rk45_accumulate_stage_fn, rk45_accumulate_error_fn,
                weaknorm_elem_fn, weaknorm_reduce_fn, rhs_mode_avg_real_fn, rhs_mode_avg_env_fn,
                apply_time_window_fn, plasma_fraction_fn, plasma_phase_fn,
                plasma_current_fn, plasma_polarization_fn,
            })
        }
    }).as_ref().map_err(|e| e.clone())
}

// Bind context to calling thread
pub fn activate_context() -> Result<&'static GpuContext, String> {
    let ctx = get_gpu_context().ok_or_else(|| "GPU context not initialized".to_string())?;
    let driver = get_driver_api()?;
    let res = unsafe { (driver.cuCtxSetCurrent)(ctx.context) };
    if res != 0 {
        return Err(format!("cuCtxSetCurrent failed: {}", res));
    }
    Ok(ctx)
}

// RAII GPU Memory Buffer
#[derive(Debug)]
pub struct GpuBuffer {
    pub dptr: CUdeviceptr,
    pub size: usize,
}

impl GpuBuffer {
    pub fn alloc(size: usize) -> Result<Self, String> {
        activate_context()?;
        let driver = get_driver_api()?;
        let mut dptr = 0;
        let res = unsafe { (driver.cuMemAlloc_v2)(&mut dptr, size) };
        if res != 0 {
            return Err(format!("cuMemAlloc_v2 failed: {}", res));
        }
        Ok(Self { dptr, size })
    }

    pub fn copy_to_device<T>(&self, src: &[T]) -> Result<(), String> {
        activate_context()?;
        let driver = get_driver_api()?;
        let bytes = src.len() * std::mem::size_of::<T>();
        assert!(bytes <= self.size);
        let res = unsafe { (driver.cuMemcpyHtoD_v2)(self.dptr, src.as_ptr() as *const libc::c_void, bytes) };
        if res != 0 {
            return Err(format!("cuMemcpyHtoD_v2 failed: {}", res));
        }
        Ok(())
    }

    pub fn copy_to_host<T>(&self, dst: &mut [T]) -> Result<(), String> {
        activate_context()?;
        let driver = get_driver_api()?;
        let bytes = dst.len() * std::mem::size_of::<T>();
        assert!(bytes <= self.size);
        let res = unsafe { (driver.cuMemcpyDtoH_v2)(dst.as_mut_ptr() as *mut libc::c_void, self.dptr, bytes) };
        if res != 0 {
            return Err(format!("cuMemcpyDtoH_v2 failed: {}", res));
        }
        Ok(())
    }

    /// Copies data from another `GpuBuffer` to this one
    pub fn copy_from_device(&mut self, src: &GpuBuffer) -> Result<(), String> {
        let driver = get_driver_api()?;
        if self.size != src.size {
            return Err("Size mismatch in copy_from_device".to_string());
        }
        let res = unsafe { (driver.cuMemcpyDtoD_v2)(self.dptr, src.dptr, self.size) };
        if res != 0 {
            return Err(format!("cuMemcpyDtoD_v2 failed: {}", res));
        }
        Ok(())
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if let Ok(driver) = get_driver_api() {
            // Best effort drop, ignore context activation failures
            let _ = activate_context();
            unsafe { (driver.cuMemFree_v2)(self.dptr); }
        }
    }
}
