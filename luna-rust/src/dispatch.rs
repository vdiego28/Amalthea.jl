#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum HardwarePath {
    Auto = 0,           // Auto-detect best hardware (default)
    CpuX86AVX512 = 1,   // Force x86 AVX-512
    CpuX86AVX2 = 2,     // Force x86 AVX2
    CpuArmNeon = 3,     // Force ARM NEON
    CpuArmAMX = 4,      // Force Apple AMX
    GpuCuda = 5,        // Force NVIDIA CUDA
    GpuVulkan = 6,      // Force Vulkan/wgpu
    CpuPortable = 7,    // Fallback to standard scalar loops (no vectorization)
}

#[repr(C)]
pub struct SimulationEngine {
    pub active_path: HardwarePath,
    pub thread_pool_size: usize,
}

fn rayon_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn is_avx512_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn is_avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn is_neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

fn is_apple_amx_available() -> bool {
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    {
        true
    }
    #[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
    {
        false
    }
}


fn is_vulkan_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        // Try dynamic load of libvulkan.so.1 via dlopen
        unsafe {
            if let Ok(c_name) = std::ffi::CString::new("libvulkan.so.1") {
                let handle = libc::dlopen(c_name.as_ptr(), libc::RTLD_LAZY);
                if !handle.is_null() {
                    libc::dlclose(handle);
                    return true;
                }
            }
        }
        false
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

impl SimulationEngine {
    pub fn try_init_cuda() -> Result<Self, String> {
        match crate::cuda::init_gpu_context() {
            Ok(_) => Ok(Self {
                active_path: HardwarePath::GpuCuda,
                thread_pool_size: 1,
            }),
            Err(e) => Err(format!("CUDA driver or device not found: {}", e)),
        }
    }

    pub fn try_init_vulkan() -> Result<Self, String> {
        if is_vulkan_available() {
            Ok(Self {
                active_path: HardwarePath::GpuVulkan,
                thread_pool_size: 1,
            })
        } else {
            Err("Vulkan driver not found".to_string())
        }
    }

    pub fn try_init_apple_amx() -> Result<Self, String> {
        if is_apple_amx_available() {
            Ok(Self {
                active_path: HardwarePath::CpuArmAMX,
                thread_pool_size: 4,
            })
        } else {
            Err("Apple AMX not supported on this platform".to_string())
        }
    }

    pub fn try_init_x86_avx512() -> Result<Self, String> {
        if is_avx512_available() {
            Ok(Self {
                active_path: HardwarePath::CpuX86AVX512,
                thread_pool_size: rayon_thread_count(),
            })
        } else {
            Err("AVX-512 vector extension not supported".to_string())
        }
    }

    pub fn try_init_x86_avx2() -> Result<Self, String> {
        if is_avx2_available() {
            Ok(Self {
                active_path: HardwarePath::CpuX86AVX2,
                thread_pool_size: rayon_thread_count(),
            })
        } else {
            Err("AVX2 vector extension not supported".to_string())
        }
    }

    pub fn try_init_arm_neon() -> Result<Self, String> {
        if is_neon_available() {
            Ok(Self {
                active_path: HardwarePath::CpuArmNeon,
                thread_pool_size: rayon_thread_count(),
            })
        } else {
            Err("ARM NEON vector extension not supported".to_string())
        }
    }

    pub fn init_portable() -> Self {
        Self {
            active_path: HardwarePath::CpuPortable,
            thread_pool_size: 1,
        }
    }

    pub fn init_path(path: HardwarePath) -> Self {
        match path {
            HardwarePath::Auto => Self::initialize(HardwarePath::Auto),
            HardwarePath::GpuCuda => Self::try_init_cuda().unwrap_or_else(|_| Self::init_portable()),
            HardwarePath::GpuVulkan => Self::try_init_vulkan().unwrap_or_else(|_| Self::init_portable()),
            HardwarePath::CpuArmAMX => Self::try_init_apple_amx().unwrap_or_else(|_| Self::init_portable()),
            HardwarePath::CpuX86AVX512 => Self::try_init_x86_avx512().unwrap_or_else(|_| Self::init_portable()),
            HardwarePath::CpuX86AVX2 => Self::try_init_x86_avx2().unwrap_or_else(|_| Self::init_portable()),
            HardwarePath::CpuArmNeon => Self::try_init_arm_neon().unwrap_or_else(|_| Self::init_portable()),
            HardwarePath::CpuPortable => Self::init_portable(),
        }
    }

    pub fn initialize(preferred: HardwarePath) -> Self {
        match preferred {
            HardwarePath::Auto => {
                if let Ok(engine) = Self::try_init_cuda() {
                    return engine;
                }
                if let Ok(engine) = Self::try_init_vulkan() {
                    return engine;
                }
                if let Ok(engine) = Self::try_init_apple_amx() {
                    return engine;
                }
                if let Ok(engine) = Self::try_init_x86_avx512() {
                    return engine;
                }
                if let Ok(engine) = Self::try_init_x86_avx2() {
                    return engine;
                }
                if let Ok(engine) = Self::try_init_arm_neon() {
                    return engine;
                }
                Self::init_portable()
            }
            HardwarePath::GpuCuda => {
                Self::try_init_cuda().unwrap_or_else(|err| {
                    eprintln!("Warning: CUDA initialization failed ({:?}). Falling back to Vulkan...", err);
                    Self::initialize(HardwarePath::GpuVulkan)
                })
            }
            HardwarePath::GpuVulkan => {
                Self::try_init_vulkan().unwrap_or_else(|err| {
                    eprintln!("Warning: Vulkan initialization failed ({:?}). Falling back to CPU...", err);
                    #[cfg(target_arch = "x86_64")]
                    return Self::initialize(HardwarePath::CpuX86AVX512);
                    #[cfg(target_arch = "aarch64")]
                    return Self::initialize(HardwarePath::CpuArmNeon);
                    #[allow(unreachable_code)]
                    Self::initialize(HardwarePath::CpuPortable)
                })
            }
            HardwarePath::CpuX86AVX512 => {
                Self::try_init_x86_avx512().unwrap_or_else(|err| {
                    eprintln!("Warning: AVX-512 init failed ({:?}). Falling back to AVX2...", err);
                    Self::initialize(HardwarePath::CpuX86AVX2)
                })
            }
            HardwarePath::CpuX86AVX2 => {
                Self::try_init_x86_avx2().unwrap_or_else(|err| {
                    eprintln!("Warning: AVX2 init failed ({:?}). Falling back to portable...", err);
                    Self::initialize(HardwarePath::CpuPortable)
                })
            }
            HardwarePath::CpuArmAMX => {
                Self::try_init_apple_amx().unwrap_or_else(|err| {
                    eprintln!("Warning: Apple AMX init failed ({:?}). Falling back to NEON...", err);
                    Self::initialize(HardwarePath::CpuArmNeon)
                })
            }
            HardwarePath::CpuArmNeon => {
                Self::try_init_arm_neon().unwrap_or_else(|err| {
                    eprintln!("Warning: ARM NEON init failed ({:?}). Falling back to portable...", err);
                    Self::initialize(HardwarePath::CpuPortable)
                })
            }
            path => {
                Self::init_path(path)
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn init_simulation_engine(preferred_path_code: i32) -> *mut SimulationEngine {
    let preferred = match preferred_path_code {
        0 => HardwarePath::Auto,
        1 => HardwarePath::CpuX86AVX512,
        2 => HardwarePath::CpuX86AVX2,
        3 => HardwarePath::CpuArmNeon,
        4 => HardwarePath::CpuArmAMX,
        5 => HardwarePath::GpuCuda,
        6 => HardwarePath::GpuVulkan,
        _ => HardwarePath::CpuPortable,
    };
    let engine = Box::new(SimulationEngine::initialize(preferred));
    Box::into_raw(engine)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_simulation_engine(engine: *mut SimulationEngine) {
    if !engine.is_null() {
        unsafe {
            let _ = Box::from_raw(engine);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_active_hardware_path(engine: *const SimulationEngine) -> i32 {
    if engine.is_null() {
        return -1;
    }
    let engine_ref = unsafe { &*engine };
    match engine_ref.active_path {
        HardwarePath::Auto => 0,
        HardwarePath::CpuX86AVX512 => 1,
        HardwarePath::CpuX86AVX2 => 2,
        HardwarePath::CpuArmNeon => 3,
        HardwarePath::CpuArmAMX => 4,
        HardwarePath::GpuCuda => 5,
        HardwarePath::GpuVulkan => 6,
        HardwarePath::CpuPortable => 7,
    }
}
