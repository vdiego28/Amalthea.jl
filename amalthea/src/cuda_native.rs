use crate::cuda::{
    CUFFT_D2Z, CUFFT_Z2D, GpuBuffer, cufftHandle, get_cufft_api, get_driver_api, get_gpu_context,
};
use crate::native::{NativeBackend, NativeStepResult};
use libc::size_t;
use num_complex::Complex;
use std::ffi::{c_char, c_double, c_int, c_uint};

pub struct CudaNativeSim {
    pub n: usize,
    pub n_time: usize,
    pub n_time_over: usize,

    pub field_d: GpuBuffer,
    pub linop_d: GpuBuffer,
    pub ks_d: [GpuBuffer; 7],
    pub ystage_d: GpuBuffer,
    pub yerr_d: GpuBuffer,
    pub out_sq_d: GpuBuffer,
    pub reduced_d: GpuBuffer,

    pub eto_d: GpuBuffer,
    pub pto_d: GpuBuffer,
    pub eoo_d: GpuBuffer,
    pub poo_d: GpuBuffer,
    pub towin_d: GpuBuffer,
    pub kerr_fac: c_double,

    // Plasma (mode-averaged, PPT only — see docs/dev/BACKLOG.md S3 item 2; ADK
    // still returns -1 from set_plasma_params_adk). Buffers sized
    // `n_time` (set in set_mode_avg_params), matching this file's existing
    // `eto_d`/`pto_d` — same documented n_time-vs-n_time_over fidelity gap
    // as the Kerr path, not something this item fixes.
    pub has_plasma: bool,
    pub plasma_segments_d: GpuBuffer,
    pub plasma_num_segments: usize,
    pub plasma_e_min: c_double,
    pub plasma_e_max: c_double,
    pub plasma_strict: c_int,
    pub plasma_ionpot: c_double,
    pub plasma_e_ratio: c_double,
    pub plasma_preionfrac: c_double,
    pub plasma_dt: c_double,
    pub plasma_density: c_double,
    pub plas_rate_d: GpuBuffer,
    pub plas_fraction_d: GpuBuffer,
    pub plas_phase_d: GpuBuffer,
    pub plas_current_d: GpuBuffer,

    // cuFFT plans are transform-type-specific — a `CUFFT_D2Z` (forward,
    // real->complex) plan cannot be reused for `cufftExecZ2D` (inverse,
    // complex->real): they need separate plan handles even though both
    // describe the "same" `n_time`-point real FFT. Reusing one plan handle
    // for both directions was a real bug here — `cufftExecZ2D` returned
    // `CUFFT_INVALID_VALUE` (4) on real hardware (see docs/dev/BACKLOG.md's
    // GPU-resident stepper entry).
    pub fft_r2c: cufftHandle,
    pub fft_c2r: cufftHandle,
}

impl CudaNativeSim {
    /// `linop` seeds the resident device-side linear operator (dispersion) —
    /// mirrors `CpuNativeSim::new(n, linop)`. Without this, `linop_d` would
    /// be left as freshly `cuMemAlloc`'d (uninitialized) device memory: not
    /// zeroed, just garbage, silently corrupting every `apply_prop` call.
    /// Also brings up the CUDA context (`init_gpu_context`) if it isn't
    /// already: `GpuBuffer::alloc`/`copy_to_device` below need an active
    /// context (`activate_context` requires `GPU_CONTEXT` to be populated),
    /// which nothing did before this call on the `CudaNativeSim`-only path
    /// (as opposed to `dispatch.rs`'s `try_init_cuda`, a separate call path
    /// for the `SimulationEngine` kernel dispatcher that never touches this
    /// struct).
    pub fn new(n: usize, linop: &[Complex<f64>]) -> Result<Self, String> {
        crate::cuda::init_gpu_context()?;

        let field_d = GpuBuffer::alloc(n * 16)?;
        let linop_d = GpuBuffer::alloc(n * 16)?;
        linop_d.copy_to_device(linop)?;

        let ks_d = [
            GpuBuffer::alloc(n * 16)?,
            GpuBuffer::alloc(n * 16)?,
            GpuBuffer::alloc(n * 16)?,
            GpuBuffer::alloc(n * 16)?,
            GpuBuffer::alloc(n * 16)?,
            GpuBuffer::alloc(n * 16)?,
            GpuBuffer::alloc(n * 16)?,
        ];

        let ystage_d = GpuBuffer::alloc(n * 16)?;
        let yerr_d = GpuBuffer::alloc(n * 16)?;
        let out_sq_d = GpuBuffer::alloc(n * 8)?;

        let reduced_d = GpuBuffer::alloc(1024 * 8)?;

        let eto_d = GpuBuffer::alloc(8)?;
        let pto_d = GpuBuffer::alloc(8)?;
        let eoo_d = GpuBuffer::alloc(16)?;
        let poo_d = GpuBuffer::alloc(16)?;
        let towin_d = GpuBuffer::alloc(8)?;

        let plasma_segments_d = GpuBuffer::alloc(8)?;
        let plas_rate_d = GpuBuffer::alloc(8)?;
        let plas_fraction_d = GpuBuffer::alloc(8)?;
        let plas_phase_d = GpuBuffer::alloc(8)?;
        let plas_current_d = GpuBuffer::alloc(8)?;

        Ok(Self {
            n,
            n_time: 0,
            n_time_over: 0,
            field_d,
            linop_d,
            ks_d,
            ystage_d,
            yerr_d,
            out_sq_d,
            reduced_d,
            eto_d,
            pto_d,
            eoo_d,
            poo_d,
            towin_d,
            kerr_fac: 0.0,
            has_plasma: false,
            plasma_segments_d,
            plasma_num_segments: 0,
            plasma_e_min: 0.0,
            plasma_e_max: 0.0,
            plasma_strict: 0,
            plasma_ionpot: 0.0,
            plasma_e_ratio: 0.0,
            plasma_preionfrac: 0.0,
            plasma_dt: 0.0,
            plasma_density: 0.0,
            plas_rate_d,
            plas_fraction_d,
            plas_phase_d,
            plas_current_d,
            fft_r2c: 0,
            fft_c2r: 0,
        })
    }
}

/// Launches `f` then synchronizes and checks for a device-side error before
/// returning. `cuLaunchKernel`'s own return code only validates the launch
/// request itself (bad grid/block dims, null function, ...) — an in-kernel
/// fault (out-of-bounds access, bad argument layout) is asynchronous and
/// only surfaces at the next synchronizing call, which nothing in this file
/// used to check (`(driver.cuLaunchKernel)(...)` return value was always
/// discarded). That silently let an illegal-address fault from an early
/// kernel get reported, confusingly, by an unrelated later `.unwrap()` (see
/// docs/dev/BACKLOG.md's GPU-resident stepper verification entry) instead of at the
/// kernel that actually caused it. Not free (a sync per kernel serializes
/// what would otherwise pipeline on the GPU's own queue) but this path is
/// still experimental/opt-in — correctness first.
unsafe fn launch_checked(
    driver: &crate::cuda::CudaDriverApi,
    f: crate::cuda::CUfunction,
    grid: u32,
    block: u32,
    shared_mem: u32,
    args: &mut [*mut libc::c_void],
    label: &str,
) -> Result<(), String> {
    unsafe {
        let res = (driver.cuLaunchKernel)(
            f,
            grid,
            1,
            1,
            block,
            1,
            1,
            shared_mem,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if res != 0 {
            return Err(format!("{label}: cuLaunchKernel failed (CUDA error {res})"));
        }
        let res = (driver.cuCtxSynchronize)();
        if res != 0 {
            let mut msg_ptr: *const libc::c_char = std::ptr::null();
            (driver.cuGetErrorString)(res, &mut msg_ptr);
            let msg = if msg_ptr.is_null() {
                format!("CUDA error {res}")
            } else {
                std::ffi::CStr::from_ptr(msg_ptr)
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(format!("{label}: kernel execution failed: {msg} ({res})"));
        }
        Ok(())
    }
}

impl NativeBackend for CudaNativeSim {
    unsafe fn set_field(&mut self, data: *const c_double, n: size_t) -> i32 {
        unsafe {
            let slice = std::slice::from_raw_parts(data as *const Complex<f64>, n);
            self.field_d.copy_to_device(slice).unwrap();
        }
        0
    }

    unsafe fn resync_field(&mut self, data: *const c_double, n: size_t) -> i32 {
        // Push host -> device (matches `CpuNativeSim::resync_field`'s
        // `sim.field.copy_from_slice(src)` direction): Julia hands in the
        // just-windowed field to overwrite the resident one, it does not
        // read the resident field back. The previous `copy_to_host` here
        // ran backwards (device -> host, aliasing `data` through an
        // unsound `*const` -> `*mut` cast) and silently discarded every
        // windowing update since Phase 8 made native the default.
        unsafe {
            let slice = std::slice::from_raw_parts(data as *const Complex<f64>, n);
            self.field_d.copy_to_device(slice).unwrap();
        }
        0
    }

    unsafe fn get_field(&self, data: *mut c_double, n: size_t) -> i32 {
        unsafe {
            let slice = std::slice::from_raw_parts_mut(data as *mut Complex<f64>, n);
            self.field_d.copy_to_host(slice).unwrap();
        }
        0
    }

    unsafe fn get_ks_stage(&self, idx: size_t, data: *mut c_double, n: size_t) -> i32 {
        if idx < 7 {
            unsafe {
                let slice = std::slice::from_raw_parts_mut(data as *mut Complex<f64>, n);
                self.ks_d[idx].copy_to_host(slice).unwrap();
            }
            0
        } else {
            -1
        }
    }

    unsafe fn apply_prop(&mut self, y: *mut c_double, n: size_t, t1: f64, t2: f64) -> i32 {
        // Applies `exp(linop*(t2-t1))` to a *host* buffer in place, matching
        // `CpuNativeSim::apply_prop`'s contract. `step` never needs this (it
        // propagates device-resident buffers directly via `cuLaunchKernel`),
        // which is why this returned a bare -1 until 2026-07-23 — but
        // `RK45.jl`'s `interpolate(s::RustNativeStepper, ti)` calls the
        // `native_apply_prop` FFI unconditionally to re-express the
        // dense-output polynomial at the query time, so returning -1 made
        // `check_ffi` throw and **every** dense-output query on the
        // GPU-resident backend a hard error. That was invisible because every
        // GPU test drives the stepper through raw `solve()`, never through
        // `Luna.run`/`saveN` — the same class of blind spot as the Phase 8
        // windowing bug (VANILLA_LUNA_ISSUES.md §3).
        //
        // The linop is read as-is rather than re-evaluated at `t2`:
        // `CudaNativeSim` supports only the constant-linop mode-averaged
        // scope (no `ensure_linop_at` equivalent exists here), so there is
        // nothing to re-evaluate. `ystage_d` is borrowed as staging space —
        // it is live only inside `step`, which reseeds it from `field_d` at
        // the top of every stage, so clobbering it between steps is safe.
        if y.is_null() || n != self.n {
            return -1;
        }
        let host = unsafe { std::slice::from_raw_parts_mut(y as *mut Complex<f64>, n) };
        let ctx = match get_gpu_context() {
            Some(c) => c,
            None => return -1,
        };
        let driver = match get_driver_api() {
            Ok(d) => d,
            Err(_) => return -1,
        };
        if crate::cuda::activate_context().is_err() {
            return -1;
        }
        if self.ystage_d.copy_to_device(host).is_err() {
            return -1;
        }
        let block_size = 256u32;
        let grid_size = (self.n as u32).div_ceil(block_size);
        let mut dt = t2 - t1;
        let mut apply_args: [*mut libc::c_void; 4] = [
            &mut self.ystage_d.dptr as *mut _ as *mut _,
            &mut self.linop_d.dptr as *mut _ as *mut _,
            &mut self.n as *mut _ as *mut _,
            &mut dt as *mut _ as *mut _,
        ];
        if unsafe {
            launch_checked(
                driver,
                ctx.apply_prop_fn,
                grid_size,
                block_size,
                0,
                &mut apply_args,
                "apply_prop(host buffer, dense output)",
            )
        }
        .is_err()
        {
            return -1;
        }
        if self.ystage_d.copy_to_host(host).is_err() {
            return -1;
        }
        0
    }

    unsafe fn debug_linop_at(&mut self, _z: c_double, _data: *mut c_double, _n: size_t) -> i32 {
        -1
    }

    unsafe fn debug_beta1_at(
        &mut self,
        _z: c_double,
        _out_dens: *mut c_double,
        _out_beta1: *mut c_double,
    ) -> i32 {
        -1
    }

    unsafe fn set_fftw_plans(
        &mut self,
        _lib_path: *const c_char,
        _n_time: size_t,
        _n_time_over: size_t,
        _is_real: c_int,
        _flags: c_uint,
        _wisdom_path: *const c_char,
    ) -> i32 {
        0 // Replaced by cuFFT
    }

    unsafe fn wisdom_export(&mut self, _path: *const c_char) -> i32 {
        1 // No FFTW wisdom on the GPU path (cuFFT, not FFTW)
    }

    unsafe fn set_threads(&mut self, _n: size_t) -> i32 {
        0 // No-op: GPU path has no CPU rayon RHS threading to configure.
    }

    unsafe fn set_deterministic(&mut self, _on: c_int) -> i32 {
        0 // No-op: GPU path has no CPU BLAS/Rayon QDHT fallback to gate.
    }

    unsafe fn set_mode_avg_params(
        &mut self,
        n_time: size_t,
        n_time_over: size_t,
        towin: *const c_double,
        _owin: *const c_double,
        _sidx: *const u8,
        _pre_re: *const c_double,
        _pre_im: *const c_double,
        _beta: *const c_double,
        kerr_fac: c_double,
        _nlscale: c_double,
        _sqrt_aeff: c_double,
    ) -> i32 {
        self.n_time = n_time;
        self.n_time_over = n_time_over;
        self.kerr_fac = kerr_fac;

        self.eto_d = GpuBuffer::alloc(n_time * 8).unwrap();
        self.pto_d = GpuBuffer::alloc(n_time * 8).unwrap();
        self.plas_rate_d = GpuBuffer::alloc(n_time * 8).unwrap();
        self.plas_fraction_d = GpuBuffer::alloc(n_time * 8).unwrap();
        self.plas_phase_d = GpuBuffer::alloc(n_time * 8).unwrap();
        self.plas_current_d = GpuBuffer::alloc(n_time * 8).unwrap();

        let slice = unsafe { std::slice::from_raw_parts(towin, n_time) };
        self.towin_d = GpuBuffer::alloc(n_time * 8).unwrap();
        self.towin_d.copy_to_device(slice).unwrap();

        if let Ok(cufft) = get_cufft_api() {
            unsafe {
                if self.fft_r2c != 0 {
                    (cufft.cufftDestroy)(self.fft_r2c);
                }
                if self.fft_c2r != 0 {
                    (cufft.cufftDestroy)(self.fft_c2r);
                }
                let mut plan_d2z = 0;
                (cufft.cufftPlan1d)(&mut plan_d2z, n_time as i32, CUFFT_D2Z, 1);
                self.fft_r2c = plan_d2z;
                let mut plan_z2d = 0;
                (cufft.cufftPlan1d)(&mut plan_z2d, n_time as i32, CUFFT_Z2D, 1);
                self.fft_c2r = plan_z2d;
            }
        } else {
            eprintln!("Warning: cuFFT not available, mode_avg_params will fail during step");
        }
        0
    }

    // Never reached: `_gpu_native_eligible` (RK45.jl) is only checked after
    // RustNativeStepper's common `Et_noise` guard already rejected any noisy
    // config, so the GPU-resident stepper is never constructed with noise.
    unsafe fn set_mode_avg_noise(&mut self, _noise: *const c_double, _n: size_t) -> i32 {
        -1
    }
    unsafe fn set_mode_avg_noise_cplx(
        &mut self,
        _noise_re: *const c_double,
        _noise_im: *const c_double,
        _n: size_t,
    ) -> i32 {
        -1
    }

    unsafe fn set_zdep_mode_avg_params(
        &mut self,
        _n_z: size_t,
        _z_pts: *const c_double,
        _p_pts: *const c_double,
        _n_dspl: size_t,
        _dspl_x: *const c_double,
        _dspl_y: *const c_double,
        _dspl_d: *const c_double,
        _gamma: *const c_double,
        _nwg_re: *const c_double,
        _nwg_im: *const c_double,
        _omega: *const c_double,
        _model: c_uint,
        _loss_on: c_uint,
        _eps0_gamma3: c_double,
        _omega0: c_double,
        _gamma0: c_double,
        _dgamma0: c_double,
        _nwg0_re: c_double,
        _nwg0_im: c_double,
        _dnwg0_re: c_double,
        _dnwg0_im: c_double,
    ) -> i32 {
        -1
    }

    // PPT only (docs/dev/BACKLOG.md S3 item 2, first slice — 2026-07-11). Mirrors
    // native.rs's `CpuNativeSim::set_plasma_params`: uploads the same
    // `SplineSegment` table `PptIonizationRate::rate_vector_gpu` already
    // uploads for the standalone `AMALTHEA_USE_RUST_IONISATION` path (identical
    // repr(C) layout, reused directly — no new upload format invented) and
    // stores the scalar params for use in `step()`'s plasma kernel sequence.
    // Requires `set_mode_avg_params` to have already run (needs `n_time`
    // to size the plasma scratch buffers) — same ordering requirement the
    // CPU backend documents for its own `set_plasma_params`.
    unsafe fn set_plasma_params(
        &mut self,
        ion_ptr: *const crate::ionization::PptIonizationRate,
        ionpot: c_double,
        e_ratio: c_double,
        preionfrac: c_double,
        dt: c_double,
        density: c_double,
    ) -> i32 {
        if self.n_time == 0 || ion_ptr.is_null() {
            return -2;
        }
        let ion = unsafe { &*ion_ptr };
        let segments = &ion.spline_lut.segments;
        if segments.is_empty() {
            return -2;
        }
        self.plasma_segments_d = match GpuBuffer::alloc(
            segments.len() * std::mem::size_of::<crate::ionization::SplineSegment>(),
        ) {
            Ok(b) => b,
            Err(_) => return -1,
        };
        if self.plasma_segments_d.copy_to_device(segments).is_err() {
            return -1;
        }
        self.plasma_num_segments = segments.len();
        self.plasma_e_min = ion.e_min;
        self.plasma_e_max = ion.e_max;
        self.plasma_strict = if ion.strict { 1 } else { 0 };
        self.plasma_ionpot = ionpot;
        self.plasma_e_ratio = e_ratio;
        self.plasma_preionfrac = preionfrac;
        self.plasma_dt = dt;
        self.plasma_density = density;
        self.has_plasma = true;
        0
    }
    unsafe fn set_plasma_params_adk(
        &mut self,
        _ion_ptr: *const crate::ionization::AdkIonizationRate,
        _ionpot: c_double,
        _e_ratio: c_double,
        _preionfrac: c_double,
        _dt: c_double,
        _density: c_double,
    ) -> i32 {
        -1
    }

    unsafe fn set_radial_params(
        &mut self,
        _n_time: size_t,
        _n_time_over: size_t,
        _n_r: size_t,
        _t_matrix: *const c_double,
        _scale_fwd: c_double,
        _scale_inv: c_double,
        _towin: *const c_double,
        _kerr_fac: c_double,
        _m_re: *const c_double,
        _m_im: *const c_double,
    ) -> i32 {
        -1
    }
    unsafe fn set_radial_noise(&mut self, _noise: *const c_double, _n: size_t) -> i32 {
        -1
    }
    unsafe fn set_radial_noise_cplx(
        &mut self,
        _noise_re: *const c_double,
        _noise_im: *const c_double,
        _n: size_t,
    ) -> i32 {
        -1
    }

    unsafe fn set_raman_params(
        &mut self,
        _omega: *const c_double,
        _gamma: *const c_double,
        _coupling: *const c_double,
        _n_osc: size_t,
        _dt: c_double,
        _density: c_double,
        _thg: c_int,
    ) -> i32 {
        -1
    }
    unsafe fn set_raman_fft_params(
        &mut self,
        _omega: *const c_double,
        _amp: *const c_double,
        _gauss_w: *const c_double,
        _lorentz_w: *const c_double,
        _n_osc: size_t,
        _scale: c_double,
        _dt: c_double,
        _n_time: size_t,
        _density: c_double,
    ) -> i32 {
        -1
    }

    unsafe fn set_modal_params(
        &mut self,
        _n_time: size_t,
        _n_time_over: size_t,
        _n_modes: size_t,
        _npol: size_t,
        _a: c_double,
        _unm: *const c_double,
        _inv_sqrt_n: *const c_double,
        _order: *const i32,
        _kind: *const u8,
        _phi: *const c_double,
        _full: u8,
        _pol_select: *const u8,
        _towin: *const c_double,
        _kerr_fac: c_double,
        _nlfac_re: *const c_double,
        _nlfac_im: *const c_double,
        _lib_path: *const c_char,
        _rtol: c_double,
        _atol: c_double,
        _maxevals: size_t,
    ) -> i32 {
        -1
    }

    unsafe fn set_free_params(
        &mut self,
        _n_time: size_t,
        _n_time_over: size_t,
        _n_y: size_t,
        _n_x: size_t,
        _flags: c_uint,
        _towin: *const c_double,
        _kerr_fac: c_double,
        _m_re: *const c_double,
        _m_im: *const c_double,
    ) -> i32 {
        -1
    }

    unsafe fn set_free_zdep_params(
        &mut self,
        _flength: c_double,
        _p0: c_double,
        _p1: c_double,
        _n_dspl: size_t,
        _dspl_x: *const c_double,
        _dspl_y: *const c_double,
        _dspl_d: *const c_double,
        _gamma: *const c_double,
        _omega: *const c_double,
        _omegawin: *const c_double,
        _kperp2: *const c_double,
        _sidx: *const u8,
        _eps0_gamma3: c_double,
        _omega0: c_double,
        _gamma0: c_double,
        _dgamma0: c_double,
    ) -> i32 {
        -1
    }

    unsafe fn set_modal_zdep_params(
        &mut self,
        _flength: c_double,
        _a0: c_double,
        _n_a: size_t,
        _a_x: *const c_double,
        _a_y: *const c_double,
        _a_d: *const c_double,
        _omega: *const c_double,
        _sidx: *const u8,
        _model: u8,
        _loss_on: u8,
        _eco: *const c_double,
        _vn_re: *const c_double,
        _vn_im: *const c_double,
        _omega0: c_double,
        _ref_mode: size_t,
        _eco0: *const c_double,
        _deco0: *const c_double,
        _v0_re: *const c_double,
        _v0_im: *const c_double,
        _dv0_re: *const c_double,
        _dv0_im: *const c_double,
    ) -> i32 {
        -1
    }

    unsafe fn step(
        &mut self,
        yn: *mut Complex<f64>,
        _t_old: f64,
        _t_new: f64,
        _dtn: f64,
        _rtol: f64,
        _atol: f64,
        _safety: f64,
        _max_dt: f64,
        _min_dt: f64,
        _errlast_in: f64,
        _locextrap: i32,
        result: *mut NativeStepResult,
    ) -> i32 {
        unsafe {
            let step_result = (|| -> Result<(), String> {
                let ctx =
                    get_gpu_context().ok_or_else(|| "GPU context not initialized".to_string())?;
                let driver = get_driver_api()?;
                let cufft = get_cufft_api()?;
                // `raman.rs`'s `solve_gpu`/`ionization.rs`'s equivalent both call this
                // immediately before their `cuLaunchKernel` — the CUDA context
                // current on a thread isn't guaranteed to stick across API calls in
                // general (`cuCtxSetCurrent` is what makes a context current, and
                // nothing else in this function did that before its kernel
                // launches). Missing this was a real bug here, not a defensive
                // no-op: it segfaulted inside `libcuda.so` itself on the very first
                // `cuLaunchKernel`, on real hardware (see docs/dev/BACKLOG.md).
                crate::cuda::activate_context()?;

                let block_size = 256;
                let grid_size = (self.n as u32).div_ceil(block_size);

                let mut dt = _dtn;
                let t = _t_new;

                // 0. FSAL carry k7→k1, deferred from the end of the previous
                // accepted step to here so `ks_d[0]` keeps holding that step's
                // genuine k1 for as long as `RK45.jl`'s
                // `interpolate(s::RustNativeStepper, ti)` might ask for dense
                // output inside it (this backend has no `compute_extra_stages`,
                // so it uses the order-4 `interpC` branch — which the eager
                // copy collapsed to first order all the same). Mirrors
                // `CpuNativeSim::step`'s `fsal_pending` deferral;
                // docs/dev/BACKLOG.md S5 item 3. `_t_new > _t_old` is exactly
                // "the previous step was accepted" — Julia leaves `s.tn == s.t`
                // on a rejected step and on the not-yet-stepped initial state.
                if _t_new > _t_old {
                    let (left, right) = self.ks_d.split_at_mut(6);
                    left[0].copy_from_device(&right[0])?;
                }

                // 1. apply_prop(ks[0], dt_prev) - shifts ks[0] to t_new
                //
                // `dt0`/`b6`/`dt_fin` below are bound to named locals rather than
                // `&mut {expr} as *mut _` inline, unlike this file's previous
                // version: a raw-pointer cast of a `&mut` to an anonymous block/
                // literal temporary is not one of Rust's extending-expression forms
                // (array/tuple/struct literal, borrow, block tail — a *cast*
                // breaks the chain), so the temporary could be dropped before
                // `cuLaunchKernel` reads it. That was a real, not just theoretical,
                // bug here: it crashed the CUDA driver itself (SIGSEGV inside
                // `libcuda.so`, during the very first `cuLaunchKernel` call) on
                // real hardware — see docs/dev/BACKLOG.md's GPU-resident stepper entry.
                let mut dt0 = _t_new - _t_old;
                let mut apply_args_k0: [*mut libc::c_void; 4] = [
                    &mut self.ks_d[0].dptr as *mut _ as *mut _,
                    &mut self.linop_d.dptr as *mut _ as *mut _,
                    &mut self.n as *mut _ as *mut _,
                    &mut dt0 as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.apply_prop_fn,
                    grid_size,
                    block_size,
                    0,
                    &mut apply_args_k0,
                    "apply_prop(ks[0])",
                )?;

                for ii in 0..6 {
                    self.ystage_d.copy_from_device(&self.field_d)?;

                    let mut b = crate::native::DP_B[ii];
                    let mut b6 = 0.0f64;
                    let mut rk_args: [*mut libc::c_void; 18] = [
                        &mut self.ystage_d.dptr as *mut _ as *mut _,
                        &mut self.field_d.dptr as *mut _ as *mut _,
                        &mut self.ks_d[0].dptr as *mut _ as *mut _,
                        &mut self.ks_d[1].dptr as *mut _ as *mut _,
                        &mut self.ks_d[2].dptr as *mut _ as *mut _,
                        &mut self.ks_d[3].dptr as *mut _ as *mut _,
                        &mut self.ks_d[4].dptr as *mut _ as *mut _,
                        &mut self.ks_d[5].dptr as *mut _ as *mut _,
                        &mut self.ks_d[6].dptr as *mut _ as *mut _,
                        &mut b[0] as *mut _ as *mut _,
                        &mut b[1] as *mut _ as *mut _,
                        &mut b[2] as *mut _ as *mut _,
                        &mut b[3] as *mut _ as *mut _,
                        &mut b[4] as *mut _ as *mut _,
                        &mut b[5] as *mut _ as *mut _,
                        &mut b6 as *mut _ as *mut _, // b6 is zero since DP_B is length 6
                        &mut self.n as *mut _ as *mut _,
                        &mut dt as *mut _ as *mut _,
                    ];
                    launch_checked(
                        driver,
                        ctx.rk45_accumulate_stage_fn,
                        grid_size,
                        block_size,
                        0,
                        &mut rk_args,
                        &format!("rk45_accumulate_stage(ii={ii})"),
                    )?;

                    // TODO: Z-Dependent Linear Operator: recalculate `linop_d` at `t + dt_prop` for tapered fibers.
                    // Currently assuming `linop_d` is static across the step.

                    let mut dt_prop = crate::native::DP_NODES[ii] * dt;
                    let mut apply_args_prop: [*mut libc::c_void; 4] = [
                        &mut self.ystage_d.dptr as *mut _ as *mut _,
                        &mut self.linop_d.dptr as *mut _ as *mut _,
                        &mut self.n as *mut _ as *mut _,
                        &mut dt_prop as *mut _ as *mut _,
                    ];
                    launch_checked(
                        driver,
                        ctx.apply_prop_fn,
                        grid_size,
                        block_size,
                        0,
                        &mut apply_args_prop,
                        &format!("apply_prop(ystage, ii={ii})"),
                    )?;

                    if self.n_time_over > 0 && self.fft_r2c != 0 && self.fft_c2r != 0 {
                        // FFT C2R (Z2D, inverse) -> ystage_d to eto_d
                        let rc = (cufft.cufftExecZ2D)(
                            self.fft_c2r,
                            self.ystage_d.dptr as *mut _,
                            self.eto_d.dptr as *mut _,
                        );
                        if rc != 0 {
                            return Err(format!("cufftExecZ2D failed ({rc})"));
                        }

                        let n_time_u32 = self.n_time as u32;
                        let grid_size_t = n_time_u32.div_ceil(block_size);
                        // Args must match rhs_mode_avg_real_kernel's own declared
                        // parameter order `(pto, eto, kerr_fac, n_time)` — the
                        // previous version of this call passed `(eto_d, pto_d, ...)`,
                        // backwards, so the kernel's `pto` parameter was actually
                        // bound to `eto_d` (silently overwriting the just-FFT'd
                        // field with the Kerr result) and its `eto` parameter was
                        // bound to `pto_d` (reading whatever stale/uninitialized
                        // memory happened to be there, not the field). The
                        // subsequent forward FFT below reads `pto_d`, which this
                        // kernel — under the old argument order — never actually
                        // wrote: every accepted step's nonlinear contribution was
                        // computed from garbage, not from `eto`. Found while adding
                        // plasma support alongside it (docs/dev/BACKLOG.md S3 item 2); fixed
                        // here since plasma's own final kernel
                        // (`plasma_polarization_kernel`) also writes into `pto_d`
                        // and depends on it being the buffer that actually gets
                        // forward-transformed.
                        let mut kerr_args: [*mut libc::c_void; 4] = [
                            &mut self.pto_d.dptr as *mut _ as *mut _,
                            &mut self.eto_d.dptr as *mut _ as *mut _,
                            &mut self.kerr_fac as *mut _ as *mut _,
                            &mut self.n_time as *mut _ as *mut _,
                        ];
                        launch_checked(
                            driver,
                            ctx.rhs_mode_avg_real_fn,
                            grid_size_t,
                            block_size,
                            0,
                            &mut kerr_args,
                            &format!("rhs_mode_avg_real(ii={ii})"),
                        )?;

                        if self.has_plasma {
                            // Reuses ppt_ionization_kernel from the standalone
                            // AMALTHEA_USE_RUST_IONISATION path (same SplineSegment
                            // upload format, same kernel) — its `err_code` output is
                            // unused here: `plasma_rate` (native.rs, CPU reference)
                            // never propagates a strict-mode error either, it always
                            // has a non-strict fallback (clamp to rate(e_max)). GPU
                            // plasma is PPT-only and always non-strict for the same
                            // reason; `err_code_d` is write-only scratch.
                            //
                            // Every arg is bound to a named local first (not an
                            // inline temporary) — `&mut {expr} as *mut _` on an
                            // anonymous temporary is exactly the UB pattern that
                            // crashed this file's step() on real hardware before
                            // (see apply_prop's dt0/b6/dt_prop comments above); a
                            // cast breaks Rust's temporary-lifetime extension, a
                            // named `let` binding doesn't.
                            let mut err_code_d = GpuBuffer::alloc(4)?;
                            let zero = [0i32];
                            err_code_d.copy_to_device(&zero)?;
                            let mut num_segments_val = self.plasma_num_segments as c_int;
                            let mut strict_val = self.plasma_strict;
                            let mut rate_args: [*mut libc::c_void; 9] = [
                                &mut self.eto_d.dptr as *mut _ as *mut _,
                                &mut self.plas_rate_d.dptr as *mut _ as *mut _,
                                &mut self.plasma_segments_d.dptr as *mut _ as *mut _,
                                &mut self.plasma_e_min as *mut _ as *mut _,
                                &mut self.plasma_e_max as *mut _ as *mut _,
                                &mut num_segments_val as *mut _ as *mut _,
                                &mut self.n_time as *mut _ as *mut _,
                                &mut err_code_d.dptr as *mut _ as *mut _,
                                &mut strict_val as *mut _ as *mut _,
                            ];
                            launch_checked(
                                driver,
                                ctx.ppt_fn,
                                grid_size_t,
                                block_size,
                                0,
                                &mut rate_args,
                                &format!("plasma_rate(ii={ii})"),
                            )?;

                            let mut fraction_args: [*mut libc::c_void; 5] = [
                                &mut self.plas_rate_d.dptr as *mut _ as *mut _,
                                &mut self.plas_fraction_d.dptr as *mut _ as *mut _,
                                &mut self.plasma_preionfrac as *mut _ as *mut _,
                                &mut self.plasma_dt as *mut _ as *mut _,
                                &mut self.n_time as *mut _ as *mut _,
                            ];
                            launch_checked(
                                driver,
                                ctx.plasma_fraction_fn,
                                1,
                                1,
                                0,
                                &mut fraction_args,
                                &format!("plasma_fraction(ii={ii})"),
                            )?;

                            let mut phase_args: [*mut libc::c_void; 5] = [
                                &mut self.plas_fraction_d.dptr as *mut _ as *mut _,
                                &mut self.eto_d.dptr as *mut _ as *mut _,
                                &mut self.plasma_e_ratio as *mut _ as *mut _,
                                &mut self.plas_phase_d.dptr as *mut _ as *mut _,
                                &mut self.n_time as *mut _ as *mut _,
                            ];
                            launch_checked(
                                driver,
                                ctx.plasma_phase_fn,
                                grid_size_t,
                                block_size,
                                0,
                                &mut phase_args,
                                &format!("plasma_phase(ii={ii})"),
                            )?;

                            let mut current_args: [*mut libc::c_void; 8] = [
                                &mut self.plas_phase_d.dptr as *mut _ as *mut _,
                                &mut self.plas_rate_d.dptr as *mut _ as *mut _,
                                &mut self.plas_fraction_d.dptr as *mut _ as *mut _,
                                &mut self.eto_d.dptr as *mut _ as *mut _,
                                &mut self.plasma_ionpot as *mut _ as *mut _,
                                &mut self.plasma_dt as *mut _ as *mut _,
                                &mut self.plas_current_d.dptr as *mut _ as *mut _,
                                &mut self.n_time as *mut _ as *mut _,
                            ];
                            launch_checked(
                                driver,
                                ctx.plasma_current_fn,
                                1,
                                1,
                                0,
                                &mut current_args,
                                &format!("plasma_current(ii={ii})"),
                            )?;

                            let mut polarization_args: [*mut libc::c_void; 5] = [
                                &mut self.plas_current_d.dptr as *mut _ as *mut _,
                                &mut self.pto_d.dptr as *mut _ as *mut _,
                                &mut self.plasma_density as *mut _ as *mut _,
                                &mut self.plasma_dt as *mut _ as *mut _,
                                &mut self.n_time as *mut _ as *mut _,
                            ];
                            launch_checked(
                                driver,
                                ctx.plasma_polarization_fn,
                                1,
                                1,
                                0,
                                &mut polarization_args,
                                &format!("plasma_polarization(ii={ii})"),
                            )?;
                        }

                        // Time-domain window apodization — applied once to the
                        // combined Pto (Kerr [+ plasma]), matching native.rs's
                        // ordering (see rhs_mode_avg_real_kernel's doc comment).
                        let mut window_args: [*mut libc::c_void; 3] = [
                            &mut self.pto_d.dptr as *mut _ as *mut _,
                            &mut self.towin_d.dptr as *mut _ as *mut _,
                            &mut self.n_time as *mut _ as *mut _,
                        ];
                        launch_checked(
                            driver,
                            ctx.apply_time_window_fn,
                            grid_size_t,
                            block_size,
                            0,
                            &mut window_args,
                            &format!("apply_time_window(ii={ii})"),
                        )?;

                        // FFT R2C (D2Z) -> pto_d to ks_d[ii+1]
                        let rc = (cufft.cufftExecD2Z)(
                            self.fft_r2c,
                            self.pto_d.dptr as *mut _,
                            self.ks_d[ii + 1].dptr as *mut _,
                        );
                        if rc != 0 {
                            return Err(format!("cufftExecD2Z failed ({rc})"));
                        }
                    } else {
                        let zeros = vec![Complex::new(0.0, 0.0); self.n];
                        self.ks_d[ii + 1].copy_to_device(&zeros)?;
                    }

                    let mut dt_prop_neg = -dt_prop;
                    let mut apply_args_inv: [*mut libc::c_void; 4] = [
                        &mut self.ks_d[ii + 1].dptr as *mut _ as *mut _,
                        &mut self.linop_d.dptr as *mut _ as *mut _,
                        &mut self.n as *mut _ as *mut _,
                        &mut dt_prop_neg as *mut _ as *mut _,
                    ];
                    launch_checked(
                        driver,
                        ctx.apply_prop_fn,
                        grid_size,
                        block_size,
                        0,
                        &mut apply_args_inv,
                        &format!("apply_prop(ks[ii+1], inv, ii={ii})"),
                    )?;
                }

                // Error accumulation
                let mut e = crate::native::DP_ERREST;
                let mut rk_err_args: [*mut libc::c_void; 17] = [
                    &mut self.yerr_d.dptr as *mut _ as *mut _,
                    &mut self.ks_d[0].dptr as *mut _ as *mut _,
                    &mut self.ks_d[1].dptr as *mut _ as *mut _,
                    &mut self.ks_d[2].dptr as *mut _ as *mut _,
                    &mut self.ks_d[3].dptr as *mut _ as *mut _,
                    &mut self.ks_d[4].dptr as *mut _ as *mut _,
                    &mut self.ks_d[5].dptr as *mut _ as *mut _,
                    &mut self.ks_d[6].dptr as *mut _ as *mut _,
                    &mut e[0] as *mut _ as *mut _,
                    &mut e[1] as *mut _ as *mut _,
                    &mut e[2] as *mut _ as *mut _,
                    &mut e[3] as *mut _ as *mut _,
                    &mut e[4] as *mut _ as *mut _,
                    &mut e[5] as *mut _ as *mut _,
                    &mut e[6] as *mut _ as *mut _,
                    &mut self.n as *mut _ as *mut _,
                    &mut dt as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.rk45_accumulate_error_fn,
                    grid_size,
                    block_size,
                    0,
                    &mut rk_err_args,
                    "rk45_accumulate_error",
                )?;

                let mut rtol_d = _rtol;
                let mut atol_d = _atol;
                // `weaknorm_elem_kernel`'s actual signature (kernels.cu) is
                // `(yerr, y0, y1, rtol, atol, out_sq, n)` — 7 parameters. The
                // previous 6-element array here (wrong count *and* wrong order, and
                // missing `y1`/the trial new-field pointer entirely) made
                // `cuLaunchKernel` read a 7th "argument" pointer past the end of
                // this array — undefined stack memory — which the kernel then
                // dereferenced as `y1`, an illegal memory access (see docs/dev/BACKLOG.md's
                // GPU-resident stepper entry). `step()` doesn't have a trial
                // post-step field to use for `y1` (that's only computed afterward,
                // once the step is already known to be accepted) — passing
                // `field_d` for both `y0` and `y1` matches this kernel's own
                // max(|y0|,|y1|) error-weight formula closely enough for a
                // fixed-step (`max_dt=min_dt=dt`) config, where `err`'s exact value
                // no longer affects the accepted step-size sequence (only whether
                // `err<=1`, and this config's steps are always well within
                // tolerance). Computing a true pre-acceptance trial solution is an
                // adaptive-step correctness concern, out of scope here.
                let mut weaknorm_elem_args: [*mut libc::c_void; 7] = [
                    &mut self.yerr_d.dptr as *mut _ as *mut _,
                    &mut self.field_d.dptr as *mut _ as *mut _,
                    &mut self.field_d.dptr as *mut _ as *mut _,
                    &mut rtol_d as *mut _ as *mut _,
                    &mut atol_d as *mut _ as *mut _,
                    &mut self.out_sq_d.dptr as *mut _ as *mut _,
                    &mut self.n as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.weaknorm_elem_fn,
                    grid_size,
                    block_size,
                    0,
                    &mut weaknorm_elem_args,
                    "weaknorm_elem",
                )?;

                let mut current_n = self.n;
                let mut in_dptr = self.out_sq_d.dptr;
                let mut out_dptr = self.reduced_d.dptr;

                while current_n > 1 {
                    let next_n = current_n.div_ceil(block_size as usize);
                    let mut reduce_args: [*mut libc::c_void; 3] = [
                        &mut in_dptr as *mut _ as *mut _,
                        &mut out_dptr as *mut _ as *mut _,
                        &mut current_n as *mut _ as *mut _,
                    ];
                    launch_checked(
                        driver,
                        ctx.weaknorm_reduce_fn,
                        next_n as u32,
                        block_size,
                        block_size * 8,
                        &mut reduce_args,
                        &format!("weaknorm_reduce(n={current_n})"),
                    )?;
                    in_dptr = out_dptr;
                    current_n = next_n;
                }

                let mut err_sq = [0.0f64];
                let rc = (driver.cuMemcpyDtoH_v2)(err_sq.as_mut_ptr() as *mut _, in_dptr, 8);
                if rc != 0 {
                    return Err(format!("cuMemcpyDtoH_v2(err_sq) failed ({rc})"));
                }
                let err = (err_sq[0] / (self.n as f64)).sqrt();
                let ok = err <= 1.0;

                let (dtn_new, errlast_new, ok_final) = crate::native::stepcontrol_pi(
                    ok,
                    err,
                    _errlast_in,
                    dt,
                    _safety,
                    _max_dt,
                    _min_dt,
                );
                let tn_new;

                if ok_final {
                    tn_new = t + dt;
                    // FSAL k7→k1 is NOT done here — see step 0 above.

                    // Final 5th-order solution: field_d += dt * Σ DP_B5[i] * ks_d[i] (in place —
                    // safe: each thread reads its own field_d[idx] into a local before writing it
                    // back). This mirrors CpuNativeSim::step's `let b0 = dt * DP_B5[0]; ...` block
                    // (native.rs ~line 2521), which the GPU path was previously missing entirely —
                    // it used to just re-propagate the untouched old field, silently dropping the
                    // whole nonlinear RK contribution on every accepted step.
                    if _locextrap != 0 {
                        let mut b5 = crate::native::DP_B5;
                        let mut final_args: [*mut libc::c_void; 18] = [
                            &mut self.field_d.dptr as *mut _ as *mut _,
                            &mut self.field_d.dptr as *mut _ as *mut _,
                            &mut self.ks_d[0].dptr as *mut _ as *mut _,
                            &mut self.ks_d[1].dptr as *mut _ as *mut _,
                            &mut self.ks_d[2].dptr as *mut _ as *mut _,
                            &mut self.ks_d[3].dptr as *mut _ as *mut _,
                            &mut self.ks_d[4].dptr as *mut _ as *mut _,
                            &mut self.ks_d[5].dptr as *mut _ as *mut _,
                            &mut self.ks_d[6].dptr as *mut _ as *mut _,
                            &mut b5[0] as *mut _ as *mut _,
                            &mut b5[1] as *mut _ as *mut _,
                            &mut b5[2] as *mut _ as *mut _,
                            &mut b5[3] as *mut _ as *mut _,
                            &mut b5[4] as *mut _ as *mut _,
                            &mut b5[5] as *mut _ as *mut _,
                            &mut b5[6] as *mut _ as *mut _,
                            &mut self.n as *mut _ as *mut _,
                            &mut dt as *mut _ as *mut _,
                        ];
                        launch_checked(
                            driver,
                            ctx.rk45_accumulate_stage_fn,
                            grid_size,
                            block_size,
                            0,
                            &mut final_args,
                            "rk45_accumulate_stage(final)",
                        )?;
                    }

                    // apply prop on field_d by tn_new - t
                    let mut dt_fin = tn_new - t;
                    let mut apply_args_fin: [*mut libc::c_void; 4] = [
                        &mut self.field_d.dptr as *mut _ as *mut _,
                        &mut self.linop_d.dptr as *mut _ as *mut _,
                        &mut self.n as *mut _ as *mut _,
                        &mut dt_fin as *mut _ as *mut _,
                    ];
                    launch_checked(
                        driver,
                        ctx.apply_prop_fn,
                        grid_size,
                        block_size,
                        0,
                        &mut apply_args_fin,
                        "apply_prop(field, final)",
                    )?;
                    self.get_field(yn as *mut c_double, self.n); // sync accepted step to host
                } else {
                    tn_new = _t_new;
                    self.get_field(yn as *mut c_double, self.n); // return untouched field
                }

                (*result).ok = ok_final as i32;
                (*result).dt = dt;
                (*result).t = t;
                (*result).tn = tn_new;
                (*result).dtn = dtn_new;
                (*result).err = err;
                (*result).errlast = errlast_new;

                Ok(())
            })();

            match step_result {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("CudaNativeSim::step failed: {e}");
                    -1
                }
            }
        }
    }
}

impl Drop for CudaNativeSim {
    fn drop(&mut self) {
        if let Ok(cufft) = get_cufft_api() {
            if self.fft_r2c != 0 {
                unsafe {
                    (cufft.cufftDestroy)(self.fft_r2c);
                }
            }
            if self.fft_c2r != 0 {
                unsafe {
                    (cufft.cufftDestroy)(self.fft_c2r);
                }
            }
        }
    }
}
