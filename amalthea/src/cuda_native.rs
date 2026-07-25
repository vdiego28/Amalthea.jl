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
    /// Oversampled spectral length `n_time_over/2+1` (RealGrid r2c
    /// convention) — mirrors `CpuNativeSim::n_spec_over`. Zero until
    /// `set_mode_avg_params` runs.
    pub n_spec_over: usize,

    pub field_d: GpuBuffer,
    pub linop_d: GpuBuffer,
    pub ks_d: [GpuBuffer; 7],
    pub ystage_d: GpuBuffer,
    pub yerr_d: GpuBuffer,
    pub out_sq_d: GpuBuffer,
    pub reduced_d: GpuBuffer,

    // `eto_d`/`pto_d` are real, length `n_time_over` (oversampled real-space
    // grid) — mirrors CpuNativeSim's `eto`/`pto`. `eoo_d`/`poo_d` are
    // complex, length `n_spec_over` — mirrors `eoo`/`poo`. All four are
    // resized in `set_mode_avg_params` once `n_time_over` is known (BACKLOG
    // S3 item 6 — this item fixes the sizing fidelity gap, see
    // portlog-inbox/gpu-nonlinearity.md).
    pub eto_d: GpuBuffer,
    pub pto_d: GpuBuffer,
    pub eoo_d: GpuBuffer,
    pub poo_d: GpuBuffer,
    pub towin_d: GpuBuffer,
    pub kerr_fac: c_double,

    // ── CPU oracle Steps 1/2/5/6/7 (native.rs::rhs_mode_avg_real) ──────────
    /// Step 1's `scale_fwd = (n_spec_over-1)/(n_spec-1)`.
    pub scale_fwd: c_double,
    /// Step 5's `scale_inv = (n_spec-1)/(n_spec_over-1)`.
    pub scale_inv: c_double,
    /// Combined Step 1 (`1/n_time_over`, cuFFT's unnormalized inverse
    /// transform) and Step 2 (`1/(nlscale·sqrt_aeff)`) scalar, applied to
    /// `eto_d` in one pass right after the inverse FFT.
    pub inv_nto_sc: c_double,
    pub nlscale: c_double,
    pub sqrt_aeff: c_double,
    /// Step 6 `pre[i]/beta[i]*sqrt_aeff`, folded to `1+0i` outside `sidx` —
    /// identical formula/order to `CpuNativeSim::norm_pre_beta`. Length
    /// `n` (=`n_spec`), complex.
    pub norm_pre_beta_d: GpuBuffer,
    /// Step 7 window, folded to `1.0` outside `sidx` — identical to
    /// `CpuNativeSim::owin`. Length `n` (=`n_spec`), real.
    pub owin_d: GpuBuffer,

    // Plasma (mode-averaged, PPT only — see docs/dev/BACKLOG.md S3 item 2; ADK
    // still returns -1 from set_plasma_params_adk). Buffers sized
    // `n_time_over` (set in set_mode_avg_params), matching `eto_d`/`pto_d`.
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
        let norm_pre_beta_d = GpuBuffer::alloc(16)?;
        let owin_d = GpuBuffer::alloc(8)?;

        let plasma_segments_d = GpuBuffer::alloc(8)?;
        let plas_rate_d = GpuBuffer::alloc(8)?;
        let plas_fraction_d = GpuBuffer::alloc(8)?;
        let plas_phase_d = GpuBuffer::alloc(8)?;
        let plas_current_d = GpuBuffer::alloc(8)?;

        Ok(Self {
            n,
            n_time: 0,
            n_time_over: 0,
            n_spec_over: 0,
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
            scale_fwd: 1.0,
            scale_inv: 1.0,
            inv_nto_sc: 0.0,
            nlscale: 1.0,
            sqrt_aeff: 1.0,
            norm_pre_beta_d,
            owin_d,
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

impl CudaNativeSim {
    /// Full CPU-oracle RHS pipeline — mirrors
    /// `CpuNativeSim::rhs_mode_avg_real` (`native.rs:897-971`) Steps 1-7
    /// exactly, step-numbered in the comments below for cross-checking.
    /// Reads the spectral stage input from `self.ystage_d` (length `n` =
    /// `n_spec`) and writes the result into `self.ks_d[idx]`.
    ///
    /// Callers: `step()`'s per-stage loop (after copying the propagated
    /// stage state into `ystage_d`, for `idx = ii+1`), and `set_field`
    /// (after copying the initial field into `ystage_d`, for `idx = 0`) —
    /// the latter mirrors `CpuNativeSim::set_field`'s
    /// `rhs_mode_avg_real(0, &field)` call, which seeds the FSAL stage-0
    /// derivative for the initial condition (see
    /// `docs/dev/native-port/portlog-inbox/gpu-nonlinearity.md` for why this
    /// was a second, previously-undiagnosed bug: without it `ks_d[0]` is
    /// whatever `cuMemAlloc` happened to return, not the true k1).
    ///
    /// # Safety
    /// `self.ystage_d` must already hold the current stage's spectral field
    /// (length `n`), and `idx < 7`.
    unsafe fn compute_rhs_mode_avg(&mut self, idx: usize) -> Result<(), String> {
        if self.n_time_over == 0 || self.fft_r2c == 0 || self.fft_c2r == 0 {
            // No FFT plans configured (set_mode_avg_params not called yet,
            // or plan creation failed) — zero-fill, matching this file's
            // pre-existing fallback for the same condition.
            let zeros = vec![Complex::new(0.0, 0.0); self.n];
            self.ks_d[idx].copy_to_device(&zeros)?;
            return Ok(());
        }
        let ctx = get_gpu_context().ok_or_else(|| "GPU context not initialized".to_string())?;
        let driver = get_driver_api()?;
        let cufft = get_cufft_api()?;
        unsafe {
            crate::cuda::activate_context()?;

            let block_size = 256u32;
            let grid_size_spec = (self.n as u32).div_ceil(block_size);
            let grid_size_over = (self.n_spec_over as u32).div_ceil(block_size);
            let grid_size_t = (self.n_time_over as u32).div_ceil(block_size);

            let mut n_spec_i = self.n as i32;
            let mut n_spec_over_i = self.n_spec_over as i32;
            let mut n_time_over_i = self.n_time_over as i32;

            // ── Step 1: zero-pad + scale ystage_d[n_spec] -> eoo_d[n_spec_over],
            // then inverse rfft (Z2D) eoo_d -> eto_d. cuFFT's out-of-place Z2D
            // may clobber its input buffer (unlike FFTW's PRESERVE_INPUT c2r
            // plan native.rs relies on) — safe here because eoo_d is rebuilt
            // from ystage_d fresh on every call, never reused across calls.
            let mut scale_fwd = self.scale_fwd;
            let mut expand_args: [*mut libc::c_void; 5] = [
                &mut self.ystage_d.dptr as *mut _ as *mut _,
                &mut self.eoo_d.dptr as *mut _ as *mut _,
                &mut scale_fwd as *mut _ as *mut _,
                &mut n_spec_i as *mut _ as *mut _,
                &mut n_spec_over_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.expand_spectrum_fn,
                grid_size_over,
                block_size,
                0,
                &mut expand_args,
                "expand_spectrum",
            )?;

            let rc = (cufft.cufftExecZ2D)(
                self.fft_c2r,
                self.eoo_d.dptr as *mut _,
                self.eto_d.dptr as *mut _,
            );
            if rc != 0 {
                return Err(format!("cufftExecZ2D failed ({rc})"));
            }

            // ── Step 1 (cuFFT's 1/n_time_over unnormalized-inverse factor)
            // combined with Step 2 (1/(nlscale*sqrt_aeff)) into one scalar
            // multiply of eto_d — both are plain scalar rescales of the same
            // buffer, so fusing changes nothing about the result.
            let mut inv_nto_sc = self.inv_nto_sc;
            let mut scale_args: [*mut libc::c_void; 3] = [
                &mut self.eto_d.dptr as *mut _ as *mut _,
                &mut inv_nto_sc as *mut _ as *mut _,
                &mut n_time_over_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.scale_real_fn,
                grid_size_t,
                block_size,
                0,
                &mut scale_args,
                "scale_eto(step1+2)",
            )?;

            // ── Step 3: Kerr RHS. Reuses rhs_mode_avg_real_kernel unchanged
            // (see its own doc comment in kernels.cu), now correctly sized to
            // n_time_over (was n_time).
            let mut kerr_fac = self.kerr_fac;
            let mut kerr_args: [*mut libc::c_void; 4] = [
                &mut self.pto_d.dptr as *mut _ as *mut _,
                &mut self.eto_d.dptr as *mut _ as *mut _,
                &mut kerr_fac as *mut _ as *mut _,
                &mut n_time_over_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.rhs_mode_avg_real_fn,
                grid_size_t,
                block_size,
                0,
                &mut kerr_args,
                "rhs_mode_avg_real(step3)",
            )?;

            // ── Step 3b: plasma polarisation (PPT only), if enabled — same
            // 5-kernel sequence as before, buffers now n_time_over-sized.
            if self.has_plasma {
                let mut err_code_d = GpuBuffer::alloc(4)?;
                let zero = [0i32];
                err_code_d.copy_to_device(&zero)?;
                let mut num_segments_val = self.plasma_num_segments as c_int;
                let mut strict_val = self.plasma_strict;
                let mut e_min = self.plasma_e_min;
                let mut e_max = self.plasma_e_max;
                let mut rate_args: [*mut libc::c_void; 9] = [
                    &mut self.eto_d.dptr as *mut _ as *mut _,
                    &mut self.plas_rate_d.dptr as *mut _ as *mut _,
                    &mut self.plasma_segments_d.dptr as *mut _ as *mut _,
                    &mut e_min as *mut _ as *mut _,
                    &mut e_max as *mut _ as *mut _,
                    &mut num_segments_val as *mut _ as *mut _,
                    &mut n_time_over_i as *mut _ as *mut _,
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
                    "plasma_rate",
                )?;

                let mut preionfrac = self.plasma_preionfrac;
                let mut plasma_dt = self.plasma_dt;
                let mut fraction_args: [*mut libc::c_void; 5] = [
                    &mut self.plas_rate_d.dptr as *mut _ as *mut _,
                    &mut self.plas_fraction_d.dptr as *mut _ as *mut _,
                    &mut preionfrac as *mut _ as *mut _,
                    &mut plasma_dt as *mut _ as *mut _,
                    &mut n_time_over_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.plasma_fraction_fn,
                    1,
                    1,
                    0,
                    &mut fraction_args,
                    "plasma_fraction",
                )?;

                let mut e_ratio = self.plasma_e_ratio;
                let mut phase_args: [*mut libc::c_void; 5] = [
                    &mut self.plas_fraction_d.dptr as *mut _ as *mut _,
                    &mut self.eto_d.dptr as *mut _ as *mut _,
                    &mut e_ratio as *mut _ as *mut _,
                    &mut self.plas_phase_d.dptr as *mut _ as *mut _,
                    &mut n_time_over_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.plasma_phase_fn,
                    grid_size_t,
                    block_size,
                    0,
                    &mut phase_args,
                    "plasma_phase",
                )?;

                let mut ionpot = self.plasma_ionpot;
                let mut current_args: [*mut libc::c_void; 8] = [
                    &mut self.plas_phase_d.dptr as *mut _ as *mut _,
                    &mut self.plas_rate_d.dptr as *mut _ as *mut _,
                    &mut self.plas_fraction_d.dptr as *mut _ as *mut _,
                    &mut self.eto_d.dptr as *mut _ as *mut _,
                    &mut ionpot as *mut _ as *mut _,
                    &mut plasma_dt as *mut _ as *mut _,
                    &mut self.plas_current_d.dptr as *mut _ as *mut _,
                    &mut n_time_over_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.plasma_current_fn,
                    1,
                    1,
                    0,
                    &mut current_args,
                    "plasma_current",
                )?;

                let mut density = self.plasma_density;
                let mut polarization_args: [*mut libc::c_void; 5] = [
                    &mut self.plas_current_d.dptr as *mut _ as *mut _,
                    &mut self.pto_d.dptr as *mut _ as *mut _,
                    &mut density as *mut _ as *mut _,
                    &mut plasma_dt as *mut _ as *mut _,
                    &mut n_time_over_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.plasma_polarization_fn,
                    1,
                    1,
                    0,
                    &mut polarization_args,
                    "plasma_polarization",
                )?;
            }

            // ── Step 4: time-domain window apodization on the combined Pto.
            let mut window_args: [*mut libc::c_void; 3] = [
                &mut self.pto_d.dptr as *mut _ as *mut _,
                &mut self.towin_d.dptr as *mut _ as *mut _,
                &mut n_time_over_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.apply_time_window_fn,
                grid_size_t,
                block_size,
                0,
                &mut window_args,
                "apply_time_window(step4)",
            )?;

            // ── Step 5: forward rfft (D2Z) pto_d -> poo_d[n_spec_over], then
            // crop to n_spec and scale by scale_inv, folded together with
            // Step 6 (norm_pre_beta) and Step 7 (owin) into one kernel.
            let rc = (cufft.cufftExecD2Z)(
                self.fft_r2c,
                self.pto_d.dptr as *mut _,
                self.poo_d.dptr as *mut _,
            );
            if rc != 0 {
                return Err(format!("cufftExecD2Z failed ({rc})"));
            }

            let mut scale_inv = self.scale_inv;
            let mut finalize_args: [*mut libc::c_void; 6] = [
                &mut self.poo_d.dptr as *mut _ as *mut _,
                &mut self.ks_d[idx].dptr as *mut _ as *mut _,
                &mut self.norm_pre_beta_d.dptr as *mut _ as *mut _,
                &mut self.owin_d.dptr as *mut _ as *mut _,
                &mut scale_inv as *mut _ as *mut _,
                &mut n_spec_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.finalize_spectrum_fn,
                grid_size_spec,
                block_size,
                0,
                &mut finalize_args,
                "finalize_spectrum(step5+6+7)",
            )?;

            Ok(())
        }
    }
}

impl NativeBackend for CudaNativeSim {
    unsafe fn set_field(&mut self, data: *const c_double, n: size_t) -> i32 {
        unsafe {
            let slice = std::slice::from_raw_parts(data as *const Complex<f64>, n);
            self.field_d.copy_to_device(slice).unwrap();
            // Seed ks_d[0] with the true FSAL stage-0 derivative for this
            // initial condition — mirrors `CpuNativeSim::set_field`'s
            // `rhs_mode_avg_real(0, &field)` call. Without this, `ks_d[0]`
            // at the first `step()` is whatever `cuMemAlloc` happened to
            // return (not necessarily zeroed), corrupting the first
            // internal stage (DP_B[0]=0.2, nonzero) once the RHS itself is
            // nonzero. See portlog-inbox/gpu-nonlinearity.md.
            if self.ystage_d.copy_from_device(&self.field_d).is_err() {
                return -1;
            }
            if self.compute_rhs_mode_avg(0).is_err() {
                return -1;
            }
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
        owin: *const c_double,
        sidx: *const u8,
        pre_re: *const c_double,
        pre_im: *const c_double,
        beta: *const c_double,
        kerr_fac: c_double,
        nlscale: c_double,
        sqrt_aeff: c_double,
    ) -> i32 {
        // `n_spec` is the ODE state length (`self.n`), matching
        // `CpuNativeSim::set_mode_avg_params`'s `s.n_spec = s.n`. All
        // buffers below that used to be sized `n_time` (skipping the
        // oversampling/anti-aliasing grid Julia/CPU both use for the
        // nonlinear evaluation — BACKLOG.md S3 item 6) are now sized
        // `n_time_over`, and the new `n_spec_over`-sized `eoo_d`/`poo_d`
        // scratch (already existed as fields, previously left at their
        // placeholder 16-byte allocation) close the crop/pad gap.
        let n_spec = self.n;
        self.n_time = n_time;
        self.n_time_over = n_time_over;
        self.n_spec_over = n_time_over / 2 + 1;
        self.kerr_fac = kerr_fac;
        self.nlscale = nlscale;
        self.sqrt_aeff = sqrt_aeff;

        let sc = nlscale * sqrt_aeff;
        if sc == 0.0 {
            eprintln!(
                "Amalthea GPU error: nlscale*sqrt_aeff == 0 in set_mode_avg_params; \
                 refusing to configure a divide-by-zero RHS scaling."
            );
            return -2;
        }
        // Combined Step 1 (cuFFT's unnormalized-inverse `1/n_time_over`) and
        // Step 2 (`1/(nlscale*sqrt_aeff)`) scalar — see
        // `compute_rhs_mode_avg`'s doc.
        self.inv_nto_sc = (1.0 / n_time_over as f64) * (1.0 / sc);
        self.scale_fwd = (self.n_spec_over as f64 - 1.0) / (n_spec as f64 - 1.0);
        self.scale_inv = (n_spec as f64 - 1.0) / (self.n_spec_over as f64 - 1.0);

        self.eto_d = GpuBuffer::alloc(n_time_over * 8).unwrap();
        self.pto_d = GpuBuffer::alloc(n_time_over * 8).unwrap();
        self.eoo_d = GpuBuffer::alloc(self.n_spec_over * 16).unwrap();
        self.poo_d = GpuBuffer::alloc(self.n_spec_over * 16).unwrap();
        self.plas_rate_d = GpuBuffer::alloc(n_time_over * 8).unwrap();
        self.plas_fraction_d = GpuBuffer::alloc(n_time_over * 8).unwrap();
        self.plas_phase_d = GpuBuffer::alloc(n_time_over * 8).unwrap();
        self.plas_current_d = GpuBuffer::alloc(n_time_over * 8).unwrap();

        // towin: length n_time_over — matches `CpuNativeSim`'s own
        // `set_mode_avg_params` (`s.towin = ...from_raw_parts(towin,
        // n_time_over)`). Previously read as `n_time` elements here, which
        // (for n_time_over > n_time, the normal oversampled case) silently
        // read only a prefix of the true window and left the resident
        // buffer sized for the wrong grid entirely.
        let towin_vec: Vec<f64> = if !towin.is_null() {
            unsafe { std::slice::from_raw_parts(towin, n_time_over) }.to_vec()
        } else {
            vec![1.0; n_time_over]
        };
        self.towin_d = GpuBuffer::alloc(n_time_over * 8).unwrap();
        self.towin_d.copy_to_device(&towin_vec).unwrap();

        // sidx: length n_spec — de-branch exactly like CpuNativeSim does
        // (BACKLOG.md S1 item 4): fold sidx into owin/norm_pre_beta once
        // here, host-side, so the GPU kernel is a plain vectorizable
        // multiply with no per-element branch, identical in spirit to the
        // CPU path's own `norm_pre_beta`/`owin` precomputation.
        let sidx_vec: Vec<bool> = if !sidx.is_null() {
            unsafe { std::slice::from_raw_parts(sidx, n_spec) }
                .iter()
                .map(|&x| x != 0)
                .collect()
        } else {
            vec![true; n_spec]
        };

        let mut owin_vec: Vec<f64> = if !owin.is_null() {
            unsafe { std::slice::from_raw_parts(owin, n_spec) }.to_vec()
        } else {
            vec![1.0; n_spec]
        };
        for i in 0..n_spec {
            if !sidx_vec[i] {
                owin_vec[i] = 1.0;
            }
        }

        let pre_vec: Vec<Complex<f64>> = if !pre_re.is_null() && !pre_im.is_null() {
            let re = unsafe { std::slice::from_raw_parts(pre_re, n_spec) };
            let im = unsafe { std::slice::from_raw_parts(pre_im, n_spec) };
            re.iter()
                .zip(im.iter())
                .map(|(&r, &i)| Complex::new(r, i))
                .collect()
        } else {
            vec![Complex::new(0.0, 0.0); n_spec]
        };
        let beta_vec: Vec<f64> = if !beta.is_null() {
            unsafe { std::slice::from_raw_parts(beta, n_spec) }.to_vec()
        } else {
            vec![1.0; n_spec]
        };
        let norm_pre_beta_vec: Vec<Complex<f64>> = (0..n_spec)
            .map(|i| {
                if sidx_vec[i] {
                    pre_vec[i] / beta_vec[i] * sqrt_aeff
                } else {
                    Complex::new(1.0, 0.0)
                }
            })
            .collect();

        self.owin_d = GpuBuffer::alloc(n_spec * 8).unwrap();
        self.owin_d.copy_to_device(&owin_vec).unwrap();
        self.norm_pre_beta_d = GpuBuffer::alloc(n_spec * 16).unwrap();
        self.norm_pre_beta_d
            .copy_to_device(&norm_pre_beta_vec)
            .unwrap();

        if let Ok(cufft) = get_cufft_api() {
            unsafe {
                if self.fft_r2c != 0 {
                    (cufft.cufftDestroy)(self.fft_r2c);
                    self.fft_r2c = 0;
                }
                if self.fft_c2r != 0 {
                    (cufft.cufftDestroy)(self.fft_c2r);
                    self.fft_c2r = 0;
                }
                // Both cufftPlan1d return codes are now checked (previously
                // discarded) — a silent plan failure used to leave
                // `fft_r2c`/`fft_c2r` at 0, which *did* disable the
                // nonlinear block via the existing `!= 0` guard, but with
                // no diagnostic distinguishing "plan failed" from
                // "never configured".
                let mut plan_d2z = 0;
                let rc1 = (cufft.cufftPlan1d)(&mut plan_d2z, n_time_over as i32, CUFFT_D2Z, 1);
                if rc1 != 0 {
                    eprintln!("Amalthea GPU error: cufftPlan1d (D2Z) failed: {rc1}");
                    return -1;
                }
                self.fft_r2c = plan_d2z;
                let mut plan_z2d = 0;
                let rc2 = (cufft.cufftPlan1d)(&mut plan_z2d, n_time_over as i32, CUFFT_Z2D, 1);
                if rc2 != 0 {
                    eprintln!("Amalthea GPU error: cufftPlan1d (Z2D) failed: {rc2}");
                    (cufft.cufftDestroy)(self.fft_r2c);
                    self.fft_r2c = 0;
                    return -1;
                }
                self.fft_c2r = plan_z2d;
            }
        } else {
            eprintln!("Warning: cuFFT not available, mode_avg_params will fail during step");
            return -1;
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
                // cuFFT handles are no longer touched directly in this closure —
                // the full RHS pipeline (FFTs included) now lives in
                // `compute_rhs_mode_avg`, called once per stage below.
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

                    // Full CPU-oracle RHS pipeline (Steps 1-7) — see
                    // `compute_rhs_mode_avg`'s doc and
                    // `docs/dev/native-port/portlog-inbox/gpu-nonlinearity.md`
                    // for the step-by-step correspondence. This replaces the
                    // previous inline "Kerr [+plasma] +window, FFT sized to
                    // n_time" block, which skipped Steps 1/2/5/6/7 entirely
                    // (the root cause of the GPU RHS computing ~zero
                    // nonlinearity — BACKLOG.md S3 item 0).
                    self.compute_rhs_mode_avg(ii + 1)?;

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
