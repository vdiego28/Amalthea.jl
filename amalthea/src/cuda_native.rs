use crate::cuda::{
    CUFFT_D2Z, CUFFT_FORWARD, CUFFT_INVERSE, CUFFT_Z2D, CUFFT_Z2Z, GpuBuffer, cufftHandle,
    get_cufft_api, get_driver_api, get_gpu_context,
};
use crate::native::{NativeBackend, NativeStepResult};
use crate::raman::PrecomputedStepCoeffs;
use libc::size_t;
use num_complex::Complex;
use std::ffi::{c_char, c_double, c_int, c_uint};
#[cfg(test)]
use std::sync::Mutex;
#[cfg(test)]
use std::sync::atomic::{AtomicU8, Ordering};

include!(concat!(env!("OUT_DIR"), "/cuda_raman_limits.rs"));

#[cfg(test)]
static MODE_AVG_SETUP_FAIL_POINT: AtomicU8 = AtomicU8::new(0);
#[cfg(test)]
static MODE_AVG_SETUP_TEST_LOCK: Mutex<()> = Mutex::new(());
#[cfg(test)]
static RAMAN_FFT_SETUP_FAIL_POINT: AtomicU8 = AtomicU8::new(0);

const MODE_AVG_FAIL_ALLOC: u8 = 1;
const MODE_AVG_FAIL_COPY: u8 = 2;
const MODE_AVG_FAIL_SECOND_PLAN: u8 = 3;
const RAMAN_FFT_FAIL_ALLOC: u8 = 1;
const RAMAN_FFT_FAIL_COPY: u8 = 2;
const RAMAN_FFT_FAIL_SECOND_PLAN: u8 = 3;

/// Test seam for deterministic failures at each transactional boundary. In
/// production this is an inline no-op; hardware tests can select an exact
/// stage without relying on allocator or cuFFT resource exhaustion.
#[inline]
fn mode_avg_setup_failpoint(point: u8) -> Result<(), String> {
    #[cfg(test)]
    if MODE_AVG_SETUP_FAIL_POINT.load(Ordering::SeqCst) == point {
        return Err(format!(
            "injected mode-averaged setup failure at point {point}"
        ));
    }
    let _ = point;
    Ok(())
}

#[inline]
fn raman_fft_setup_failpoint(point: u8) -> Result<(), String> {
    #[cfg(test)]
    if RAMAN_FFT_SETUP_FAIL_POINT.load(Ordering::SeqCst) == point {
        return Err(format!("injected Raman FFT setup failure at point {point}"));
    }
    let _ = point;
    Ok(())
}

/// Fully prepared replacement for the mode-averaged device configuration.
/// It owns every allocation and cuFFT handle until `commit_mode_avg_setup`
/// swaps it into the live simulation, so an allocation/copy/planning error
/// cannot damage a configuration which is still usable by the caller.
struct ModeAvgSetup {
    n_time: usize,
    n_time_over: usize,
    n_spec_over: usize,
    eto_d: Option<GpuBuffer>,
    pto_d: Option<GpuBuffer>,
    eoo_d: Option<GpuBuffer>,
    poo_d: Option<GpuBuffer>,
    towin_d: Option<GpuBuffer>,
    norm_pre_beta_d: Option<GpuBuffer>,
    owin_d: Option<GpuBuffer>,
    plas_rate_d: Option<GpuBuffer>,
    plas_fraction_d: Option<GpuBuffer>,
    plas_phase_d: Option<GpuBuffer>,
    plas_current_d: Option<GpuBuffer>,
    plas_scan_sums_d: Option<GpuBuffer>,
    kerr_fac: c_double,
    scale_fwd: c_double,
    scale_inv: c_double,
    inv_nto_sc: c_double,
    nlscale: c_double,
    sqrt_aeff: c_double,
    fft_r2c: cufftHandle,
    fft_c2r: cufftHandle,
    fft_c2c: cufftHandle,
}

/// Fully prepared replacement for the EnvGrid intermediate-broadening Raman
/// convolution state. The response spectrum, zero-padded intensity scratch,
/// and both cuFFT plans remain device-resident after commit; staging keeps the
/// active configuration usable if allocation, planning, or setup execution
/// fails partway through.
struct RamanFftSetup {
    e2_d: Option<GpuBuffer>,
    ew_d: Option<GpuBuffer>,
    hw_d: Option<GpuBuffer>,
    density: c_double,
    fft_r2c: cufftHandle,
    fft_c2r: cufftHandle,
}

/// Fully prepared replacement for the CUDA radial RealGrid configuration.
/// The QDHT matrix and normalization are transferred once at setup; the RHS
/// keeps every field buffer and both temporal FFT plans resident on the device.
/// `Option` ownership makes setup transactional: until commit, dropping this
/// value releases only the staged resources and cannot disturb the active
/// configuration.
struct RadialSetup {
    n_time: usize,
    n_time_over: usize,
    n_spec_over: usize,
    n_r: usize,
    eto_d: Option<GpuBuffer>,
    pto_d: Option<GpuBuffer>,
    qdht_d: Option<GpuBuffer>,
    eoo_d: Option<GpuBuffer>,
    poo_d: Option<GpuBuffer>,
    qdht_matrix_d: Option<GpuBuffer>,
    towin_d: Option<GpuBuffer>,
    norm_d: Option<GpuBuffer>,
    kerr_fac: c_double,
    scale_fwd: c_double,
    scale_inv: c_double,
    fft_r2c: cufftHandle,
    fft_c2r: cufftHandle,
    fft_c2c: cufftHandle,
}

impl Drop for RadialSetup {
    fn drop(&mut self) {
        if let Ok(cufft) = get_cufft_api() {
            unsafe {
                if self.fft_r2c != 0 {
                    (cufft.cufftDestroy)(self.fft_r2c);
                }
                if self.fft_c2r != 0 {
                    (cufft.cufftDestroy)(self.fft_c2r);
                }
                if self.fft_c2c != 0 {
                    (cufft.cufftDestroy)(self.fft_c2c);
                }
            }
        }
    }
}

impl Drop for RamanFftSetup {
    fn drop(&mut self) {
        if let Ok(cufft) = get_cufft_api() {
            unsafe {
                if self.fft_r2c != 0 {
                    (cufft.cufftDestroy)(self.fft_r2c);
                }
                if self.fft_c2r != 0 {
                    (cufft.cufftDestroy)(self.fft_c2r);
                }
            }
        }
    }
}

impl Drop for ModeAvgSetup {
    fn drop(&mut self) {
        if let Ok(cufft) = get_cufft_api() {
            unsafe {
                if self.fft_r2c != 0 {
                    (cufft.cufftDestroy)(self.fft_r2c);
                }
                if self.fft_c2r != 0 {
                    (cufft.cufftDestroy)(self.fft_c2r);
                }
                if self.fft_c2c != 0 {
                    (cufft.cufftDestroy)(self.fft_c2c);
                }
            }
        }
    }
}

fn checked_bytes(elements: usize, element_size: usize) -> Result<usize, String> {
    elements
        .checked_mul(element_size)
        .ok_or_else(|| "mode-averaged CUDA buffer size overflow".to_string())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlasmaRateKind {
    Ppt,
    Adk,
}

pub struct CudaNativeSim {
    pub n: usize,
    /// `true` for RealGrid/r2c; `false` for EnvGrid/c2c. Set by
    /// `native_set_fftw_plans` before mode-averaged parameters are staged.
    pub is_real: bool,
    pub n_time: usize,
    pub n_time_over: usize,
    /// Oversampled spectral length `n_time_over/2+1` (RealGrid r2c
    /// convention) — mirrors `CpuNativeSim::n_spec_over`. Zero until
    /// `set_mode_avg_params` runs.
    pub n_spec_over: usize,

    // ── Plan 08: radial RealGrid resident QDHT + scalar Kerr ──────────────
    // This state is separate from the mode-averaged buffers above.  A radial
    // field is column-major `(n_spec,n_r)` and needs the same temporal FFT
    // applied independently to each radial column, plus a device-resident
    // QDHT matrix and a separate real scratch output for its non-elementwise
    // matrix multiply.
    pub is_radial: bool,
    pub n_r: usize,
    pub radial_eto_d: GpuBuffer,
    pub radial_pto_d: GpuBuffer,
    pub radial_qdht_d: GpuBuffer,
    pub radial_eoo_d: GpuBuffer,
    pub radial_poo_d: GpuBuffer,
    pub radial_qdht_matrix_d: GpuBuffer,
    pub radial_towin_d: GpuBuffer,
    pub radial_norm_d: GpuBuffer,
    pub radial_kerr_fac: c_double,
    pub radial_scale_fwd: c_double,
    pub radial_scale_inv: c_double,
    pub radial_fft_r2c: cufftHandle,
    pub radial_fft_c2r: cufftHandle,
    pub radial_fft_c2c: cufftHandle,

    pub field_d: GpuBuffer,
    pub linop_d: GpuBuffer,
    pub ks_d: [GpuBuffer; 7],
    pub ystage_d: GpuBuffer,
    pub yerr_d: GpuBuffer,
    pub out_sq_d: GpuBuffer,
    pub y0_sq_d: GpuBuffer,
    pub y1_sq_d: GpuBuffer,
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

    // Plasma (mode-averaged PPT or ADK). Buffers sized
    // `n_time_over` (set in set_mode_avg_params), matching `eto_d`/`pto_d`.
    pub has_plasma: bool,
    plasma_rate_kind: PlasmaRateKind,
    pub plasma_segments_d: GpuBuffer,
    pub plasma_num_segments: usize,
    pub plasma_e_min: c_double,
    pub plasma_e_max: c_double,
    pub plasma_strict: c_int,
    // ADK's seven Julia-precomputed constants.  They are copied verbatim from
    // `AdkIonizationRate`, keeping the CUDA formula tied to the CPU oracle.
    plasma_adk_occupancy: c_double,
    plasma_adk_omega_p: c_double,
    plasma_adk_cn_sq: c_double,
    plasma_adk_nstar: c_double,
    plasma_adk_omega_t_prefac: c_double,
    plasma_adk_thr: c_double,
    plasma_adk_avfac: c_double,
    pub plasma_ionpot: c_double,
    pub plasma_e_ratio: c_double,
    pub plasma_preionfrac: c_double,
    pub plasma_dt: c_double,
    pub plasma_density: c_double,
    pub plas_rate_d: GpuBuffer,
    pub plas_fraction_d: GpuBuffer,
    pub plas_phase_d: GpuBuffer,
    pub plas_current_d: GpuBuffer,
    pub plas_scan_sums_d: GpuBuffer,

    // cuFFT plans are transform-type-specific — a `CUFFT_D2Z` (forward,
    // real->complex) plan cannot be reused for `cufftExecZ2D` (inverse,
    // complex->real): they need separate plan handles even though both
    // describe the "same" `n_time`-point real FFT. Reusing one plan handle
    // for both directions was a real bug here — `cufftExecZ2D` returned
    // `CUFFT_INVALID_VALUE` (4) on real hardware (see docs/dev/BACKLOG.md's
    // GPU-resident stepper entry).
    pub fft_r2c: cufftHandle,
    pub fft_c2r: cufftHandle,
    pub fft_c2c: cufftHandle,

    // Resident SDO Raman state. The coefficient layout is repr(C) and shared
    // with `raman_ade_kernel`; no host/device transfer occurs during a step.
    pub has_raman: bool,
    pub raman_num_osc: usize,
    pub raman_density: c_double,
    pub raman_thg: bool,
    pub raman_coeffs_d: GpuBuffer,
    pub raman_intensity_d: GpuBuffer,
    pub raman_p_d: GpuBuffer,
    pub raman_hilbert_a_d: GpuBuffer,
    pub raman_hilbert_b_d: GpuBuffer,
    pub raman_hilbert_fft: cufftHandle,

    // Resident EnvGrid intermediate-broadening (`:SiO2`) Raman convolution.
    // `raman_fft_e2_d` has length 2*n_time_over and is real; both complex
    // spectra have length n_time_over+1. The response spectrum is prepared
    // once at setup and multiplied with each padded intensity spectrum in the
    // RHS without a host transfer.
    pub has_raman_fft: bool,
    pub raman_fft_density: c_double,
    pub raman_fft_e2_d: GpuBuffer,
    pub raman_fft_ew_d: GpuBuffer,
    pub raman_fft_hw_d: GpuBuffer,
    pub raman_fft_r2c: cufftHandle,
    pub raman_fft_c2r: cufftHandle,
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
        let y0_sq_d = GpuBuffer::alloc(n * 8)?;
        let y1_sq_d = GpuBuffer::alloc(n * 8)?;
        // Full-sized so reduction passes can safely ping-pong between the
        // metric array and scratch at arbitrary n.
        let reduced_d = GpuBuffer::alloc(n * 8)?;

        // Sized down to one element until Plan 08's radial setter stages a
        // real configuration. Keeping valid RAII buffers here lets the
        // resident handle remain constructible before geometry-specific setup.
        let radial_eto_d = GpuBuffer::alloc(8)?;
        let radial_pto_d = GpuBuffer::alloc(8)?;
        let radial_qdht_d = GpuBuffer::alloc(8)?;
        let radial_eoo_d = GpuBuffer::alloc(16)?;
        let radial_poo_d = GpuBuffer::alloc(16)?;
        let radial_qdht_matrix_d = GpuBuffer::alloc(8)?;
        let radial_towin_d = GpuBuffer::alloc(8)?;
        let radial_norm_d = GpuBuffer::alloc(16)?;

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
        let plas_scan_sums_d = GpuBuffer::alloc(8)?;
        let raman_coeffs_d = GpuBuffer::alloc(std::mem::size_of::<PrecomputedStepCoeffs>())?;
        let raman_intensity_d = GpuBuffer::alloc(8)?;
        let raman_p_d = GpuBuffer::alloc(8)?;
        let raman_hilbert_a_d = GpuBuffer::alloc(16)?;
        let raman_hilbert_b_d = GpuBuffer::alloc(16)?;
        let raman_fft_e2_d = GpuBuffer::alloc(8)?;
        let raman_fft_ew_d = GpuBuffer::alloc(16)?;
        let raman_fft_hw_d = GpuBuffer::alloc(16)?;

        Ok(Self {
            n,
            is_real: true,
            n_time: 0,
            n_time_over: 0,
            n_spec_over: 0,
            is_radial: false,
            n_r: 0,
            radial_eto_d,
            radial_pto_d,
            radial_qdht_d,
            radial_eoo_d,
            radial_poo_d,
            radial_qdht_matrix_d,
            radial_towin_d,
            radial_norm_d,
            radial_kerr_fac: 0.0,
            radial_scale_fwd: 1.0,
            radial_scale_inv: 1.0,
            radial_fft_r2c: 0,
            radial_fft_c2r: 0,
            radial_fft_c2c: 0,
            field_d,
            linop_d,
            ks_d,
            ystage_d,
            yerr_d,
            out_sq_d,
            y0_sq_d,
            y1_sq_d,
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
            plasma_rate_kind: PlasmaRateKind::Ppt,
            plasma_segments_d,
            plasma_num_segments: 0,
            plasma_e_min: 0.0,
            plasma_e_max: 0.0,
            plasma_strict: 0,
            plasma_adk_occupancy: 0.0,
            plasma_adk_omega_p: 0.0,
            plasma_adk_cn_sq: 0.0,
            plasma_adk_nstar: 0.0,
            plasma_adk_omega_t_prefac: 0.0,
            plasma_adk_thr: 0.0,
            plasma_adk_avfac: 1.0,
            plasma_ionpot: 0.0,
            plasma_e_ratio: 0.0,
            plasma_preionfrac: 0.0,
            plasma_dt: 0.0,
            plasma_density: 0.0,
            plas_rate_d,
            plas_fraction_d,
            plas_phase_d,
            plas_current_d,
            plas_scan_sums_d,
            fft_r2c: 0,
            fft_c2r: 0,
            fft_c2c: 0,
            has_raman: false,
            raman_num_osc: 0,
            raman_density: 0.0,
            raman_thg: true,
            raman_coeffs_d,
            raman_intensity_d,
            raman_p_d,
            raman_hilbert_a_d,
            raman_hilbert_b_d,
            raman_hilbert_fft: 0,
            has_raman_fft: false,
            raman_fft_density: 0.0,
            raman_fft_e2_d,
            raman_fft_ew_d,
            raman_fft_hw_d,
            raman_fft_r2c: 0,
            raman_fft_c2r: 0,
        })
    }
}

impl CudaNativeSim {
    /// Stage the resident CUDA radial RealGrid buffers and QDHT matrix.
    ///
    /// The matrix is copied from Julia's `HT.T` into row-major storage, the
    /// same convention as `QdhtFfiHandle`; no transform convention is
    /// reconstructed in Rust.  The two rank-1 cuFFT plans are intentionally
    /// separate because cuFFT requires transform-specific handles for D2Z and
    /// Z2D.  Every pointer read and allocation happens before commit.
    unsafe fn stage_radial_real_setup(
        &self,
        n_time: usize,
        n_time_over: usize,
        n_r: usize,
        t_matrix: *const c_double,
        scale_fwd: c_double,
        scale_inv: c_double,
        towin: *const c_double,
        kerr_fac: c_double,
        m_re: *const c_double,
        m_im: *const c_double,
    ) -> Result<RadialSetup, String> {
        if n_time == 0
            || n_time_over < n_time
            || !n_time.is_multiple_of(2)
            || !n_time_over.is_multiple_of(2)
            || n_r == 0
            || !self.n.is_multiple_of(n_r)
            || t_matrix.is_null()
            || m_re.is_null()
            || m_im.is_null()
            || !scale_fwd.is_finite()
            || !scale_inv.is_finite()
            || !kerr_fac.is_finite()
        {
            return Err("invalid CUDA radial dimensions or parameters".to_string());
        }
        let n_spec = self.n / n_r;
        if n_spec != n_time / 2 + 1 {
            return Err("CUDA radial RealGrid spectral dimension mismatch".to_string());
        }
        let n_spec_over = n_time_over / 2 + 1;
        let _n_r_i32 = i32::try_from(n_r)
            .map_err(|_| "CUDA radial radial dimension exceeds kernel i32 range".to_string())?;
        let _n_spec_i32 = i32::try_from(n_spec)
            .map_err(|_| "CUDA radial spectral dimension exceeds kernel i32 range".to_string())?;
        let _n_spec_over_i32 = i32::try_from(n_spec_over).map_err(|_| {
            "CUDA radial oversampled spectral dimension exceeds kernel i32 range".to_string()
        })?;
        let n_time_over_i32 = i32::try_from(n_time_over)
            .map_err(|_| "CUDA radial time dimension exceeds cuFFT i32 range".to_string())?;
        let matrix_len = n_r
            .checked_mul(n_r)
            .ok_or_else(|| "CUDA radial QDHT matrix size overflow".to_string())?;
        let field_time_len = n_time_over
            .checked_mul(n_r)
            .ok_or_else(|| "CUDA radial time buffer size overflow".to_string())?;
        let field_spec_over_len = n_spec_over
            .checked_mul(n_r)
            .ok_or_else(|| "CUDA radial spectral buffer size overflow".to_string())?;
        let field_spec_len = n_spec
            .checked_mul(n_r)
            .ok_or_else(|| "CUDA radial normalization size overflow".to_string())?;

        let t_host = unsafe { std::slice::from_raw_parts(t_matrix, matrix_len) };
        if t_host.iter().any(|value| !value.is_finite()) {
            return Err("non-finite CUDA radial QDHT matrix".to_string());
        }
        // Julia's Matrix is column-major.  The CUDA kernel consumes rows, so
        // transpose the storage once, exactly as QdhtFfiHandle::new does.
        let mut matrix = vec![0.0; matrix_len];
        for r in 0..n_r {
            for s in 0..n_r {
                matrix[r * n_r + s] = t_host[r + n_r * s];
            }
        }
        let towin_vec = if towin.is_null() {
            vec![1.0; n_time_over]
        } else {
            unsafe { std::slice::from_raw_parts(towin, n_time_over) }.to_vec()
        };
        let m_re_host = unsafe { std::slice::from_raw_parts(m_re, field_spec_len) };
        let m_im_host = unsafe { std::slice::from_raw_parts(m_im, field_spec_len) };
        if towin_vec.iter().any(|value| !value.is_finite())
            || m_re_host
                .iter()
                .chain(m_im_host.iter())
                .any(|value| !value.is_finite())
        {
            return Err("non-finite CUDA radial window or normalization".to_string());
        }
        let norm = m_re_host
            .iter()
            .zip(m_im_host.iter())
            .map(|(&re, &im)| Complex::new(re, im))
            .collect::<Vec<_>>();

        let eto_d = GpuBuffer::alloc(checked_bytes(field_time_len, 8)?)?;
        let pto_d = GpuBuffer::alloc(checked_bytes(field_time_len, 8)?)?;
        let qdht_d = GpuBuffer::alloc(checked_bytes(field_time_len, 8)?)?;
        let eoo_d = GpuBuffer::alloc(checked_bytes(field_spec_over_len, 16)?)?;
        let poo_d = GpuBuffer::alloc(checked_bytes(field_spec_over_len, 16)?)?;
        let qdht_matrix_d = GpuBuffer::alloc(checked_bytes(matrix_len, 8)?)?;
        let towin_d = GpuBuffer::alloc(checked_bytes(n_time_over, 8)?)?;
        let norm_d = GpuBuffer::alloc(checked_bytes(field_spec_len, 16)?)?;

        qdht_matrix_d.copy_to_device(&matrix)?;
        towin_d.copy_to_device(&towin_vec)?;
        norm_d.copy_to_device(&norm)?;

        let cufft = get_cufft_api()?;
        crate::cuda::activate_context()?;
        let mut fft_r2c = 0;
        let rc = unsafe { (cufft.cufftPlan1d)(&mut fft_r2c, n_time_over_i32, CUFFT_D2Z, 1) };
        if rc != 0 {
            return Err(format!("cufftPlan1d (radial D2Z) failed: {rc}"));
        }
        let mut fft_c2r = 0;
        let rc = unsafe { (cufft.cufftPlan1d)(&mut fft_c2r, n_time_over_i32, CUFFT_Z2D, 1) };
        if rc != 0 {
            unsafe { (cufft.cufftDestroy)(fft_r2c) };
            return Err(format!("cufftPlan1d (radial Z2D) failed: {rc}"));
        }

        Ok(RadialSetup {
            n_time,
            n_time_over,
            n_spec_over,
            n_r,
            eto_d: Some(eto_d),
            pto_d: Some(pto_d),
            qdht_d: Some(qdht_d),
            eoo_d: Some(eoo_d),
            poo_d: Some(poo_d),
            qdht_matrix_d: Some(qdht_matrix_d),
            towin_d: Some(towin_d),
            norm_d: Some(norm_d),
            kerr_fac,
            scale_fwd,
            scale_inv,
            fft_r2c,
            fft_c2r,
            fft_c2c: 0,
        })
    }

    /// Stage the resident CUDA radial EnvGrid buffers.  EnvGrid keeps the
    /// complete c2c spectrum in each radial column, so the time-domain
    /// scratch buffers are complex and the oversampled spectral length is the
    /// full `n_time_over`, unlike RealGrid's `n_time_over/2+1`.
    unsafe fn stage_radial_env_setup(
        &self,
        n_time: usize,
        n_time_over: usize,
        n_r: usize,
        t_matrix: *const c_double,
        scale_fwd: c_double,
        scale_inv: c_double,
        towin: *const c_double,
        kerr_fac: c_double,
        m_re: *const c_double,
        m_im: *const c_double,
    ) -> Result<RadialSetup, String> {
        if self.is_real
            || n_time == 0
            || n_time_over < n_time
            || !n_time.is_multiple_of(2)
            || !n_time_over.is_multiple_of(2)
            || n_r == 0
            || !self.n.is_multiple_of(n_r)
            || t_matrix.is_null()
            || m_re.is_null()
            || m_im.is_null()
            || !scale_fwd.is_finite()
            || !scale_inv.is_finite()
            || !kerr_fac.is_finite()
        {
            return Err("invalid CUDA radial EnvGrid dimensions or parameters".to_string());
        }
        let n_spec = self.n / n_r;
        if n_spec != n_time {
            return Err("CUDA radial EnvGrid spectral dimension mismatch".to_string());
        }
        let _n_r_i32 = i32::try_from(n_r)
            .map_err(|_| "CUDA radial radial dimension exceeds kernel i32 range".to_string())?;
        let _n_spec_i32 = i32::try_from(n_spec)
            .map_err(|_| "CUDA radial spectral dimension exceeds kernel i32 range".to_string())?;
        let _n_time_over_i32 = i32::try_from(n_time_over)
            .map_err(|_| "CUDA radial time dimension exceeds cuFFT i32 range".to_string())?;
        let matrix_len = n_r
            .checked_mul(n_r)
            .ok_or_else(|| "CUDA radial QDHT matrix size overflow".to_string())?;
        let field_time_len = n_time_over
            .checked_mul(n_r)
            .ok_or_else(|| "CUDA radial EnvGrid time buffer size overflow".to_string())?;
        let field_spec_len = n_spec
            .checked_mul(n_r)
            .ok_or_else(|| "CUDA radial EnvGrid normalization size overflow".to_string())?;
        let field_spec_over_len = n_time_over
            .checked_mul(n_r)
            .ok_or_else(|| "CUDA radial EnvGrid spectrum size overflow".to_string())?;

        let t_host = unsafe { std::slice::from_raw_parts(t_matrix, matrix_len) };
        if t_host.iter().any(|value| !value.is_finite()) {
            return Err("non-finite CUDA radial EnvGrid QDHT matrix".to_string());
        }
        let mut matrix = vec![0.0; matrix_len];
        for r in 0..n_r {
            for s in 0..n_r {
                matrix[r * n_r + s] = t_host[r + n_r * s];
            }
        }
        let towin_vec = if towin.is_null() {
            vec![1.0; n_time_over]
        } else {
            unsafe { std::slice::from_raw_parts(towin, n_time_over) }.to_vec()
        };
        let m_re_host = unsafe { std::slice::from_raw_parts(m_re, field_spec_len) };
        let m_im_host = unsafe { std::slice::from_raw_parts(m_im, field_spec_len) };
        if towin_vec.iter().any(|value| !value.is_finite())
            || m_re_host
                .iter()
                .chain(m_im_host.iter())
                .any(|value| !value.is_finite())
        {
            return Err("non-finite CUDA radial EnvGrid window or normalization".to_string());
        }
        let norm = m_re_host
            .iter()
            .zip(m_im_host.iter())
            .map(|(&re, &im)| Complex::new(re, im))
            .collect::<Vec<_>>();

        let eto_d = GpuBuffer::alloc(checked_bytes(field_time_len, 16)?)?;
        let pto_d = GpuBuffer::alloc(checked_bytes(field_time_len, 16)?)?;
        let qdht_d = GpuBuffer::alloc(checked_bytes(field_time_len, 16)?)?;
        let eoo_d = GpuBuffer::alloc(checked_bytes(field_spec_over_len, 16)?)?;
        let poo_d = GpuBuffer::alloc(checked_bytes(field_spec_over_len, 16)?)?;
        let qdht_matrix_d = GpuBuffer::alloc(checked_bytes(matrix_len, 8)?)?;
        let towin_d = GpuBuffer::alloc(checked_bytes(n_time_over, 8)?)?;
        let norm_d = GpuBuffer::alloc(checked_bytes(field_spec_len, 16)?)?;
        qdht_matrix_d.copy_to_device(&matrix)?;
        towin_d.copy_to_device(&towin_vec)?;
        norm_d.copy_to_device(&norm)?;

        let cufft = get_cufft_api()?;
        crate::cuda::activate_context()?;
        let mut fft_c2c = 0;
        let n_time_over_i32 = i32::try_from(n_time_over).map_err(|_| {
            "CUDA radial EnvGrid time dimension exceeds cuFFT i32 range".to_string()
        })?;
        let rc = unsafe { (cufft.cufftPlan1d)(&mut fft_c2c, n_time_over_i32, CUFFT_Z2Z, 1) };
        if rc != 0 {
            return Err(format!("cufftPlan1d (radial EnvGrid Z2Z) failed: {rc}"));
        }

        Ok(RadialSetup {
            n_time,
            n_time_over,
            n_spec_over: n_time_over,
            n_r,
            eto_d: Some(eto_d),
            pto_d: Some(pto_d),
            qdht_d: Some(qdht_d),
            eoo_d: Some(eoo_d),
            poo_d: Some(poo_d),
            qdht_matrix_d: Some(qdht_matrix_d),
            towin_d: Some(towin_d),
            norm_d: Some(norm_d),
            kerr_fac,
            scale_fwd,
            scale_inv,
            fft_r2c: 0,
            fft_c2r: 0,
            fft_c2c,
        })
    }

    unsafe fn stage_radial_setup(
        &self,
        n_time: usize,
        n_time_over: usize,
        n_r: usize,
        t_matrix: *const c_double,
        scale_fwd: c_double,
        scale_inv: c_double,
        towin: *const c_double,
        kerr_fac: c_double,
        m_re: *const c_double,
        m_im: *const c_double,
    ) -> Result<RadialSetup, String> {
        if self.is_real {
            unsafe {
                self.stage_radial_real_setup(
                    n_time,
                    n_time_over,
                    n_r,
                    t_matrix,
                    scale_fwd,
                    scale_inv,
                    towin,
                    kerr_fac,
                    m_re,
                    m_im,
                )
            }
        } else {
            unsafe {
                self.stage_radial_env_setup(
                    n_time,
                    n_time_over,
                    n_r,
                    t_matrix,
                    scale_fwd,
                    scale_inv,
                    towin,
                    kerr_fac,
                    m_re,
                    m_im,
                )
            }
        }
    }

    fn commit_radial_setup(&mut self, mut staged: RadialSetup) {
        let cufft = get_cufft_api().ok();
        let old_r2c = std::mem::replace(
            &mut self.radial_fft_r2c,
            std::mem::replace(&mut staged.fft_r2c, 0),
        );
        let old_c2r = std::mem::replace(
            &mut self.radial_fft_c2r,
            std::mem::replace(&mut staged.fft_c2r, 0),
        );
        let old_c2c = std::mem::replace(
            &mut self.radial_fft_c2c,
            std::mem::replace(&mut staged.fft_c2c, 0),
        );
        self.n_time = staged.n_time;
        self.n_time_over = staged.n_time_over;
        self.n_spec_over = staged.n_spec_over;
        self.is_radial = true;
        self.n_r = staged.n_r;
        self.radial_eto_d = staged.eto_d.take().expect("staged radial eto buffer");
        self.radial_pto_d = staged.pto_d.take().expect("staged radial pto buffer");
        self.radial_qdht_d = staged.qdht_d.take().expect("staged radial QDHT buffer");
        self.radial_eoo_d = staged.eoo_d.take().expect("staged radial eoo buffer");
        self.radial_poo_d = staged.poo_d.take().expect("staged radial poo buffer");
        self.radial_qdht_matrix_d = staged
            .qdht_matrix_d
            .take()
            .expect("staged radial QDHT matrix");
        self.radial_towin_d = staged.towin_d.take().expect("staged radial time window");
        self.radial_norm_d = staged.norm_d.take().expect("staged radial normalization");
        self.radial_kerr_fac = staged.kerr_fac;
        self.radial_scale_fwd = staged.scale_fwd;
        self.radial_scale_inv = staged.scale_inv;
        self.has_plasma = false;
        self.has_raman = false;
        self.has_raman_fft = false;
        if let Some(cufft) = cufft {
            unsafe {
                if old_r2c != 0 {
                    (cufft.cufftDestroy)(old_r2c);
                }
                if old_c2r != 0 {
                    (cufft.cufftDestroy)(old_c2r);
                }
                if old_c2c != 0 {
                    (cufft.cufftDestroy)(old_c2c);
                }
            }
        }
    }

    /// Stage all resources for a replacement mode-averaged configuration.
    /// Nothing in `self` is changed here; callers may therefore return an
    /// error at any point without invalidating the active setup.
    unsafe fn stage_mode_avg_setup(
        &self,
        n_time: usize,
        n_time_over: usize,
        towin: *const c_double,
        owin: *const c_double,
        sidx: *const u8,
        pre_re: *const c_double,
        pre_im: *const c_double,
        beta: *const c_double,
        kerr_fac: c_double,
        nlscale: c_double,
        sqrt_aeff: c_double,
    ) -> Result<ModeAvgSetup, String> {
        let n_spec = self.n;
        if n_time == 0
            || n_time_over < n_time
            || (self.is_real && n_spec != n_time / 2 + 1)
            || (!self.is_real && n_spec != n_time)
            || n_spec < 2
            || (pre_re.is_null() != pre_im.is_null())
            || !nlscale.is_finite()
            || !sqrt_aeff.is_finite()
            || nlscale == 0.0
            || sqrt_aeff == 0.0
            || !kerr_fac.is_finite()
        {
            return Err("invalid mode-averaged CUDA dimensions or pre pair".to_string());
        }
        let n_spec_over = if self.is_real {
            n_time_over
                .checked_div(2)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| "mode-averaged CUDA spectral dimension overflow".to_string())?
        } else {
            n_time_over
        };
        let n_time_over_i32 = i32::try_from(n_time_over)
            .map_err(|_| "mode-averaged CUDA time dimension exceeds cuFFT i32 range".to_string())?;
        let sc = nlscale * sqrt_aeff;
        if !sc.is_finite() || sc == 0.0 {
            return Err("nlscale*sqrt_aeff must be nonzero".to_string());
        }

        // All pointer-based host reads happen only after every dimension and
        // optional-pair condition has been checked.
        let towin_vec = if towin.is_null() {
            vec![1.0; n_time_over]
        } else {
            unsafe { std::slice::from_raw_parts(towin, n_time_over) }.to_vec()
        };
        let sidx_vec: Vec<bool> = if sidx.is_null() {
            vec![true; n_spec]
        } else {
            unsafe { std::slice::from_raw_parts(sidx, n_spec) }
                .iter()
                .map(|&x| x != 0)
                .collect()
        };
        let mut owin_vec = if owin.is_null() {
            vec![1.0; n_spec]
        } else {
            unsafe { std::slice::from_raw_parts(owin, n_spec) }.to_vec()
        };
        let pre_vec: Vec<Complex<f64>> = if pre_re.is_null() {
            vec![Complex::new(0.0, 0.0); n_spec]
        } else {
            let re = unsafe { std::slice::from_raw_parts(pre_re, n_spec) };
            let im = unsafe { std::slice::from_raw_parts(pre_im, n_spec) };
            re.iter()
                .zip(im.iter())
                .map(|(&re, &im)| Complex::new(re, im))
                .collect()
        };
        let beta_vec = if beta.is_null() {
            vec![1.0; n_spec]
        } else {
            unsafe { std::slice::from_raw_parts(beta, n_spec) }.to_vec()
        };
        if (0..n_spec).any(|i| {
            sidx_vec[i]
                && (!pre_vec[i].re.is_finite()
                    || !pre_vec[i].im.is_finite()
                    || !beta_vec[i].is_finite()
                    || beta_vec[i] == 0.0)
        }) {
            return Err("non-finite active mode-averaged coefficient".to_string());
        }
        let norm_pre_beta_vec = (0..n_spec)
            .map(|i| {
                if sidx_vec[i] {
                    pre_vec[i] / beta_vec[i] * sqrt_aeff
                } else {
                    owin_vec[i] = 1.0;
                    Complex::new(1.0, 0.0)
                }
            })
            .collect::<Vec<_>>();

        mode_avg_setup_failpoint(MODE_AVG_FAIL_ALLOC)?;
        let time_bytes = if self.is_real { 8 } else { 16 };
        let eto_d = GpuBuffer::alloc(checked_bytes(n_time_over, time_bytes)?)?;
        let pto_d = GpuBuffer::alloc(checked_bytes(n_time_over, time_bytes)?)?;
        let eoo_d = GpuBuffer::alloc(checked_bytes(n_spec_over, 16)?)?;
        let poo_d = GpuBuffer::alloc(checked_bytes(n_spec_over, 16)?)?;
        let towin_d = GpuBuffer::alloc(checked_bytes(n_time_over, 8)?)?;
        let norm_pre_beta_d = GpuBuffer::alloc(checked_bytes(n_spec, 16)?)?;
        let owin_d = GpuBuffer::alloc(checked_bytes(n_spec, 8)?)?;
        let plas_rate_d = GpuBuffer::alloc(checked_bytes(n_time_over, 8)?)?;
        let plas_fraction_d = GpuBuffer::alloc(checked_bytes(n_time_over, 8)?)?;
        let plas_phase_d = GpuBuffer::alloc(checked_bytes(n_time_over, 8)?)?;
        let plas_current_d = GpuBuffer::alloc(checked_bytes(n_time_over, 8)?)?;
        let scan_len = n_time_over.div_ceil(256).max(1);
        let plas_scan_sums_d = GpuBuffer::alloc(checked_bytes(scan_len, 8)?)?;

        mode_avg_setup_failpoint(MODE_AVG_FAIL_COPY)?;
        towin_d.copy_to_device(&towin_vec)?;
        owin_d.copy_to_device(&owin_vec)?;
        norm_pre_beta_d.copy_to_device(&norm_pre_beta_vec)?;

        let cufft = get_cufft_api()?;
        let mut fft_r2c = 0;
        let mut fft_c2r = 0;
        let mut fft_c2c = 0;
        if self.is_real {
            let rc = unsafe { (cufft.cufftPlan1d)(&mut fft_r2c, n_time_over_i32, CUFFT_D2Z, 1) };
            if rc != 0 {
                return Err(format!("cufftPlan1d (D2Z) failed: {rc}"));
            }
            if let Err(e) = mode_avg_setup_failpoint(MODE_AVG_FAIL_SECOND_PLAN) {
                unsafe { (cufft.cufftDestroy)(fft_r2c) };
                return Err(e);
            }
            let rc = unsafe { (cufft.cufftPlan1d)(&mut fft_c2r, n_time_over_i32, CUFFT_Z2D, 1) };
            if rc != 0 {
                unsafe { (cufft.cufftDestroy)(fft_r2c) };
                return Err(format!("cufftPlan1d (Z2D) failed: {rc}"));
            }
        } else {
            let rc = unsafe { (cufft.cufftPlan1d)(&mut fft_c2c, n_time_over_i32, CUFFT_Z2Z, 1) };
            if rc != 0 {
                return Err(format!("cufftPlan1d (Z2Z) failed: {rc}"));
            }
        }

        Ok(ModeAvgSetup {
            n_time,
            n_time_over,
            n_spec_over,
            eto_d: Some(eto_d),
            pto_d: Some(pto_d),
            eoo_d: Some(eoo_d),
            poo_d: Some(poo_d),
            towin_d: Some(towin_d),
            norm_pre_beta_d: Some(norm_pre_beta_d),
            owin_d: Some(owin_d),
            plas_rate_d: Some(plas_rate_d),
            plas_fraction_d: Some(plas_fraction_d),
            plas_phase_d: Some(plas_phase_d),
            plas_current_d: Some(plas_current_d),
            plas_scan_sums_d: Some(plas_scan_sums_d),
            kerr_fac,
            scale_fwd: if self.is_real {
                (n_spec_over as f64 - 1.0) / (n_spec as f64 - 1.0)
            } else {
                n_time_over as f64 / n_time as f64
            },
            scale_inv: if self.is_real {
                (n_spec as f64 - 1.0) / (n_spec_over as f64 - 1.0)
            } else {
                n_time as f64 / n_time_over as f64
            },
            inv_nto_sc: (1.0 / n_time_over as f64) * (1.0 / sc),
            nlscale,
            sqrt_aeff,
            fft_r2c,
            fft_c2r,
            fft_c2c,
        })
    }

    fn commit_mode_avg_setup(&mut self, mut staged: ModeAvgSetup) {
        let cufft = get_cufft_api().ok();
        let old_r2c =
            std::mem::replace(&mut self.fft_r2c, std::mem::replace(&mut staged.fft_r2c, 0));
        let old_c2r =
            std::mem::replace(&mut self.fft_c2r, std::mem::replace(&mut staged.fft_c2r, 0));
        let old_c2c =
            std::mem::replace(&mut self.fft_c2c, std::mem::replace(&mut staged.fft_c2c, 0));
        self.n_time = staged.n_time;
        self.n_time_over = staged.n_time_over;
        self.n_spec_over = staged.n_spec_over;
        self.eto_d = staged.eto_d.take().expect("staged eto buffer");
        self.pto_d = staged.pto_d.take().expect("staged pto buffer");
        self.eoo_d = staged.eoo_d.take().expect("staged eoo buffer");
        self.poo_d = staged.poo_d.take().expect("staged poo buffer");
        self.towin_d = staged.towin_d.take().expect("staged towin buffer");
        self.norm_pre_beta_d = staged
            .norm_pre_beta_d
            .take()
            .expect("staged normalized-prefactor buffer");
        self.owin_d = staged.owin_d.take().expect("staged owin buffer");
        self.plas_rate_d = staged
            .plas_rate_d
            .take()
            .expect("staged plasma-rate buffer");
        self.plas_fraction_d = staged
            .plas_fraction_d
            .take()
            .expect("staged plasma-fraction buffer");
        self.plas_phase_d = staged
            .plas_phase_d
            .take()
            .expect("staged plasma-phase buffer");
        self.plas_current_d = staged
            .plas_current_d
            .take()
            .expect("staged plasma-current buffer");
        self.plas_scan_sums_d = staged
            .plas_scan_sums_d
            .take()
            .expect("staged plasma-scan-sums buffer");
        self.kerr_fac = staged.kerr_fac;
        self.scale_fwd = staged.scale_fwd;
        self.scale_inv = staged.scale_inv;
        self.inv_nto_sc = staged.inv_nto_sc;
        self.nlscale = staged.nlscale;
        self.sqrt_aeff = staged.sqrt_aeff;
        if let Some(cufft) = cufft {
            unsafe {
                if old_r2c != 0 {
                    (cufft.cufftDestroy)(old_r2c);
                }
                if old_c2r != 0 {
                    (cufft.cufftDestroy)(old_c2r);
                }
                if old_c2c != 0 {
                    (cufft.cufftDestroy)(old_c2c);
                }
            }
        }
    }

    /// Stage the resident r2c/c2r convolution for
    /// `RamanRespIntermediateBroadening`. The host only constructs and
    /// uploads the fixed response spectrum here; the RHS uses the staged
    /// buffers and plans entirely on the device.
    unsafe fn stage_raman_fft_setup(
        &self,
        omega: *const c_double,
        amp: *const c_double,
        gauss_w: *const c_double,
        lorentz_w: *const c_double,
        n_osc: usize,
        scale: c_double,
        dt: c_double,
        n_time: usize,
        density: c_double,
    ) -> Result<RamanFftSetup, String> {
        if self.is_real
            || self.n_time_over == 0
            || n_time != self.n_time_over
            || omega.is_null()
            || amp.is_null()
            || gauss_w.is_null()
            || lorentz_w.is_null()
            || n_osc == 0
            || !scale.is_finite()
            || !dt.is_finite()
            || dt <= 0.0
            || !density.is_finite()
        {
            return Err("invalid EnvGrid Raman FFT setup arguments".to_string());
        }

        let omegas = unsafe { std::slice::from_raw_parts(omega, n_osc) };
        let amps = unsafe { std::slice::from_raw_parts(amp, n_osc) };
        let gauss = unsafe { std::slice::from_raw_parts(gauss_w, n_osc) };
        let lorentz = unsafe { std::slice::from_raw_parts(lorentz_w, n_osc) };
        if omegas
            .iter()
            .chain(amps)
            .chain(gauss)
            .chain(lorentz)
            .any(|value| !value.is_finite())
        {
            return Err("non-finite EnvGrid Raman FFT response parameter".to_string());
        }

        let n_over = n_time
            .checked_mul(2)
            .ok_or_else(|| "EnvGrid Raman FFT length overflow".to_string())?;
        let n_spec_over = n_over
            .checked_div(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| "EnvGrid Raman FFT spectral length overflow".to_string())?;
        let n_over_i32 = i32::try_from(n_over)
            .map_err(|_| "EnvGrid Raman FFT length exceeds cuFFT i32 range".to_string())?;
        let mut h = vec![0.0f64; n_over];
        for (idx, value) in h[..n_time].iter_mut().enumerate() {
            let t = idx as f64 * dt;
            if !t.is_finite() {
                return Err("EnvGrid Raman FFT time grid is non-finite".to_string());
            }
            let mut response = 0.0;
            for oscillator in 0..n_osc {
                response += amps[oscillator]
                    * (-lorentz[oscillator] * t).exp()
                    * (-gauss[oscillator] * gauss[oscillator] * t * t / 4.0).exp()
                    * (omegas[oscillator] * t).sin();
            }
            *value = scale * response;
        }

        raman_fft_setup_failpoint(RAMAN_FFT_FAIL_ALLOC)?;
        let e2_d = GpuBuffer::alloc(checked_bytes(n_over, std::mem::size_of::<f64>())?)?;
        let ew_d = GpuBuffer::alloc(checked_bytes(
            n_spec_over,
            std::mem::size_of::<Complex<f64>>(),
        )?)?;
        let hw_d = GpuBuffer::alloc(checked_bytes(
            n_spec_over,
            std::mem::size_of::<Complex<f64>>(),
        )?)?;
        let h_d = GpuBuffer::alloc(checked_bytes(n_over, std::mem::size_of::<f64>())?)?;

        let cufft = get_cufft_api()?;
        crate::cuda::activate_context()?;
        let mut fft_r2c = 0;
        let rc = unsafe { (cufft.cufftPlan1d)(&mut fft_r2c, n_over_i32, CUFFT_D2Z, 1) };
        if rc != 0 {
            return Err(format!("cufftPlan1d (Raman D2Z) failed: {rc}"));
        }
        if let Err(e) = raman_fft_setup_failpoint(RAMAN_FFT_FAIL_SECOND_PLAN) {
            unsafe { (cufft.cufftDestroy)(fft_r2c) };
            return Err(e);
        }
        let mut fft_c2r = 0;
        let rc = unsafe { (cufft.cufftPlan1d)(&mut fft_c2r, n_over_i32, CUFFT_Z2D, 1) };
        if rc != 0 {
            unsafe { (cufft.cufftDestroy)(fft_r2c) };
            return Err(format!("cufftPlan1d (Raman Z2D) failed: {rc}"));
        }

        let staged = RamanFftSetup {
            e2_d: Some(e2_d),
            ew_d: Some(ew_d),
            hw_d: Some(hw_d),
            density,
            fft_r2c,
            fft_c2r,
        };
        raman_fft_setup_failpoint(RAMAN_FFT_FAIL_COPY)?;
        h_d.copy_to_device(&h)?;
        let rc = unsafe {
            (cufft.cufftExecD2Z)(
                staged.fft_r2c,
                h_d.dptr as *mut f64,
                staged.hw_d.as_ref().expect("staged Raman hw").dptr as *mut _,
            )
        };
        if rc != 0 {
            return Err(format!("cufftExecD2Z (Raman response) failed: {rc}"));
        }

        let ctx = get_gpu_context().ok_or_else(|| "GPU context not initialized".to_string())?;
        let driver = get_driver_api()?;
        let block_size = 256u32;
        let grid = (n_spec_over as u32).div_ceil(block_size);
        let mut hw_ptr = staged.hw_d.as_ref().expect("staged Raman hw").dptr;
        let mut scale_factor = dt / n_over as f64;
        let mut n_spec_i = n_spec_over as c_int;
        let mut scale_args: [*mut libc::c_void; 3] = [
            &mut hw_ptr as *mut _ as *mut _,
            &mut scale_factor as *mut _ as *mut _,
            &mut n_spec_i as *mut _ as *mut _,
        ];
        unsafe {
            launch_checked(
                driver,
                ctx.scale_complex_fn,
                grid,
                block_size,
                0,
                &mut scale_args,
                "raman_fft_response_scale",
            )?;
        }
        Ok(staged)
    }

    fn commit_raman_fft_setup(&mut self, mut staged: RamanFftSetup) {
        let cufft = get_cufft_api().ok();
        let old_hilbert = std::mem::replace(&mut self.raman_hilbert_fft, 0);
        let old_r2c = std::mem::replace(
            &mut self.raman_fft_r2c,
            std::mem::replace(&mut staged.fft_r2c, 0),
        );
        let old_c2r = std::mem::replace(
            &mut self.raman_fft_c2r,
            std::mem::replace(&mut staged.fft_c2r, 0),
        );
        self.raman_fft_e2_d = staged.e2_d.take().expect("staged Raman e2 buffer");
        self.raman_fft_ew_d = staged.ew_d.take().expect("staged Raman ew buffer");
        self.raman_fft_hw_d = staged.hw_d.take().expect("staged Raman hw buffer");
        self.raman_fft_density = staged.density;
        self.has_raman_fft = true;
        self.has_raman = false;
        if let Some(cufft) = cufft {
            unsafe {
                if old_hilbert != 0 {
                    (cufft.cufftDestroy)(old_hilbert);
                }
                if old_r2c != 0 {
                    (cufft.cufftDestroy)(old_r2c);
                }
                if old_c2r != 0 {
                    (cufft.cufftDestroy)(old_c2r);
                }
            }
        }
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

/// Reduces one `n`-element device array of `f64` values to a host scalar.
/// The source and full-sized scratch buffers alternate roles on successive
/// passes, avoiding the old in-place alias when more than two passes were
/// required.
unsafe fn reduce_sum(
    driver: &crate::cuda::CudaDriverApi,
    reduce_fn: crate::cuda::CUfunction,
    input_dptr: u64,
    scratch_dptr: u64,
    n: usize,
    block_size: u32,
    label: &str,
) -> Result<f64, String> {
    unsafe {
        let mut current_n = n;
        let mut in_dptr = input_dptr;
        let mut out_dptr = scratch_dptr;

        while current_n > 1 {
            let next_n = current_n.div_ceil(2 * block_size as usize);
            let mut reduce_args: [*mut libc::c_void; 3] = [
                &mut in_dptr as *mut _ as *mut _,
                &mut out_dptr as *mut _ as *mut _,
                &mut current_n as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                reduce_fn,
                next_n as u32,
                block_size,
                block_size * 8,
                &mut reduce_args,
                &format!("{label}(n={current_n})"),
            )?;
            std::mem::swap(&mut in_dptr, &mut out_dptr);
            current_n = next_n;
        }

        let mut sum = [0.0f64];
        let rc = (driver.cuMemcpyDtoH_v2)(sum.as_mut_ptr() as *mut _, in_dptr, 8);
        if rc != 0 {
            return Err(format!("cuMemcpyDtoH_v2({label}) failed ({rc})"));
        }
        Ok(sum[0])
    }
}

impl CudaNativeSim {
    /// Parallel trapezoidal prefix scan for the PPT cumulative integrals.
    /// The first launch scans 256-element blocks; the second scans the much
    /// smaller block-total array in place. A physics-specific finalizer adds
    /// the preceding-block offset.
    unsafe fn plasma_scan(
        &mut self,
        input_dptr: u64,
        output_dptr: u64,
        label: &str,
    ) -> Result<(), String> {
        unsafe {
            let ctx = get_gpu_context().ok_or_else(|| "GPU context not initialized".to_string())?;
            let driver = get_driver_api()?;
            let block_size = 256u32;
            let n_blocks = self.n_time_over.div_ceil(block_size as usize);
            let mut input = input_dptr;
            let mut output = output_dptr;
            let mut dt = self.plasma_dt;
            let mut n_time = self.n_time_over as c_int;
            let mut scan_args: [*mut libc::c_void; 5] = [
                &mut input as *mut _ as *mut _,
                &mut output as *mut _ as *mut _,
                &mut self.plas_scan_sums_d.dptr as *mut _ as *mut _,
                &mut dt as *mut _ as *mut _,
                &mut n_time as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.plasma_scan_blocks_fn,
                n_blocks as u32,
                block_size,
                block_size * 8,
                &mut scan_args,
                &format!("{label}:blocks"),
            )?;

            let mut n_blocks_i = n_blocks as c_int;
            let mut sums_args: [*mut libc::c_void; 2] = [
                &mut self.plas_scan_sums_d.dptr as *mut _ as *mut _,
                &mut n_blocks_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.plasma_scan_block_sums_fn,
                1,
                1,
                0,
                &mut sums_args,
                &format!("{label}:block_sums"),
            )
        }
    }

    /// Segmented trapezoidal prefix scan for radial plasma fields.  The
    /// device layout is column-major `(n_time_over, n_r)`, so the first
    /// launch assigns several 256-sample blocks to every column and records
    /// one total per `(column, block)`.  The radial finalizers below add the
    /// column-local preceding-block offset; no scan state is shared between
    /// adjacent radial columns.
    unsafe fn plasma_scan_radial(
        &mut self,
        input_dptr: u64,
        output_dptr: u64,
        label: &str,
    ) -> Result<(), String> {
        unsafe {
            let ctx = get_gpu_context().ok_or_else(|| "GPU context not initialized".to_string())?;
            let driver = get_driver_api()?;
            let block_size = 256u32;
            let n_blocks = self.n_time_over.div_ceil(block_size as usize);
            let grid = self
                .n_r
                .checked_mul(n_blocks)
                .ok_or_else(|| "CUDA radial plasma scan grid size overflow".to_string())?;
            let mut input = input_dptr;
            let mut output = output_dptr;
            let mut dt = self.plasma_dt;
            let mut n_time = c_int::try_from(self.n_time_over).map_err(|_| {
                "CUDA radial plasma time dimension exceeds kernel range".to_string()
            })?;
            let mut n_r = c_int::try_from(self.n_r).map_err(|_| {
                "CUDA radial plasma radial dimension exceeds kernel range".to_string()
            })?;
            let mut n_blocks_i = c_int::try_from(n_blocks)
                .map_err(|_| "CUDA radial plasma block count exceeds kernel range".to_string())?;
            let mut scan_args: [*mut libc::c_void; 7] = [
                &mut input as *mut _ as *mut _,
                &mut output as *mut _ as *mut _,
                &mut self.plas_scan_sums_d.dptr as *mut _ as *mut _,
                &mut dt as *mut _ as *mut _,
                &mut n_time as *mut _ as *mut _,
                &mut n_r as *mut _ as *mut _,
                &mut n_blocks_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.plasma_scan_radial_blocks_fn,
                u32::try_from(grid)
                    .map_err(|_| "CUDA radial plasma scan grid exceeds launch range".to_string())?,
                block_size,
                block_size * 8,
                &mut scan_args,
                &format!("{label}:radial_blocks"),
            )
        }
    }

    /// CUDA radial RealGrid RHS — mirrors `CpuNativeSim::rhs_radial` for the
    /// Plan 08 scope (scalar Kerr, no plasma/Raman/noise).  The field remains
    /// resident on the device for the complete pipeline:
    ///
    /// `expand → per-column inverse r2c → QDHT ldiv → Kerr → window →
    /// QDHT mul → per-column forward r2c → crop/normalization`.
    ///
    /// The only host-side work in this function is launching kernels and
    /// cuFFT plans.  In particular, no stage field or QDHT result is copied to
    /// the host; `get_field` remains a diagnostic/step-boundary operation.
    unsafe fn compute_rhs_radial(&mut self, idx: usize) -> Result<(), String> {
        if self.is_real {
            unsafe { self.compute_rhs_radial_real(idx) }
        } else {
            unsafe { self.compute_rhs_radial_env(idx) }
        }
    }

    unsafe fn compute_rhs_radial_real(&mut self, idx: usize) -> Result<(), String> {
        if !self.is_radial || self.n_r == 0 || self.radial_fft_r2c == 0 || self.radial_fft_c2r == 0
        {
            return Err("CUDA radial configuration is not initialized".to_string());
        }
        let ctx = get_gpu_context().ok_or_else(|| "GPU context not initialized".to_string())?;
        let driver = get_driver_api()?;
        let cufft = get_cufft_api()?;
        unsafe {
            crate::cuda::activate_context()?;
            let block_size = 256u32;
            let n_spec = self.n / self.n_r;
            let total_time = self
                .n_time_over
                .checked_mul(self.n_r)
                .ok_or_else(|| "CUDA radial time launch size overflow".to_string())?;
            let total_spec_over = self
                .n_spec_over
                .checked_mul(self.n_r)
                .ok_or_else(|| "CUDA radial spectral launch size overflow".to_string())?;
            let total_time_u32 = u32::try_from(total_time)
                .map_err(|_| "CUDA radial time launch size exceeds grid range".to_string())?;
            let total_spec_over_u32 = u32::try_from(total_spec_over)
                .map_err(|_| "CUDA radial spectral launch size exceeds grid range".to_string())?;
            let n_u32 = u32::try_from(self.n)
                .map_err(|_| "CUDA radial field size exceeds grid range".to_string())?;
            let grid_time = total_time_u32.div_ceil(block_size);
            let grid_spec_over = total_spec_over_u32.div_ceil(block_size);
            let grid_spec = n_u32.div_ceil(block_size);
            let mut n_spec_i = c_int::try_from(n_spec)
                .map_err(|_| "CUDA radial spectral dimension exceeds kernel range".to_string())?;
            let mut n_spec_over_i = c_int::try_from(self.n_spec_over).map_err(|_| {
                "CUDA radial oversampled spectral dimension exceeds kernel range".to_string()
            })?;
            let mut n_time_over_i = c_int::try_from(self.n_time_over)
                .map_err(|_| "CUDA radial time dimension exceeds kernel range".to_string())?;
            let mut n_r_i = c_int::try_from(self.n_r)
                .map_err(|_| "CUDA radial radial dimension exceeds kernel range".to_string())?;

            // Temporal zero-padding scale is independent of the QDHT
            // `scaleRK` transferred in `radial_scale_fwd`.  The CPU oracle
            // uses `(n_spec_over-1)/(n_spec-1)` here and `scaleRK` only for
            // the two QDHT directions below; conflating them was the Plan 08
            // stage-scale bug found by the non-symmetric primitive test.
            let mut scale_fwd = (self.n_spec_over - 1) as f64 / (n_spec - 1) as f64;
            let mut expand_args: [*mut libc::c_void; 6] = [
                &mut self.ystage_d.dptr as *mut _ as *mut _,
                &mut self.radial_eoo_d.dptr as *mut _ as *mut _,
                &mut scale_fwd as *mut _ as *mut _,
                &mut n_spec_i as *mut _ as *mut _,
                &mut n_spec_over_i as *mut _ as *mut _,
                &mut n_r_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.expand_radial_spectrum_fn,
                grid_spec_over,
                block_size,
                0,
                &mut expand_args,
                "expand_radial_spectrum",
            )?;

            // cuFFT's inverse is unnormalized.  Rebuild eoo on every call,
            // so its documented input-clobbering behavior is harmless.
            for r in 0..self.n_r {
                let eoo_ptr = (self.radial_eoo_d.dptr
                    + (r * self.n_spec_over * std::mem::size_of::<Complex<f64>>()) as u64)
                    as *mut libc::c_void;
                let eto_ptr = (self.radial_eto_d.dptr
                    + (r * self.n_time_over * std::mem::size_of::<f64>()) as u64)
                    as *mut f64;
                let rc = (cufft.cufftExecZ2D)(self.radial_fft_c2r, eoo_ptr, eto_ptr);
                if rc != 0 {
                    return Err(format!("cufftExecZ2D (radial column {r}) failed: {rc}"));
                }
            }
            let mut inverse_scale = 1.0 / self.n_time_over as f64;
            let mut total_time_i = c_int::try_from(total_time)
                .map_err(|_| "CUDA radial time launch size exceeds kernel range".to_string())?;
            let mut scale_args: [*mut libc::c_void; 3] = [
                &mut self.radial_eto_d.dptr as *mut _ as *mut _,
                &mut inverse_scale as *mut _ as *mut _,
                &mut total_time_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.scale_real_fn,
                grid_time,
                block_size,
                0,
                &mut scale_args,
                "scale_radial_inverse_fft",
            )?;

            let mut qdht_scale = self.radial_scale_inv;
            let mut qdht_args: [*mut libc::c_void; 6] = [
                &mut self.radial_eto_d.dptr as *mut _ as *mut _,
                &mut self.radial_qdht_d.dptr as *mut _ as *mut _,
                &mut self.radial_qdht_matrix_d.dptr as *mut _ as *mut _,
                &mut qdht_scale as *mut _ as *mut _,
                &mut n_time_over_i as *mut _ as *mut _,
                &mut n_r_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.qdht_radial_real_fn,
                grid_time,
                block_size,
                0,
                &mut qdht_args,
                "qdht_radial_ldiv",
            )?;

            let mut kerr_fac = self.radial_kerr_fac;
            let mut kerr_args: [*mut libc::c_void; 4] = [
                &mut self.radial_pto_d.dptr as *mut _ as *mut _,
                &mut self.radial_qdht_d.dptr as *mut _ as *mut _,
                &mut kerr_fac as *mut _ as *mut _,
                &mut total_time_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.rhs_mode_avg_real_fn,
                grid_time,
                block_size,
                0,
                &mut kerr_args,
                "rhs_radial_kerr",
            )?;

            // Plans 10-11: one PPT or thresholded ADK plasma response,
            // evaluated independently for every radial time column.  The
            // rate kernel is pointwise over the flattened `(time,
            // radial-column)` field; all three cumulative integrals use the
            // segmented scan above, followed by the same
            // rho/phase/current/polarization transforms as the CPU radial
            // oracle.  EnvGrid plasma and unthresholded ADK remain outside
            // this GPU slice.
            if self.has_plasma {
                if !self.is_real {
                    return Err(
                        "CUDA radial plasma currently requires RealGrid ionization".to_string()
                    );
                }
                match self.plasma_rate_kind {
                    PlasmaRateKind::Ppt => {
                        let mut err_code_d = GpuBuffer::alloc(4)?;
                        let zero = [0i32];
                        err_code_d.copy_to_device(&zero)?;
                        let mut num_segments_val = self.plasma_num_segments as c_int;
                        let mut strict_val = self.plasma_strict;
                        let mut e_min = self.plasma_e_min;
                        let mut e_max = self.plasma_e_max;
                        let mut rate_args: [*mut libc::c_void; 9] = [
                            &mut self.radial_qdht_d.dptr as *mut _ as *mut _,
                            &mut self.plas_rate_d.dptr as *mut _ as *mut _,
                            &mut self.plasma_segments_d.dptr as *mut _ as *mut _,
                            &mut e_min as *mut _ as *mut _,
                            &mut e_max as *mut _ as *mut _,
                            &mut num_segments_val as *mut _ as *mut _,
                            &mut total_time_i as *mut _ as *mut _,
                            &mut err_code_d.dptr as *mut _ as *mut _,
                            &mut strict_val as *mut _ as *mut _,
                        ];
                        launch_checked(
                            driver,
                            ctx.ppt_fn,
                            grid_time,
                            block_size,
                            0,
                            &mut rate_args,
                            "plasma_radial_rate_ppt",
                        )?;
                    }
                    PlasmaRateKind::Adk => {
                        let mut occupancy = self.plasma_adk_occupancy;
                        let mut omega_p = self.plasma_adk_omega_p;
                        let mut cn_sq = self.plasma_adk_cn_sq;
                        let mut nstar = self.plasma_adk_nstar;
                        let mut omega_t_prefac = self.plasma_adk_omega_t_prefac;
                        let mut thr = self.plasma_adk_thr;
                        let mut avfac = self.plasma_adk_avfac;
                        let mut rate_args: [*mut libc::c_void; 10] = [
                            &mut self.radial_qdht_d.dptr as *mut _ as *mut _,
                            &mut self.plas_rate_d.dptr as *mut _ as *mut _,
                            &mut occupancy as *mut _ as *mut _,
                            &mut omega_p as *mut _ as *mut _,
                            &mut cn_sq as *mut _ as *mut _,
                            &mut nstar as *mut _ as *mut _,
                            &mut omega_t_prefac as *mut _ as *mut _,
                            &mut thr as *mut _ as *mut _,
                            &mut avfac as *mut _ as *mut _,
                            &mut total_time_i as *mut _ as *mut _,
                        ];
                        launch_checked(
                            driver,
                            ctx.adk_fn,
                            grid_time,
                            block_size,
                            0,
                            &mut rate_args,
                            "plasma_radial_rate_adk",
                        )?;
                    }
                }

                let n_blocks = self.n_time_over.div_ceil(block_size as usize);
                let mut n_time_i = c_int::try_from(self.n_time_over).map_err(|_| {
                    "CUDA radial plasma time dimension exceeds kernel range".to_string()
                })?;
                let mut n_r_i = c_int::try_from(self.n_r).map_err(|_| {
                    "CUDA radial plasma radial dimension exceeds kernel range".to_string()
                })?;
                let mut n_blocks_i = c_int::try_from(n_blocks).map_err(|_| {
                    "CUDA radial plasma block count exceeds kernel range".to_string()
                })?;
                let mut preionfrac = self.plasma_preionfrac;
                let mut fraction_args: [*mut libc::c_void; 6] = [
                    &mut self.plas_fraction_d.dptr as *mut _ as *mut _,
                    &mut self.plas_scan_sums_d.dptr as *mut _ as *mut _,
                    &mut preionfrac as *mut _ as *mut _,
                    &mut n_time_i as *mut _ as *mut _,
                    &mut n_r_i as *mut _ as *mut _,
                    &mut n_blocks_i as *mut _ as *mut _,
                ];
                self.plasma_scan_radial(
                    self.plas_rate_d.dptr,
                    self.plas_fraction_d.dptr,
                    "plasma_radial_fraction_scan",
                )?;
                launch_checked(
                    driver,
                    ctx.plasma_fraction_radial_finalize_fn,
                    grid_time,
                    block_size,
                    0,
                    &mut fraction_args,
                    "plasma_radial_fraction_finalize",
                )?;

                let mut e_ratio = self.plasma_e_ratio;
                let mut phase_args: [*mut libc::c_void; 5] = [
                    &mut self.plas_fraction_d.dptr as *mut _ as *mut _,
                    &mut self.radial_qdht_d.dptr as *mut _ as *mut _,
                    &mut e_ratio as *mut _ as *mut _,
                    &mut self.plas_phase_d.dptr as *mut _ as *mut _,
                    &mut total_time_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.plasma_phase_radial_fn,
                    grid_time,
                    block_size,
                    0,
                    &mut phase_args,
                    "plasma_radial_phase",
                )?;

                self.plasma_scan_radial(
                    self.plas_phase_d.dptr,
                    self.plas_current_d.dptr,
                    "plasma_radial_current_scan",
                )?;
                let mut ionpot = self.plasma_ionpot;
                let mut current_args: [*mut libc::c_void; 9] = [
                    &mut self.plas_current_d.dptr as *mut _ as *mut _,
                    &mut self.plas_scan_sums_d.dptr as *mut _ as *mut _,
                    &mut self.plas_rate_d.dptr as *mut _ as *mut _,
                    &mut self.plas_fraction_d.dptr as *mut _ as *mut _,
                    &mut self.radial_qdht_d.dptr as *mut _ as *mut _,
                    &mut ionpot as *mut _ as *mut _,
                    &mut n_time_i as *mut _ as *mut _,
                    &mut n_r_i as *mut _ as *mut _,
                    &mut n_blocks_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.plasma_current_radial_finalize_fn,
                    grid_time,
                    block_size,
                    0,
                    &mut current_args,
                    "plasma_radial_current_finalize",
                )?;

                self.plasma_scan_radial(
                    self.plas_current_d.dptr,
                    self.plas_phase_d.dptr,
                    "plasma_radial_polarization_scan",
                )?;
                let mut density = self.plasma_density;
                let mut polarization_args: [*mut libc::c_void; 7] = [
                    &mut self.plas_phase_d.dptr as *mut _ as *mut _,
                    &mut self.plas_scan_sums_d.dptr as *mut _ as *mut _,
                    &mut self.radial_pto_d.dptr as *mut _ as *mut _,
                    &mut density as *mut _ as *mut _,
                    &mut n_time_i as *mut _ as *mut _,
                    &mut n_r_i as *mut _ as *mut _,
                    &mut n_blocks_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.plasma_polarization_radial_finalize_fn,
                    grid_time,
                    block_size,
                    0,
                    &mut polarization_args,
                    "plasma_radial_polarization_finalize",
                )?;
            }

            let mut n_time_i = self.n_time_over as c_int;
            let mut window_args: [*mut libc::c_void; 4] = [
                &mut self.radial_pto_d.dptr as *mut _ as *mut _,
                &mut self.radial_towin_d.dptr as *mut _ as *mut _,
                &mut n_time_i as *mut _ as *mut _,
                &mut n_r_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.apply_radial_time_window_fn,
                grid_time,
                block_size,
                0,
                &mut window_args,
                "apply_radial_time_window",
            )?;

            let mut qdht_scale = self.radial_scale_fwd;
            let mut qdht_args: [*mut libc::c_void; 6] = [
                &mut self.radial_pto_d.dptr as *mut _ as *mut _,
                &mut self.radial_qdht_d.dptr as *mut _ as *mut _,
                &mut self.radial_qdht_matrix_d.dptr as *mut _ as *mut _,
                &mut qdht_scale as *mut _ as *mut _,
                &mut n_time_over_i as *mut _ as *mut _,
                &mut n_r_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.qdht_radial_real_fn,
                grid_time,
                block_size,
                0,
                &mut qdht_args,
                "qdht_radial_mul",
            )?;

            for r in 0..self.n_r {
                let qdht_ptr = (self.radial_qdht_d.dptr
                    + (r * self.n_time_over * std::mem::size_of::<f64>()) as u64)
                    as *mut f64;
                let poo_ptr = (self.radial_poo_d.dptr
                    + (r * self.n_spec_over * std::mem::size_of::<Complex<f64>>()) as u64)
                    as *mut libc::c_void;
                let rc = (cufft.cufftExecD2Z)(self.radial_fft_r2c, qdht_ptr, poo_ptr);
                if rc != 0 {
                    return Err(format!("cufftExecD2Z (radial column {r}) failed: {rc}"));
                }
            }

            let mut scale_inv = (n_spec - 1) as f64 / (self.n_spec_over - 1) as f64;
            let mut finalize_args: [*mut libc::c_void; 7] = [
                &mut self.radial_poo_d.dptr as *mut _ as *mut _,
                &mut self.ks_d[idx].dptr as *mut _ as *mut _,
                &mut self.radial_norm_d.dptr as *mut _ as *mut _,
                &mut scale_inv as *mut _ as *mut _,
                &mut n_spec_i as *mut _ as *mut _,
                &mut n_spec_over_i as *mut _ as *mut _,
                &mut n_r_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.finalize_radial_spectrum_fn,
                grid_spec,
                block_size,
                0,
                &mut finalize_args,
                "finalize_radial_spectrum",
            )
        }
    }

    /// CUDA radial EnvGrid RHS — mirrors `CpuNativeSim::rhs_radial_env` for
    /// scalar envelope Kerr.  Each radial column uses the full complex c2c
    /// spectrum, complex QDHT directions, the `3/4` envelope Kerr factor from
    /// `rhs_mode_avg_env_kernel`, and the same crop/normalization order as the
    /// CPU oracle.
    unsafe fn compute_rhs_radial_env(&mut self, idx: usize) -> Result<(), String> {
        if !self.is_radial || self.n_r == 0 || self.radial_fft_c2c == 0 {
            return Err("CUDA radial EnvGrid configuration is not initialized".to_string());
        }
        let ctx = get_gpu_context().ok_or_else(|| "GPU context not initialized".to_string())?;
        let driver = get_driver_api()?;
        let cufft = get_cufft_api()?;
        unsafe {
            crate::cuda::activate_context()?;
            let block_size = 256u32;
            let n_spec = self.n / self.n_r;
            let total_time = self
                .n_time_over
                .checked_mul(self.n_r)
                .ok_or_else(|| "CUDA radial EnvGrid time launch size overflow".to_string())?;
            let total_spec = self
                .n_spec_over
                .checked_mul(self.n_r)
                .ok_or_else(|| "CUDA radial EnvGrid spectral launch size overflow".to_string())?;
            let total_time_u32 = u32::try_from(total_time).map_err(|_| {
                "CUDA radial EnvGrid time launch size exceeds grid range".to_string()
            })?;
            let total_spec_u32 = u32::try_from(total_spec).map_err(|_| {
                "CUDA radial EnvGrid spectral launch size exceeds grid range".to_string()
            })?;
            let n_u32 = u32::try_from(self.n)
                .map_err(|_| "CUDA radial EnvGrid field size exceeds grid range".to_string())?;
            let grid_time = total_time_u32.div_ceil(block_size);
            let grid_spec_over = total_spec_u32.div_ceil(block_size);
            let grid_spec = n_u32.div_ceil(block_size);
            let mut n_spec_i = c_int::try_from(n_spec).map_err(|_| {
                "CUDA radial EnvGrid spectral dimension exceeds kernel range".to_string()
            })?;
            let mut n_spec_over_i = c_int::try_from(self.n_spec_over).map_err(|_| {
                "CUDA radial EnvGrid oversampled spectral dimension exceeds kernel range"
                    .to_string()
            })?;
            let mut n_time_over_i = c_int::try_from(self.n_time_over).map_err(|_| {
                "CUDA radial EnvGrid time dimension exceeds kernel range".to_string()
            })?;
            let mut n_r_i = c_int::try_from(self.n_r).map_err(|_| {
                "CUDA radial EnvGrid radial dimension exceeds kernel range".to_string()
            })?;

            // CPU rhs_radial_env's to_time! copies the low and high halves and
            // scales by no/n.  The radial kernel applies that rule per column.
            let mut scale_fwd = self.n_spec_over as f64 / n_spec as f64;
            let mut expand_args: [*mut libc::c_void; 6] = [
                &mut self.ystage_d.dptr as *mut _ as *mut _,
                &mut self.radial_eoo_d.dptr as *mut _ as *mut _,
                &mut scale_fwd as *mut _ as *mut _,
                &mut n_spec_i as *mut _ as *mut _,
                &mut n_spec_over_i as *mut _ as *mut _,
                &mut n_r_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.expand_radial_spectrum_env_fn,
                grid_spec_over,
                block_size,
                0,
                &mut expand_args,
                "expand_radial_spectrum_env",
            )?;

            for r in 0..self.n_r {
                let eoo_ptr = self.radial_eoo_d.dptr
                    + (r * self.n_spec_over * std::mem::size_of::<Complex<f64>>()) as u64;
                let eto_ptr = self.radial_eto_d.dptr
                    + (r * self.n_time_over * std::mem::size_of::<Complex<f64>>()) as u64;
                let rc = (cufft.cufftExecZ2Z)(
                    self.radial_fft_c2c,
                    eoo_ptr as *mut _,
                    eto_ptr as *mut _,
                    CUFFT_INVERSE,
                );
                if rc != 0 {
                    return Err(format!(
                        "cufftExecZ2Z (radial EnvGrid inverse column {r}) failed: {rc}"
                    ));
                }
            }

            let mut inverse_scale = 1.0 / self.n_time_over as f64;
            // Keep the launch argument alive for the complete call; the
            // explicit local avoids passing a pointer to a cast temporary.
            let mut total_time_i = c_int::try_from(total_time).map_err(|_| {
                "CUDA radial EnvGrid time launch size exceeds kernel range".to_string()
            })?;
            let mut scale_args: [*mut libc::c_void; 3] = [
                &mut self.radial_eto_d.dptr as *mut _ as *mut _,
                &mut inverse_scale as *mut _ as *mut _,
                &mut total_time_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.scale_complex_fn,
                grid_time,
                block_size,
                0,
                &mut scale_args,
                "scale_radial_env_inverse_fft",
            )?;

            let mut qdht_scale = self.radial_scale_inv;
            let mut qdht_args: [*mut libc::c_void; 6] = [
                &mut self.radial_eto_d.dptr as *mut _ as *mut _,
                &mut self.radial_qdht_d.dptr as *mut _ as *mut _,
                &mut self.radial_qdht_matrix_d.dptr as *mut _ as *mut _,
                &mut qdht_scale as *mut _ as *mut _,
                &mut n_time_over_i as *mut _ as *mut _,
                &mut n_r_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.qdht_radial_complex_fn,
                grid_time,
                block_size,
                0,
                &mut qdht_args,
                "qdht_radial_env_ldiv",
            )?;

            let mut kerr_fac = self.radial_kerr_fac;
            let mut kerr_args: [*mut libc::c_void; 4] = [
                &mut self.radial_pto_d.dptr as *mut _ as *mut _,
                &mut self.radial_qdht_d.dptr as *mut _ as *mut _,
                &mut kerr_fac as *mut _ as *mut _,
                &mut total_time_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.rhs_mode_avg_env_fn,
                grid_time,
                block_size,
                0,
                &mut kerr_args,
                "rhs_radial_env_kerr",
            )?;

            let mut window_args: [*mut libc::c_void; 4] = [
                &mut self.radial_pto_d.dptr as *mut _ as *mut _,
                &mut self.radial_towin_d.dptr as *mut _ as *mut _,
                &mut n_time_over_i as *mut _ as *mut _,
                &mut n_r_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.apply_radial_time_window_complex_fn,
                grid_time,
                block_size,
                0,
                &mut window_args,
                "apply_radial_env_time_window",
            )?;

            let mut qdht_scale = self.radial_scale_fwd;
            let mut qdht_args: [*mut libc::c_void; 6] = [
                &mut self.radial_pto_d.dptr as *mut _ as *mut _,
                &mut self.radial_qdht_d.dptr as *mut _ as *mut _,
                &mut self.radial_qdht_matrix_d.dptr as *mut _ as *mut _,
                &mut qdht_scale as *mut _ as *mut _,
                &mut n_time_over_i as *mut _ as *mut _,
                &mut n_r_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.qdht_radial_complex_fn,
                grid_time,
                block_size,
                0,
                &mut qdht_args,
                "qdht_radial_env_mul",
            )?;

            for r in 0..self.n_r {
                let qdht_ptr = self.radial_qdht_d.dptr
                    + (r * self.n_time_over * std::mem::size_of::<Complex<f64>>()) as u64;
                let poo_ptr = self.radial_poo_d.dptr
                    + (r * self.n_spec_over * std::mem::size_of::<Complex<f64>>()) as u64;
                let rc = (cufft.cufftExecZ2Z)(
                    self.radial_fft_c2c,
                    qdht_ptr as *mut _,
                    poo_ptr as *mut _,
                    CUFFT_FORWARD,
                );
                if rc != 0 {
                    return Err(format!(
                        "cufftExecZ2Z (radial EnvGrid forward column {r}) failed: {rc}"
                    ));
                }
            }

            // The unnormalized forward c2c transform is cropped back from no
            // to n samples, matching rhs_radial_env's n/no factor.
            let mut scale_inv = n_spec as f64 / self.n_spec_over as f64;
            let mut finalize_args: [*mut libc::c_void; 7] = [
                &mut self.radial_poo_d.dptr as *mut _ as *mut _,
                &mut self.ks_d[idx].dptr as *mut _ as *mut _,
                &mut self.radial_norm_d.dptr as *mut _ as *mut _,
                &mut scale_inv as *mut _ as *mut _,
                &mut n_spec_i as *mut _ as *mut _,
                &mut n_spec_over_i as *mut _ as *mut _,
                &mut n_r_i as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.finalize_radial_spectrum_env_fn,
                grid_spec,
                block_size,
                0,
                &mut finalize_args,
                "finalize_radial_spectrum_env",
            )
        }
    }

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
        if self.is_real {
            unsafe { self.compute_rhs_mode_avg_real(idx) }
        } else {
            unsafe { self.compute_rhs_mode_avg_env(idx) }
        }
    }

    unsafe fn compute_rhs_mode_avg_real(&mut self, idx: usize) -> Result<(), String> {
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

            // ── Step 3b: plasma polarisation. PPT and ADK share the completed
            // fraction/current/polarization scan/finalizer pipeline; only the
            // pointwise rate kernel differs.
            if self.has_plasma {
                match self.plasma_rate_kind {
                    PlasmaRateKind::Ppt => {
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
                            "plasma_rate_ppt",
                        )?;
                    }
                    PlasmaRateKind::Adk => {
                        let mut occupancy = self.plasma_adk_occupancy;
                        let mut omega_p = self.plasma_adk_omega_p;
                        let mut cn_sq = self.plasma_adk_cn_sq;
                        let mut nstar = self.plasma_adk_nstar;
                        let mut omega_t_prefac = self.plasma_adk_omega_t_prefac;
                        let mut thr = self.plasma_adk_thr;
                        let mut avfac = self.plasma_adk_avfac;
                        let mut rate_args: [*mut libc::c_void; 10] = [
                            &mut self.eto_d.dptr as *mut _ as *mut _,
                            &mut self.plas_rate_d.dptr as *mut _ as *mut _,
                            &mut occupancy as *mut _ as *mut _,
                            &mut omega_p as *mut _ as *mut _,
                            &mut cn_sq as *mut _ as *mut _,
                            &mut nstar as *mut _ as *mut _,
                            &mut omega_t_prefac as *mut _ as *mut _,
                            &mut thr as *mut _ as *mut _,
                            &mut avfac as *mut _ as *mut _,
                            &mut n_time_over_i as *mut _ as *mut _,
                        ];
                        launch_checked(
                            driver,
                            ctx.adk_fn,
                            grid_size_t,
                            block_size,
                            0,
                            &mut rate_args,
                            "plasma_rate_adk",
                        )?;
                    }
                }

                // Parallel cumtrapz(rate) then rho transform.
                let rate_dptr = self.plas_rate_d.dptr;
                let fraction_dptr = self.plas_fraction_d.dptr;
                self.plasma_scan(rate_dptr, fraction_dptr, "plasma_fraction_scan")?;
                let mut preionfrac = self.plasma_preionfrac;
                let mut fraction_args: [*mut libc::c_void; 4] = [
                    &mut self.plas_fraction_d.dptr as *mut _ as *mut _,
                    &mut self.plas_scan_sums_d.dptr as *mut _ as *mut _,
                    &mut preionfrac as *mut _ as *mut _,
                    &mut n_time_over_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.plasma_fraction_finalize_fn,
                    grid_size_t,
                    block_size,
                    0,
                    &mut fraction_args,
                    "plasma_fraction_finalize",
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

                // Parallel cumtrapz(phase), then add the ionization-loss
                // current term elementwise.
                let phase_dptr = self.plas_phase_d.dptr;
                let current_dptr = self.plas_current_d.dptr;
                self.plasma_scan(phase_dptr, current_dptr, "plasma_current_scan")?;
                let mut ionpot = self.plasma_ionpot;
                let mut current_args: [*mut libc::c_void; 7] = [
                    &mut self.plas_current_d.dptr as *mut _ as *mut _,
                    &mut self.plas_scan_sums_d.dptr as *mut _ as *mut _,
                    &mut self.plas_rate_d.dptr as *mut _ as *mut _,
                    &mut self.plas_fraction_d.dptr as *mut _ as *mut _,
                    &mut self.eto_d.dptr as *mut _ as *mut _,
                    &mut ionpot as *mut _ as *mut _,
                    &mut n_time_over_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.plasma_current_finalize_fn,
                    grid_size_t,
                    block_size,
                    0,
                    &mut current_args,
                    "plasma_current_finalize",
                )?;

                // `plas_phase_d` is no longer needed after the current has
                // been formed, so reuse it for cumtrapz(current).
                let current_dptr = self.plas_current_d.dptr;
                let polarization_dptr = self.plas_phase_d.dptr;
                self.plasma_scan(current_dptr, polarization_dptr, "plasma_polarization_scan")?;
                let mut density = self.plasma_density;
                let mut polarization_args: [*mut libc::c_void; 5] = [
                    &mut self.plas_phase_d.dptr as *mut _ as *mut _,
                    &mut self.plas_scan_sums_d.dptr as *mut _ as *mut _,
                    &mut self.pto_d.dptr as *mut _ as *mut _,
                    &mut density as *mut _ as *mut _,
                    &mut n_time_over_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.plasma_polarization_finalize_fn,
                    grid_size_t,
                    block_size,
                    0,
                    &mut polarization_args,
                    "plasma_polarization_finalize",
                )?;
            }

            // ── Step 3c: resident Raman SDO ADE. `thg=true` uses the
            // carrier field square directly; `thg=false` follows
            // RamanPolarField's analytic-signal Hilbert convention using a
            // dedicated c2c plan and resident complex scratch.
            if self.has_raman {
                let mut thg = self.raman_thg as c_int;
                let mut intensity_args: [*mut libc::c_void; 4] = [
                    &mut self.eto_d.dptr as *mut _ as *mut _,
                    &mut self.raman_intensity_d.dptr as *mut _ as *mut _,
                    &mut n_time_over_i as *mut _ as *mut _,
                    &mut thg as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.raman_intensity_real_fn,
                    grid_size_t,
                    block_size,
                    0,
                    &mut intensity_args,
                    "raman_intensity_real",
                )?;
                if !self.raman_thg {
                    let mut pack_args: [*mut libc::c_void; 3] = [
                        &mut self.eto_d.dptr as *mut _ as *mut _,
                        &mut self.raman_hilbert_a_d.dptr as *mut _ as *mut _,
                        &mut n_time_over_i as *mut _ as *mut _,
                    ];
                    launch_checked(
                        driver,
                        ctx.raman_hilbert_pack_fn,
                        grid_size_t,
                        block_size,
                        0,
                        &mut pack_args,
                        "raman_hilbert_pack",
                    )?;
                    let rc = (cufft.cufftExecZ2Z)(
                        self.raman_hilbert_fft,
                        self.raman_hilbert_a_d.dptr as *mut _,
                        self.raman_hilbert_b_d.dptr as *mut _,
                        CUFFT_FORWARD,
                    );
                    if rc != 0 {
                        return Err(format!("raman Hilbert forward failed ({rc})"));
                    }
                    let mut filter_args: [*mut libc::c_void; 2] = [
                        &mut self.raman_hilbert_b_d.dptr as *mut _ as *mut _,
                        &mut n_time_over_i as *mut _ as *mut _,
                    ];
                    launch_checked(
                        driver,
                        ctx.raman_hilbert_filter_fn,
                        grid_size_t,
                        block_size,
                        0,
                        &mut filter_args,
                        "raman_hilbert_filter",
                    )?;
                    let rc = (cufft.cufftExecZ2Z)(
                        self.raman_hilbert_fft,
                        self.raman_hilbert_b_d.dptr as *mut _,
                        self.raman_hilbert_a_d.dptr as *mut _,
                        CUFFT_INVERSE,
                    );
                    if rc != 0 {
                        return Err(format!("raman Hilbert inverse failed ({rc})"));
                    }
                    let mut hilbert_scale = 1.0 / self.n_time_over as f64;
                    let mut hilbert_scale_args: [*mut libc::c_void; 3] = [
                        &mut self.raman_hilbert_a_d.dptr as *mut _ as *mut _,
                        &mut hilbert_scale as *mut _ as *mut _,
                        &mut n_time_over_i as *mut _ as *mut _,
                    ];
                    launch_checked(
                        driver,
                        ctx.scale_complex_fn,
                        grid_size_t,
                        block_size,
                        0,
                        &mut hilbert_scale_args,
                        "raman Hilbert inverse scale",
                    )?;
                    let mut intensity_args: [*mut libc::c_void; 3] = [
                        &mut self.raman_hilbert_a_d.dptr as *mut _ as *mut _,
                        &mut self.raman_intensity_d.dptr as *mut _ as *mut _,
                        &mut n_time_over_i as *mut _ as *mut _,
                    ];
                    launch_checked(
                        driver,
                        ctx.raman_hilbert_intensity_fn,
                        grid_size_t,
                        block_size,
                        0,
                        &mut intensity_args,
                        "raman_hilbert_intensity",
                    )?;
                }
                self.launch_raman_ade(driver, ctx, 1, self.n_time_over)?;
                let mut density = self.raman_density;
                let mut accumulate_args: [*mut libc::c_void; 5] = [
                    &mut self.pto_d.dptr as *mut _ as *mut _,
                    &mut self.eto_d.dptr as *mut _ as *mut _,
                    &mut self.raman_p_d.dptr as *mut _ as *mut _,
                    &mut density as *mut _ as *mut _,
                    &mut n_time_over_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.raman_accumulate_real_fn,
                    grid_size_t,
                    block_size,
                    0,
                    &mut accumulate_args,
                    "raman_accumulate_real",
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

    /// EnvGrid counterpart of `compute_rhs_mode_avg_real`. It mirrors
    /// `CpuNativeSim::rhs_mode_avg_env`: copy the low/high spectral halves into
    /// an oversampled c2c buffer, apply the inverse transform and scaling,
    /// evaluate Kerr plus Raman in complex time domain, window, then copy the
    /// low/high output halves back to the ODE state.
    unsafe fn compute_rhs_mode_avg_env(&mut self, idx: usize) -> Result<(), String> {
        if self.n_time_over == 0 || self.fft_c2c == 0 {
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
            let grid_spec = (self.n as u32).div_ceil(block_size);
            let grid_over = (self.n_spec_over as u32).div_ceil(block_size);
            let mut n_spec = self.n as c_int;
            let mut n_spec_over = self.n_spec_over as c_int;
            let mut n_time_over = self.n_time_over as c_int;
            let mut scale_fwd = self.scale_fwd;
            let mut expand_args: [*mut libc::c_void; 5] = [
                &mut self.ystage_d.dptr as *mut _ as *mut _,
                &mut self.eoo_d.dptr as *mut _ as *mut _,
                &mut scale_fwd as *mut _ as *mut _,
                &mut n_spec as *mut _ as *mut _,
                &mut n_spec_over as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.expand_spectrum_env_fn,
                grid_over,
                block_size,
                0,
                &mut expand_args,
                "expand_spectrum_env",
            )?;

            let rc = (cufft.cufftExecZ2Z)(
                self.fft_c2c,
                self.eoo_d.dptr as *mut _,
                self.eto_d.dptr as *mut _,
                CUFFT_INVERSE,
            );
            if rc != 0 {
                return Err(format!("cufftExecZ2Z inverse failed ({rc})"));
            }

            let mut inv_scale = self.inv_nto_sc;
            let mut scale_args: [*mut libc::c_void; 3] = [
                &mut self.eto_d.dptr as *mut _ as *mut _,
                &mut inv_scale as *mut _ as *mut _,
                &mut n_time_over as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.scale_complex_fn,
                grid_over,
                block_size,
                0,
                &mut scale_args,
                "scale_eto_env(step1+2)",
            )?;

            let mut kerr_fac = self.kerr_fac;
            let mut kerr_args: [*mut libc::c_void; 4] = [
                &mut self.pto_d.dptr as *mut _ as *mut _,
                &mut self.eto_d.dptr as *mut _ as *mut _,
                &mut kerr_fac as *mut _ as *mut _,
                &mut n_time_over as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.rhs_mode_avg_env_fn,
                grid_over,
                block_size,
                0,
                &mut kerr_args,
                "rhs_mode_avg_env(step3)",
            )?;

            if self.has_raman {
                let mut intensity_args: [*mut libc::c_void; 3] = [
                    &mut self.eto_d.dptr as *mut _ as *mut _,
                    &mut self.raman_intensity_d.dptr as *mut _ as *mut _,
                    &mut n_time_over as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.raman_intensity_env_fn,
                    grid_over,
                    block_size,
                    0,
                    &mut intensity_args,
                    "raman_intensity_env",
                )?;
                self.launch_raman_ade(driver, ctx, 1, n_time_over as usize)?;
                let mut density = self.raman_density;
                let mut accumulate_args: [*mut libc::c_void; 5] = [
                    &mut self.pto_d.dptr as *mut _ as *mut _,
                    &mut self.eto_d.dptr as *mut _ as *mut _,
                    &mut self.raman_p_d.dptr as *mut _ as *mut _,
                    &mut density as *mut _ as *mut _,
                    &mut n_time_over as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.raman_accumulate_env_fn,
                    grid_over,
                    block_size,
                    0,
                    &mut accumulate_args,
                    "raman_accumulate_env",
                )?;
            }

            // ── Step 3c: resident EnvGrid intermediate-broadening Raman.
            // The padded 0.5*|E|² and response impulse are real, so use the
            // resident D2Z/Z2D pair at length 2*n_time_over. The response
            // spectrum already includes dt/n_over and is multiplied in place;
            // no host transfer occurs during an RHS evaluation.
            if self.has_raman_fft {
                let n_over = self
                    .n_time_over
                    .checked_mul(2)
                    .ok_or_else(|| "EnvGrid Raman FFT length overflow".to_string())?;
                let n_over_i = n_over as c_int;
                let n_spec_over = self.n_time_over + 1;
                let n_spec_over_i = n_spec_over as c_int;
                let grid_padded = (n_over as u32).div_ceil(block_size);
                let grid_spectrum = (n_spec_over as u32).div_ceil(block_size);

                let mut n_time_over_i = self.n_time_over as c_int;
                let mut pack_n_over_i = n_over_i;
                let mut pack_args: [*mut libc::c_void; 4] = [
                    &mut self.eto_d.dptr as *mut _ as *mut _,
                    &mut self.raman_fft_e2_d.dptr as *mut _ as *mut _,
                    &mut n_time_over_i as *mut _ as *mut _,
                    &mut pack_n_over_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.raman_fft_pack_env_fn,
                    grid_padded,
                    block_size,
                    0,
                    &mut pack_args,
                    "raman_fft_pack_env",
                )?;

                let rc = (cufft.cufftExecD2Z)(
                    self.raman_fft_r2c,
                    self.raman_fft_e2_d.dptr as *mut _,
                    self.raman_fft_ew_d.dptr as *mut _,
                );
                if rc != 0 {
                    return Err(format!("cufftExecD2Z (Raman convolution) failed: {rc}"));
                }

                let mut n_spec_i = n_spec_over_i;
                let mut multiply_args: [*mut libc::c_void; 3] = [
                    &mut self.raman_fft_ew_d.dptr as *mut _ as *mut _,
                    &mut self.raman_fft_hw_d.dptr as *mut _ as *mut _,
                    &mut n_spec_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.raman_fft_multiply_fn,
                    grid_spectrum,
                    block_size,
                    0,
                    &mut multiply_args,
                    "raman_fft_multiply",
                )?;

                let rc = (cufft.cufftExecZ2D)(
                    self.raman_fft_c2r,
                    self.raman_fft_ew_d.dptr as *mut _,
                    self.raman_fft_e2_d.dptr as *mut _,
                );
                if rc != 0 {
                    return Err(format!("cufftExecZ2D (Raman convolution) failed: {rc}"));
                }

                let mut density = self.raman_fft_density;
                let mut accumulate_args: [*mut libc::c_void; 5] = [
                    &mut self.pto_d.dptr as *mut _ as *mut _,
                    &mut self.eto_d.dptr as *mut _ as *mut _,
                    &mut self.raman_fft_e2_d.dptr as *mut _ as *mut _,
                    &mut density as *mut _ as *mut _,
                    &mut n_time_over_i as *mut _ as *mut _,
                ];
                launch_checked(
                    driver,
                    ctx.raman_accumulate_env_fn,
                    grid_over,
                    block_size,
                    0,
                    &mut accumulate_args,
                    "raman_fft_accumulate_env",
                )?;
            }

            let mut window_args: [*mut libc::c_void; 3] = [
                &mut self.pto_d.dptr as *mut _ as *mut _,
                &mut self.towin_d.dptr as *mut _ as *mut _,
                &mut n_time_over as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.apply_time_window_complex_fn,
                grid_over,
                block_size,
                0,
                &mut window_args,
                "apply_time_window_env(step4)",
            )?;

            let rc = (cufft.cufftExecZ2Z)(
                self.fft_c2c,
                self.pto_d.dptr as *mut _,
                self.poo_d.dptr as *mut _,
                CUFFT_FORWARD,
            );
            if rc != 0 {
                return Err(format!("cufftExecZ2Z forward failed ({rc})"));
            }
            let mut scale_inv = self.scale_inv;
            let mut finalize_args: [*mut libc::c_void; 7] = [
                &mut self.poo_d.dptr as *mut _ as *mut _,
                &mut self.ks_d[idx].dptr as *mut _ as *mut _,
                &mut self.norm_pre_beta_d.dptr as *mut _ as *mut _,
                &mut self.owin_d.dptr as *mut _ as *mut _,
                &mut scale_inv as *mut _ as *mut _,
                &mut n_spec as *mut _ as *mut _,
                &mut n_spec_over as *mut _ as *mut _,
            ];
            launch_checked(
                driver,
                ctx.finalize_spectrum_env_fn,
                grid_spec,
                block_size,
                0,
                &mut finalize_args,
                "finalize_spectrum_env(step5+6+7)",
            )?;
            Ok(())
        }
    }

    unsafe fn launch_raman_ade(
        &self,
        driver: &crate::cuda::CudaDriverApi,
        ctx: &crate::cuda::GpuContext,
        grid: u32,
        n_time: usize,
    ) -> Result<(), String> {
        let mut intensity = self.raman_intensity_d.dptr;
        let mut polarization = self.raman_p_d.dptr;
        let mut coeffs = self.raman_coeffs_d.dptr;
        let mut num_osc = self.raman_num_osc as c_int;
        let mut n_time_i = n_time as c_int;
        let mut n_series = 1i32;
        let mut args: [*mut libc::c_void; 6] = [
            &mut intensity as *mut _ as *mut _,
            &mut polarization as *mut _ as *mut _,
            &mut coeffs as *mut _ as *mut _,
            &mut num_osc as *mut _ as *mut _,
            &mut n_time_i as *mut _ as *mut _,
            &mut n_series as *mut _ as *mut _,
        ];
        unsafe {
            launch_checked(
                driver,
                ctx.raman_fn,
                grid,
                1,
                0,
                &mut args,
                "raman_ade_resident",
            )
        }
    }
}

impl NativeBackend for CudaNativeSim {
    unsafe fn set_field(&mut self, data: *const c_double, n: size_t) -> i32 {
        if data.is_null() || n != self.n {
            return -1;
        }
        unsafe {
            let slice = std::slice::from_raw_parts(data as *const Complex<f64>, n);
            if self.field_d.copy_to_device(slice).is_err() {
                return -1;
            }
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
            let rhs_result = if self.is_radial {
                self.compute_rhs_radial(0)
            } else {
                self.compute_rhs_mode_avg(0)
            };
            if rhs_result.is_err() {
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
        if data.is_null() || n != self.n {
            return -1;
        }
        unsafe {
            let slice = std::slice::from_raw_parts(data as *const Complex<f64>, n);
            if self.field_d.copy_to_device(slice).is_err() {
                return -1;
            }
        }
        0
    }

    unsafe fn get_field(&self, data: *mut c_double, n: size_t) -> i32 {
        if data.is_null() || n != self.n {
            return -1;
        }
        unsafe {
            let slice = std::slice::from_raw_parts_mut(data as *mut Complex<f64>, n);
            if self.field_d.copy_to_host(slice).is_err() {
                return -1;
            }
        }
        0
    }

    unsafe fn get_ks_stage(&self, idx: size_t, data: *mut c_double, n: size_t) -> i32 {
        if data.is_null() || idx >= 7 || n != self.n {
            -1
        } else {
            unsafe {
                let slice = std::slice::from_raw_parts_mut(data as *mut Complex<f64>, n);
                if self.ks_d[idx].copy_to_host(slice).is_err() {
                    return -1;
                }
            }
            0
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
        is_real: c_int,
        _flags: c_uint,
        _wisdom_path: *const c_char,
    ) -> i32 {
        self.is_real = is_real != 0;
        0 // Replaced by cuFFT; retain the grid kind for mode-avg setup.
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
        match unsafe {
            self.stage_mode_avg_setup(
                n_time,
                n_time_over,
                towin,
                owin,
                sidx,
                pre_re,
                pre_im,
                beta,
                kerr_fac,
                nlscale,
                sqrt_aeff,
            )
        } {
            Ok(staged) => {
                self.commit_mode_avg_setup(staged);
                0
            }
            Err(e) => {
                eprintln!("Amalthea GPU error: mode-averaged setup failed: {e}");
                -1
            }
        }
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

    // PPT branch of the shared plasma setup. Mirrors native.rs's
    // `CpuNativeSim::set_plasma_params`: uploads the same
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
        if self.n_time == 0 || ion_ptr.is_null() || !dt.is_finite() {
            return -2;
        }
        if self.is_radial && (!self.is_real || self.n_r == 0) {
            return -2;
        }
        let ion = unsafe { &*ion_ptr };
        let segments = &ion.spline_lut.segments;
        if segments.is_empty() {
            return -2;
        }
        let n_series = if self.is_radial { self.n_r } else { 1 };
        let scratch_len = match self.n_time_over.checked_mul(n_series) {
            Some(len) if len > 0 => len,
            _ => return -2,
        };
        let n_blocks = self.n_time_over.div_ceil(256);
        let scan_len = match n_blocks.checked_mul(n_series) {
            Some(len) if len > 0 => len,
            _ => return -2,
        };
        let segments_bytes = match checked_bytes(
            segments.len(),
            std::mem::size_of::<crate::ionization::SplineSegment>(),
        ) {
            Ok(bytes) => bytes,
            Err(_) => return -2,
        };
        let scratch_bytes = match checked_bytes(scratch_len, 8) {
            Ok(bytes) => bytes,
            Err(_) => return -2,
        };
        let scan_bytes = match checked_bytes(scan_len, 8) {
            Ok(bytes) => bytes,
            Err(_) => return -2,
        };
        let segments_d = match GpuBuffer::alloc(segments_bytes) {
            Ok(b) => b,
            Err(_) => return -1,
        };
        let plas_rate_d = match GpuBuffer::alloc(scratch_bytes) {
            Ok(b) => b,
            Err(_) => return -1,
        };
        let plas_fraction_d = match GpuBuffer::alloc(scratch_bytes) {
            Ok(b) => b,
            Err(_) => return -1,
        };
        let plas_phase_d = match GpuBuffer::alloc(scratch_bytes) {
            Ok(b) => b,
            Err(_) => return -1,
        };
        let plas_current_d = match GpuBuffer::alloc(scratch_bytes) {
            Ok(b) => b,
            Err(_) => return -1,
        };
        let plas_scan_sums_d = match GpuBuffer::alloc(scan_bytes) {
            Ok(b) => b,
            Err(_) => return -1,
        };
        if segments_d.copy_to_device(segments).is_err() {
            return -1;
        }
        self.plasma_segments_d = segments_d;
        self.plas_rate_d = plas_rate_d;
        self.plas_fraction_d = plas_fraction_d;
        self.plas_phase_d = plas_phase_d;
        self.plas_current_d = plas_current_d;
        self.plas_scan_sums_d = plas_scan_sums_d;
        self.plasma_num_segments = segments.len();
        self.plasma_rate_kind = PlasmaRateKind::Ppt;
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
        ion_ptr: *const crate::ionization::AdkIonizationRate,
        ionpot: c_double,
        e_ratio: c_double,
        preionfrac: c_double,
        dt: c_double,
        density: c_double,
    ) -> i32 {
        if self.n_time == 0 || ion_ptr.is_null() {
            return -2;
        }
        if self.is_radial && (!self.is_real || self.n_r == 0) {
            return -2;
        }
        let ion = unsafe { &*ion_ptr };
        // The rate function's documented non-finite *field* handling belongs
        // on the device. Parameter non-finites instead mean setup is invalid:
        // accepting them would poison every later resident step.
        if !ion.occupancy.is_finite()
            || !ion.omega_p.is_finite()
            || !ion.cn_sq.is_finite()
            || !ion.nstar.is_finite()
            || !ion.omega_t_prefac.is_finite()
            || !ion.thr.is_finite()
            || ion.thr <= 0.0
            || !ion.avfac.is_finite()
            || ion.omega_t_prefac == 0.0
            || !ionpot.is_finite()
            || !e_ratio.is_finite()
            || !preionfrac.is_finite()
            || !dt.is_finite()
            || !density.is_finite()
        {
            return -2;
        }

        // Mode-averaged setup allocates these resident arrays in
        // `set_mode_avg_params`.  A radial setup has no mode-averaged
        // allocation to reuse, so stage the column-segmented scratch here
        // before changing any live plasma state.  This keeps invalid ADK
        // setup and allocation failures transactional, just like the PPT
        // setter above, and lets the same scan/finalizer pipeline serve both
        // rate kernels.
        let radial_buffers = if self.is_radial {
            let n_series = self.n_r;
            let scratch_len = match self.n_time_over.checked_mul(n_series) {
                Some(len) if len > 0 => len,
                _ => return -2,
            };
            let n_blocks = self.n_time_over.div_ceil(256);
            let scan_len = match n_blocks.checked_mul(n_series) {
                Some(len) if len > 0 => len,
                _ => return -2,
            };
            let scratch_bytes = match checked_bytes(scratch_len, 8) {
                Ok(bytes) => bytes,
                Err(_) => return -2,
            };
            let scan_bytes = match checked_bytes(scan_len, 8) {
                Ok(bytes) => bytes,
                Err(_) => return -2,
            };
            let plas_rate_d = match GpuBuffer::alloc(scratch_bytes) {
                Ok(b) => b,
                Err(_) => return -1,
            };
            let plas_fraction_d = match GpuBuffer::alloc(scratch_bytes) {
                Ok(b) => b,
                Err(_) => return -1,
            };
            let plas_phase_d = match GpuBuffer::alloc(scratch_bytes) {
                Ok(b) => b,
                Err(_) => return -1,
            };
            let plas_current_d = match GpuBuffer::alloc(scratch_bytes) {
                Ok(b) => b,
                Err(_) => return -1,
            };
            let plas_scan_sums_d = match GpuBuffer::alloc(scan_bytes) {
                Ok(b) => b,
                Err(_) => return -1,
            };
            Some((
                plas_rate_d,
                plas_fraction_d,
                plas_phase_d,
                plas_current_d,
                plas_scan_sums_d,
            ))
        } else {
            None
        };

        if let Some((
            plas_rate_d,
            plas_fraction_d,
            plas_phase_d,
            plas_current_d,
            plas_scan_sums_d,
        )) = radial_buffers
        {
            self.plas_rate_d = plas_rate_d;
            self.plas_fraction_d = plas_fraction_d;
            self.plas_phase_d = plas_phase_d;
            self.plas_current_d = plas_current_d;
            self.plas_scan_sums_d = plas_scan_sums_d;
            self.plasma_num_segments = 0;
        }
        self.plasma_rate_kind = PlasmaRateKind::Adk;
        self.plasma_adk_occupancy = ion.occupancy;
        self.plasma_adk_omega_p = ion.omega_p;
        self.plasma_adk_cn_sq = ion.cn_sq;
        self.plasma_adk_nstar = ion.nstar;
        self.plasma_adk_omega_t_prefac = ion.omega_t_prefac;
        self.plasma_adk_thr = ion.thr;
        self.plasma_adk_avfac = ion.avfac;
        self.plasma_ionpot = ionpot;
        self.plasma_e_ratio = e_ratio;
        self.plasma_preionfrac = preionfrac;
        self.plasma_dt = dt;
        self.plasma_density = density;
        self.has_plasma = true;
        0
    }

    unsafe fn set_radial_params(
        &mut self,
        n_time: size_t,
        n_time_over: size_t,
        n_r: size_t,
        t_matrix: *const c_double,
        scale_fwd: c_double,
        scale_inv: c_double,
        towin: *const c_double,
        kerr_fac: c_double,
        m_re: *const c_double,
        m_im: *const c_double,
    ) -> i32 {
        unsafe {
            match self.stage_radial_setup(
                n_time,
                n_time_over,
                n_r,
                t_matrix,
                scale_fwd,
                scale_inv,
                towin,
                kerr_fac,
                m_re,
                m_im,
            ) {
                Ok(staged) => {
                    self.commit_radial_setup(staged);
                    0
                }
                Err(_) => -1,
            }
        }
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
        omega: *const c_double,
        gamma: *const c_double,
        coupling: *const c_double,
        n_osc: size_t,
        dt: c_double,
        density: c_double,
        thg: c_int,
    ) -> i32 {
        if self.n_time_over == 0
            || omega.is_null()
            || gamma.is_null()
            || coupling.is_null()
            || n_osc == 0
            || n_osc > CUDA_RAMAN_MAX_OSCILLATORS
            || !dt.is_finite()
            || dt == 0.0
            || !density.is_finite()
        {
            return -2;
        }
        let omegas = unsafe { std::slice::from_raw_parts(omega, n_osc) };
        let gammas = unsafe { std::slice::from_raw_parts(gamma, n_osc) };
        let couplings = unsafe { std::slice::from_raw_parts(coupling, n_osc) };
        let mut coeffs = Vec::with_capacity(n_osc);
        for i in 0..n_osc {
            if !omegas[i].is_finite()
                || !gammas[i].is_finite()
                || !couplings[i].is_finite()
                || omegas[i] == 0.0
            {
                return -2;
            }
            let osc = crate::raman::RamanOscillator {
                omega: omegas[i],
                gamma: gammas[i],
                coupling: couplings[i],
            };
            coeffs.push(PrecomputedStepCoeffs::compute(&osc, dt));
        }

        let coeff_bytes =
            match checked_bytes(coeffs.len(), std::mem::size_of::<PrecomputedStepCoeffs>()) {
                Ok(bytes) => bytes,
                Err(_) => return -1,
            };
        let time_bytes = match checked_bytes(self.n_time_over, std::mem::size_of::<f64>()) {
            Ok(bytes) => bytes,
            Err(_) => return -1,
        };
        let complex_time_bytes =
            match checked_bytes(self.n_time_over, std::mem::size_of::<Complex<f64>>()) {
                Ok(bytes) => bytes,
                Err(_) => return -1,
            };
        let coeffs_d = match GpuBuffer::alloc(coeff_bytes) {
            Ok(b) => b,
            Err(_) => return -1,
        };
        let intensity_d = match GpuBuffer::alloc(time_bytes) {
            Ok(b) => b,
            Err(_) => return -1,
        };
        let p_d = match GpuBuffer::alloc(time_bytes) {
            Ok(b) => b,
            Err(_) => return -1,
        };
        let hilbert_a_d = match GpuBuffer::alloc(complex_time_bytes) {
            Ok(b) => b,
            Err(_) => return -1,
        };
        let hilbert_b_d = match GpuBuffer::alloc(complex_time_bytes) {
            Ok(b) => b,
            Err(_) => return -1,
        };
        if coeffs_d.copy_to_device(&coeffs).is_err() {
            return -1;
        }

        let mut hilbert_fft = 0;
        if self.is_real && thg == 0 {
            let cufft = match get_cufft_api() {
                Ok(api) => api,
                Err(_) => return -1,
            };
            let n_i32 = match i32::try_from(self.n_time_over) {
                Ok(n) => n,
                Err(_) => return -1,
            };
            let rc = unsafe { (cufft.cufftPlan1d)(&mut hilbert_fft, n_i32, CUFFT_Z2Z, 1) };
            if rc != 0 {
                return -1;
            }
        }

        if let Ok(cufft) = get_cufft_api() {
            unsafe {
                if self.raman_hilbert_fft != 0 {
                    (cufft.cufftDestroy)(self.raman_hilbert_fft);
                    self.raman_hilbert_fft = 0;
                }
                if self.raman_fft_r2c != 0 {
                    (cufft.cufftDestroy)(self.raman_fft_r2c);
                    self.raman_fft_r2c = 0;
                }
                if self.raman_fft_c2r != 0 {
                    (cufft.cufftDestroy)(self.raman_fft_c2r);
                    self.raman_fft_c2r = 0;
                }
            }
        }
        self.raman_coeffs_d = coeffs_d;
        self.raman_intensity_d = intensity_d;
        self.raman_p_d = p_d;
        self.raman_hilbert_a_d = hilbert_a_d;
        self.raman_hilbert_b_d = hilbert_b_d;
        self.raman_hilbert_fft = hilbert_fft;
        self.raman_num_osc = n_osc;
        self.raman_density = density;
        self.raman_thg = thg != 0;
        self.has_raman_fft = false;
        self.has_raman = true;
        0
    }
    unsafe fn set_raman_fft_params(
        &mut self,
        omega: *const c_double,
        amp: *const c_double,
        gauss_w: *const c_double,
        lorentz_w: *const c_double,
        n_osc: size_t,
        scale: c_double,
        dt: c_double,
        n_time: size_t,
        density: c_double,
    ) -> i32 {
        if self.is_real || self.n_time_over == 0 || n_time != self.n_time_over {
            return -2;
        }
        if omega.is_null()
            || amp.is_null()
            || gauss_w.is_null()
            || lorentz_w.is_null()
            || n_osc == 0
        {
            return -1;
        }
        match unsafe {
            self.stage_raman_fft_setup(
                omega, amp, gauss_w, lorentz_w, n_osc, scale, dt, n_time, density,
            )
        } {
            Ok(staged) => {
                self.commit_raman_fft_setup(staged);
                0
            }
            Err(error) => {
                eprintln!("Amalthea GPU error: EnvGrid Raman FFT setup failed: {error}");
                -1
            }
        }
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

                    // `compute_rhs_mode_avg` below propagates `ystage_d` in
                    // place. When local extrapolation is disabled, preserve
                    // the final interaction-picture stage before that
                    // transform so it can become the trial state. `yerr_d`
                    // is still dead here and is overwritten by the error
                    // kernel immediately after the stage loop.
                    if _locextrap == 0 && ii == 5 {
                        self.yerr_d.copy_from_device(&self.ystage_d)?;
                    }

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

                    // Dispatch the complete resident RHS for the configured
                    // geometry.  Both branches keep the stage on device;
                    // radial additionally applies the resident QDHT matrix.
                    if self.is_radial {
                        self.compute_rhs_radial(ii + 1)?;
                    } else {
                        self.compute_rhs_mode_avg(ii + 1)?;
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

                if _locextrap == 0 {
                    self.ystage_d.copy_from_device(&self.yerr_d)?;
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

                // Form the genuine fifth-order trial state *before* the
                // acceptance decision, exactly like CpuNativeSim::step.
                // `ystage_d` is dead after the seven RK stages, so it doubles
                // as a transactional trial buffer: rejection leaves
                // `field_d` untouched; acceptance propagates and swaps this
                // buffer into the resident field.
                if _locextrap != 0 {
                    self.ystage_d.copy_from_device(&self.field_d)?;
                    let mut b5 = crate::native::DP_B5;
                    let mut trial_args: [*mut libc::c_void; 18] = [
                        &mut self.ystage_d.dptr as *mut _ as *mut _,
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
                        &mut trial_args,
                        "rk45_accumulate_stage(trial)",
                    )?;
                }
                // With local extrapolation disabled, the stage loop already
                // left the final internal RK stage in `ystage_d`. Retain it
                // as the transactional trial instead of replacing it with
                // the old `field_d`; this mirrors Julia and the CPU backend.

                // Emit the three squared-magnitude arrays required by
                // native.rs::weaknorm_c64. The former kernel used an
                // element-wise tolerance (a different norm entirely) and
                // also received `field_d` for both old and trial states.
                let mut weaknorm_elem_args: [*mut libc::c_void; 7] = [
                    &mut self.yerr_d.dptr as *mut _ as *mut _,
                    &mut self.field_d.dptr as *mut _ as *mut _,
                    &mut self.ystage_d.dptr as *mut _ as *mut _,
                    &mut self.out_sq_d.dptr as *mut _ as *mut _,
                    &mut self.y0_sq_d.dptr as *mut _ as *mut _,
                    &mut self.y1_sq_d.dptr as *mut _ as *mut _,
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

                let syerr = reduce_sum(
                    driver,
                    ctx.weaknorm_reduce_fn,
                    self.out_sq_d.dptr,
                    self.reduced_d.dptr,
                    self.n,
                    block_size,
                    "weaknorm_reduce(yerr)",
                )?;
                let sy = reduce_sum(
                    driver,
                    ctx.weaknorm_reduce_fn,
                    self.y0_sq_d.dptr,
                    self.reduced_d.dptr,
                    self.n,
                    block_size,
                    "weaknorm_reduce(y0)",
                )?;
                let syn = reduce_sum(
                    driver,
                    ctx.weaknorm_reduce_fn,
                    self.y1_sq_d.dptr,
                    self.reduced_d.dptr,
                    self.n,
                    block_size,
                    "weaknorm_reduce(y1)",
                )?;
                let errwt = f64::max(f64::max(sy.sqrt(), syn.sqrt()), _atol);
                let err = syerr.sqrt() / _rtol / errwt;
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

                    // The accepted trial is still in the interaction
                    // picture. Propagate it to `tn_new`, then make it the
                    // resident field with an O(1) ownership swap.
                    let mut dt_fin = tn_new - t;
                    let mut apply_args_fin: [*mut libc::c_void; 4] = [
                        &mut self.ystage_d.dptr as *mut _ as *mut _,
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
                        "apply_prop(trial, final)",
                    )?;
                    std::mem::swap(&mut self.field_d, &mut self.ystage_d);
                    if self.get_field(yn as *mut c_double, self.n) != 0 {
                        return Err("get_field failed after accepted CUDA step".to_string());
                    }
                } else {
                    tn_new = _t_new;
                    if self.get_field(yn as *mut c_double, self.n) != 0 {
                        return Err("get_field failed after rejected CUDA step".to_string());
                    }
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
            if self.fft_c2c != 0 {
                unsafe {
                    (cufft.cufftDestroy)(self.fft_c2c);
                }
            }
            if self.radial_fft_r2c != 0 {
                unsafe {
                    (cufft.cufftDestroy)(self.radial_fft_r2c);
                }
            }
            if self.radial_fft_c2r != 0 {
                unsafe {
                    (cufft.cufftDestroy)(self.radial_fft_c2r);
                }
            }
            if self.radial_fft_c2c != 0 {
                unsafe {
                    (cufft.cufftDestroy)(self.radial_fft_c2c);
                }
            }
            if self.raman_hilbert_fft != 0 {
                unsafe {
                    (cufft.cufftDestroy)(self.raman_hilbert_fft);
                }
            }
            if self.raman_fft_r2c != 0 {
                unsafe {
                    (cufft.cufftDestroy)(self.raman_fft_r2c);
                }
            }
            if self.raman_fft_c2r != 0 {
                unsafe {
                    (cufft.cufftDestroy)(self.raman_fft_c2r);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adk_rate_for_test(fields: &[f64], ion: &crate::ionization::AdkIonizationRate) -> Vec<f64> {
        let ctx = crate::cuda::activate_context().expect("activate CUDA context");
        let driver = get_driver_api().expect("CUDA driver API");
        let n = fields.len();
        let mut fields_d = GpuBuffer::alloc(n * std::mem::size_of::<f64>()).unwrap();
        let mut rates_d = GpuBuffer::alloc(n * std::mem::size_of::<f64>()).unwrap();
        fields_d.copy_to_device(fields).unwrap();
        let mut occupancy = ion.occupancy;
        let mut omega_p = ion.omega_p;
        let mut cn_sq = ion.cn_sq;
        let mut nstar = ion.nstar;
        let mut omega_t_prefac = ion.omega_t_prefac;
        let mut thr = ion.thr;
        let mut avfac = ion.avfac;
        let mut n_i = i32::try_from(n).unwrap();
        let mut args: [*mut libc::c_void; 10] = [
            &mut fields_d.dptr as *mut _ as *mut _,
            &mut rates_d.dptr as *mut _ as *mut _,
            &mut occupancy as *mut _ as *mut _,
            &mut omega_p as *mut _ as *mut _,
            &mut cn_sq as *mut _ as *mut _,
            &mut nstar as *mut _ as *mut _,
            &mut omega_t_prefac as *mut _ as *mut _,
            &mut thr as *mut _ as *mut _,
            &mut avfac as *mut _ as *mut _,
            &mut n_i as *mut _ as *mut _,
        ];
        unsafe {
            launch_checked(
                driver,
                ctx.adk_fn,
                (n as u32).div_ceil(256),
                256,
                0,
                &mut args,
                "test_adk_ionization",
            )
            .unwrap();
        }
        let mut result = vec![0.0; n];
        rates_d.copy_to_host(&mut result).unwrap();
        result
    }

    fn cuda_or_skip(test_name: &str) -> bool {
        if let Err(e) = crate::cuda::init_gpu_context() {
            assert!(
                !crate::cuda::tests_require_cuda(),
                "{test_name}: CUDA is required but unavailable: {e}"
            );
            eprintln!("Skipping {test_name}: {e}");
            return false;
        }
        true
    }

    #[test]
    fn plasma_scan_matches_sequential_across_partial_blocks() {
        if !cuda_or_skip("CUDA plasma scan test") {
            return;
        }

        // 513 spans two full 256-sample blocks plus one sample, so this
        // catches both missing block offsets and partial-final-block errors.
        let n = 513usize;
        let dt = 0.125;
        let linop = [Complex::new(0.0, 0.0)];
        let mut sim = CudaNativeSim::new(1, &linop).expect("CudaNativeSim::new");
        sim.n_time_over = n;
        sim.plasma_dt = dt;
        sim.plas_scan_sums_d = GpuBuffer::alloc(n.div_ceil(256) * 8).unwrap();

        let input: Vec<f64> = (0..n)
            .map(|i| ((i * 37 + 11) % 101) as f64 / 100.0)
            .collect();
        let input_d = GpuBuffer::alloc(n * 8).unwrap();
        let output_d = GpuBuffer::alloc(n * 8).unwrap();
        input_d.copy_to_device(&input).unwrap();

        unsafe {
            sim.plasma_scan(input_d.dptr, output_d.dptr, "test_plasma_scan")
                .unwrap();
        }

        let mut got = vec![0.0; n];
        output_d.copy_to_host(&mut got).unwrap();
        let mut block_sums = vec![0.0; n.div_ceil(256)];
        sim.plas_scan_sums_d.copy_to_host(&mut block_sums).unwrap();
        for (i, value) in got.iter_mut().enumerate() {
            if i / 256 > 0 {
                *value += block_sums[i / 256 - 1];
            }
        }

        let mut expected = vec![0.0; n];
        for i in 1..n {
            expected[i] = expected[i - 1] + 0.5 * (input[i - 1] + input[i]) * dt;
        }
        let max_abs = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_abs < 1e-12, "max_abs={max_abs:e}");
    }

    #[test]
    fn adk_ionization_kernel_matches_cpu_boundaries_signs_and_cycle_average() {
        if !cuda_or_skip("CUDA ADK ionization kernel test") {
            return;
        }
        let fields = [
            0.0,
            0.5,
            -0.5,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0_f64.next_down(),
            1.0,
            1.0_f64.next_up(),
            -1.0,
            1.75,
            -1.75,
        ];
        let mut unaveraged_at_peak = 0.0;
        let mut averaged_at_peak = 0.0;
        for avfac in [1.0, 1.7] {
            let ion = crate::ionization::AdkIonizationRate {
                occupancy: 2.0,
                omega_p: 1.3,
                cn_sq: 0.8,
                nstar: 1.2,
                omega_t_prefac: 0.9,
                thr: 1.0,
                avfac,
            };
            let got = adk_rate_for_test(&fields, &ion);
            for (&field, &gpu) in fields.iter().zip(&got) {
                let cpu = ion.rate(field).unwrap();
                if !field.is_finite() || field.abs() < ion.thr {
                    assert_eq!(gpu, 0.0, "field={field:?}, avfac={avfac}");
                } else {
                    let scale = cpu.abs().max(1.0);
                    assert!(
                        (gpu - cpu).abs() / scale < 1e-13,
                        "field={field}, avfac={avfac}: gpu={gpu:e}, cpu={cpu:e}"
                    );
                }
            }
            // ±E must produce the same rate, and the exact threshold is
            // active while its predecessor is not.
            assert_eq!(got[6], 0.0);
            assert!((got[7] - got[9]).abs() < 1e-13);
            assert!((got[10] - got[11]).abs() < 1e-13);
            if avfac == 1.0 {
                unaveraged_at_peak = got[10];
            } else {
                averaged_at_peak = got[10];
            }
        }
        assert_ne!(averaged_at_peak, unaveraged_at_peak);
    }

    #[test]
    fn field_transfers_reject_invalid_ffi_arguments() {
        if !cuda_or_skip("CUDA field-transfer contract test") {
            return;
        }

        let n = 4usize;
        let linop = vec![Complex::new(0.0, 0.0); n];
        let mut sim = CudaNativeSim::new(n, &linop).expect("CudaNativeSim::new");
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(i as f64, -(i as f64)))
            .collect();
        let mut output = vec![Complex::new(0.0, 0.0); n];

        unsafe {
            assert_eq!(sim.set_field(std::ptr::null(), n), -1);
            assert_eq!(sim.set_field(input.as_ptr() as *const c_double, n + 1), -1);
            assert_eq!(sim.resync_field(std::ptr::null(), n), -1);
            assert_eq!(
                sim.resync_field(input.as_ptr() as *const c_double, n + 1),
                -1
            );
            assert_eq!(sim.get_field(std::ptr::null_mut(), n), -1);
            assert_eq!(
                sim.get_field(output.as_mut_ptr() as *mut c_double, n + 1),
                -1
            );
            assert_eq!(sim.get_ks_stage(0, std::ptr::null_mut(), n), -1);
            assert_eq!(
                sim.get_ks_stage(7, output.as_mut_ptr() as *mut c_double, n),
                -1
            );
            assert_eq!(
                sim.get_ks_stage(0, output.as_mut_ptr() as *mut c_double, n + 1),
                -1
            );

            assert_eq!(sim.resync_field(input.as_ptr() as *const c_double, n), 0);
            assert_eq!(sim.get_field(output.as_mut_ptr() as *mut c_double, n), 0);
        }
        assert_eq!(output, input);
    }

    #[test]
    fn mode_avg_setup_failures_preserve_the_active_cuda_configuration() {
        if !cuda_or_skip("CUDA mode-averaged setup transaction test") {
            return;
        }
        let _serial = MODE_AVG_SETUP_TEST_LOCK.lock().unwrap();
        let n = 4usize; // RealGrid Nt=6 -> Nt/2+1 spectral bins.
        let mut sim =
            CudaNativeSim::new(n, &vec![Complex::new(0.0, 0.0); n]).expect("CudaNativeSim::new");
        let towin = [1.0; 8];
        let owin = vec![1.0; n];
        let sidx = vec![1u8; n];
        let pre = vec![0.0; n];
        let beta = vec![1.0; n];
        unsafe {
            assert_eq!(
                sim.set_mode_avg_params(
                    6,
                    8,
                    towin.as_ptr(),
                    owin.as_ptr(),
                    sidx.as_ptr(),
                    pre.as_ptr(),
                    pre.as_ptr(),
                    beta.as_ptr(),
                    0.0,
                    1.0,
                    1.0,
                ),
                0
            );
        }
        let field: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(i as f64 + 1.0, -0.25 * i as f64))
            .collect();
        unsafe {
            assert_eq!(sim.set_field(field.as_ptr() as *const c_double, n), 0);
        }

        for point in [
            MODE_AVG_FAIL_ALLOC,
            MODE_AVG_FAIL_COPY,
            MODE_AVG_FAIL_SECOND_PLAN,
        ] {
            MODE_AVG_SETUP_FAIL_POINT.store(point, Ordering::SeqCst);
            let rc = unsafe {
                sim.set_mode_avg_params(
                    6,
                    8,
                    towin.as_ptr(),
                    owin.as_ptr(),
                    sidx.as_ptr(),
                    pre.as_ptr(),
                    pre.as_ptr(),
                    beta.as_ptr(),
                    0.0,
                    1.0,
                    1.0,
                )
            };
            MODE_AVG_SETUP_FAIL_POINT.store(0, Ordering::SeqCst);
            assert_ne!(rc, 0, "fault point {point} must fail setup");

            // The previous plans/buffers remain live: reseeding the field
            // recomputes its RHS through the old configuration, then the
            // resident field round-trips unchanged.
            unsafe {
                assert_eq!(sim.set_field(field.as_ptr() as *const c_double, n), 0);
            }
            let mut got = vec![Complex::new(0.0, 0.0); n];
            unsafe {
                assert_eq!(sim.get_field(got.as_mut_ptr() as *mut c_double, n), 0);
            }
            assert_eq!(
                got, field,
                "fault point {point} damaged active field/config"
            );
        }
    }

    #[test]
    fn raman_fft_setup_failures_preserve_the_active_cuda_configuration() {
        if !cuda_or_skip("CUDA Raman FFT setup transaction test") {
            return;
        }
        let _serial = MODE_AVG_SETUP_TEST_LOCK.lock().unwrap();
        let n = 4usize;
        let mut sim =
            CudaNativeSim::new(n, &vec![Complex::new(0.0, 0.0); n]).expect("CudaNativeSim::new");
        let towin = [1.0; 8];
        let owin = vec![1.0; n];
        let sidx = vec![1u8; n];
        let pre = vec![0.0; n];
        let beta = vec![1.0; n];
        unsafe {
            assert_eq!(
                sim.set_fftw_plans(std::ptr::null(), 4, 8, 0, 0, std::ptr::null()),
                0
            );
            assert_eq!(
                sim.set_mode_avg_params(
                    4,
                    8,
                    towin.as_ptr(),
                    owin.as_ptr(),
                    sidx.as_ptr(),
                    pre.as_ptr(),
                    pre.as_ptr(),
                    beta.as_ptr(),
                    0.0,
                    1.0,
                    1.0,
                ),
                0
            );
        }
        let field: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(i as f64 + 1.0, -0.25 * i as f64))
            .collect();
        let omega = [1.0];
        let amp = [1.0];
        let gauss = [0.0];
        let lorentz = [0.0];
        unsafe {
            assert_eq!(
                sim.set_raman_fft_params(
                    omega.as_ptr(),
                    amp.as_ptr(),
                    gauss.as_ptr(),
                    lorentz.as_ptr(),
                    1,
                    1.0,
                    1e-3,
                    8,
                    1.0,
                ),
                0
            );
            assert_eq!(sim.set_field(field.as_ptr() as *const c_double, n), 0);
        }

        for point in [
            RAMAN_FFT_FAIL_ALLOC,
            RAMAN_FFT_FAIL_COPY,
            RAMAN_FFT_FAIL_SECOND_PLAN,
        ] {
            RAMAN_FFT_SETUP_FAIL_POINT.store(point, Ordering::SeqCst);
            let rc = unsafe {
                sim.set_raman_fft_params(
                    omega.as_ptr(),
                    amp.as_ptr(),
                    gauss.as_ptr(),
                    lorentz.as_ptr(),
                    1,
                    1.0,
                    1e-3,
                    8,
                    1.0,
                )
            };
            RAMAN_FFT_SETUP_FAIL_POINT.store(0, Ordering::SeqCst);
            assert_ne!(rc, 0, "fault point {point} must fail setup");

            // The old response spectrum and scratch buffers remain active.
            unsafe {
                assert_eq!(sim.set_field(field.as_ptr() as *const c_double, n), 0);
            }
            let mut got = vec![Complex::new(0.0, 0.0); n];
            unsafe {
                assert_eq!(sim.get_field(got.as_mut_ptr() as *mut c_double, n), 0);
            }
            assert_eq!(
                got, field,
                "fault point {point} damaged active Raman configuration"
            );
        }
    }
}
