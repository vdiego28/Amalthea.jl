/// Represents a single damped Raman oscillator transition (Single Damped Oscillator - SDO)
#[derive(Debug, Clone, Copy)]
pub struct RamanOscillator {
    pub omega: f64,   // Oscillator transition frequency Ω_i
    pub gamma: f64,   // Damping rate Γ_i
    pub coupling: f64, // Coupling coefficient K_i
}

/// Precomputed matrix exponential coefficients for a single oscillator at a fixed time step Δt
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PrecomputedStepCoeffs {
    // 2x2 matrix exponential A = exp(M * dt)
    pub a11: f64,
    pub a12: f64,
    pub a21: f64,
    pub a22: f64,
    
    // Integral vector B0
    pub b0_1: f64,
    pub b0_2: f64,
    
    // Integral vector B1
    pub b1_1: f64,
    pub b1_2: f64,
}

impl PrecomputedStepCoeffs {
    /// Computes the exact exponential integrator coefficients for a given time step
    pub fn compute(osc: &RamanOscillator, dt: f64) -> Self {
        let omega = osc.omega;
        let gamma = osc.gamma;
        let coupling = osc.coupling;
        
        let omega_d_sq = omega * omega + gamma * gamma; // ω_d^2
        let exp_gamma = (-gamma * dt).exp();
        
        // exp(M * dt) elements
        let a11 = exp_gamma * (omega * dt).cos() + (gamma / omega) * exp_gamma * (omega * dt).sin();
        let a12 = (1.0 / omega) * exp_gamma * (omega * dt).sin();
        let a21 = -(omega_d_sq / omega) * exp_gamma * (omega * dt).sin();
        let a22 = exp_gamma * (omega * dt).cos() - (gamma / omega) * exp_gamma * (omega * dt).sin();
        
        // M inverse matrix elements
        // M = [[0, 1], [-ω_d^2, -2Γ]]
        // M^-1 = [[-2Γ/ω_d^2, -1/ω_d^2], [1, 0]]
        let m_inv11 = -2.0 * gamma / omega_d_sq;
        let m_inv12 = -1.0 / omega_d_sq;
        let m_inv21 = 1.0;
        let m_inv22 = 0.0;
        
        // M^-2 elements
        let m_inv_sq11 = (4.0 * gamma * gamma) / (omega_d_sq * omega_d_sq) - 1.0 / omega_d_sq;
        let m_inv_sq12 = 2.0 * gamma / (omega_d_sq * omega_d_sq);
        let m_inv_sq21 = -2.0 * gamma / omega_d_sq;
        let m_inv_sq22 = -1.0 / omega_d_sq;
        
        // Driving vector coupling constant
        let f_re2 = coupling * omega;
        
        // Recalculating matrix multiplications explicitly:
        // (A - I) * [0, f_re2] = [a12 * f_re2, (a22 - 1.0) * f_re2]
        let temp1 = a12 * f_re2;
        let temp2 = (a22 - 1.0) * f_re2;
        
        let c0_1 = m_inv11 * temp1 + m_inv12 * temp2;
        let c0_2 = m_inv21 * temp1 + m_inv22 * temp2;
        
        // C1 = 1/dt * M^-2 * (A - I - M*dt) * [0, f_re2]
        // M * dt = [[0, dt], [-ω_d^2 * dt, -2Γ * dt]]
        // (A - I - M*dt) * [0, f_re2] = [temp1_prime, temp2_prime]
        let temp1_prime = (a12 - dt) * f_re2;
        let temp2_prime = (a22 - 1.0 + 2.0 * gamma * dt) * f_re2;
        
        let c1_1 = (m_inv_sq11 * temp1_prime + m_inv_sq12 * temp2_prime) / dt;
        let c1_2 = (m_inv_sq21 * temp1_prime + m_inv_sq22 * temp2_prime) / dt;
        
        let b0_1 = c0_1 - c1_1;
        let b0_2 = c0_2 - c1_2;
        let b1_1 = c1_1;
        let b1_2 = c1_2;
        
        Self {
            a11, a12, a21, a22,
            b0_1, b0_2,
            b1_1, b1_2,
        }
    }
}

/// Multi-oscillator Raman solver
#[derive(Debug, Clone)]
pub struct TimeDomainRamanSolver {
    pub oscillators: Vec<RamanOscillator>,
    pub step_coeffs: Vec<PrecomputedStepCoeffs>,
    pub states: Vec<(f64, f64)>, // Q_i, dQ_i for each oscillator
    pub dt: f64,
}

impl TimeDomainRamanSolver {
    pub fn new(oscillators: Vec<RamanOscillator>, dt: f64) -> Self {
        let step_coeffs: Vec<PrecomputedStepCoeffs> = oscillators
            .iter()
            .map(|osc| PrecomputedStepCoeffs::compute(osc, dt))
            .collect();
            
        let states = vec![(0.0, 0.0); oscillators.len()];
        
        Self { oscillators, step_coeffs, states, dt }
    }

    /// Resets the internal state of all oscillators back to equilibrium (0.0)
    pub fn reset_state(&mut self) {
        for state in self.states.iter_mut() {
            *state = (0.0, 0.0);
        }
    }

    /// Evaluates the total Raman response vector for a time-domain intensity array I
    /// Updates states in-place.
    pub fn solve(&mut self, intensity: &[f64], raman_polarization: &mut [f64]) {
        if let Some(ctx) = crate::cuda::get_gpu_context() {
            if self.solve_gpu(ctx, intensity, raman_polarization).is_ok() {
                return;
            }
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            return unsafe { self.solve_avx2(intensity, raman_polarization) };
        }
        self.solve_scalar(intensity, raman_polarization);
    }

    fn solve_gpu(&self, ctx: &crate::cuda::GpuContext, intensity: &[f64], raman_polarization: &mut [f64]) -> Result<(), String> {
        let n_t = intensity.len();
        let num_oscillators = self.oscillators.len();
        
        let d_intensity = crate::cuda::GpuBuffer::alloc(n_t * std::mem::size_of::<f64>())?;
        let d_polarization = crate::cuda::GpuBuffer::alloc(n_t * std::mem::size_of::<f64>())?;
        let d_coeffs = crate::cuda::GpuBuffer::alloc(num_oscillators * std::mem::size_of::<PrecomputedStepCoeffs>())?;
        
        d_intensity.copy_to_device(intensity)?;
        d_coeffs.copy_to_device(&self.step_coeffs)?;
        
        let mut d_intensity_ptr = d_intensity.dptr;
        let mut d_polarization_ptr = d_polarization.dptr;
        let mut d_coeffs_ptr = d_coeffs.dptr;
        let mut num_osc_val = num_oscillators as libc::c_int;
        let mut n_t_val = n_t as libc::c_int;
        let mut n_series_val = 1 as libc::c_int;
        
        let mut args: [*mut libc::c_void; 6] = [
            &mut d_intensity_ptr as *mut _ as *mut libc::c_void,
            &mut d_polarization_ptr as *mut _ as *mut libc::c_void,
            &mut d_coeffs_ptr as *mut _ as *mut libc::c_void,
            &mut num_osc_val as *mut _ as *mut libc::c_void,
            &mut n_t_val as *mut _ as *mut libc::c_void,
            &mut n_series_val as *mut _ as *mut libc::c_void,
        ];
        
        crate::cuda::activate_context()?;
        let driver = crate::cuda::get_driver_api()?;
        
        unsafe {
            let res = (driver.cuLaunchKernel)(
                ctx.raman_fn,
                1, 1, 1,
                1, 1, 1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            
            if res != 0 {
                return Err(format!("cuLaunchKernel for raman_ade_kernel failed: {}", res));
            }
        }
        
        d_polarization.copy_to_host(raman_polarization)?;
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn solve_avx2(&mut self, intensity: &[f64], raman_polarization: &mut [f64]) {
        self.solve_scalar(intensity, raman_polarization);
    }

    fn solve_scalar(&mut self, intensity: &[f64], raman_polarization: &mut [f64]) {
        let n_t = intensity.len();
        assert_eq!(raman_polarization.len(), n_t);
        
        self.reset_state();
        raman_polarization[0] = 0.0;
        
        // Outer loop: time steps
        for n in 0..(n_t - 1) {
            let i_n = intensity[n];
            let i_np1 = intensity[n + 1];
            
            let mut total_q = 0.0;
            
            // Inner loop: oscillators (usually 1 for simple gas, or ~10 for multi-oscillator glass/silica)
            // Contour indices are contiguous, ensuring L1 cache-hits and autovectorization.
            for i in 0..self.oscillators.len() {
                let coeffs = unsafe { self.step_coeffs.get_unchecked(i) };
                let (q, dq) = unsafe { *self.states.get_unchecked(i) };
                
                // Explicit matrix multiply-add
                let q_new = coeffs.a11 * q + coeffs.a12 * dq + coeffs.b0_1 * i_n + coeffs.b1_1 * i_np1;
                let dq_new = coeffs.a21 * q + coeffs.a22 * dq + coeffs.b0_2 * i_n + coeffs.b1_2 * i_np1;
                
                unsafe {
                    *self.states.get_unchecked_mut(i) = (q_new, dq_new);
                }
                
                total_q += q_new;
            }
            
            raman_polarization[n + 1] = total_q;
        }
    }
}
