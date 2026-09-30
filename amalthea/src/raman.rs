include!(concat!(env!("OUT_DIR"), "/cuda_raman_limits.rs"));

/// Represents a single damped Raman oscillator transition (Single Damped Oscillator - SDO)
#[derive(Debug, Clone, Copy)]
pub struct RamanOscillator {
    pub omega: f64,    // Oscillator transition frequency Ω_i
    pub gamma: f64,    // Damping rate Γ_i
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
    /// Computes the exponential update for intensity interpolated linearly over `dt`.
    /// Small-step forcing integrals use a series to avoid cancellation in `A-I-M*dt`.
    pub fn compute(osc: &RamanOscillator, dt: f64) -> Self {
        let omega = osc.omega;
        let gamma = osc.gamma;
        let coupling = osc.coupling;

        let omega_d_sq = omega * omega + gamma * gamma; // ω_d^2
        let exp_gamma = (-gamma * dt).exp();

        // exp(M * dt), including the finite limit at omega=0.
        let phase = omega * dt;
        let sinc = if phase == 0.0 {
            1.0
        } else {
            phase.sin() / phase
        };
        let a12 = dt * exp_gamma * sinc;
        let a11 = exp_gamma * phase.cos() + gamma * a12;
        let a21 = -omega_d_sq * a12;
        let a22 = exp_gamma * phase.cos() - gamma * a12;

        if (omega.abs() + gamma.abs()) * dt.abs() <= 0.5 {
            // With v_n=(M*dt)^n * b*dt/n!, integrate each power directly:
            // B0=sum v_n/(n+2), B1=sum v_n/((n+1)*(n+2)).
            // B0 multiplies the old intensity and B1 the new intensity.
            // In coordinates (q,dt*q'), ||M*dt||_inf <= 1.25 here;
            // 24 terms put the truncation tail well below f64 roundoff.
            let mut term_q = 0.0;
            let mut term_dq = coupling * omega * dt;
            let (mut b0_1, mut b0_2, mut b1_1, mut b1_2) = (0.0, 0.0, 0.0, 0.0);
            for n in 0..24 {
                let old_weight = 1.0 / (n + 2) as f64;
                let new_weight = old_weight / (n + 1) as f64;
                b0_1 += old_weight * term_q;
                b0_2 += old_weight * term_dq;
                b1_1 += new_weight * term_q;
                b1_2 += new_weight * term_dq;
                let next_q = term_dq * dt / (n + 1) as f64;
                let next_dq = -(omega_d_sq * term_q + 2.0 * gamma * term_dq) * dt / (n + 1) as f64;
                term_q = next_q;
                term_dq = next_dq;
            }
            return Self {
                a11,
                a12,
                a21,
                a22,
                b0_1,
                b0_2,
                b1_1,
                b1_2,
            };
        }

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
            a11,
            a12,
            a21,
            a22,
            b0_1,
            b0_2,
            b1_1,
            b1_2,
        }
    }
}

/// Multi-oscillator Raman solver
#[derive(Debug, Clone)]
struct StepCoeffsSoa {
    a11: Vec<f64>,
    a12: Vec<f64>,
    a21: Vec<f64>,
    a22: Vec<f64>,
    b0_1: Vec<f64>,
    b0_2: Vec<f64>,
    b1_1: Vec<f64>,
    b1_2: Vec<f64>,
}

impl StepCoeffsSoa {
    fn from_packed(packed: &[PrecomputedStepCoeffs]) -> Self {
        let mut out = Self {
            a11: Vec::with_capacity(packed.len()),
            a12: Vec::with_capacity(packed.len()),
            a21: Vec::with_capacity(packed.len()),
            a22: Vec::with_capacity(packed.len()),
            b0_1: Vec::with_capacity(packed.len()),
            b0_2: Vec::with_capacity(packed.len()),
            b1_1: Vec::with_capacity(packed.len()),
            b1_2: Vec::with_capacity(packed.len()),
        };
        for c in packed {
            out.a11.push(c.a11);
            out.a12.push(c.a12);
            out.a21.push(c.a21);
            out.a22.push(c.a22);
            out.b0_1.push(c.b0_1);
            out.b0_2.push(c.b0_2);
            out.b1_1.push(c.b1_1);
            out.b1_2.push(c.b1_2);
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct TimeDomainRamanSolver {
    pub oscillators: Vec<RamanOscillator>,
    /// Packed representation retained exclusively for the CUDA kernel ABI.
    pub gpu_step_coeffs: Vec<PrecomputedStepCoeffs>,
    step_coeffs: StepCoeffsSoa,
    states_q: Vec<f64>,
    states_dq: Vec<f64>,
    pub dt: f64,
}

impl TimeDomainRamanSolver {
    pub fn new(oscillators: Vec<RamanOscillator>, dt: f64) -> Self {
        let gpu_step_coeffs: Vec<PrecomputedStepCoeffs> = oscillators
            .iter()
            .map(|osc| PrecomputedStepCoeffs::compute(osc, dt))
            .collect();
        let step_coeffs = StepCoeffsSoa::from_packed(&gpu_step_coeffs);
        let states_q = vec![0.0; oscillators.len()];
        let states_dq = vec![0.0; oscillators.len()];

        Self {
            oscillators,
            gpu_step_coeffs,
            step_coeffs,
            states_q,
            states_dq,
            dt,
        }
    }

    /// Resets the internal state of all oscillators back to equilibrium (0.0)
    pub fn reset_state(&mut self) {
        self.states_q.fill(0.0);
        self.states_dq.fill(0.0);
    }

    /// Evaluates the total Raman response vector for a time-domain intensity array I
    /// Updates states in-place.
    pub fn solve(&mut self, intensity: &[f64], raman_polarization: &mut [f64]) {
        if let Some(ctx) = crate::cuda::get_gpu_context()
            && self.solve_gpu(ctx, intensity, raman_polarization).is_ok()
        {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            return unsafe { self.solve_avx2(intensity, raman_polarization) };
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { self.solve_neon(intensity, raman_polarization) };
        }
        self.solve_scalar(intensity, raman_polarization);
    }

    fn solve_gpu(
        &self,
        ctx: &crate::cuda::GpuContext,
        intensity: &[f64],
        raman_polarization: &mut [f64],
    ) -> Result<(), String> {
        let n_t = intensity.len();
        let num_oscillators = self.oscillators.len();
        if num_oscillators == 0 || num_oscillators > CUDA_RAMAN_MAX_OSCILLATORS {
            return Err(format!(
                "CUDA Raman oscillator count {} exceeds capacity {}",
                num_oscillators, CUDA_RAMAN_MAX_OSCILLATORS
            ));
        }

        let d_intensity = crate::cuda::GpuBuffer::alloc(std::mem::size_of_val(intensity))?;
        let d_polarization = crate::cuda::GpuBuffer::alloc(std::mem::size_of_val(intensity))?;
        let coeff_bytes = num_oscillators
            .checked_mul(std::mem::size_of::<PrecomputedStepCoeffs>())
            .ok_or_else(|| "CUDA Raman coefficient buffer size overflow".to_string())?;
        let d_coeffs = crate::cuda::GpuBuffer::alloc(coeff_bytes)?;

        d_intensity.copy_to_device(intensity)?;
        d_coeffs.copy_to_device(&self.gpu_step_coeffs)?;

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
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );

            if res != 0 {
                return Err(format!(
                    "cuLaunchKernel for raman_ade_kernel failed: {}",
                    res
                ));
            }
        }

        d_polarization.copy_to_host(raman_polarization)?;
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    // Every pointer is derived from equal-length SoA vectors and `lanes_end`
    // stops before the first partial four-lane group.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn solve_avx2(&mut self, intensity: &[f64], raman_polarization: &mut [f64]) {
        use std::arch::x86_64::*;
        self.reset_output(intensity, raman_polarization);
        let lanes_end = self.oscillators.len() / 4 * 4;
        for n in 0..intensity.len() - 1 {
            let i_n = _mm256_set1_pd(intensity[n]);
            let i_np1 = _mm256_set1_pd(intensity[n + 1]);
            let mut total_q = 0.0;
            for i in (0..lanes_end).step_by(4) {
                let q = _mm256_loadu_pd(self.states_q.as_ptr().add(i));
                let dq = _mm256_loadu_pd(self.states_dq.as_ptr().add(i));
                let q_new = _mm256_add_pd(
                    _mm256_add_pd(
                        _mm256_add_pd(
                            _mm256_mul_pd(_mm256_loadu_pd(self.step_coeffs.a11.as_ptr().add(i)), q),
                            _mm256_mul_pd(
                                _mm256_loadu_pd(self.step_coeffs.a12.as_ptr().add(i)),
                                dq,
                            ),
                        ),
                        _mm256_mul_pd(_mm256_loadu_pd(self.step_coeffs.b0_1.as_ptr().add(i)), i_n),
                    ),
                    _mm256_mul_pd(
                        _mm256_loadu_pd(self.step_coeffs.b1_1.as_ptr().add(i)),
                        i_np1,
                    ),
                );
                let dq_new = _mm256_add_pd(
                    _mm256_add_pd(
                        _mm256_add_pd(
                            _mm256_mul_pd(_mm256_loadu_pd(self.step_coeffs.a21.as_ptr().add(i)), q),
                            _mm256_mul_pd(
                                _mm256_loadu_pd(self.step_coeffs.a22.as_ptr().add(i)),
                                dq,
                            ),
                        ),
                        _mm256_mul_pd(_mm256_loadu_pd(self.step_coeffs.b0_2.as_ptr().add(i)), i_n),
                    ),
                    _mm256_mul_pd(
                        _mm256_loadu_pd(self.step_coeffs.b1_2.as_ptr().add(i)),
                        i_np1,
                    ),
                );
                _mm256_storeu_pd(self.states_q.as_mut_ptr().add(i), q_new);
                _mm256_storeu_pd(self.states_dq.as_mut_ptr().add(i), dq_new);
                let mut lanes = [0.0; 4];
                _mm256_storeu_pd(lanes.as_mut_ptr(), q_new);
                for value in lanes {
                    total_q += value;
                }
            }
            total_q += self.update_scalar_tail(lanes_end, intensity[n], intensity[n + 1]);
            raman_polarization[n + 1] = total_q;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    // As above, with a two-lane bound for NEON.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn solve_neon(&mut self, intensity: &[f64], raman_polarization: &mut [f64]) {
        use std::arch::aarch64::*;
        self.reset_output(intensity, raman_polarization);
        let lanes_end = self.oscillators.len() / 2 * 2;
        for n in 0..intensity.len() - 1 {
            let i_n = vdupq_n_f64(intensity[n]);
            let i_np1 = vdupq_n_f64(intensity[n + 1]);
            let mut total_q = 0.0;
            for i in (0..lanes_end).step_by(2) {
                let q = vld1q_f64(self.states_q.as_ptr().add(i));
                let dq = vld1q_f64(self.states_dq.as_ptr().add(i));
                let q_new = vaddq_f64(
                    vaddq_f64(
                        vaddq_f64(
                            vmulq_f64(vld1q_f64(self.step_coeffs.a11.as_ptr().add(i)), q),
                            vmulq_f64(vld1q_f64(self.step_coeffs.a12.as_ptr().add(i)), dq),
                        ),
                        vmulq_f64(vld1q_f64(self.step_coeffs.b0_1.as_ptr().add(i)), i_n),
                    ),
                    vmulq_f64(vld1q_f64(self.step_coeffs.b1_1.as_ptr().add(i)), i_np1),
                );
                let dq_new = vaddq_f64(
                    vaddq_f64(
                        vaddq_f64(
                            vmulq_f64(vld1q_f64(self.step_coeffs.a21.as_ptr().add(i)), q),
                            vmulq_f64(vld1q_f64(self.step_coeffs.a22.as_ptr().add(i)), dq),
                        ),
                        vmulq_f64(vld1q_f64(self.step_coeffs.b0_2.as_ptr().add(i)), i_n),
                    ),
                    vmulq_f64(vld1q_f64(self.step_coeffs.b1_2.as_ptr().add(i)), i_np1),
                );
                vst1q_f64(self.states_q.as_mut_ptr().add(i), q_new);
                vst1q_f64(self.states_dq.as_mut_ptr().add(i), dq_new);
                let mut lanes = [0.0; 2];
                vst1q_f64(lanes.as_mut_ptr(), q_new);
                for value in lanes {
                    total_q += value;
                }
            }
            total_q += self.update_scalar_tail(lanes_end, intensity[n], intensity[n + 1]);
            raman_polarization[n + 1] = total_q;
        }
    }

    fn reset_output(&mut self, intensity: &[f64], raman_polarization: &mut [f64]) {
        assert_eq!(raman_polarization.len(), intensity.len());
        assert!(!intensity.is_empty());
        self.reset_state();
        raman_polarization[0] = 0.0;
    }

    #[inline]
    fn update_scalar_tail(&mut self, start: usize, i_n: f64, i_np1: f64) -> f64 {
        let mut total_q = 0.0;
        for i in start..self.oscillators.len() {
            let q = self.states_q[i];
            let dq = self.states_dq[i];
            let q_new = self.step_coeffs.a11[i] * q
                + self.step_coeffs.a12[i] * dq
                + self.step_coeffs.b0_1[i] * i_n
                + self.step_coeffs.b1_1[i] * i_np1;
            let dq_new = self.step_coeffs.a21[i] * q
                + self.step_coeffs.a22[i] * dq
                + self.step_coeffs.b0_2[i] * i_n
                + self.step_coeffs.b1_2[i] * i_np1;
            self.states_q[i] = q_new;
            self.states_dq[i] = dq_new;
            total_q += q_new;
        }
        total_q
    }

    fn solve_scalar(&mut self, intensity: &[f64], raman_polarization: &mut [f64]) {
        let n_t = intensity.len();
        self.reset_output(intensity, raman_polarization);

        // Outer loop: time steps
        for n in 0..(n_t - 1) {
            let i_n = intensity[n];
            let i_np1 = intensity[n + 1];

            let mut total_q = 0.0;

            total_q += self.update_scalar_tail(0, i_n, i_np1);

            raman_polarization[n + 1] = total_q;
        }
    }
}

#[cfg(test)]
#[path = "raman_math_tests.rs"]
mod math_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn oscillators(n: usize) -> Vec<RamanOscillator> {
        (0..n)
            .map(|i| RamanOscillator {
                omega: 0.8e12 + i as f64 * 1.7e10,
                gamma: 1.1e11 + i as f64 * 3.0e8,
                coupling: (0.02 + i as f64 * 2.0e-4) * if i % 3 == 0 { -1.0 } else { 1.0 },
            })
            .collect()
    }

    fn intensity(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let x = i as f64;
                2.0e14 * (-(x - 17.0).powi(2) / 80.0).exp() + 3.0e12 * (0.17 * x).sin()
            })
            .collect()
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            let scale = e.abs().max(1.0);
            assert!(
                (a - e).abs() <= 2.0e-13 * scale,
                "SIMD/scalar mismatch at {i}: {a} vs {e}"
            );
        }
    }

    #[test]
    fn simd_matches_scalar_with_vector_boundaries_and_tails() {
        for n_osc in [1, 2, 3, 4, 5, 49, 50, 65] {
            for n_t in [2, 7, 67] {
                let drive = intensity(n_t);
                let mut scalar = TimeDomainRamanSolver::new(oscillators(n_osc), 0.7e-14);
                let mut expected = vec![0.0; n_t];
                scalar.solve_scalar(&drive, &mut expected);

                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                    let mut simd = TimeDomainRamanSolver::new(oscillators(n_osc), 0.7e-14);
                    let mut actual = vec![0.0; n_t];
                    unsafe { simd.solve_avx2(&drive, &mut actual) };
                    assert_close(&actual, &expected);
                }

                #[cfg(target_arch = "aarch64")]
                if std::arch::is_aarch64_feature_detected!("neon") {
                    let mut simd = TimeDomainRamanSolver::new(oscillators(n_osc), 0.7e-14);
                    let mut actual = vec![0.0; n_t];
                    unsafe { simd.solve_neon(&drive, &mut actual) };
                    assert_close(&actual, &expected);
                }
            }
        }
    }
}
