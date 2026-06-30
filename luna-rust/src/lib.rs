pub mod grid;
pub mod ffi;
pub mod fftw;
pub mod native;
pub mod dispersion;
pub mod cuda;
pub mod diffraction;
pub mod ionization;
pub mod raman;
pub mod integrator;
pub mod stepper;
pub mod dispatch;
pub mod io;
pub mod scans;

#[cfg(test)]
mod tests {
    use super::ffi::process_field_inplace;
    use super::dispersion::{ChebyshevDispersion, SellmeierGas, ZeisbergerNeff};
    use super::diffraction::Qdht;
    use super::ionization::PptIonizationRate;
    use super::raman::{RamanOscillator, TimeDomainRamanSolver};
    use super::integrator::integrate_integrating_factor;
    use super::stepper::Dopri5Stepper;
    use super::dispatch::{HardwarePath, SimulationEngine};
    use super::io::{Hdf5Writer, get_hdf5_api};
    use super::scans::ScanQueue;
    use num_complex::Complex;

    #[test]
    fn test_inplace_processing() {
        let mut data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
        ];
        
        unsafe {
            process_field_inplace(
                data.as_mut_ptr() as *mut f64,
                data.len(),
                2.0,
            );
        }
        
        assert_eq!(data[0], Complex::new(2.0, 4.0));
        assert_eq!(data[1], Complex::new(6.0, 8.0));
    }

    #[test]
    fn test_sellmeier_gas_derivatives() {
        // Dummy parameters for a Sellmeier gas
        let gas = SellmeierGas {
            b1: 0.1,
            c1: 100.0,
        };

        let ω = 2.0; // test frequency
        let h = 1e-5; // finite difference step

        let n = gas.refractive_index(ω);
        let dn_exact = gas.dn_dω(ω);
        let d2n_exact = gas.d2n_dω2(ω);

        let n_plus = gas.refractive_index(ω + h);
        let n_minus = gas.refractive_index(ω - h);

        // Finite difference approximations
        let dn_fd = (n_plus - n_minus) / (2.0 * h);
        let d2n_fd = (n_plus - 2.0 * n + n_minus) / (h * h);

        // Assert exact matches finite difference up to some numerical tolerance
        assert!((dn_exact - dn_fd).abs() < 1e-8, "First derivative mismatch: exact {}, fd {}", dn_exact, dn_fd);
        assert!((d2n_exact - d2n_fd).abs() < 1e-5, "Second derivative mismatch: exact {}, fd {}", d2n_exact, d2n_fd);
    }

    #[test]
    fn test_chebyshev_dispersion() {
        // Fit f(ω) = 3ω^2 - 4ω + 5 on [1.0, 10.0]
        // Analytical derivatives:
        // f'(ω) = 6ω - 4
        // f''(ω) = 6
        let ω_min = 1.0;
        let ω_max = 10.0;
        let degree = 8;
        
        let cheb = ChebyshevDispersion::fit(ω_min, ω_max, degree, |ω| {
            3.0 * ω * ω - 4.0 * ω + 5.0
        });
        
        let test_ω = 5.5;
        let (val, deriv1, deriv2) = cheb.evaluate_derivatives(test_ω);
        
        // Assert value and derivatives are accurate to within machine epsilon scale
        assert!((val - (3.0 * test_ω * test_ω - 4.0 * test_ω + 5.0)).abs() < 1e-12);
        assert!((deriv1 - (6.0 * test_ω - 4.0)).abs() < 1e-12);
        assert!((deriv2 - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_qdht_self_inverse() {
        let n_zeros = 64;
        let radius = 10.0;
        let qdht = Qdht::new(n_zeros, radius);
        
        let n_grid = qdht.n_r;
        
        // Define a simple input: Gaussian profile f(r) = exp(-r^2)
        let mut input = vec![Complex::new(0.0, 0.0); n_grid];
        for i in 0..n_grid {
            let r = qdht.r[i];
            input[i] = Complex::new((-r * r).exp(), 0.0);
        }
        
        let mut forward = vec![Complex::new(0.0, 0.0); n_grid];
        let mut backward = vec![Complex::new(0.0, 0.0); n_grid];
        
        // Run forward transform
        qdht.transform(&input, &mut forward);
        
        // Run backward transform
        qdht.transform(&forward, &mut backward);
        
        // Find maximum difference
        let mut max_diff = 0.0;
        for i in 0..n_grid {
            let diff = (backward[i].re - input[i].re).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        println!("Max QDHT reconstruction difference: {:.2e}", max_diff);
        
        // Assert against a safe bounds check
        assert!(max_diff < 0.05); // ensure it is reasonably close
        for i in 0..n_grid {
            assert!(backward[i].im.abs() < 1e-12);
        }
    }

    /// Verify that `PptIonizationRate::from_samples` + `rate_vector` round-trips
    /// correctly and that the FFI-level functions behave identically to the
    /// closure-based `new` constructor for the same underlying data.
    #[test]
    fn test_ppt_from_samples_ffi() {
        use super::ffi::{init_ppt_ionization_lut, free_ppt_ionization_lut, ppt_ionization_rate_vector};

        let e_min    = 1.0e8_f64;
        let e_max    = 5.0e9_f64;
        let n_points = 64_usize;
        let rate_fn  = |e: f64| 1.2_f64 * (e / 1.0e9).exp();

        // Build reference via closure constructor
        let reference = PptIonizationRate::new(e_min, e_max, n_points, rate_fn);

        // Build the same LUT via from_samples — simulates what Julia will pass
        let h = (e_max - e_min) / ((n_points - 1) as f64);
        let e_vals: Vec<f64>    = (0..n_points).map(|i| e_min + i as f64 * h).collect();
        let rate_vals: Vec<f64> = e_vals.iter().map(|&e| rate_fn(e)).collect();

        let lut = PptIonizationRate::from_samples(e_min, e_max, &e_vals, &rate_vals);

        // Both should agree at mid-range points
        let test_fields: Vec<f64> = (0..20)
            .map(|i| e_min + (i as f64 + 0.5) * (e_max - e_min) / 20.0)
            .collect();
        for &ef in &test_fields {
            let r_ref = reference.rate(ef).unwrap();
            let r_new = lut.rate(ef).unwrap();
            let rel_err = (r_ref - r_new).abs() / r_ref;
            assert!(rel_err < 1e-4,
                "from_samples vs new mismatch at E={:.2e}: ref={:.6e} new={:.6e} rel_err={:.2e}",
                ef, r_ref, r_new, rel_err);
        }

        // --- Test FFI entry points ---
        let ptr = unsafe {
            init_ppt_ionization_lut(
                e_min, e_max, n_points,
                e_vals.as_ptr(), rate_vals.as_ptr(),
            )
        };
        assert!(!ptr.is_null(), "init_ppt_ionization_lut returned null");

        let mut out_rates = vec![0.0_f64; test_fields.len()];
        let ret = unsafe {
            ppt_ionization_rate_vector(
                ptr as *const _,
                test_fields.as_ptr(),
                out_rates.as_mut_ptr(),
                test_fields.len(),
            )
        };
        assert_eq!(ret, 0, "ppt_ionization_rate_vector returned error {}", ret);

        for (i, (&ef, &r_ffi)) in test_fields.iter().zip(out_rates.iter()).enumerate() {
            let r_ref = reference.rate(ef).unwrap();
            let rel_err = (r_ffi - r_ref).abs() / r_ref;
            assert!(rel_err < 1e-4,
                "FFI rate mismatch at index {} E={:.2e}: ref={:.6e} ffi={:.6e}",
                i, ef, r_ref, r_ffi);
        }

        // Below-cutoff must return 0.0
        let low_field = vec![e_min * 0.1];
        let mut low_out = vec![-1.0_f64];
        let ret2 = unsafe {
            ppt_ionization_rate_vector(ptr as *const _, low_field.as_ptr(), low_out.as_mut_ptr(), 1)
        };
        assert_eq!(ret2, 0);
        assert_eq!(low_out[0], 0.0, "Below-cutoff rate must be exactly 0.0");

        // Null-pointer guard must return -1
        let ret3 = unsafe {
            ppt_ionization_rate_vector(std::ptr::null(), test_fields.as_ptr(), out_rates.as_mut_ptr(), 1)
        };
        assert_eq!(ret3, -1);

        unsafe { free_ppt_ionization_lut(ptr); }
    }

    #[test]
    fn test_ppt_ionization_cutoff() {
        let e_min = 1.0e8; // 100 MV/m
        let e_max = 5.0e9; // 5 GV/m
        let n_points = 128;
        
        // Define a simple mock PPT rate function: w(E) = 1.2 * exp(E / 1.0e9)
        let rate_fn = |e: f64| 1.2 * (e / 1.0e9).exp();
        
        let ppt = PptIonizationRate::new(e_min, e_max, n_points, rate_fn);
        
        // Below cutoff: rate must be strictly 0.0
        let low_rate = ppt.rate(5.0e7).unwrap();
        assert_eq!(low_rate, 0.0);
        
        // Within range: rate should interpolate accurately
        let mid_e = 2.5e9;
        let mid_rate = ppt.rate(mid_e).unwrap();
        let expected = rate_fn(mid_e);
        assert!((mid_rate - expected).abs() / expected < 1e-3); // within spline accuracy
        
        // Out of bounds: should return error
        assert!(ppt.rate(6.0e9).is_err());
    }

    #[test]
    fn test_raman_ade_solver() {
        // Define a single oscillator transition (e.g. He Raman model)
        let osc = RamanOscillator {
            omega: 1e12,    // 1 THz
            gamma: 2e11,    // 200 GHz dephasing
            coupling: 0.1,
        };
        
        let dt = 1e-14; // 10 fs step size
        let mut solver = TimeDomainRamanSolver::new(vec![osc], dt);
        
        // Simple step intensity profile
        let mut intensity = vec![0.0; 100];
        for i in 10..100 {
            intensity[i] = 1e15; // 1 PW/cm^2 drive
        }
        
        let mut raman_pol = vec![0.0; 100];
        solver.solve(&intensity, &mut raman_pol);
        
        // Verification: check that polarization stays 0 before intensity starts
        assert_eq!(raman_pol[0], 0.0);
        assert_eq!(raman_pol[9], 0.0);
        
        // check that polarization is excited and non-zero after step starts
        assert!(raman_pol[11] > 0.0);
    }

    #[test]
    fn test_raman_solver_ffi() {
        use super::ffi::{init_raman_solver, free_raman_solver, raman_solve};

        let omega    = 1e13_f64; // 10 THz oscillator
        let gamma    = 1e12_f64; // 1 THz damping (τ2 = 1 ps)
        let coupling = 1.0_f64;
        let dt       = 1e-14_f64; // 10 fs time step
        let n_t      = 256_usize;

        // n_osc=0 must return null
        let ptr_zero = unsafe { init_raman_solver(&omega, &gamma, &coupling, 0, dt) };
        assert!(ptr_zero.is_null(), "n_osc=0 should return null");

        // null omega must return null
        let ptr_null = unsafe { init_raman_solver(std::ptr::null(), &gamma, &coupling, 1, dt) };
        assert!(ptr_null.is_null(), "null omega should return null");

        // Normal init
        let ptr = unsafe { init_raman_solver(&omega, &gamma, &coupling, 1, dt) };
        assert!(!ptr.is_null(), "init_raman_solver returned null");

        // Null ptr to raman_solve → -1
        let mut dummy_out = [0.0_f64; 1];
        let dummy_in = [0.0_f64; 1];
        let ret_null = unsafe {
            raman_solve(std::ptr::null_mut(), dummy_in.as_ptr(), dummy_out.as_mut_ptr(), 1)
        };
        assert_eq!(ret_null, -1, "null ptr should return -1");

        // Zero intensity → zero polarization (solver resets state each call)
        let intensity_zero = vec![0.0_f64; n_t];
        let mut polar = vec![9.9_f64; n_t]; // pre-fill non-zero
        let ret = unsafe {
            raman_solve(ptr, intensity_zero.as_ptr(), polar.as_mut_ptr(), n_t)
        };
        assert_eq!(ret, 0);
        assert!(polar.iter().all(|&x| x == 0.0), "zero intensity must give zero polarization");

        // Non-zero intensity → non-zero polarization
        let intensity_step: Vec<f64> = vec![1e14_f64; n_t];
        let mut polar_step = vec![0.0_f64; n_t];
        let ret2 = unsafe {
            raman_solve(ptr, intensity_step.as_ptr(), polar_step.as_mut_ptr(), n_t)
        };
        assert_eq!(ret2, 0);
        assert!(polar_step.iter().any(|&x| x.abs() > 0.0), "non-zero drive must yield non-zero response");

        // FFI output must exactly match a direct TimeDomainRamanSolver::solve call
        let oscs = vec![RamanOscillator { omega, gamma, coupling }];
        let mut ref_solver = TimeDomainRamanSolver::new(oscs, dt);
        let mut ref_polar = vec![0.0_f64; n_t];
        ref_solver.solve(&intensity_step, &mut ref_polar);
        for i in 0..n_t {
            assert_eq!(polar_step[i], ref_polar[i],
                "FFI vs direct mismatch at step {}: {} vs {}", i, polar_step[i], ref_polar[i]);
        }

        unsafe { free_raman_solver(ptr); }
    }

    #[test]
    fn test_integrating_factor_quadrature() {
        let z_n = 0.0;
        let h = 2.0;
        
        // Define grid frequencies
        let omega = vec![1.0, 2.0, 3.0];
        
        // Simple z-dependent operator: L(z, ω) = -i * ω * z^2
        // Integral over [0, 2] is:
        // \int_0^2 (-i * ω * z^2) dz = -i * ω * [z^3/3]_0^2 = -i * ω * 8/3
        // Exact Integrating Factor: exp(-i * ω * 8/3)
        let factors = integrate_integrating_factor(z_n, h, &omega, |z, ω| {
            Complex::new(0.0, -ω * z * z)
        });
        
        for i in 0..omega.len() {
            let ω = omega[i];
            let exact = Complex::new(0.0, -ω * 8.0 / 3.0).exp();
            
            assert!((factors[i].re - exact.re).abs() < 1e-12);
            assert!((factors[i].im - exact.im).abs() < 1e-12);
        }
    }

    #[test]
    fn test_dopri5_stepper_basic() {
        let size = 4;
        let mut stepper = Dopri5Stepper::new(size, 1e-4, 1e-6);
        let mut y = vec![Complex::new(1.0, 0.0); size];
        let mut fsal_active = false;

        // dy/dz = -0.5 * y
        // exact solution y(z) = y0 * exp(-0.5 * z)
        let lin_op = |_z1: f64, _z2: f64, y_in: &[Complex<f64>], y_out: &mut [Complex<f64>]| {
            y_out.copy_from_slice(y_in);
        };
        let rhs = |_z: f64, y_in: &[Complex<f64>], dy_out: &mut [Complex<f64>]| {
            for i in 0..y_in.len() {
                dy_out[i] = -0.5 * y_in[i];
            }
        };

        let res = stepper.step(&mut y, 0.0, 0.1, lin_op, rhs, &mut fsal_active);
        assert!(res.is_ok());
        let (h_next, accepted) = res.unwrap();
        assert!(accepted);
        assert!(h_next > 0.0);
        for val in y.iter() {
            assert!((val.re - (-0.05f64).exp()).abs() < 1e-3);
        }
    }

    #[test]
    fn test_simulation_engine_dispatch() {
        let engine = SimulationEngine::initialize(HardwarePath::Auto);
        println!("Auto-selected hardware path: {:?}", engine.active_path);
        
        let portable = SimulationEngine::initialize(HardwarePath::CpuPortable);
        assert_eq!(portable.active_path, HardwarePath::CpuPortable);
        assert_eq!(portable.thread_pool_size, 1);
        
        let avx2_engine = SimulationEngine::initialize(HardwarePath::CpuX86AVX2);
        assert!(avx2_engine.thread_pool_size >= 1);
    }

    #[test]
    fn test_hdf5_io_basic() {
        let api_res = get_hdf5_api();
        if let Err(err) = api_res {
            println!("Skipping HDF5 IO test because library could not be loaded: {}", err);
            return;
        }
        let api = api_res.unwrap();
        let test_file = "test_io_out.h5";
        
        let writer = Hdf5Writer::open_or_create(test_file).unwrap();
        let group_id = writer.create_group("stats").unwrap();
        
        let dset_id = writer.create_dataset_2d(group_id, "energy", api.h5t_native_double, &[5], &[5]).unwrap();
        writer.write_dataset_f64(dset_id, &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        writer.close_dataset(dset_id);
        
        let dset_c_id = writer.create_dataset_2d(group_id, "field", api.h5t_complex, &[2], &[2]).unwrap();
        writer.write_dataset_complex(dset_c_id, &[Complex::new(1.1, 2.2), Complex::new(3.3, 4.4)]).unwrap();
        writer.close_dataset(dset_c_id);
        
        writer.close_group(group_id);
        drop(writer);
        
        let _ = std::fs::remove_file(test_file);
    }

    #[test]
    fn test_scan_queue_flock() {
        let api_res = get_hdf5_api();
        if let Err(err) = api_res {
            println!("Skipping ScanQueue flock test because HDF5 could not be loaded: {}", err);
            return;
        }
        let qfile = "test_queue.h5";
        let queue = ScanQueue::new(qfile, 4);
        
        let idx0 = queue.checkout_next_index().unwrap();
        assert_eq!(idx0, Some(0));
        
        let idx1 = queue.checkout_next_index().unwrap();
        assert_eq!(idx1, Some(1));
        
        queue.mark_completed(0, true).unwrap();
        queue.mark_completed(1, false).unwrap();
        
        let idx2 = queue.checkout_next_index().unwrap();
        assert_eq!(idx2, Some(2));
        let idx3 = queue.checkout_next_index().unwrap();
        assert_eq!(idx3, Some(3));
        
        let idx4 = queue.checkout_next_index().unwrap();
        assert_eq!(idx4, None);
        
        queue.mark_completed(2, true).unwrap();
        queue.mark_completed(3, true).unwrap();
        
        let idx5 = queue.checkout_next_index().unwrap();
        assert_eq!(idx5, None);
        assert!(!std::path::Path::new(qfile).exists());
    }

    #[test]
    fn test_gpu_qdht_numerical_equivalence() {
        if let Err(err) = crate::cuda::init_gpu_context() {
            println!("Skipping GPU QDHT test: GPU context not available: {}", err);
            return;
        }
        
        let n_zeros = 64;
        let radius = 10.0;
        let qdht = Qdht::new(n_zeros, radius);
        
        let n_grid = qdht.n_r;
        let mut input = vec![Complex::new(0.0, 0.0); n_grid];
        for i in 0..n_grid {
            let r = qdht.r[i];
            input[i] = Complex::new((-r * r).exp(), 0.0);
        }
        
        let mut gpu_output = vec![Complex::new(0.0, 0.0); n_grid];
        qdht.transform(&input, &mut gpu_output);
        
        let mut cpu_output = vec![Complex::new(0.0, 0.0); n_grid];
        for i in 0..n_grid {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            let row_offset = i * n_grid;
            for j in 0..n_grid {
                let t_val = qdht.t_matrix[row_offset + j];
                let val = input[j];
                sum_re += t_val * val.re;
                sum_im += t_val * val.im;
            }
            cpu_output[i] = Complex::new(sum_re, sum_im);
        }
        
        for i in 0..n_grid {
            assert!((gpu_output[i].re - cpu_output[i].re).abs() < 1e-12);
            assert!((gpu_output[i].im - cpu_output[i].im).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gpu_raman_numerical_equivalence() {
        if let Err(err) = crate::cuda::init_gpu_context() {
            println!("Skipping GPU Raman test: GPU context not available: {}", err);
            return;
        }
        
        let osc = RamanOscillator {
            omega: 1e12,
            gamma: 2e11,
            coupling: 0.1,
        };
        
        let dt = 1e-14;
        let mut solver = TimeDomainRamanSolver::new(vec![osc], dt);
        
        let mut intensity = vec![0.0; 100];
        for i in 10..100 {
            intensity[i] = 1e15;
        }
        
        let mut gpu_pol = vec![0.0; 100];
        solver.solve(&intensity, &mut gpu_pol);
        
        let mut cpu_pol = vec![0.0; 100];
        let mut states = vec![(0.0, 0.0); solver.oscillators.len()];
        cpu_pol[0] = 0.0;
        for n in 0..99 {
            let i_n = intensity[n];
            let i_np1 = intensity[n + 1];
            let mut total_q = 0.0;
            for i in 0..solver.oscillators.len() {
                let coeffs = &solver.step_coeffs[i];
                let (q, dq) = states[i];
                let q_new = coeffs.a11 * q + coeffs.a12 * dq + coeffs.b0_1 * i_n + coeffs.b1_1 * i_np1;
                let dq_new = coeffs.a21 * q + coeffs.a22 * dq + coeffs.b0_2 * i_n + coeffs.b1_2 * i_np1;
                states[i] = (q_new, dq_new);
                total_q += q_new;
            }
            cpu_pol[n + 1] = total_q;
        }
        
        for i in 0..100 {
            assert!((gpu_pol[i] - cpu_pol[i]).abs() < 1e-12);
        }
    }

    /// Verify that the Zeisberger FFI trio (`init_zeisberger_neff` /
    /// `free_zeisberger_neff` / `zeisberger_neff_vector`) produces results that
    /// match a direct `ZeisbergerNeff::neff_one` call to within ~1e-13.
    ///
    /// Covers all four kind codes (0=HE, 1=EH, 2=TE, 3=TM) and all three loss
    /// encodings (Val{true}, Val{false}, Number).
    #[test]
    fn test_zeisberger_neff_ffi() {
        use super::ffi::{init_zeisberger_neff, free_zeisberger_neff, zeisberger_neff_vector};
        use num_complex::Complex;

        // Physically realistic hollow-core fibre parameters
        let radius        = 150e-6_f64;   // 150 µm core radius
        let wallthickness = 550e-9_f64;   // 550 nm strut thickness
        let unm_he11      = 2.404_825_56_f64; // first zero of J0

        // Sellmeier-like gas (He): nco ≈ 1 + tiny correction
        let nco_re_val = 1.000_035_f64;
        let nco_im_val = 0.0_f64;
        // SiO2 glass cladding at ~800 nm: ncl ≈ 1.453
        let ncl_re_val = 1.453_f64;
        let ncl_im_val = 0.0_f64;

        // Test grid: a few representative frequencies
        let omegas: Vec<f64> = vec![
            2.0e15, 2.35e15, 2.7e15, 3.0e15,
        ];
        let n = omegas.len();
        let nco_re: Vec<f64> = vec![nco_re_val; n];
        let nco_im: Vec<f64> = vec![nco_im_val; n];
        let ncl_re: Vec<f64> = vec![ncl_re_val; n];
        let ncl_im: Vec<f64> = vec![ncl_im_val; n];

        // ── helper to run one (kind, loss_on, loss_scale) combination ──────────
        let run_case = |kind: u32, loss_on: u32, loss_scale: f64, m_az: f64| {
            let ptr = unsafe {
                init_zeisberger_neff(
                    unm_he11, m_az, kind, wallthickness, loss_on, loss_scale,
                )
            };
            assert!(!ptr.is_null(), "init_zeisberger_neff returned null for kind={}", kind);

            let mut re_out = vec![0.0_f64; n];
            let mut im_out = vec![0.0_f64; n];
            let ret = unsafe {
                zeisberger_neff_vector(
                    ptr as *const _,
                    omegas.as_ptr(), nco_re.as_ptr(), nco_im.as_ptr(),
                    ncl_re.as_ptr(), ncl_im.as_ptr(),
                    radius,
                    re_out.as_mut_ptr(), im_out.as_mut_ptr(),
                    n,
                )
            };
            assert_eq!(ret, 0, "zeisberger_neff_vector returned {} for kind={}", ret, kind);

            // Compare FFI output against direct ZeisbergerNeff::neff_one
            let handle = ZeisbergerNeff::new(
                unm_he11, m_az, kind as u8, wallthickness, loss_on != 0, loss_scale,
            );
            for i in 0..n {
                let nco = Complex::new(nco_re[i], nco_im[i]);
                let ncl = Complex::new(ncl_re[i], ncl_im[i]);
                let ref_ne = handle.neff_one(omegas[i], nco, ncl, radius);
                let diff_re = (re_out[i] - ref_ne.re).abs();
                let diff_im = (im_out[i] - ref_ne.im).abs();
                assert!(
                    diff_re < 1e-13,
                    "Re(neff) FFI vs direct mismatch at i={}: FFI={} ref={} diff={}",
                    i, re_out[i], ref_ne.re, diff_re
                );
                assert!(
                    diff_im < 1e-13,
                    "Im(neff) FFI vs direct mismatch at i={}: FFI={} ref={} diff={}",
                    i, im_out[i], ref_ne.im, diff_im
                );
            }

            unsafe { free_zeisberger_neff(ptr); }
        };

        // ── HE11 (kind=0), all three loss modes ───────────────────────────────
        run_case(0, 1, 1.0, 1.0);   // Val{true}
        run_case(0, 0, 0.0, 1.0);   // Val{false}
        run_case(0, 1, 0.5, 1.0);   // Number(0.5)

        // ── EH11 (kind=1) ─────────────────────────────────────────────────────
        let unm_eh11 = 3.831_705_97_f64; // first zero of J1'
        let run_eh = |loss_on: u32, loss_scale: f64| {
            let ptr = unsafe {
                init_zeisberger_neff(unm_eh11, 1.0, 1, wallthickness, loss_on, loss_scale)
            };
            assert!(!ptr.is_null());
            let mut re_out = vec![0.0_f64; n];
            let mut im_out = vec![0.0_f64; n];
            let ret = unsafe {
                zeisberger_neff_vector(
                    ptr as *const _, omegas.as_ptr(),
                    nco_re.as_ptr(), nco_im.as_ptr(),
                    ncl_re.as_ptr(), ncl_im.as_ptr(),
                    radius, re_out.as_mut_ptr(), im_out.as_mut_ptr(), n,
                )
            };
            assert_eq!(ret, 0);
            let handle = ZeisbergerNeff::new(unm_eh11, 1.0, 1, wallthickness, loss_on != 0, loss_scale);
            for i in 0..n {
                let nco = Complex::new(nco_re[i], nco_im[i]);
                let ncl = Complex::new(ncl_re[i], ncl_im[i]);
                let ref_ne = handle.neff_one(omegas[i], nco, ncl, radius);
                assert!((re_out[i] - ref_ne.re).abs() < 1e-13);
                assert!((im_out[i] - ref_ne.im).abs() < 1e-13);
            }
            unsafe { free_zeisberger_neff(ptr); }
        };
        run_eh(1, 1.0);
        run_eh(0, 0.0);

        // ── TE01 (kind=2, m_az irrelevant) ────────────────────────────────────
        let unm_te01 = 3.831_705_97_f64;
        run_case(2, 1, 1.0, 0.0); // m_az=0 ok for TE/TM
        run_case(2, 0, 0.0, 0.0);

        // ── TM01 (kind=3) ─────────────────────────────────────────────────────
        run_case(3, 1, 1.0, 0.0);
        let _ = unm_te01; // silence unused warning

        // ── Null-pointer guards ────────────────────────────────────────────────
        let ptr = unsafe {
            init_zeisberger_neff(unm_he11, 1.0, 0, wallthickness, 1, 1.0)
        };
        assert!(!ptr.is_null());

        // null ptr → -1
        let mut re_dummy = [0.0_f64; 1];
        let mut im_dummy = [0.0_f64; 1];
        let omega_dummy = [2.0e15_f64];
        let nco_re_d = [1.0_f64]; let nco_im_d = [0.0_f64];
        let ncl_re_d = [1.45_f64]; let ncl_im_d = [0.0_f64];
        let ret_null = unsafe {
            zeisberger_neff_vector(
                std::ptr::null(),
                omega_dummy.as_ptr(), nco_re_d.as_ptr(), nco_im_d.as_ptr(),
                ncl_re_d.as_ptr(), ncl_im_d.as_ptr(),
                radius, re_dummy.as_mut_ptr(), im_dummy.as_mut_ptr(), 1,
            )
        };
        assert_eq!(ret_null, -1, "null ptr should return -1");

        unsafe { free_zeisberger_neff(ptr); }

        // null free is a no-op (must not crash)
        unsafe { free_zeisberger_neff(std::ptr::null_mut()); }
    }

    #[test]
    fn test_marcatili_neff_ffi() {
        use super::ffi::{init_marcatili_neff, free_marcatili_neff, marcatili_neff_vector};
        use super::dispersion::MarcatiliNeff;

        // Physically realistic parameters for a hollow capillary
        let unm_he11 = 2.404_825_56_f64;  // first zero of J0
        let radius   = 150e-6_f64;        // 150 µm core radius

        // Precomputed nwg from SiO2 cladding at a few representative frequencies.
        // nwg = (unm/(k*a))^2 * (1 − i*vn/(k*a))^2 (full model, HE11)
        // For the test we just use plausible small complex values.
        let n = 4_usize;
        let nwg_re: Vec<f64> = vec![1.2e-5, 1.3e-5, 1.4e-5, 1.5e-5];
        let nwg_im: Vec<f64> = vec![2.0e-9, 2.1e-9, 2.2e-9, 2.3e-9];

        // Gas refractive index (He at 1 bar, nearly 1.0 at optical frequencies)
        let nco_re: Vec<f64> = vec![1.000_035_f64; n];
        let nco_im: Vec<f64> = vec![0.0_f64; n];

        // ── helper: run FFI and compare with MarcatiliNeff::neff_vector directly ──
        let run_case = |model: u32, loss_on: u32| {
            let ptr = unsafe {
                init_marcatili_neff(
                    nwg_re.as_ptr(), nwg_im.as_ptr(), n,
                    model, loss_on,
                )
            };
            assert!(!ptr.is_null(), "init_marcatili_neff returned null (model={}, loss_on={})", model, loss_on);

            let mut re_out = vec![0.0_f64; n];
            let mut im_out = vec![0.0_f64; n];
            let ret = unsafe {
                marcatili_neff_vector(
                    ptr as *const _,
                    nco_re.as_ptr(), nco_im.as_ptr(),
                    re_out.as_mut_ptr(), im_out.as_mut_ptr(),
                    n,
                )
            };
            assert_eq!(ret, 0, "marcatili_neff_vector returned {} (model={}, loss_on={})", ret, model, loss_on);

            // Compare against direct in-process call
            let handle = MarcatiliNeff::new(
                nwg_re.clone(), nwg_im.clone(), model as u8, loss_on != 0,
            );
            let mut ref_re = vec![0.0_f64; n];
            let mut ref_im = vec![0.0_f64; n];
            handle.neff_vector(&nco_re, &nco_im, &mut ref_re, &mut ref_im);

            for i in 0..n {
                let diff_re = (re_out[i] - ref_re[i]).abs();
                let diff_im = (im_out[i] - ref_im[i]).abs();
                assert!(diff_re < 1e-15,
                    "Re(neff) FFI vs direct at i={}: FFI={} ref={} diff={}", i, re_out[i], ref_re[i], diff_re);
                assert!(diff_im < 1e-15,
                    "Im(neff) FFI vs direct at i={}: FFI={} ref={} diff={}", i, im_out[i], ref_im[i], diff_im);
            }

            unsafe { free_marcatili_neff(ptr); }
        };

        // :full model (0), loss_on=1 (Val{true})
        run_case(0, 1);
        // :full model (0), loss_on=0 (Val{false}) — Im(neff) must be zero
        run_case(0, 0);
        // :reduced model (1), loss_on=1
        run_case(1, 1);
        // :reduced model (1), loss_on=0
        run_case(1, 0);

        // loss_on=0 must zero out imaginary part
        let ptr = unsafe { init_marcatili_neff(nwg_re.as_ptr(), nwg_im.as_ptr(), n, 0, 0) };
        assert!(!ptr.is_null());
        let mut re_out = vec![0.0_f64; n];
        let mut im_out = vec![0.0_f64; n];
        let ret = unsafe {
            marcatili_neff_vector(ptr as *const _, nco_re.as_ptr(), nco_im.as_ptr(),
                                  re_out.as_mut_ptr(), im_out.as_mut_ptr(), n)
        };
        assert_eq!(ret, 0);
        for i in 0..n {
            assert_eq!(im_out[i], 0.0, "Im(neff) must be 0 when loss_on=0 at i={}", i);
        }
        unsafe { free_marcatili_neff(ptr); }

        // null-pointer guard: init with null nwg → null return
        let null_ptr = unsafe { init_marcatili_neff(std::ptr::null(), nwg_im.as_ptr(), n, 0, 1) };
        assert!(null_ptr.is_null(), "init with null nwg_re must return null");

        // n=0 → null
        let zero_ptr = unsafe { init_marcatili_neff(nwg_re.as_ptr(), nwg_im.as_ptr(), 0, 0, 1) };
        assert!(zero_ptr.is_null(), "init with n=0 must return null");

        // null ptr to vector → -1
        let mut re_d = [0.0_f64; 1]; let mut im_d = [0.0_f64; 1];
        let nc_r = [1.0_f64]; let nc_i = [0.0_f64];
        let ret_null = unsafe {
            marcatili_neff_vector(
                std::ptr::null(), nc_r.as_ptr(), nc_i.as_ptr(),
                re_d.as_mut_ptr(), im_d.as_mut_ptr(), 1,
            )
        };
        assert_eq!(ret_null, -1, "null handle ptr should return -1");

        // free(null) is a no-op (must not crash)
        unsafe { free_marcatili_neff(std::ptr::null_mut()); }

        let _ = (unm_he11, radius); // suppress unused-variable warnings
    }

    /// Verify the QDHT FFI batch-transform functions against a direct reference implementation.
    ///
    /// Uses a small arbitrary symmetric matrix as T (not a true QDHT matrix, but sufficient
    /// to exercise all code paths).  Checks: real mul!, real ldiv!, complex mul!, complex ldiv!,
    /// null-pointer guards, n_r mismatch guard, and that free(null) is a no-op.
    #[test]
    fn test_qdht_ffi() {
        use super::ffi::{
            init_qdht_ffi, free_qdht_ffi,
            qdht_ffi_mul_real, qdht_ffi_ldiv_real,
            qdht_ffi_mul_cplx, qdht_ffi_ldiv_cplx,
        };

        let n_r    = 8_usize;
        let n_time = 16_usize;

        // Build a symmetric n_r×n_r test matrix (col-major Julia layout: T[r,s] at r + n_r*s)
        let mut t_col = vec![0.0f64; n_r * n_r];
        for r in 0..n_r {
            for s in 0..n_r {
                let val = 1.0 / (1.0 + ((r as f64) - (s as f64)).powi(2));
                t_col[r + n_r * s] = val;
            }
        }

        let scale_fwd = 2.5_f64;
        let scale_inv = 1.0 / scale_fwd;

        // ── initialise handle ──────────────────────────────────────────────────
        let ptr = unsafe {
            init_qdht_ffi(t_col.as_ptr(), n_r, scale_fwd, scale_inv, n_time)
        };
        assert!(!ptr.is_null(), "init_qdht_ffi returned null");

        // ── reference: B[t,r] = scale × Σ_s T[r,s] × A[t,s] ─────────────────
        // T[r,s] = t_col[r + n_r*s]  (Julia col-major)
        // data layout: col-major (n_time, n_r): data[t + n_time*r]
        let compute_ref_real = |data: &[f64], scale: f64| -> Vec<f64> {
            let mut out = vec![0.0f64; n_time * n_r];
            for r in 0..n_r {
                for t in 0..n_time {
                    let mut sum = 0.0;
                    for s in 0..n_r {
                        sum += t_col[r + n_r * s] * data[t + n_time * s];
                    }
                    out[t + n_time * r] = scale * sum;
                }
            }
            out
        };

        // ── real mul! ─────────────────────────────────────────────────────────
        let mut data_real: Vec<f64> = (0..n_time * n_r)
            .map(|i| (i as f64).sin())
            .collect();
        let ref_fwd = compute_ref_real(&data_real, scale_fwd);
        let ret = unsafe { qdht_ffi_mul_real(ptr, data_real.as_mut_ptr(), n_time, n_r) };
        assert_eq!(ret, 0, "qdht_ffi_mul_real returned {}", ret);
        for (i, (&got, &exp)) in data_real.iter().zip(ref_fwd.iter()).enumerate() {
            let diff = (got - exp).abs();
            assert!(diff < 1e-14,
                "mul_real mismatch at i={}: got={} exp={} diff={}", i, got, exp, diff);
        }

        // ── real ldiv! ────────────────────────────────────────────────────────
        let data_orig: Vec<f64> = (0..n_time * n_r).map(|i| (i as f64).cos()).collect();
        let ref_inv = compute_ref_real(&data_orig, scale_inv);
        let mut data_inv = data_orig.clone();
        let ret2 = unsafe { qdht_ffi_ldiv_real(ptr, data_inv.as_mut_ptr(), n_time, n_r) };
        assert_eq!(ret2, 0, "qdht_ffi_ldiv_real returned {}", ret2);
        for (i, (&got, &exp)) in data_inv.iter().zip(ref_inv.iter()).enumerate() {
            let diff = (got - exp).abs();
            assert!(diff < 1e-14,
                "ldiv_real mismatch at i={}: got={} exp={} diff={}", i, got, exp, diff);
        }

        // ── complex mul! ─────────────────────────────────────────────────────
        // Interleaved layout: data[2*t + 2*n_time*r] = Re, data[2*t+1 + ...] = Im
        let mut data_cplx: Vec<f64> = (0..2 * n_time * n_r)
            .map(|i| (i as f64) * 0.1)
            .collect();

        // Reference for complex mul! (same T, applied to re and im separately)
        let mut ref_cre = vec![0.0f64; n_time * n_r];
        let mut ref_cim = vec![0.0f64; n_time * n_r];
        for r in 0..n_r {
            for t in 0..n_time {
                let mut sr = 0.0f64; let mut si = 0.0f64;
                for s in 0..n_r {
                    let tv = t_col[r + n_r * s];
                    sr += tv * data_cplx[2 * t     + 2 * n_time * s];
                    si += tv * data_cplx[2 * t + 1 + 2 * n_time * s];
                }
                ref_cre[t + n_time * r] = scale_fwd * sr;
                ref_cim[t + n_time * r] = scale_fwd * si;
            }
        }

        let ret3 = unsafe { qdht_ffi_mul_cplx(ptr, data_cplx.as_mut_ptr(), n_time, n_r) };
        assert_eq!(ret3, 0, "qdht_ffi_mul_cplx returned {}", ret3);
        for r in 0..n_r {
            for t in 0..n_time {
                let got_re = data_cplx[2 * t     + 2 * n_time * r];
                let got_im = data_cplx[2 * t + 1 + 2 * n_time * r];
                let exp_re = ref_cre[t + n_time * r];
                let exp_im = ref_cim[t + n_time * r];
                assert!((got_re - exp_re).abs() < 1e-14,
                    "cplx mul! Re mismatch t={} r={}: got={} exp={}", t, r, got_re, exp_re);
                assert!((got_im - exp_im).abs() < 1e-14,
                    "cplx mul! Im mismatch t={} r={}: got={} exp={}", t, r, got_im, exp_im);
            }
        }

        // ── complex ldiv! ────────────────────────────────────────────────────
        let data_cplx_orig = data_cplx.clone();  // already transformed by mul_cplx
        let mut data_cplx2: Vec<f64> = (0..2 * n_time * n_r)
            .map(|i| (i as f64) * 0.05)
            .collect();
        let ret4 = unsafe { qdht_ffi_ldiv_cplx(ptr, data_cplx2.as_mut_ptr(), n_time, n_r) };
        assert_eq!(ret4, 0, "qdht_ffi_ldiv_cplx returned {}", ret4);
        // Verify ldiv uses scale_inv (not scale_fwd): check against reference with scale_inv
        let mut ref_cre_inv = vec![0.0f64; n_time * n_r];
        let mut ref_cim_inv = vec![0.0f64; n_time * n_r];
        let data_cplx2_orig: Vec<f64> = (0..2 * n_time * n_r).map(|i| (i as f64) * 0.05).collect();
        for r in 0..n_r {
            for t in 0..n_time {
                let mut sr = 0.0f64; let mut si = 0.0f64;
                for s in 0..n_r {
                    let tv = t_col[r + n_r * s];
                    sr += tv * data_cplx2_orig[2 * t     + 2 * n_time * s];
                    si += tv * data_cplx2_orig[2 * t + 1 + 2 * n_time * s];
                }
                ref_cre_inv[t + n_time * r] = scale_inv * sr;
                ref_cim_inv[t + n_time * r] = scale_inv * si;
            }
        }
        for r in 0..n_r {
            for t in 0..n_time {
                let got_re = data_cplx2[2 * t     + 2 * n_time * r];
                let got_im = data_cplx2[2 * t + 1 + 2 * n_time * r];
                let exp_re = ref_cre_inv[t + n_time * r];
                let exp_im = ref_cim_inv[t + n_time * r];
                assert!((got_re - exp_re).abs() < 1e-14,
                    "cplx ldiv! Re mismatch t={} r={}: got={} exp={}", t, r, got_re, exp_re);
                assert!((got_im - exp_im).abs() < 1e-14,
                    "cplx ldiv! Im mismatch t={} r={}: got={} exp={}", t, r, got_im, exp_im);
            }
        }

        // ── null-pointer guards ───────────────────────────────────────────────
        let mut dummy = vec![0.0f64; n_time * n_r];
        let ret_n1 = unsafe { qdht_ffi_mul_real(std::ptr::null_mut(), dummy.as_mut_ptr(), n_time, n_r) };
        assert_eq!(ret_n1, -1, "null ptr must return -1");
        let ret_n2 = unsafe { qdht_ffi_mul_real(ptr, std::ptr::null_mut(), n_time, n_r) };
        assert_eq!(ret_n2, -1, "null data must return -1");
        let ret_n3 = unsafe { qdht_ffi_ldiv_real(std::ptr::null_mut(), dummy.as_mut_ptr(), n_time, n_r) };
        assert_eq!(ret_n3, -1);
        let ret_n4 = unsafe { qdht_ffi_mul_cplx(std::ptr::null_mut(), dummy.as_mut_ptr(), n_time, n_r) };
        assert_eq!(ret_n4, -1);
        let ret_n5 = unsafe { qdht_ffi_ldiv_cplx(ptr, std::ptr::null_mut(), n_time, n_r) };
        assert_eq!(ret_n5, -1);

        // ── n_r mismatch → -1 ────────────────────────────────────────────────
        let ret_mm = unsafe { qdht_ffi_mul_real(ptr, dummy.as_mut_ptr(), n_time, n_r + 1) };
        assert_eq!(ret_mm, -1, "n_r mismatch must return -1");

        // ── init guards ──────────────────────────────────────────────────────
        let null_ptr = unsafe { init_qdht_ffi(std::ptr::null(), n_r, scale_fwd, scale_inv, n_time) };
        assert!(null_ptr.is_null(), "null t_matrix must return null");
        let zero_ptr = unsafe { init_qdht_ffi(t_col.as_ptr(), 0, scale_fwd, scale_inv, n_time) };
        assert!(zero_ptr.is_null(), "n_r=0 must return null");

        // ── free(null) is a no-op ─────────────────────────────────────────────
        unsafe { free_qdht_ffi(std::ptr::null_mut()); }

        unsafe { free_qdht_ffi(ptr); }
        let _ = data_cplx_orig; // suppress unused warning
    }

    #[test]
    fn test_gpu_ionization_numerical_equivalence() {
        if let Err(err) = crate::cuda::init_gpu_context() {
            println!("Skipping GPU Ionization test: GPU context not available: {}", err);
            return;
        }
        
        let e_min = 1.0e8;
        let e_max = 5.0e9;
        let n_points = 128;
        let rate_fn = |e: f64| 1.2 * (e / 1.0e9).exp();
        
        let ppt = PptIonizationRate::new(e_min, e_max, n_points, rate_fn);
        
        let mut fields = vec![0.0; 100];
        for i in 0..100 {
            fields[i] = e_min + (i as f64) * (e_max - e_min) / 101.0;
        }
        
        let mut gpu_rates = vec![0.0; 100];
        ppt.rate_vector(&fields, &mut gpu_rates).unwrap();
        
        for i in 0..100 {
            let cpu_rate = ppt.rate(fields[i]).unwrap();
            assert!((gpu_rates[i] - cpu_rate).abs() / cpu_rate < 1e-12);
        }
    }
}
