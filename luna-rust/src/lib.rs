pub mod grid;
pub mod ffi;
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
    use super::dispersion::{ChebyshevDispersion, SellmeierGas};
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
