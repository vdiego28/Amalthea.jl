use super::*;

#[test]
fn portable_real_mode_average_resolves_kerr_harmonics() {
    let n = 16;
    let no = 32;
    let ns = n / 2 + 1;
    let mut sim =
        CpuNativeSim::new_portable(&vec![Complex::new(0.0, 0.0); ns], n, no, true).unwrap();
    let ones = vec![1.0; no];
    let zeros = vec![0.0; ns];
    let active = vec![1u8; ns];
    assert_eq!(
        unsafe {
            sim.set_mode_avg_params(
                n,
                no,
                ones.as_ptr(),
                ones.as_ptr(),
                active.as_ptr(),
                zeros.as_ptr(),
                ones.as_ptr(),
                ones.as_ptr(),
                0.2,
                1.0,
                1.0,
            )
        },
        0
    );
    let mut field = vec![Complex::new(0.0, 0.0); ns];
    field[1] = Complex::new(n as f64 / 2.0, 0.0);
    sim.rhs_mode_avg_real(0, &field);
    let mut expected = vec![Complex::new(0.0, 0.0); ns];
    expected[1] = Complex::new(0.0, 0.2 * 3.0 * n as f64 / 8.0);
    expected[3] = Complex::new(0.0, 0.2 * n as f64 / 8.0);
    let err = sim.ks[0]
        .iter()
        .zip(&expected)
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(err < 1e-13, "Kerr harmonic error {err:e}");
    assert!(expected[3].norm() > 0.1);
    assert!(sim.fftw_api.is_none());
}

#[test]
fn portable_hilbert_has_correct_intensity_and_reuses_inputs() {
    let n = 64;
    let fft = ComplexFft1d::portable(n);
    let mut scratch = FftScratch::default();
    fft.prepare(&mut scratch);
    let field: Vec<_> = (0..n)
        .map(|j| {
            let x = std::f64::consts::TAU * j as f64 / n as f64;
            0.3 + x.cos() + 0.2 * (3.0 * x).sin()
        })
        .collect();
    let mut a = vec![Complex::new(0.0, 0.0); n];
    let mut b = a.clone();
    let mut out = vec![0.0; n];
    for _ in 0..3 {
        hilbert_intensity(
            &fft,
            &mut scratch,
            &mut a,
            &mut b,
            &field,
            &mut out,
            1.0 / n as f64,
        );
        for j in 0..n {
            let x = std::f64::consts::TAU * j as f64 / n as f64;
            let imaginary = x.sin() - 0.2 * (3.0 * x).cos();
            let expected = 0.5 * (field[j] * field[j] + imaginary * imaginary);
            assert!((out[j] - expected).abs() < 1e-13);
        }
    }
}

#[test]
fn portable_modal_worker_scratch_is_independent() {
    for is_real in [false, true] {
        let n = 32;
        let no = 64;
        let nm = 2;
        let ns = if is_real { n / 2 + 1 } else { n };
        let mut sim =
            CpuNativeSim::new_portable(&vec![Complex::new(0.0, 0.0); ns * nm], n, no, is_real)
                .unwrap();
        // Fixed-node evaluation does not load or call libcubature.
        sim.n_modes = nm;
        sim.npol = 2;
        sim.n_spec = ns;
        sim.n_time = n;
        sim.n_time_over = no;
        sim.n_spec_over = if is_real { no / 2 + 1 } else { no };
        sim.modal_a = 1.0;
        sim.modal_full = true;
        sim.modal_unm = vec![2.404825557695773, 5.520078110286311];
        sim.modal_order = vec![0, 0];
        sim.modal_kind = vec![0, 0];
        sim.modal_phi = vec![0.2, 1.1];
        sim.modal_inv_sqrt_n = vec![1.0, 0.7];
        sim.modal_pol_select = vec![0, 1];
        sim.modal_nlfac = vec![Complex::new(0.0, 0.4); ns];
        sim.modal_kerr_fac = 0.2;
        sim.towin = (0..no)
            .map(|j| 0.8 + 0.2 * (j as f64 * 0.17).cos())
            .collect();
        sim.modal_emega = (0..ns * nm)
            .map(|j| Complex::new((j as f64 * 0.21).sin(), (j as f64 * 0.13).cos()))
            .collect();
        if is_real {
            let omega = [1.2];
            let tau = [2.1];
            let amp = [0.3];
            assert_eq!(
                unsafe {
                    sim.set_raman_params(omega.as_ptr(), tau.as_ptr(), amp.as_ptr(), 1, 0.1, 1.0, 0)
                },
                0
            );
        }
        let coords: Vec<_> = (0..17)
            .map(|j| (0.1 + 0.8 * j as f64 / 17.0, 0.1 * j as f64))
            .collect();
        let fdim = 2 * ns * nm;
        let mut serial = vec![0.0; fdim * coords.len()];
        sim.modal_eval_pairs(&coords, &mut serial, fdim);
        assert!(serial.iter().map(|v| v * v).sum::<f64>() > 1e-8);
        sim.n_threads = 4;
        for _ in 0..5 {
            let mut parallel = vec![0.0; serial.len()];
            sim.modal_eval_pairs(&coords, &mut parallel, fdim);
            assert_eq!(parallel, serial);
            assert_eq!(sim.modal_scratch_pool.len(), 4);
        }
        assert!(sim.fftw_api.is_none());
    }
}

#[test]
fn portable_real_raman_matches_direct_causal_sum_and_resets_tail() {
    let n = 32;
    let ns = n / 2 + 1;
    let dt = 0.03;
    let mut sim = CpuNativeSim::new_portable(&vec![Complex::new(0., 0.); ns], n, n, true).unwrap();
    let ones = vec![1.; n];
    let zeros = vec![0.; ns];
    let active = vec![1u8; ns];
    unsafe {
        assert_eq!(
            sim.set_mode_avg_params(
                n,
                n,
                ones.as_ptr(),
                ones.as_ptr(),
                active.as_ptr(),
                ones.as_ptr(),
                zeros.as_ptr(),
                ones.as_ptr(),
                0.,
                1.,
                1.
            ),
            0
        );
    }
    let samples: Vec<_> = (0..n)
        .map(|i| (i as f64 * dt * 2.1).sin() * (-(i as f64) * dt * 0.7).exp())
        .collect();
    let mut padded = samples.clone();
    padded.resize(2 * n, 0.);
    sim.configure_raman_samples(padded, dt, 1.7);
    let mut scratch = FftScratch::default();
    let fft = RealFft1d::portable(n);
    fft.prepare(&mut scratch);
    for phase in [0., 0.4, 0., 0.9] {
        let time: Vec<_> = (0..n)
            .map(|i| 0.4 + (std::f64::consts::TAU * i as f64 / n as f64 + phase).cos())
            .collect();
        let mut input = time.clone();
        let mut spectrum = vec![Complex::new(0., 0.); ns];
        fft.forward(&mut input, &mut spectrum, &mut scratch);
        sim.rhs_mode_avg_real(0, &spectrum);
        let mut response = vec![0.; n];
        for i in 0..n {
            let sum: f64 = (0..=i).map(|j| samples[i - j] * time[j] * time[j]).sum();
            response[i] = 1.7 * time[i] * sum * dt;
        }
        let mut expected = vec![Complex::new(0., 0.); ns];
        fft.forward(&mut response, &mut expected, &mut scratch);
        let norm = expected.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        let error = sim.ks[0]
            .iter()
            .zip(expected)
            .map(|(a, b)| (*a - b).norm_sqr())
            .sum::<f64>()
            .sqrt()
            / norm;
        assert!(
            norm > 0.1 && error < 1e-13,
            "Raman causal sum error {error:e}"
        );
    }
}
