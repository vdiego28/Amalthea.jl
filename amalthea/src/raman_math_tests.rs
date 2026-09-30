//! Analytic checks independent of the matrix-exponential forcing formula.
use super::*;

// Composite Gauss-Legendre integration of the physical impulse response.
// These nodes/weights are independent of the production coefficient series.
fn integrate(f: impl Fn(f64) -> f64, end: f64) -> f64 {
    let nodes = [
        0.1834346424956498,
        0.5255324099163290,
        0.7966664774136267,
        0.9602898564975363,
    ];
    let weights = [
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763,
    ];
    let half = end / 16.0;
    let mut sum = 0.0;
    for panel in 0..8 {
        let mid = (2 * panel + 1) as f64 * half;
        for (&node, &weight) in nodes.iter().zip(&weights) {
            sum += weight * (f(mid - half * node) + f(mid + half * node));
        }
    }
    half * sum
}

fn impulse(osc: &RamanOscillator, t: f64) -> (f64, f64) {
    let scale = osc.coupling * (-osc.gamma * t).exp();
    let (sin, cos) = (osc.omega * t).sin_cos();
    (scale * sin, scale * (osc.omega * cos - osc.gamma * sin))
}

fn relative(actual: f64, expected: f64) -> f64 {
    assert!(actual.is_finite() && expected.is_finite());
    assert_ne!(expected, 0.0);
    (actual - expected).abs() / expected.abs()
}

#[test]
fn forcing_coefficients_match_impulse_integrals() {
    let mut worst: f64 = 0.0;
    let mut worst_case = String::new();
    // The damping-dominated cases also probe both sides of the branch boundary.
    for &(omega, gamma) in &[(1.0, 0.0), (1.0, 0.3), (0.01, 1.0), (8e12, 1.1e11)] {
        for phase in [1e-8, 1e-6, 1e-4, 0.01, 0.499999, 0.500001, 1.0, 4.0] {
            let dt = phase / (omega + gamma);
            for coupling in [0.7, -0.2, 1e-50] {
                let osc = RamanOscillator {
                    omega,
                    gamma,
                    coupling,
                };
                let c = PrecomputedStepCoeffs::compute(&osc, dt);
                let actual = [c.b0_1, c.b0_2, c.b1_1, c.b1_2];
                let expected = [
                    integrate(|u| impulse(&osc, u).0 * (u / dt), dt),
                    integrate(|u| impulse(&osc, u).1 * (u / dt), dt),
                    integrate(|u| impulse(&osc, u).0 * (1.0 - u / dt), dt),
                    integrate(|u| impulse(&osc, u).1 * (1.0 - u / dt), dt),
                ];
                for (index, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                    let error = relative(a, e);
                    if !error.is_finite() || error > worst {
                        worst = error;
                        worst_case = format!(
                            "omega={omega}, gamma={gamma}, phase={phase}, component={index}"
                        );
                    }
                }
            }
        }
    }
    println!("Raman forcing worst relative error: {worst:e} ({worst_case})");
    assert!(
        worst.is_finite() && worst < 2e-13,
        "{worst:e}: {worst_case}"
    );
}

#[test]
fn homogeneous_map_matches_damped_sinusoid() {
    let mut worst: f64 = 0.0;
    for &(omega, gamma) in &[(1.0, 0.0), (1.0, 0.3), (8e12, 1.1e11)] {
        let osc = RamanOscillator {
            omega,
            gamma,
            coupling: 0.7,
        };
        let t0 = 0.37 / omega;
        let initial = impulse(&osc, t0);
        for phase in [1e-8, 0.1, 0.7] {
            let dt = phase / omega;
            let c = PrecomputedStepCoeffs::compute(&osc, dt);
            let actual = (
                c.a11 * initial.0 + c.a12 * initial.1,
                c.a21 * initial.0 + c.a22 * initial.1,
            );
            let expected = impulse(&osc, t0 + dt);
            worst = worst.max(relative(actual.0, expected.0));
            worst = worst.max(relative(actual.1, expected.1));
        }
    }
    println!("Raman homogeneous worst relative error: {worst:e}");
    assert!(worst < 2e-14);
}

#[test]
fn zero_step_and_zero_frequency_have_finite_limits() {
    for omega in [0.0, 1.0] {
        let osc = RamanOscillator {
            omega,
            gamma: 0.3,
            coupling: 0.7,
        };
        let c = PrecomputedStepCoeffs::compute(&osc, 0.0);
        assert_eq!([c.a11, c.a12, c.a21, c.a22], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!([c.b0_1, c.b0_2, c.b1_1, c.b1_2], [0.0; 4]);
    }
    for gamma in [0.0, 0.3, 3.0] {
        let osc = RamanOscillator {
            omega: 0.0,
            gamma,
            coupling: 0.7,
        };
        let dt = 0.4;
        let c = PrecomputedStepCoeffs::compute(&osc, dt);
        let decay = (-gamma * dt).exp();
        let expected = [
            decay * (1.0 + gamma * dt),
            dt * decay,
            -gamma * gamma * dt * decay,
            decay * (1.0 - gamma * dt),
        ];
        for (&a, &e) in [c.a11, c.a12, c.a21, c.a22].iter().zip(&expected) {
            assert!((a - e).abs() <= 2e-15 * e.abs().max(1.0));
        }
        assert_eq!([c.b0_1, c.b0_2, c.b1_1, c.b1_2], [0.0; 4]);
    }
}

#[test]
fn affine_drive_trajectory_matches_direct_convolution() {
    let mut worst: f64 = 0.0;
    #[cfg(target_arch = "x86_64")]
    println!(
        "Raman analytic trajectory AVX2 enabled: {}",
        is_x86_feature_detected!("avx2")
    );
    for &(omega, gamma, end) in &[
        (1.0, 0.0, 1e-4),
        (1.0, 0.3, 1e-4),
        (1.0, 0.3, 2.0),
        (8e12, 1.1e11, 2.5e-13),
    ] {
        // Five unequal oscillators exercise the AVX2 group and scalar tail.
        let oscillators: Vec<_> = (0..5)
            .map(|i| RamanOscillator {
                omega: omega * (1.0 + 0.1 * i as f64),
                gamma,
                coupling: 0.7 / (1 + i) as f64,
            })
            .collect();
        for steps in [2, 16, 32, 64] {
            let dt = end / steps as f64;
            let drive: Vec<_> = (0..=steps)
                .map(|i| 0.2 + 0.8 * i as f64 / steps as f64)
                .collect();
            let expected: Vec<_> = (0..=steps)
                .map(|i| {
                    let t = i as f64 * dt;
                    oscillators
                        .iter()
                        .map(|osc| {
                            integrate(|u| impulse(osc, u).0 * (0.2 + 0.8 * (t - u) / end), t)
                        })
                        .sum::<f64>()
                })
                .collect();
            let scale = expected.iter().copied().map(f64::abs).fold(0.0, f64::max);
            assert!(
                scale > 1e-12 / omega,
                "feature effect must exceed the relative error floor"
            );
            let check = |actual: &[f64]| {
                assert!(actual.iter().chain(&expected).all(|v| v.is_finite()));
                actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, e)| (a - e).abs() / scale)
                    .fold(0.0, f64::max)
            };
            let mut solver = TimeDomainRamanSolver::new(oscillators.clone(), dt);
            let mut actual = vec![0.0; steps + 1];
            solver.solve_scalar(&drive, &mut actual);
            worst = worst.max(check(&actual));
            // solve resets the oscillator state on every RHS call.
            solver.solve_scalar(&drive, &mut actual);
            worst = worst.max(check(&actual));
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") {
                unsafe { solver.solve_avx2(&drive, &mut actual) };
                worst = worst.max(check(&actual));
            }
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe { solver.solve_neon(&drive, &mut actual) };
                worst = worst.max(check(&actual));
            }
        }
    }
    println!("Raman affine trajectory worst relative error: {worst:e}");
    assert!(worst < 2e-12);
}

#[test]
fn constant_drive_matches_undamped_closed_form() {
    let osc = RamanOscillator {
        omega: 1.0,
        gamma: 0.0,
        coupling: 0.7,
    };
    let mut worst: f64 = 0.0;
    for end in [1e-4, 2.0] {
        for steps in [2, 16, 32, 64] {
            let dt = end / steps as f64;
            let mut solver = TimeDomainRamanSolver::new(vec![osc], dt);
            let mut actual = vec![0.0; steps + 1];
            solver.solve_scalar(&vec![1.0; steps + 1], &mut actual);
            let expected = 2.0 * osc.coupling / osc.omega * (0.5 * osc.omega * end).sin().powi(2);
            worst = worst.max(relative(actual[steps], expected));
        }
    }
    println!("Raman constant trajectory worst relative error: {worst:e}");
    assert!(worst < 2e-12);
}
