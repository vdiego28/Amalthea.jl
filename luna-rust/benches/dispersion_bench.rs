use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use luna_rust::dispersion::ChebyshevDispersion;

fn bench_dispersion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Chebyshev Dispersion Evaluation");
    let ω_min = 1.0;
    let ω_max = 10.0;
    let degree = 8;
    
    let cheb = ChebyshevDispersion::fit(ω_min, ω_max, degree, |ω| {
        3.0 * ω * ω - 4.0 * ω + 5.0
    });
    
    for &n_omega in &[512, 2048] {
        let grid: Vec<f64> = (0..n_omega)
            .map(|i| ω_min + (i as f64) * (ω_max - ω_min) / (n_omega as f64))
            .collect();
            
        group.bench_with_input(
            BenchmarkId::from_parameter(n_omega),
            &n_omega,
            |b, _| {
                b.iter(|| {
                    for &ω in &grid {
                        let _ = cheb.evaluate_derivatives(ω);
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_dispersion);
criterion_main!(benches);
