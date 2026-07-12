use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use luna_rust::raman::{RamanOscillator, TimeDomainRamanSolver};

fn bench_raman(c: &mut Criterion) {
    let mut group = c.benchmark_group("Raman ADE Solver");
    let osc = RamanOscillator {
        omega: 1e12,
        gamma: 2e11,
        coupling: 0.1,
    };
    let dt = 1e-14;

    for &nt in &[1000, 4000, 16000] {
        let mut solver = TimeDomainRamanSolver::new(vec![osc], dt);
        let intensity = vec![1e15; nt];
        let mut raman_pol = vec![0.0; nt];

        group.bench_with_input(BenchmarkId::from_parameter(nt), &nt, |b, _| {
            b.iter(|| {
                solver.solve(&intensity, &mut raman_pol);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_raman);
criterion_main!(benches);
