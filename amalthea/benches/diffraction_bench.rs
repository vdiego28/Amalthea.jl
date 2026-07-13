use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use amalthea::diffraction::Qdht;
use num_complex::Complex;

fn bench_qdht(c: &mut Criterion) {
    let mut group = c.benchmark_group("QDHT Transform");
    for &n_zeros in &[64, 256, 1024] {
        let qdht = Qdht::new(n_zeros, 10.0);
        let n_grid = qdht.n_r;
        let input = vec![Complex::new(1.0, 0.5); n_grid];
        let mut output = vec![Complex::new(0.0, 0.0); n_grid];

        group.bench_with_input(BenchmarkId::from_parameter(n_zeros), &n_zeros, |b, _| {
            b.iter(|| {
                qdht.transform(&input, &mut output);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_qdht);
criterion_main!(benches);
