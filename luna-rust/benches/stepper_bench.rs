use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use luna_rust::stepper::Dopri5Stepper;
use num_complex::Complex;

fn bench_stepper(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dopri5 Stepper Step");
    let lin_op = |_z1: f64, _z2: f64, y_in: &[Complex<f64>], y_out: &mut [Complex<f64>]| {
        y_out.copy_from_slice(y_in);
    };
    let rhs = |_z: f64, y_in: &[Complex<f64>], dy_out: &mut [Complex<f64>]| {
        for i in 0..y_in.len() {
            dy_out[i] = -0.5 * y_in[i];
        }
    };

    for &size in &[512, 2048] {
        let mut stepper = Dopri5Stepper::new(size, 1e-4, 1e-6);
        let mut y = vec![Complex::new(1.0, 0.0); size];
        let mut fsal_active = false;

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let _ = stepper.step(&mut y, 0.0, 0.1, lin_op, rhs, &mut fsal_active);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_stepper);
criterion_main!(benches);
