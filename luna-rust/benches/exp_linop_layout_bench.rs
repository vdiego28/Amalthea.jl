// BACKLOG.md S1 item 6 (SUGGESTIONS.md item 3b) — timeboxed spike only.
// Compares AoS (interleaved Complex<f64>, today's native.rs layout) against
// SoA (separate re/im Vec<f64>) for the two exp-linop hot-loop shapes:
// apply_prop_cached's `y[i] *= factors[i]` and weaknorm_c64's norm_sqr
// reduction. Decision rule: only pursue the invasive guru-split-DFT FFTW
// plan rewrite (which would make SoA the resident layout end-to-end) if SoA
// beats AoS by >1.3x on both shapes; otherwise record the negative result
// and close.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex;
use std::hint::black_box;

fn apply_prop_aos(y: &mut [Complex<f64>], factors: &[Complex<f64>]) {
    for i in 0..y.len() {
        y[i] *= factors[i];
    }
}

fn apply_prop_soa(yr: &mut [f64], yi: &mut [f64], fr: &[f64], fi: &[f64]) {
    for i in 0..yr.len() {
        let (a, b, c, d) = (yr[i], yi[i], fr[i], fi[i]);
        yr[i] = a * c - b * d;
        yi[i] = a * d + b * c;
    }
}

fn weaknorm_aos(y: &[Complex<f64>]) -> f64 {
    let mut s = 0.0f64;
    for v in y {
        s += v.norm_sqr();
    }
    s
}

fn weaknorm_soa(yr: &[f64], yi: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for i in 0..yr.len() {
        s += yr[i] * yr[i] + yi[i] * yi[i];
    }
    s
}

fn bench_apply_prop(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp_linop_apply_prop");
    for &n in &[2000usize, 8000, 16000] {
        let y0: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(1.0 + i as f64 * 1e-6, -0.5))
            .collect();
        let factors: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.001).cos(), (i as f64 * 0.001).sin()))
            .collect();
        let (fr, fi): (Vec<f64>, Vec<f64>) = factors.iter().map(|c| (c.re, c.im)).collect();

        group.bench_with_input(BenchmarkId::new("aos", n), &n, |b, _| {
            let mut y = y0.clone();
            b.iter(|| {
                apply_prop_aos(black_box(&mut y), black_box(&factors));
                black_box(&y);
            });
        });

        group.bench_with_input(BenchmarkId::new("soa", n), &n, |b, _| {
            let mut yr: Vec<f64> = y0.iter().map(|c| c.re).collect();
            let mut yi: Vec<f64> = y0.iter().map(|c| c.im).collect();
            b.iter(|| {
                apply_prop_soa(black_box(&mut yr), black_box(&mut yi), black_box(&fr), black_box(&fi));
                black_box((&yr, &yi));
            });
        });
    }
    group.finish();
}

fn bench_weaknorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp_linop_weaknorm");
    for &n in &[2000usize, 8000, 16000] {
        let y0: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(1.0 + i as f64 * 1e-6, -0.5))
            .collect();
        let (yr, yi): (Vec<f64>, Vec<f64>) = y0.iter().map(|c| (c.re, c.im)).collect();

        group.bench_with_input(BenchmarkId::new("aos", n), &n, |b, _| {
            b.iter(|| black_box(weaknorm_aos(black_box(&y0))));
        });

        group.bench_with_input(BenchmarkId::new("soa", n), &n, |b, _| {
            b.iter(|| black_box(weaknorm_soa(black_box(&yr), black_box(&yi))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_apply_prop, bench_weaknorm);
criterion_main!(benches);
