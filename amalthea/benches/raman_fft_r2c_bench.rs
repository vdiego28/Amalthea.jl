// docs/dev/BACKLOG.md Phase J.3 / native-port/MATH.md §8.4 — timeboxed
// measure-first spike, Criterion-only, no production code touched by this
// file itself.
//
// Both resident FFT-convolution Raman kernels (native `rhs_mode_avg_env`
// Step 3c / `:SiO2`, and Julia's `RamanPolarEnv` fallback) pad the
// intensity `0.5·|E|²` and the impulse response `h` to a doubled grid and
// run a full complex-to-complex FFT convolution, even though both operands
// are mathematically real. This bench isolates the per-RHS-call hot
// path — forward transform, elementwise multiply by the (precomputed)
// response spectrum, inverse transform — and compares:
//   - "c2c": today's approach, `ComplexFft1d` on an `n_over`-length
//     `Complex<f64>` buffer with a zero imaginary part.
//   - "r2c": a `RealFft1d` pair on the same `n_over`-length real buffer,
//     spectrum length `n_over/2+1`.
// Response-spectrum setup (once per `solve`, not per RHS call) is
// deliberately excluded from the timed region.
//
// Decision rule (S1.6/S5.1 add/measure/revert discipline): only wire this
// into `native.rs`/`Nonlinear.jl` production code if r2c clears a
// meaningful speedup (>1.3x, matching the S1 item 6 SoA-spike bar) at the
// grid sizes real configs actually hit.
use amalthea::fftw::{ComplexFft1d, FftwApi, RealFft1d};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use num_complex::Complex;
use std::hint::black_box;

const FFTW_ESTIMATE: u32 = 1 << 6;

fn load_api() -> FftwApi {
    let path = std::env::var("AMALTHEA_FFTW_LIB").ok();
    FftwApi::load(path.as_deref()).expect(
        "libfftw3 not found — set AMALTHEA_FFTW_LIB or install libfftw3-dev \
         (this bench needs the same FFTW native.rs dlopens at runtime)",
    )
}

fn bench_raman_fft_conv(c: &mut Criterion) {
    let api = load_api();
    let mut group = c.benchmark_group("raman_fft_conv_step3c");

    // n_time_over candidates spanning the range real EnvGrid configs land
    // in (power-of-two oversampled grids, MATH.md §5.3/§8.4); n_over is the
    // doubled padding length actually FFT'd in Step 3c / RamanPolarEnv.
    for &n_time_over in &[1024usize, 2048, 4096, 8192, 16384, 32768, 65536] {
        let n_over = 2 * n_time_over;

        // ── c2c setup (mirrors set_raman_fft_params / RamanPolarEnv ctor) ──
        let c2c_plan = ComplexFft1d::new(&api, n_over, FFTW_ESTIMATE);
        let mut hw_c2c = vec![Complex::new(0.0, 0.0); n_over];
        {
            // synthetic causal, decaying "response" — shape doesn't matter
            // for timing, only that it's real and localized like a real
            // Raman response.
            let mut h = vec![Complex::new(0.0, 0.0); n_over];
            for i in 0..n_time_over {
                let t = i as f64;
                h[i] = Complex::new((-t / 50.0).exp() * (t / 10.0).sin(), 0.0);
            }
            let mut hw_scratch = vec![Complex::new(0.0, 0.0); n_over];
            c2c_plan.forward(&mut h, &mut hw_scratch);
            hw_c2c = hw_scratch;
        }
        let mut e2_c2c = vec![Complex::new(0.0, 0.0); n_over];
        for i in 0..n_time_over {
            e2_c2c[i] = Complex::new(1.0 + (i as f64 * 0.001).sin().abs(), 0.0);
        }
        let mut ew_c2c = vec![Complex::new(0.0, 0.0); n_over];

        group.bench_with_input(BenchmarkId::new("c2c", n_time_over), &n_time_over, |b, _| {
            b.iter(|| {
                let e2 = black_box(&mut e2_c2c);
                c2c_plan.forward(e2, &mut ew_c2c);
                for k in 0..ew_c2c.len() {
                    ew_c2c[k] *= hw_c2c[k];
                }
                c2c_plan.inverse(&mut ew_c2c, e2);
                black_box(&e2_c2c);
            });
        });

        // ── r2c setup ────────────────────────────────────────────────────
        let r2c_plan = RealFft1d::new(&api, n_over, FFTW_ESTIMATE);
        let nspec = r2c_plan.nspec();
        let mut hw_r2c = vec![Complex::new(0.0, 0.0); nspec];
        {
            let mut h = vec![0.0f64; n_over];
            for i in 0..n_time_over {
                let t = i as f64;
                h[i] = (-t / 50.0).exp() * (t / 10.0).sin();
            }
            let mut hw_scratch = vec![Complex::new(0.0, 0.0); nspec];
            r2c_plan.forward(&mut h, &mut hw_scratch);
            hw_r2c = hw_scratch;
        }
        let mut e2_r2c = vec![0.0f64; n_over];
        for i in 0..n_time_over {
            e2_r2c[i] = 1.0 + (i as f64 * 0.001).sin().abs();
        }
        let mut ew_r2c = vec![Complex::new(0.0, 0.0); nspec];

        group.bench_with_input(BenchmarkId::new("r2c", n_time_over), &n_time_over, |b, _| {
            b.iter(|| {
                let e2 = black_box(&mut e2_r2c);
                r2c_plan.forward(e2, &mut ew_r2c);
                for k in 0..ew_r2c.len() {
                    ew_r2c[k] *= hw_r2c[k];
                }
                r2c_plan.inverse(&mut ew_r2c, e2);
                black_box(&e2_r2c);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_raman_fft_conv);
criterion_main!(benches);
