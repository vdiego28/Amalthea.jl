//! Handle-local FFT selection with immutable plans and exclusively owned scratch.
//! Both directions are unnormalized; callers retain the physical normalization.
use crate::fftw::{self, FftwApi};
use num_complex::Complex64 as C;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Scratch may be reused by different plan sizes, but never by concurrent calls.
#[derive(Default)]
pub struct FftScratch {
    work: Vec<C>,
    real: Vec<f64>,
    spectrum: Vec<C>,
}

impl FftScratch {
    fn reserve(&mut self, work: usize, real: usize, spectrum: usize) {
        if self.work.len() < work {
            self.work.resize(work, C::default());
        }
        if self.real.len() < real {
            self.real.resize(real, 0.0);
        }
        if self.spectrum.len() < spectrum {
            self.spectrum.resize(spectrum, C::default());
        }
    }
}

pub struct PortableComplex {
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

pub enum ComplexFft1d {
    Fftw(fftw::ComplexFft1d),
    Portable(PortableComplex),
}

impl PortableComplex {
    pub fn new(n: usize) -> Self {
        assert!(n > 0);
        let mut planner = FftPlanner::new();
        Self {
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        }
    }
    pub fn prepare(&self, scratch: &mut FftScratch) {
        scratch.reserve(
            self.forward
                .get_inplace_scratch_len()
                .max(self.inverse.get_inplace_scratch_len()),
            0,
            0,
        );
    }
    pub fn forward(&self, input: &[C], output: &mut [C], scratch: &mut FftScratch) {
        Self::process(&self.forward, input, output, scratch);
    }
    pub fn inverse(&self, input: &[C], output: &mut [C], scratch: &mut FftScratch) {
        Self::process(&self.inverse, input, output, scratch);
    }
    fn process(plan: &Arc<dyn Fft<f64>>, input: &[C], output: &mut [C], scratch: &mut FftScratch) {
        assert_eq!(input.len(), plan.len());
        assert_eq!(output.len(), plan.len());
        scratch.reserve(plan.get_inplace_scratch_len(), 0, 0);
        output.copy_from_slice(input);
        plan.process_with_scratch(output, &mut scratch.work);
    }
}
impl ComplexFft1d {
    pub fn new(api: &FftwApi, n: usize, flags: u32) -> Self {
        Self::Fftw(fftw::ComplexFft1d::new(api, n, flags))
    }
    pub fn portable(n: usize) -> Self {
        Self::Portable(PortableComplex::new(n))
    }
    pub fn prepare(&self, scratch: &mut FftScratch) {
        if let Self::Portable(plan) = self {
            plan.prepare(scratch);
        }
    }
    pub fn forward(&self, input: &mut [C], output: &mut [C], scratch: &mut FftScratch) {
        match self {
            Self::Fftw(plan) => plan.forward(input, output),
            Self::Portable(plan) => plan.forward(input, output, scratch),
        }
    }
    pub fn inverse(&self, input: &mut [C], output: &mut [C], scratch: &mut FftScratch) {
        match self {
            Self::Fftw(plan) => plan.inverse(input, output),
            Self::Portable(plan) => plan.inverse(input, output, scratch),
        }
    }
}
pub struct PortableReal {
    forward: Arc<dyn RealToComplex<f64>>,
    inverse: Arc<dyn ComplexToReal<f64>>,
}

pub enum RealFft1d {
    Fftw(fftw::RealFft1d),
    Portable(PortableReal),
}

impl PortableReal {
    pub fn new(n: usize) -> Self {
        assert!(n > 0);
        let mut planner = RealFftPlanner::new();
        Self {
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        }
    }
    pub fn nspec(&self) -> usize {
        self.forward.len() / 2 + 1
    }
    pub fn prepare(&self, scratch: &mut FftScratch) {
        scratch.reserve(
            self.forward
                .get_scratch_len()
                .max(self.inverse.get_scratch_len()),
            self.forward.len(),
            self.nspec(),
        );
    }
    pub fn forward(&self, input: &[f64], output: &mut [C], scratch: &mut FftScratch) {
        let n = self.forward.len();
        assert_eq!(input.len(), n);
        assert_eq!(output.len(), self.nspec());
        self.prepare(scratch);
        scratch.real[..n].copy_from_slice(input);
        self.forward
            .process_with_scratch(&mut scratch.real[..n], output, &mut scratch.work)
            .expect("validated real FFT dimensions");
    }
    pub fn inverse(&self, input: &[C], output: &mut [f64], scratch: &mut FftScratch) {
        let n = self.inverse.len();
        let ns = self.nspec();
        assert_eq!(input.len(), ns);
        assert_eq!(output.len(), n);
        self.prepare(scratch);
        let spectrum = &mut scratch.spectrum[..ns];
        spectrum.copy_from_slice(input);
        // FFTW ignores imaginary self-conjugate bins without mutating input.
        spectrum[0].im = 0.0;
        if n % 2 == 0 {
            spectrum[n / 2].im = 0.0;
        }
        self.inverse
            .process_with_scratch(spectrum, output, &mut scratch.work)
            .expect("validated real FFT dimensions and endpoints");
    }
}
impl RealFft1d {
    pub fn new(api: &FftwApi, n: usize, flags: u32) -> Self {
        Self::Fftw(fftw::RealFft1d::new(api, n, flags))
    }
    pub fn portable(n: usize) -> Self {
        Self::Portable(PortableReal::new(n))
    }
    pub fn prepare(&self, scratch: &mut FftScratch) {
        if let Self::Portable(plan) = self {
            plan.prepare(scratch);
        }
    }
    pub fn forward(&self, input: &mut [f64], output: &mut [C], scratch: &mut FftScratch) {
        match self {
            Self::Fftw(plan) => plan.forward(input, output),
            Self::Portable(plan) => plan.forward(input, output, scratch),
        }
    }
    pub fn inverse(&self, input: &mut [C], output: &mut [f64], scratch: &mut FftScratch) {
        match self {
            Self::Fftw(plan) => plan.inverse(input, output),
            Self::Portable(plan) => plan.inverse(input, output, scratch),
        }
    }
}
impl RealFft1d {
    pub fn nspec(&self) -> usize {
        match self {
            Self::Fftw(plan) => plan.nspec(),
            Self::Portable(plan) => plan.nspec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dft(x: &[C], inverse: bool) -> Vec<C> {
        (0..x.len())
            .map(|k| {
                x.iter()
                    .enumerate()
                    .map(|(j, v)| {
                        *v * C::from_polar(
                            1.0,
                            if inverse { 1.0 } else { -1.0 }
                                * std::f64::consts::TAU
                                * (j * k) as f64
                                / x.len() as f64,
                        )
                    })
                    .sum()
            })
            .collect()
    }
    fn close(x: &[C], y: &[C]) {
        let error = x
            .iter()
            .zip(y)
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f64>()
            .sqrt()
            / y.iter()
                .map(|v| v.norm_sqr())
                .sum::<f64>()
                .sqrt()
                .max(1e-30);
        assert!(error < 1e-13, "relative error {error:e}");
    }
    #[test]
    fn portable_direct_dft_and_input_preservation() {
        for n in [1, 2, 3, 7, 8, 15, 16, 31, 32, 63] {
            let x: Vec<C> = (0..n)
                .map(|j| C::new((j as f64 * 0.7).sin(), (j as f64 * 0.3).cos()))
                .collect();
            let plan = PortableComplex::new(n);
            let mut work = FftScratch::default();
            plan.prepare(&mut work);
            let ptr = work.work.as_ptr();
            let mut y = vec![C::default(); n];
            let mut z = y.clone();
            for _ in 0..3 {
                plan.forward(&x, &mut y, &mut work);
                close(&y, &dft(&x, false));
                plan.inverse(&y, &mut z, &mut work);
                close(&z, &x.iter().map(|v| *v * n as f64).collect::<Vec<_>>());
                assert_eq!(ptr, work.work.as_ptr());
            }
            let plan = RealFft1d::portable(n);
            let mut real: Vec<_> = x.iter().map(|v| v.re + 0.5).collect();
            let original = real.clone();
            let mut spec = vec![C::default(); n / 2 + 1];
            let mut out = vec![0.0; n];
            plan.prepare(&mut work);
            let ptrs = (
                work.work.as_ptr(),
                work.real.as_ptr(),
                work.spectrum.as_ptr(),
            );
            for _ in 0..3 {
                plan.forward(&mut real, &mut spec, &mut work);
                assert_eq!(real, original);
                close(
                    &spec,
                    &dft(
                        &original.iter().map(|v| C::new(*v, 0.0)).collect::<Vec<_>>(),
                        false,
                    )[..n / 2 + 1],
                );
                spec[0].im = 4.0;
                if n % 2 == 0 {
                    spec[n / 2].im = -7.0;
                }
                let preserved = spec.clone();
                plan.inverse(&mut spec, &mut out, &mut work);
                assert_eq!(spec, preserved);
                close(
                    &out.iter().map(|v| C::new(*v, 0.0)).collect::<Vec<_>>(),
                    &original
                        .iter()
                        .map(|v| C::new(*v * n as f64, 0.0))
                        .collect::<Vec<_>>(),
                );
                assert_eq!(
                    ptrs,
                    (
                        work.work.as_ptr(),
                        work.real.as_ptr(),
                        work.spectrum.as_ptr()
                    )
                );
            }
        }
    }
    #[test]
    fn shared_portable_plan_distinct_worker_scratch() {
        use rayon::prelude::*;
        let plan = ComplexFft1d::portable(128);
        let inputs: Vec<Vec<C>> = (0..8)
            .map(|i| {
                (0..128)
                    .map(|j| C::new(((i + j) as f64 * 0.17).sin(), 0.0))
                    .collect()
            })
            .collect();
        let mut expected = vec![vec![C::default(); 128]; 8];
        let mut scratch = FftScratch::default();
        for (input, output) in inputs.iter().zip(&mut expected) {
            plan.forward(&mut input.clone(), output, &mut scratch);
        }
        let mut work: Vec<_> = (0..8)
            .map(|_| {
                let mut s = FftScratch::default();
                plan.prepare(&mut s);
                s
            })
            .collect();
        let mut actual = vec![vec![C::default(); 128]; 8];
        for _ in 0..5 {
            actual
                .par_iter_mut()
                .zip(inputs.par_iter())
                .zip(work.par_iter_mut())
                .for_each(|((out, input), s)| plan.forward(&mut input.clone(), out, s));
            assert_eq!(actual, expected);
        }
    }
}
