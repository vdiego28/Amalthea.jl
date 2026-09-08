//! Private portable FFT bindings. These do not yet drive the resident engine.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner, num_complex::Complex64};
use std::sync::Arc;
mod solver;

fn valid_length(n: usize) -> PyResult<()> {
    if n == 0 || n > (1 << 24) {
        return Err(PyValueError::new_err("FFT length must be in 1..=2**24"));
    }
    Ok(())
}

fn valid_complex(values: &[Complex64], n: usize) -> PyResult<()> {
    if values.len() != n
        || values
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(PyValueError::new_err(
            "FFT input has wrong length or nonfinite values",
        ));
    }
    Ok(())
}

#[pyclass(module = "amalthea_native._native")]
struct ComplexFft {
    n: usize,
    forward_plan: Arc<dyn Fft<f64>>,
    inverse_plan: Arc<dyn Fft<f64>>,
    scratch: Vec<Complex64>,
    buffer: Vec<Complex64>,
}

#[pymethods]
impl ComplexFft {
    #[new]
    fn new(n: usize) -> PyResult<Self> {
        valid_length(n)?;
        let mut planner = FftPlanner::<f64>::new();
        let forward_plan = planner.plan_fft_forward(n);
        let inverse_plan = planner.plan_fft_inverse(n);
        let scratch_len = forward_plan
            .get_inplace_scratch_len()
            .max(inverse_plan.get_inplace_scratch_len());
        Ok(Self {
            n,
            forward_plan,
            inverse_plan,
            scratch: vec![Complex64::default(); scratch_len],
            buffer: vec![Complex64::default(); n],
        })
    }

    #[pyo3(signature = (values, inverse=false))]
    fn transform(&mut self, values: Vec<Complex64>, inverse: bool) -> PyResult<Vec<Complex64>> {
        valid_complex(&values, self.n)?;
        self.buffer.copy_from_slice(&values);
        let plan = if inverse {
            &self.inverse_plan
        } else {
            &self.forward_plan
        };
        plan.process_with_scratch(&mut self.buffer, &mut self.scratch);
        Ok(self.buffer.clone())
    }
}

#[pyclass(module = "amalthea_native._native")]
struct RealFft {
    n: usize,
    forward_plan: Arc<dyn RealToComplex<f64>>,
    inverse_plan: Arc<dyn ComplexToReal<f64>>,
    scratch: Vec<Complex64>,
    time: Vec<f64>,
    spectrum: Vec<Complex64>,
}

#[pymethods]
impl RealFft {
    #[new]
    fn new(n: usize) -> PyResult<Self> {
        valid_length(n)?;
        let mut planner = RealFftPlanner::<f64>::new();
        let forward_plan = planner.plan_fft_forward(n);
        let inverse_plan = planner.plan_fft_inverse(n);
        let scratch_len = forward_plan
            .get_scratch_len()
            .max(inverse_plan.get_scratch_len());
        Ok(Self {
            n,
            forward_plan,
            inverse_plan,
            scratch: vec![Complex64::default(); scratch_len],
            time: vec![0.0; n],
            spectrum: vec![Complex64::default(); n / 2 + 1],
        })
    }

    fn forward(&mut self, values: Vec<f64>) -> PyResult<Vec<Complex64>> {
        if values.len() != self.n || values.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "FFT input has wrong length or nonfinite values",
            ));
        }
        self.time.copy_from_slice(&values);
        self.forward_plan
            .process_with_scratch(&mut self.time, &mut self.spectrum, &mut self.scratch)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(self.spectrum.clone())
    }

    fn inverse(&mut self, values: Vec<Complex64>) -> PyResult<Vec<f64>> {
        valid_complex(&values, self.n / 2 + 1)?;
        self.spectrum.copy_from_slice(&values);
        // FFTW c2r ignores imaginary self-conjugate bins. Match that convention
        // explicitly; RealFFT otherwise rejects them after performing the FFT.
        self.spectrum[0].im = 0.0;
        if self.n % 2 == 0 {
            self.spectrum[self.n / 2].im = 0.0;
        }
        self.inverse_plan
            .process_with_scratch(&mut self.spectrum, &mut self.time, &mut self.scratch)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(self.time.clone())
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ComplexFft>()?;
    m.add_class::<RealFft>()?;
    m.add_function(wrap_pyfunction!(solver::solve, m)?)?;
    Ok(())
}
