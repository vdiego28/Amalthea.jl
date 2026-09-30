//! Private bindings sharing the resident engine's portable transforms.
use amalthea::transforms::{FftScratch, PortableComplex, PortableReal};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustfft::num_complex::Complex64;
mod points;
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
    plan: PortableComplex,
    scratch: FftScratch,
    buffer: Vec<Complex64>,
}

#[pymethods]
impl ComplexFft {
    #[new]
    fn new(n: usize) -> PyResult<Self> {
        valid_length(n)?;
        let plan = PortableComplex::new(n);
        let mut scratch = FftScratch::default();
        plan.prepare(&mut scratch);
        Ok(Self {
            n,
            plan,
            scratch,
            buffer: vec![Complex64::default(); n],
        })
    }

    #[pyo3(signature = (values, inverse=false))]
    fn transform(&mut self, values: Vec<Complex64>, inverse: bool) -> PyResult<Vec<Complex64>> {
        valid_complex(&values, self.n)?;
        if inverse {
            self.plan
                .inverse(&values, &mut self.buffer, &mut self.scratch);
        } else {
            self.plan
                .forward(&values, &mut self.buffer, &mut self.scratch);
        }
        Ok(self.buffer.clone())
    }
}

#[pyclass(module = "amalthea_native._native")]
struct RealFft {
    n: usize,
    plan: PortableReal,
    scratch: FftScratch,
    time: Vec<f64>,
    spectrum: Vec<Complex64>,
}

#[pymethods]
impl RealFft {
    #[new]
    fn new(n: usize) -> PyResult<Self> {
        valid_length(n)?;
        let plan = PortableReal::new(n);
        let mut scratch = FftScratch::default();
        plan.prepare(&mut scratch);
        Ok(Self {
            n,
            plan,
            scratch,
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
        self.plan
            .forward(&values, &mut self.spectrum, &mut self.scratch);
        Ok(self.spectrum.clone())
    }

    fn inverse(&mut self, values: Vec<Complex64>) -> PyResult<Vec<f64>> {
        valid_complex(&values, self.n / 2 + 1)?;
        self.plan
            .inverse(&values, &mut self.time, &mut self.scratch);
        Ok(self.time.clone())
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<points::ModalPoints>()?;
    m.add_class::<ComplexFft>()?;
    m.add_class::<RealFft>()?;
    m.add_function(wrap_pyfunction!(solver::solve, m)?)?;
    m.add_function(wrap_pyfunction!(solver::solve_envelope, m)?)?;
    m.add_function(wrap_pyfunction!(solver::envelope_rhs, m)?)?;
    m.add_function(wrap_pyfunction!(solver::solve_real, m)?)?;
    m.add_function(wrap_pyfunction!(solver::real_rhs, m)?)?;
    Ok(())
}
