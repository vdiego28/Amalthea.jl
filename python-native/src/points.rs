use amalthea::points::{PointBatch, PointConfig};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustfft::num_complex::Complex64;

#[pyclass(unsendable, module = "amalthea_native._native")]
pub struct ModalPoints {
    inner: PointBatch,
}

#[pymethods]
impl ModalPoints {
    #[new]
    #[pyo3(signature=(n_time, components, is_real, time_window, prefactor, kerr, dt, raman=None))]
    fn new(
        n_time: usize,
        components: usize,
        is_real: bool,
        time_window: Vec<f64>,
        prefactor: Vec<Complex64>,
        kerr: f64,
        dt: f64,
        raman: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let inner = PointBatch::new(PointConfig {
            n_time,
            components,
            is_real,
            time_window,
            prefactor,
            kerr,
            raman,
            dt,
        })
        .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    fn evaluate<'py>(
        &mut self,
        py: Python<'py>,
        fields: PyReadonlyArray1<'py, Complex64>,
        points: usize,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let input = fields
            .as_slice()
            .map_err(|_| PyValueError::new_err("native point input must be contiguous"))?;
        let result = self
            .inner
            .evaluate(input, points)
            .map_err(PyValueError::new_err)?;
        Ok(PyArray1::from_vec(py, result))
    }
}
