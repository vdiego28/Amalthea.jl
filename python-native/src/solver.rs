//! Owning, serial adapter around the existing repaired Rust callback kernel.
use amalthea::ffi::{self, PreconStepFfiHandle, PreconStepResult};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rustfft::num_complex::Complex64 as C;
use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[path = "dense.rs"]
mod dense;

struct Handle(*mut PreconStepFfiHandle);
impl Drop for Handle {
    fn drop(&mut self) {
        // This handle is exclusively owned, constructed once, and never exposed.
        unsafe { ffi::free_precon_step_ffi(self.0) }
    }
}

struct Context {
    rhs: Py<PyAny>,
    linop: Vec<C>,
    error: Option<PyErr>,
}

fn check(values: &[C], n: usize) -> PyResult<()> {
    if values.len() != n
        || values
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(PyValueError::new_err(
            "field has wrong length or nonfinite values",
        ));
    }
    Ok(())
}

impl Context {
    fn prop(&self, values: &mut [C], dt: f64) {
        for (v, l) in values.iter_mut().zip(&self.linop) {
            *v *= (*l * dt).exp();
        }
    }

    fn evaluate(&self, t0: f64, t1: f64, values: &[C]) -> PyResult<Vec<C>> {
        let mut physical = values.to_vec();
        self.prop(&mut physical, t1 - t0);
        check(&physical, self.linop.len())?;
        let mut output =
            Python::attach(|py| self.rhs.call1(py, (t1, physical))?.extract::<Vec<C>>(py))?;
        check(&output, values.len())?;
        self.prop(&mut output, t0 - t1);
        check(&output, values.len())?;
        Ok(output)
    }
}

unsafe extern "C" fn rhs_callback(
    t0: f64,
    t1: f64,
    input: *const C,
    output: *mut C,
    n: usize,
    data: *mut c_void,
) {
    // All pointers originate below, refer to disjoint vectors of exactly n
    // elements, and the stack context lives through the synchronous FFI call.
    let ctx = unsafe { &mut *data.cast::<Context>() };
    let dst = unsafe { std::slice::from_raw_parts_mut(output, n) };
    if ctx.error.is_some() {
        dst.fill(C::default());
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        ctx.evaluate(t0, t1, unsafe { std::slice::from_raw_parts(input, n) })
    }));
    match result {
        Ok(Ok(values)) => dst.copy_from_slice(&values),
        Ok(Err(error)) => {
            ctx.error = Some(error);
            dst.fill(C::default());
        }
        Err(_) => {
            ctx.error = Some(PyRuntimeError::new_err("panic inside Rust RHS adapter"));
            dst.fill(C::default());
        }
    }
}

unsafe extern "C" fn prop_callback(t0: f64, t1: f64, values: *mut C, n: usize, data: *mut c_void) {
    let ctx = unsafe { &mut *data.cast::<Context>() };
    if ctx.error.is_some() {
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let values = unsafe { std::slice::from_raw_parts_mut(values, n) };
        ctx.prop(values, t1 - t0);
        check(values, n)
    }));
    match result {
        Ok(Ok(())) => (),
        Ok(Err(error)) => ctx.error = Some(error),
        Err(_) => {
            ctx.error = Some(PyRuntimeError::new_err(
                "panic inside Rust propagator adapter",
            ))
        }
    }
}

fn extra(ctx: &Context, y: &[C], stages: &mut [Vec<C>], t: f64, dt: f64) -> PyResult<()> {
    for (idx, coefficients) in [(7, &dense::A7[..]), (8, &dense::A8[..])] {
        let mut trial = y.to_vec();
        for (j, a) in coefficients.iter().enumerate() {
            for (v, k) in trial.iter_mut().zip(&stages[j]) {
                *v += *k * (dt * a);
            }
        }
        stages[idx] = ctx.evaluate(t, t + 0.4 * dt, &trial)?;
    }
    Ok(())
}

fn interpolate(
    ctx: &Context,
    y: &[C],
    stages: &[Vec<C>],
    t: f64,
    dt: f64,
    sample: f64,
    fifth: bool,
) -> PyResult<Vec<C>> {
    let theta = (sample - t) / dt;
    let mut output = y.to_vec();
    for (j, stage) in stages.iter().enumerate().take(if fifth { 9 } else { 7 }) {
        let coefficients: &[f64] = if fifth { &dense::C5[j] } else { &dense::C4[j] };
        let mut weight = 0.0;
        for a in coefficients.iter().rev() {
            weight = (weight + a) * theta;
        }
        if !fifth {
            weight += theta * theta * (3.0 - 2.0 * theta) * dense::ERROR[j];
        }
        for (v, k) in output.iter_mut().zip(stage) {
            *v += *k * (dt * weight);
        }
    }
    ctx.prop(&mut output, sample - t);
    check(&output, y.len())?;
    Ok(output)
}

// Private bindings still validate inputs: callers can import _native directly.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn solve(
    rhs: Py<PyAny>,
    linop: Vec<C>,
    initial: Vec<C>,
    positions: Vec<f64>,
    dt: f64,
    rtol: f64,
    atol: f64,
    safety: f64,
    min_dt: f64,
    max_dt: f64,
    fifth: bool,
    max_attempts: usize,
    repeat_limit: usize,
    filter: Option<Py<PyAny>>,
) -> PyResult<(Vec<Vec<C>>, usize, usize, Vec<f64>)> {
    let n = initial.len();
    if n == 0 {
        return Err(PyValueError::new_err("field must not be empty"));
    }
    check(&initial, n)?;
    check(&linop, n)?;
    if positions.len() < 2
        || positions.iter().any(|x| !x.is_finite())
        || positions.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(PyValueError::new_err(
            "positions must be finite and strictly increasing",
        ));
    }
    if ![dt, rtol, atol, safety, min_dt, max_dt]
        .iter()
        .all(|x| x.is_finite() && *x > 0.0)
        || min_dt > max_dt
        || dt < min_dt
        || dt > max_dt
        || safety > 1.0
        || max_attempts == 0
    {
        return Err(PyValueError::new_err("invalid solver controls"));
    }
    let handle = Handle(unsafe { ffi::init_precon_step_ffi(n) });
    if handle.0.is_null() {
        return Err(PyRuntimeError::new_err("stepper allocation failed"));
    }
    let mut ctx = Context {
        rhs,
        linop,
        error: None,
    };
    let mut yn = initial;
    let mut y = yn.clone();
    let mut stages = vec![vec![C::default(); n]; 9];
    let mut t = positions[0];
    let mut tn = t;
    let mut next_dt = dt;
    let mut errlast = 0.0;
    stages[0] = ctx.evaluate(t, t, &yn)?;
    let mut saved = vec![yn.clone()];
    let mut accepted = Vec::new();
    let mut rejected = 0;
    let mut consecutive = 0;
    for _ in 0..max_attempts {
        if tn > *positions.last().unwrap() {
            return Ok((saved, accepted.len(), rejected, accepted));
        }
        if !next_dt.is_finite() || tn + next_dt <= tn || !(tn + next_dt).is_finite() {
            return Err(PyRuntimeError::new_err(
                "step size cannot advance finite time",
            ));
        }
        let ptrs: Vec<*mut C> = stages.iter_mut().take(7).map(|v| v.as_mut_ptr()).collect();
        let mut result = PreconStepResult {
            ok: 0,
            dt: 0.0,
            t: 0.0,
            tn: 0.0,
            dtn: 0.0,
            err: 0.0,
            errlast: 0.0,
        };
        let status = unsafe {
            ffi::precon_step_ffi(
                handle.0,
                y.as_mut_ptr(),
                yn.as_mut_ptr(),
                ptrs.as_ptr(),
                n,
                t,
                tn,
                next_dt,
                rtol,
                atol,
                safety,
                max_dt,
                min_dt,
                errlast,
                i32::from(fifth),
                rhs_callback,
                prop_callback,
                (&mut ctx as *mut Context).cast(),
                &mut result,
            )
        };
        if let Some(error) = ctx.error.take() {
            return Err(error);
        }
        if status != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "Rust step failed: {status}"
            )));
        }
        t = result.t;
        tn = result.tn;
        next_dt = result.dtn;
        errlast = result.errlast;
        if result.ok != 0 {
            check(&yn, n)?;
            accepted.push(tn);
            consecutive = 0;
            let mut extra_ready = false;
            while saved.len() < positions.len() && positions[saved.len()] < tn {
                if fifth && !extra_ready {
                    extra(&ctx, &y, &mut stages, t, result.dt)?;
                    extra_ready = true;
                }
                saved.push(interpolate(
                    &ctx,
                    &y,
                    &stages,
                    t,
                    result.dt,
                    positions[saved.len()],
                    fifth,
                )?);
            }
            if let Some(ref callback) = filter {
                yn = Python::attach(|py| {
                    callback.call1(py, (tn, yn.clone()))?.extract::<Vec<C>>(py)
                })?;
                check(&yn, n)?;
            }
        } else {
            rejected += 1;
            consecutive += 1;
            if consecutive > repeat_limit {
                return Err(PyRuntimeError::new_err("reached step repetition limit"));
            }
        }
    }
    if tn > *positions.last().unwrap() {
        Ok((saved, accepted.len(), rejected, accepted))
    } else {
        Err(PyRuntimeError::new_err("reached maximum step attempts"))
    }
}
