/// 1D Cubic Spline Interpolation Segment: S_i(x) = a + b(x - x_i) + c(x - x_i)^2 + d(x - x_i)^3
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SplineSegment {
    pub x: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

/// Cubic Spline Lookup Table for rapid physical rate evaluation
#[derive(Debug, Clone)]
pub struct CubicSplineLUT {
    pub segments: Vec<SplineSegment>,
    pub x_min: f64,
    pub x_max: f64,
}

impl CubicSplineLUT {
    /// Fits a natural cubic spline to a function over a uniform grid
    pub fn fit<F>(x_min: f64, x_max: f64, n_points: usize, mut f: F) -> Self
    where
        F: FnMut(f64) -> f64,
    {
        assert!(n_points >= 3);
        let h = (x_max - x_min) / ((n_points - 1) as f64);
        
        let mut x = vec![0.0; n_points];
        let mut a = vec![0.0; n_points];
        for i in 0..n_points {
            x[i] = x_min + (i as f64) * h;
            a[i] = f(x[i]);
        }
        
        // Solve the tridiagonal system for the second derivatives c_i
        let mut alpha = vec![0.0; n_points - 1];
        for i in 1..(n_points - 1) {
            alpha[i] = (3.0 / h) * (a[i + 1] - a[i]) - (3.0 / h) * (a[i] - a[i - 1]);
        }
        
        let mut l = vec![1.0; n_points];
        let mut mu = vec![0.0; n_points];
        let mut z = vec![0.0; n_points];
        
        for i in 1..(n_points - 1) {
            l[i] = 4.0 * h - h * mu[i - 1];
            mu[i] = h / l[i];
            z[i] = (alpha[i] - h * z[i - 1]) / l[i];
        }
        
        let mut c = vec![0.0; n_points];
        let mut b = vec![0.0; n_points];
        let mut d = vec![0.0; n_points];
        
        for j in (0..(n_points - 1)).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (a[j + 1] - a[j]) / h - h * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h);
        }
        
        let mut segments = Vec::with_capacity(n_points - 1);
        for i in 0..(n_points - 1) {
            segments.push(SplineSegment {
                x: x[i],
                a: a[i],
                b: b[i],
                c: c[i],
                d: d[i],
            });
        }
        
        Self { segments, x_min, x_max }
    }

    /// Build a natural cubic spline from pre-sampled (x, y) pairs.
    ///
    /// Supports non-uniform x grids (e.g. after filtering zero-rate points on
    /// the Julia side).  Uses the standard natural-spline tridiagonal solve:
    /// second derivatives at both endpoints are forced to zero.
    ///
    /// # Panics
    /// Panics if `x_values.len() < 3` or if lengths differ.
    pub fn from_samples(x_values: &[f64], y_values: &[f64]) -> Self {
        let n = x_values.len();
        assert!(n >= 3, "CubicSplineLUT requires at least 3 points");
        assert_eq!(n, y_values.len(), "x and y must have the same length");

        // Precompute per-segment widths
        let mut h = vec![0.0f64; n - 1];
        for i in 0..(n - 1) {
            h[i] = x_values[i + 1] - x_values[i];
            assert!(h[i] > 0.0, "x_values must be strictly increasing");
        }

        let a = y_values.to_vec();

        // Forward sweep of the Thomas algorithm for natural cubic spline
        let mut alpha = vec![0.0f64; n];
        for i in 1..(n - 1) {
            alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i])
                - (3.0 / h[i - 1]) * (a[i] - a[i - 1]);
        }

        let mut l  = vec![1.0f64; n];
        let mut mu = vec![0.0f64; n];
        let mut z  = vec![0.0f64; n];

        for i in 1..(n - 1) {
            l[i]  = 2.0 * (h[i - 1] + h[i]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i]  = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        // l[n-1] = 1, z[n-1] = 0 (natural BC: c[n-1] = 0)

        // Back-substitution
        let mut c = vec![0.0f64; n];
        let mut b = vec![0.0f64; n - 1];
        let mut d = vec![0.0f64; n - 1];

        for j in (0..(n - 1)).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (a[j + 1] - a[j]) / h[j]
                - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }

        let mut segments = Vec::with_capacity(n - 1);
        for i in 0..(n - 1) {
            segments.push(SplineSegment {
                x: x_values[i],
                a: a[i],
                b: b[i],
                c: c[i],
                d: d[i],
            });
        }

        Self {
            segments,
            x_min: x_values[0],
            x_max: *x_values.last().unwrap(),
        }
    }

    /// Evaluates the spline at x. Returns None if x lies outside [x_min, x_max]
    pub fn evaluate(&self, x: f64) -> Option<f64> {
        if x < self.x_min || x > self.x_max {
            return None;
        }
        
        // Binary search to find segment
        let mut low = 0;
        let mut high = self.segments.len() - 1;
        while low < high {
            let mid = (low + high + 1) / 2;
            if self.segments[mid].x <= x {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        
        let seg = unsafe { self.segments.get_unchecked(low) };
        let dx = x - seg.x;
        Some(seg.a + dx * (seg.b + dx * (seg.c + dx * seg.d)))
    }
}

/// PPT Ionization solver wrapper
#[derive(Debug, Clone)]
pub struct PptIonizationRate {
    pub spline_lut: CubicSplineLUT,
    pub e_min: f64, // Lower-bound cutoff boundary to prevent Keldysh divergence
    pub e_max: f64,
}

impl PptIonizationRate {
    /// Create the PPT Ionization lookup rate.
    /// This accepts a representative rate function (e.g. PPT formula evaluations)
    /// and precomputes the spline lookup table.
    pub fn new<F>(e_min: f64, e_max: f64, n_points: usize, mut ppt_rate_fun: F) -> Self
    where
        F: FnMut(f64) -> f64,
    {
        // Fit the spline to the logarithm of the rate to handle many orders of magnitude
        let spline_lut = CubicSplineLUT::fit(e_min, e_max, n_points, move |e| {
            let rate = ppt_rate_fun(e);
            if rate > 0.0 { rate.ln() } else { -100.0 } // clamp minimum log-rate
        });
        
        Self { spline_lut, e_min, e_max }
    }

    /// Create from pre-sampled (E, rate) pairs supplied by the caller (e.g. Julia).
    ///
    /// `e_values` and `rate_values` must be the same length and `e_values` must
    /// be strictly increasing.  Rates that are zero or negative are mapped to
    /// `ln(-100)` so the spline stays well-conditioned — they will never be
    /// returned because `e_min` acts as a lower cutoff.
    ///
    /// This constructor is used by the Julia FFI bridge so Julia can pass the
    /// same pre-computed knot data that its own `CSpline` uses, ensuring both
    /// paths interpolate the *same* underlying rate table.
    pub fn from_samples(e_min: f64, e_max: f64, e_values: &[f64], rate_values: &[f64]) -> Self {
        let ln_rates: Vec<f64> = rate_values.iter().map(|&r| {
            if r > 0.0 { r.ln() } else { -100.0 }
        }).collect();
        let spline_lut = CubicSplineLUT::from_samples(e_values, &ln_rates);
        Self { spline_lut, e_min, e_max }
    }

    /// Calculates the ionization rate with strict cutoff checks
    pub fn rate(&self, field_strength: f64) -> Result<f64, String> {
        let abs_e = field_strength.abs();
        
        if abs_e < self.e_min {
            // Cutoff region: rate is physically negligible and Keldysh calculation diverges
            return Ok(0.0);
        }
        
        if abs_e > self.e_max {
            return Err(format!(
                "Electric field amplitude {:.2e} V/m exceeds PPT lookup table upper bound {:.2e} V/m",
                abs_e, self.e_max
            ));
        }
        
        match self.spline_lut.evaluate(abs_e) {
            Some(ln_rate) => Ok(ln_rate.exp()),
            None => Ok(0.0), // fallback safety
        }
    }

    /// Evaluates ionization rates for a vector of field strength values, using GPU acceleration if available.
    pub fn rate_vector(&self, fields: &[f64], rates: &mut [f64]) -> Result<(), String> {
        let n_t = fields.len();
        assert_eq!(rates.len(), n_t);
        
        if let Some(ctx) = crate::cuda::get_gpu_context() {
            if self.rate_vector_gpu(ctx, fields, rates).is_ok() {
                return Ok(());
            }
        }
        
        for i in 0..n_t {
            rates[i] = self.rate(fields[i])?;
        }
        Ok(())
    }

    fn rate_vector_gpu(&self, ctx: &crate::cuda::GpuContext, fields: &[f64], rates: &mut [f64]) -> Result<(), String> {
        let n_t = fields.len();
        let num_segments = self.spline_lut.segments.len();
        
        let d_fields = crate::cuda::GpuBuffer::alloc(n_t * std::mem::size_of::<f64>())?;
        let d_rates = crate::cuda::GpuBuffer::alloc(n_t * std::mem::size_of::<f64>())?;
        let d_segments = crate::cuda::GpuBuffer::alloc(num_segments * std::mem::size_of::<SplineSegment>())?;
        let d_err = crate::cuda::GpuBuffer::alloc(std::mem::size_of::<libc::c_int>())?;
        
        d_fields.copy_to_device(fields)?;
        d_segments.copy_to_device(&self.spline_lut.segments)?;
        
        let err_initial = [0 as libc::c_int];
        d_err.copy_to_device(&err_initial)?;
        
        let mut d_fields_ptr = d_fields.dptr;
        let mut d_rates_ptr = d_rates.dptr;
        let mut d_segments_ptr = d_segments.dptr;
        let mut e_min_val = self.e_min;
        let mut e_max_val = self.e_max;
        let mut num_segments_val = num_segments as libc::c_int;
        let mut n_t_val = n_t as libc::c_int;
        let mut d_err_ptr = d_err.dptr;
        
        let mut args: [*mut libc::c_void; 8] = [
            &mut d_fields_ptr as *mut _ as *mut libc::c_void,
            &mut d_rates_ptr as *mut _ as *mut libc::c_void,
            &mut d_segments_ptr as *mut _ as *mut libc::c_void,
            &mut e_min_val as *mut _ as *mut libc::c_void,
            &mut e_max_val as *mut _ as *mut libc::c_void,
            &mut num_segments_val as *mut _ as *mut libc::c_void,
            &mut n_t_val as *mut _ as *mut libc::c_void,
            &mut d_err_ptr as *mut _ as *mut libc::c_void,
        ];
        
        let block_size = 256;
        let grid_size = (n_t + block_size - 1) / block_size;
        
        crate::cuda::activate_context()?;
        let driver = crate::cuda::get_driver_api()?;
        
        unsafe {
            let res = (driver.cuLaunchKernel)(
                ctx.ppt_fn,
                grid_size as libc::c_uint, 1, 1,
                block_size as libc::c_uint, 1, 1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            
            if res != 0 {
                return Err(format!("cuLaunchKernel for ppt_ionization_kernel failed: {}", res));
            }
        }
        
        let mut err_res = [0 as libc::c_int];
        d_err.copy_to_host(&mut err_res)?;
        if err_res[0] != 0 {
            return Err("Electric field amplitude exceeded upper bound of ionization rate lookup table".to_string());
        }
        
        d_rates.copy_to_host(rates)?;
        Ok(())
    }
}
