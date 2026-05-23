/// 1D Cubic Spline Interpolation Segment: S_i(x) = a + b(x - x_i) + c(x - x_i)^2 + d(x - x_i)^3
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
}
