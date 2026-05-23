/// Sellmeier coefficients for a gas (e.g., Helium or Argon)
#[derive(Debug, Clone, Copy)]
pub struct SellmeierGas {
    pub b1: f64,
    pub c1: f64, // in units of rad^2/s^2 or similar frequency-squared equivalent
}

impl SellmeierGas {
    /// Computes refractive index n(ω)
    pub fn refractive_index(&self, ω: f64) -> f64 {
        let den = self.c1 - ω * ω;
        let n2 = 1.0 + (self.b1 * self.c1) / den;
        n2.sqrt()
    }

    /// Exact analytical first derivative dn/dω
    pub fn dn_dω(&self, ω: f64) -> f64 {
        let n = self.refractive_index(ω);
        let den = self.c1 - ω * ω;
        let d_n2 = (2.0 * self.b1 * self.c1 * ω) / (den * den);
        d_n2 / (2.0 * n)
    }

    /// Exact analytical second derivative d^2n/dω^2
    pub fn d2n_dω2(&self, ω: f64) -> f64 {
        let n = self.refractive_index(ω);
        let dn = self.dn_dω(ω);
        let den = self.c1 - ω * ω;
        let d_n2 = (2.0 * self.b1 * self.c1 * ω) / (den * den);
        let d2_n2 = (2.0 * self.b1 * self.c1 * (self.c1 + 3.0 * ω * ω)) / (den * den * den);
        
        (d2_n2 * n - d_n2 * dn) / (2.0 * n * n)
    }
}

/// Zeisberger Analytical Dispersion Calculator
#[derive(Debug, Clone)]
pub struct ZeisbergerDispersion {
    pub gas: SellmeierGas,
    pub core_radius: f64,
    pub glass_index: f64,
    pub mode_root: f64, // root u_nm (e.g., 2.4048 for HE11)
}

impl ZeisbergerDispersion {
    pub fn new(gas: SellmeierGas, core_radius: f64, glass_index: f64, mode_root: f64) -> Self {
        Self { gas, core_radius, glass_index, mode_root }
    }

    /// Analytical propagation constant beta(ω)
    pub fn beta(&self, ω: f64) -> f64 {
        let c = 299792458.0;
        let n_co = self.gas.refractive_index(ω);
        let σ = c / (ω * n_co * self.core_radius);
        
        // Coefficients
        let a_coeff = 0.5 * self.mode_root * self.mode_root;
        let epsilon = (self.glass_index * self.glass_index) / (n_co * n_co);
        let theta = (epsilon + 1.0) / (epsilon - 1.0).sqrt();
        let b_coeff = 0.5 * self.mode_root * self.mode_root * theta;
        
        let n_eff = n_co * (1.0 - a_coeff * σ * σ - b_coeff * σ * σ * σ);
        (ω / c) * n_eff
    }
}

/// Chebyshev Differentiation and Interpolation Solver
#[derive(Debug, Clone)]
pub struct ChebyshevDispersion {
    pub coefficients: Vec<f64>,
    pub ω_min: f64,
    pub ω_max: f64,
    pub degree: usize,
}

impl ChebyshevDispersion {
    /// Fit coefficients based on an input evaluation function f(ω)
    pub fn fit<F>(ω_min: f64, ω_max: f64, degree: usize, mut f: F) -> Self
    where
        F: FnMut(f64) -> f64,
    {
        let mut coefficients = vec![0.0; degree];
        let n = degree as f64;
        
        // Compute discrete cosine transform coefficients
        for i in 0..degree {
            let mut sum = 0.0;
            for k in 1..=degree {
                let k_f = k as f64;
                let node = ((2.0 * k_f - 1.0) / (2.0 * n) * std::f64::consts::PI).cos();
                // Map node s in [-1, 1] back to frequency space
                let ω = 0.5 * (node * (ω_max - ω_min) + (ω_max + ω_min));
                sum += f(ω) * (((i as f64) * (2.0 * k_f - 1.0) / (2.0 * n) * std::f64::consts::PI).cos());
            }
            let factor = if i == 0 { 1.0 / n } else { 2.0 / n };
            coefficients[i] = factor * sum;
        }
        
        Self { coefficients, ω_min, ω_max, degree }
    }

    /// Evaluates the Chebyshev series at a mapped coordinate s in [-1, 1]
    fn evaluate_series(&self, s: f64, coeffs: &[f64]) -> f64 {
        if coeffs.is_empty() {
            return 0.0;
        }
        if coeffs.len() == 1 {
            return coeffs[0];
        }
        
        // Clenshaw's recurrence algorithm
        let mut d1 = 0.0;
        let mut d2 = 0.0;
        for i in (1..coeffs.len()).rev() {
            let temp = d1;
            d1 = 2.0 * s * d1 - d2 + coeffs[i];
            d2 = temp;
        }
        s * d1 - d2 + coeffs[0]
    }

    /// Computes the exact Chebyshev derivative coefficients using backward recurrence
    pub fn differentiate_coefficients(&self, coeffs: &[f64]) -> Vec<f64> {
        let n = coeffs.len();
        let mut deriv_coeffs = vec![0.0; n];
        if n < 2 {
            return deriv_coeffs;
        }
        
        deriv_coeffs[n - 1] = 0.0;
        if n > 2 {
            deriv_coeffs[n - 2] = 2.0 * ((n - 1) as f64) * coeffs[n - 1];
        }
        
        for i in (1..=n - 3).rev() {
            deriv_coeffs[i] = deriv_coeffs[i + 2] + 2.0 * ((i + 1) as f64) * coeffs[i + 1];
        }
        // Special case for c_0
        deriv_coeffs[0] = 0.5 * (deriv_coeffs[2] + 2.0 * coeffs[1]);
        
        deriv_coeffs
    }

    /// Returns the exact value and first/second derivatives with respect to ω
    pub fn evaluate_derivatives(&self, ω: f64) -> (f64, f64, f64) {
        let s = (2.0 * ω - (self.ω_max + self.ω_min)) / (self.ω_max - self.ω_min);
        
        // Value
        let val = self.evaluate_series(s, &self.coefficients);
        
        // Mapped derivative ds/dω
        let ds_dω = 2.0 / (self.ω_max - self.ω_min);
        
        // First derivative coefficients
        let c_prime = self.differentiate_coefficients(&self.coefficients);
        let dy_ds = self.evaluate_series(s, &c_prime);
        let dy_dω = dy_ds * ds_dω;
        
        // Second derivative coefficients
        let c_double_prime = self.differentiate_coefficients(&c_prime);
        let d2y_ds2 = self.evaluate_series(s, &c_double_prime);
        let d2y_dω2 = d2y_ds2 * ds_dω * ds_dω;
        
        (val, dy_dω, d2y_dω2)
    }
}
