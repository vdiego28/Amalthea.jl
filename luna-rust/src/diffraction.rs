use num_complex::Complex;

/// Evaluation of the 0-th order Bessel function of the first kind J0(x)
pub fn j0(x: f64) -> f64 {
    if x.abs() < 8.0 {
        // Power series representation
        let y = x * x / 4.0;
        let mut term = 1.0;
        let mut sum = 1.0;
        for i in 1..25 {
            let i_f = i as f64;
            term *= -y / (i_f * i_f);
            sum += term;
            if term.abs() < 1e-16 {
                break;
            }
        }
        sum
    } else {
        // Asymptotic expansion
        let z = x.abs();
        let chi = z - std::f64::consts::PI / 4.0;
        (2.0 / (std::f64::consts::PI * z)).sqrt() * (chi.cos() + 0.125 * chi.sin() / z)
    }
}

/// Evaluation of the 1-st order Bessel function of the first kind J1(x)
pub fn j1(x: f64) -> f64 {
    if x.abs() < 8.0 {
        // Power series representation
        let y = x * x / 4.0;
        let mut term = 0.5 * x;
        let mut sum = term;
        for i in 1..25 {
            let i_f = i as f64;
            term *= -y / (i_f * (i_f + 1.0));
            sum += term;
            if term.abs() < 1e-16 {
                break;
            }
        }
        sum
    } else {
        // Asymptotic expansion
        let z = x.abs();
        let chi = z - 3.0 * std::f64::consts::PI / 4.0;
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        sign * (2.0 / (std::f64::consts::PI * z)).sqrt() * (chi.cos() - 0.375 * chi.sin() / z)
    }
}

/// Derivative of J0(x), which is -J1(x)
fn j0_prime(x: f64) -> f64 {
    -j1(x)
}

/// Find the first N zeros of the J0 Bessel function using Newton-Raphson
pub fn find_j0_zeros(n: usize) -> Vec<f64> {
    let mut zeros = Vec::with_capacity(n);
    for i in 1..=n {
        // Good initial estimate for the i-th zero of J0
        let mut x = std::f64::consts::PI * ((i as f64) - 0.25);
        // Newton-Raphson refinement
        for _ in 0..10 {
            let val = j0(x);
            let deriv = j0_prime(x);
            let diff = val / deriv;
            x -= diff;
            if diff.abs() < 1e-14 {
                break;
            }
        }
        zeros.push(x);
    }
    zeros
}

/// Quasi-Discrete Hankel Transform (QDHT) solver
#[derive(Debug, Clone)]
pub struct Qdht {
    pub n_r: usize,
    pub aperture_radius: f64,
    pub r: Vec<f64>,  // Radial grid points (size n_r)
    pub k: Vec<f64>,  // Reciprocal/spectral grid points (size n_r)
    pub t_matrix: Vec<f64>, // Flattened symmetric transform matrix T_ij (n_r x n_r)
    pub j1_zeros: Vec<f64>, // j1 evaluated at zeros of j0 (needed for scaling factors)
}

impl Qdht {
    pub fn new(n_zeros: usize, aperture_radius: f64) -> Self {
        assert!(n_zeros >= 3);
        let zeros = find_j0_zeros(n_zeros);
        let s = zeros[n_zeros - 1]; // boundary scale (the last zero)
        
        let n_grid = n_zeros - 1; // number of active grid points (excluding boundary)
        let mut r = vec![0.0; n_grid];
        let mut k = vec![0.0; n_grid];
        let mut j1_zeros = vec![0.0; n_grid];
        
        for i in 0..n_grid {
            r[i] = zeros[i] / s * aperture_radius;
            k[i] = zeros[i] / aperture_radius;
            j1_zeros[i] = j1(zeros[i]).abs();
        }
        
        // Populate symmetric transform matrix T_ij of size n_grid x n_grid
        let mut t_matrix = vec![0.0; n_grid * n_grid];
        for i in 0..n_grid {
            for j in 0..n_grid {
                let num = 2.0 * j0((zeros[i] * zeros[j]) / s);
                let den = s * j1_zeros[i] * j1_zeros[j];
                t_matrix[i * n_grid + j] = num / den;
            }
        }
        
        Self {
            n_r: n_grid,
            aperture_radius,
            r,
            k,
            t_matrix,
            j1_zeros,
        }
    }

    /// Performs the Hankel Transform in-place using dense Level-2 BLAS matrix-vector product
    /// Since the matrix T is its own inverse, this works for both forward and backward transform.
    pub fn transform(&self, input: &[Complex<f64>], output: &mut [Complex<f64>]) {
        assert_eq!(input.len(), self.n_r);
        assert_eq!(output.len(), self.n_r);
        
        use rayon::prelude::*;
        
        output.par_iter_mut().enumerate().for_each(|(i, out_val)| {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            let row_offset = i * self.n_r;
            
            for j in 0..self.n_r {
                let t_val = unsafe { *self.t_matrix.get_unchecked(row_offset + j) };
                let val = unsafe { input.get_unchecked(j) };
                sum_re += t_val * val.re;
                sum_im += t_val * val.im;
            }
            
            *out_val = Complex::new(sum_re, sum_im);
        });
    }
}
