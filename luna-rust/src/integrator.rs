use num_complex::Complex;

/// Gauss-Legendre 4-point quadrature nodes and weights
const GL_X: [f64; 4] = [
    -0.8611363115940526,
    -0.3399810435848563,
     0.3399810435848563,
     0.8611363115940526,
];

const GL_W: [f64; 4] = [
    0.3478548451374538,
    0.6521451548625461,
    0.6521451548625461,
    0.3478548451374538,
];

/// Integrates the linear operator L(z, ω) over the step [z_n, z_n + h] for each frequency
/// using 4-point Gauss-Legendre quadrature. Returns the diagonal exponential factors.
///
/// * `z_n` - Start coordinate
/// * `h` - Step size
/// * `omega` - Vector of grid frequencies (size N)
/// * `operator_evaluator` - A function/closure of type (z: f64, ω: f64) -> Complex<f64>
pub fn integrate_integrating_factor<F>(
    z_n: f64,
    h: f64,
    omega: &[f64],
    mut operator_evaluator: F,
) -> Vec<Complex<f64>>
where
    F: FnMut(f64, f64) -> Complex<f64>,
{
    let n_w = omega.len();
    let mut factor = vec![Complex::new(0.0, 0.0); n_w];
    
    let half_h = 0.5 * h;
    let mid_z = z_n + half_h;
    
    // Pre-evaluate the z coordinates of the 4 nodes
    let z_coords = [
        mid_z + half_h * GL_X[0],
        mid_z + half_h * GL_X[1],
        mid_z + half_h * GL_X[2],
        mid_z + half_h * GL_X[3],
    ];
    
    // Outer loop: frequencies (easily parallelized if needed, but vectorization handles it)
    for i in 0..n_w {
        let ω = unsafe { *omega.get_unchecked(i) };
        let mut sum = Complex::new(0.0, 0.0);
        
        for k in 0..4 {
            let l_val = operator_evaluator(z_coords[k], ω);
            sum += l_val * GL_W[k];
        }
        
        // Exact Integrating Factor exponent multiplication
        factor[i] = (sum * half_h).exp();
    }
    
    factor
}
