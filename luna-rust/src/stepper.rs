use num_complex::Complex;

// Dormand-Prince 5(4) Butcher Tableau coefficients
const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;
const A71: f64 = 35.0 / 384.0;
const A73: f64 = 500.0 / 1113.0;
const A74: f64 = 125.0 / 192.0;
const A75: f64 = -2187.0 / 6784.0;
const A76: f64 = 11.0 / 84.0;

// Embedded error estimator weights (E_i = b_i - b_i^*)
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

/// Adaptive Dormand-Prince 5(4) Stepper in the Interaction Picture
pub struct Dopri5Stepper {
    pub rtol: f64,
    pub atol: f64,
    pub safety: f64,
    // Lund PI controller parameters
    pub beta1: f64,
    pub beta2: f64,
    // Solver state buffers
    k1: Vec<Complex<f64>>,
    k2: Vec<Complex<f64>>,
    k3: Vec<Complex<f64>>,
    k4: Vec<Complex<f64>>,
    k5: Vec<Complex<f64>>,
    k6: Vec<Complex<f64>>,
    k7: Vec<Complex<f64>>,
    y_stage: Vec<Complex<f64>>,
    y_err: Vec<Complex<f64>>,
    // Internal propagation buffers for zero allocations
    k1_prop: Vec<Complex<f64>>,
    k2_prop: Vec<Complex<f64>>,
    k3_prop: Vec<Complex<f64>>,
    k4_prop: Vec<Complex<f64>>,
    k5_prop: Vec<Complex<f64>>,
    y_prop: Vec<Complex<f64>>,
    last_error: f64,
}

impl Dopri5Stepper {
    pub fn new(size: usize, rtol: f64, atol: f64) -> Self {
        Self {
            rtol,
            atol,
            safety: 0.9,
            beta1: 0.12,
            beta2: -0.04,
            k1: vec![Complex::new(0.0, 0.0); size],
            k2: vec![Complex::new(0.0, 0.0); size],
            k3: vec![Complex::new(0.0, 0.0); size],
            k4: vec![Complex::new(0.0, 0.0); size],
            k5: vec![Complex::new(0.0, 0.0); size],
            k6: vec![Complex::new(0.0, 0.0); size],
            k7: vec![Complex::new(0.0, 0.0); size],
            y_stage: vec![Complex::new(0.0, 0.0); size],
            y_err: vec![Complex::new(0.0, 0.0); size],
            k1_prop: vec![Complex::new(0.0, 0.0); size],
            k2_prop: vec![Complex::new(0.0, 0.0); size],
            k3_prop: vec![Complex::new(0.0, 0.0); size],
            k4_prop: vec![Complex::new(0.0, 0.0); size],
            k5_prop: vec![Complex::new(0.0, 0.0); size],
            y_prop: vec![Complex::new(0.0, 0.0); size],
            last_error: 1.0e-4,
        }
    }

    /// Takes a single step of size h in the interaction picture.
    /// Returns Ok((h_next, accepted)) if successful, or Err if step size underflows.
    ///
    /// * `y` - State vector in the interaction picture (modified in-place if accepted)
    /// * `z` - Current propagation coordinate
    /// * `h` - Suggested step size
    /// * `lin_op` - A function of type (z1: f64, z2: f64, y_in: &[Complex<f64>], y_out: &mut [Complex<f64>]) that applies the linear Integrating Factor.
    /// * `rhs` - A function of type (z: f64, y_in: &[Complex<f64>], dy_out: &mut [Complex<f64>]) that evaluates the nonlinear RHS.
    pub fn step<L, R>(
        &mut self,
        y: &mut [Complex<f64>],
        z: f64,
        h: f64,
        mut lin_op: L,
        mut rhs: R,
        fsal_active: &mut bool,
    ) -> Result<(f64, bool), String>
    where
        L: FnMut(f64, f64, &[Complex<f64>], &mut [Complex<f64>]),
        R: FnMut(f64, &[Complex<f64>], &mut [Complex<f64>]),
    {
        let size = y.len();
        
        // Stage 1 (FSAL: First Same As Last)
        if !*fsal_active {
            rhs(z, y, &mut self.k1);
            *fsal_active = true;
        }

        // Stage 2: g2 = lin_op(z, z + A21*h, y + h*A21*k1)
        for i in 0..size {
            self.y_stage[i] = y[i] + h * A21 * self.k1[i];
        }
        lin_op(z, z + A21 * h, &self.y_stage, &mut self.k2_prop);
        rhs(z + A21 * h, &self.k2_prop, &mut self.k2);

        // Stage 3: g3 = lin_op(z, z + 0.3*h, y + h*(A31*k1 + A32*k2))
        lin_op(z, z + 0.3 * h, &self.k1, &mut self.k1_prop);
        lin_op(z + A21 * h, z + 0.3 * h, &self.k2, &mut self.k2_prop);
        lin_op(z, z + 0.3 * h, y, &mut self.y_prop);

        for i in 0..size {
            self.y_stage[i] = self.y_prop[i] + h * (A31 * self.k1_prop[i] + A32 * self.k2_prop[i]);
        }
        rhs(z + 0.3 * h, &self.y_stage, &mut self.k3);

        // Stage 4: g4
        lin_op(z, z + 0.8 * h, &self.k1, &mut self.k1_prop);
        lin_op(z + A21 * h, z + 0.8 * h, &self.k2, &mut self.k2_prop);
        lin_op(z + 0.3 * h, z + 0.8 * h, &self.k3, &mut self.k3_prop);
        lin_op(z, z + 0.8 * h, y, &mut self.y_prop);

        for i in 0..size {
            self.y_stage[i] = self.y_prop[i] + h * (A41 * self.k1_prop[i] + A42 * self.k2_prop[i] + A43 * self.k3_prop[i]);
        }
        rhs(z + 0.8 * h, &self.y_stage, &mut self.k4);

        // Stage 5: g5
        let a5_z = z + (8.0 / 9.0) * h;
        lin_op(z, a5_z, &self.k1, &mut self.k1_prop);
        lin_op(z + A21 * h, a5_z, &self.k2, &mut self.k2_prop);
        lin_op(z + 0.3 * h, a5_z, &self.k3, &mut self.k3_prop);
        lin_op(z + 0.8 * h, a5_z, &self.k4, &mut self.k4_prop);
        lin_op(z, a5_z, y, &mut self.y_prop);

        for i in 0..size {
            self.y_stage[i] = self.y_prop[i] + h * (A51 * self.k1_prop[i] + A52 * self.k2_prop[i] + A53 * self.k3_prop[i] + A54 * self.k4_prop[i]);
        }
        rhs(a5_z, &self.y_stage, &mut self.k5);

        // Stage 6: g6
        lin_op(z, z + h, &self.k1, &mut self.k1_prop);
        lin_op(z + A21 * h, z + h, &self.k2, &mut self.k2_prop);
        lin_op(z + 0.3 * h, z + h, &self.k3, &mut self.k3_prop);
        lin_op(z + 0.8 * h, z + h, &self.k4, &mut self.k4_prop);
        lin_op(a5_z, z + h, &self.k5, &mut self.k5_prop);
        lin_op(z, z + h, y, &mut self.y_prop);

        for i in 0..size {
            self.y_stage[i] = self.y_prop[i] + h * (A61 * self.k1_prop[i] + A62 * self.k2_prop[i] + A63 * self.k3_prop[i] + A64 * self.k4_prop[i] + A65 * self.k5_prop[i]);
        }
        rhs(z + h, &self.y_stage, &mut self.k6);

        // Stage 7: g7 (FSAL prediction)
        lin_op(z, z + h, &self.k1, &mut self.k1_prop);
        lin_op(z + 0.3 * h, z + h, &self.k3, &mut self.k3_prop);
        lin_op(z + 0.8 * h, z + h, &self.k4, &mut self.k4_prop);
        lin_op(a5_z, z + h, &self.k5, &mut self.k5_prop);
        lin_op(z, z + h, y, &mut self.y_prop);

        for i in 0..size {
            let k6_p = self.k6[i];
            self.y_stage[i] = self.y_prop[i] + h * (A71 * self.k1_prop[i] + A73 * self.k3_prop[i] + A74 * self.k4_prop[i] + A75 * self.k5_prop[i] + A76 * k6_p);
        }
        rhs(z + h, &self.y_stage, &mut self.k7);

        // Error estimation
        lin_op(z, z + h, &self.k1, &mut self.k1_prop);
        lin_op(z + 0.3 * h, z + h, &self.k3, &mut self.k3_prop);
        lin_op(z + 0.8 * h, z + h, &self.k4, &mut self.k4_prop);
        lin_op(a5_z, z + h, &self.k5, &mut self.k5_prop);

        let mut err_sum = 0.0;
        for i in 0..size {
            let k6_p = self.k6[i];
            let k7_p = self.k7[i];

            let err = h * (E1 * self.k1_prop[i] + E3 * self.k3_prop[i] + E4 * self.k4_prop[i] + E5 * self.k5_prop[i] + E6 * k6_p + E7 * k7_p);
            self.y_err[i] = err;
            
            // Standard scale factor for error evaluation: tol = atol + max(|y|, |y_stage|) * rtol
            let norm_y = y[i].norm();
            let norm_stage = self.y_stage[i].norm();
            let max_val = if norm_y > norm_stage { norm_y } else { norm_stage };
            let tol = self.atol + max_val * self.rtol;
            
            let term = err.norm() / tol;
            err_sum += term * term;
        }

        // Root Mean Square Error (L2 Norm)
        let error = (err_sum / (size as f64)).sqrt();

        // Check step acceptability
        let accepted = error <= 1.0;
        
        // Calculate next step size using the Lund PI step size controller
        let factor = if error > 0.0 {
            let mut fac = self.safety * (1.0 / error).powf(self.beta1) * (self.last_error / error).powf(self.beta2);
            if fac < 0.2 { fac = 0.2; }
            if fac > 5.0 { fac = 5.0; }
            fac
        } else {
            5.0
        };
        let h_next = h * factor;

        if accepted {
            // Apply Integrating Factor linear step to input field vector
            lin_op(z, z + h, y, &mut self.y_prop);
            for i in 0..size {
                y[i] = self.y_prop[i] + h * (A71 * self.k1[i] + A73 * self.k3[i] + A74 * self.k4[i] + A75 * self.k5[i] + A76 * self.k6[i]);
            }
            
            // FSAL transition: k7 becomes the next step's k1
            self.k1.copy_from_slice(&self.k7);
            self.last_error = error.max(1.0e-4);
            Ok((h_next, true))
        } else {
            *fsal_active = false; // Reset FSAL since step was rejected
            if h_next < 1.0e-12 {
                return Err("Step size underflow in Dopri5 solver".to_string());
            }
            Ok((h_next, false))
        }
    }
}
