use num_complex::Complex;

/// Convenience alias used throughout this module for complex arithmetic.
type C64 = Complex<f64>;

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
        Self {
            gas,
            core_radius,
            glass_index,
            mode_root,
        }
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
                sum += f(ω)
                    * (((i as f64) * (2.0 * k_f - 1.0) / (2.0 * n) * std::f64::consts::PI).cos());
            }
            let factor = if i == 0 { 1.0 / n } else { 2.0 / n };
            coefficients[i] = factor * sum;
        }

        Self {
            coefficients,
            ω_min,
            ω_max,
            degree,
        }
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

// ─── Zeisberger full-parity neff (eq. 15, Zeisberger & Schmidt 2017) ─────────

/// Geometry-only struct that computes the complete complex effective index
/// of an anti-resonant hollow-core fibre mode from externally supplied `nco`/`ncl`.
///
/// Implements all four mode-kind branches (HE/EH/TE/TM) and all three loss
/// encodings (Val{true}, Val{false}, Number) of Julia's `Antiresonant.__neff`.
///
/// `nco` and `ncl` are passed as complex numbers so that absorbing / off-resonance
/// cladding is handled correctly. For the typical real-Sellmeier gas/glass case
/// the imaginary parts are zero and the arithmetic is effectively real.
///
/// # Kind encoding
/// - `0` = HE  (`s = -1`)
/// - `1` = EH  (`s = +1`)
/// - `2` = TE
/// - `3` = TM
///
/// # Loss encoding
/// - `loss_on = false` → drop imaginary part → `Val{false}` behaviour
/// - `loss_on = true,  loss_scale = 1.0` → `Val{true}` behaviour
/// - `loss_on = true,  loss_scale = L`   → `Number L`   behaviour
#[derive(Debug, Clone, Copy)]
pub struct ZeisbergerNeff {
    /// Bessel root `u_nm` (e.g. 2.4048 for the fundamental HE11 mode)
    pub unm: f64,
    /// Azimuthal mode index `m` (used in the C/s terms for HE/EH)
    pub m_az: f64,
    /// Mode kind: 0=HE, 1=EH, 2=TE, 3=TM
    pub kind: u8,
    /// Anti-resonant strut wall thickness (m)
    pub wallthickness: f64,
    /// If false, the imaginary part of neff is forced to zero (Val{false})
    pub loss_on: bool,
    /// Scaling factor applied to the D·σ⁴ imaginary term (1.0 for Val{true})
    pub loss_scale: f64,
}

impl ZeisbergerNeff {
    /// Construct a new handle.
    pub fn new(
        unm: f64,
        m_az: f64,
        kind: u8,
        wallthickness: f64,
        loss_on: bool,
        loss_scale: f64,
    ) -> Self {
        Self {
            unm,
            m_az,
            kind,
            wallthickness,
            loss_on,
            loss_scale,
        }
    }

    /// Compute the complex effective index at a single frequency `omega` (rad/s).
    ///
    /// `nco` and `ncl` are the core and cladding refractive indices (supplied by
    /// Julia from its multi-term Sellmeier; this function does only geometry).
    /// `radius` is the capillary core radius at the current propagation position.
    ///
    /// Mirrors Julia's `Antiresonant._neff` / `__neff` verbatim.
    pub fn neff_one(&self, omega: f64, nco: C64, ncl: C64, radius: f64) -> C64 {
        const CLIGHT: f64 = 299_792_458.0;
        let one = C64::new(1.0, 0.0);

        // Intermediate quantities — eqs as in Julia _neff
        let eps = ncl * ncl / (nco * nco); // ϵ = ncl²/nco²
        let k0 = omega / CLIGHT; // k₀ = ω/c
        let ka = C64::new(k0, 0.0) * nco; // kₐ = k₀·nco
        let phi = C64::new(k0 * self.wallthickness, 0.0)               // ϕ = k₀·t·√(ncl²-nco²)
                      * (ncl * ncl - nco * nco).sqrt();
        let sigma = one / (ka * radius); // σ = 1/(kₐ·a)
        let tan_phi = phi.tan();
        let eps_m1 = eps - one; // ϵ-1
        let sqrt_em1 = eps_m1.sqrt(); // √(ϵ-1)
        let unm = self.unm;
        let m = self.m_az;
        let tan2 = tan_phi * tan_phi;
        let cot2 = one / tan2; // 1/tan²(ϕ)

        // Compute A, B, C, D for the chosen mode kind
        let (a, b, c_coeff, d): (C64, C64, C64, C64) = match self.kind {
            2 => {
                // ─── TE ───────────────────────────────────────────────────────
                // A = u²/2
                // B = A / (√(ϵ-1)·tan(ϕ))
                // C = u⁴/8 + 2u²/(ϵ-1)/tan²(ϕ)
                // D = u³·(1 + 1/tan²(ϕ))/(ϵ-1)
                let a = C64::new(unm * unm / 2.0, 0.0);
                let b = a / (sqrt_em1 * tan_phi);
                let c = C64::new(unm.powi(4) / 8.0, 0.0)
                    + C64::new(2.0 * unm * unm, 0.0) / (eps_m1 * tan2);
                let d = C64::new(unm.powi(3), 0.0) * (one + cot2) / eps_m1;
                (a, b, c, d)
            }
            3 => {
                // ─── TM ───────────────────────────────────────────────────────
                // A = u²/2
                // B = A·ϵ / (√(ϵ-1)·tan(ϕ))
                // C = u⁴/8 + 2u²·ϵ²/(ϵ-1)/tan²(ϕ)
                // D = u³·ϵ²·(1 + 1/tan²(ϕ))/(ϵ-1)
                let a = C64::new(unm * unm / 2.0, 0.0);
                let b = a * eps / (sqrt_em1 * tan_phi);
                let c = C64::new(unm.powi(4) / 8.0, 0.0)
                    + C64::new(2.0 * unm * unm, 0.0) * eps * eps / (eps_m1 * tan2);
                let d = C64::new(unm.powi(3), 0.0) * eps * eps * (one + cot2) / eps_m1;
                (a, b, c, d)
            }
            kind => {
                // ─── HE (kind=0, s=-1) or EH (kind=1, s=+1) ─────────────────
                // s: EH=+1, HE=-1
                // A = u²/2
                // B = u²·(ϵ+1) / (√(ϵ-1)·tan(ϕ))
                // C = u⁴/8 + u²·m·s/2
                //     + (u²/4·(2+m·s)·(ϵ+1)²/(ϵ-1) - u⁴·(ϵ-1)/(8·m)) / tan²(ϕ)
                // D = u³/2·(ϵ²+1)/(ϵ-1)·(1 + 1/tan²(ϕ))
                let s = if kind == 1 { 1.0_f64 } else { -1.0_f64 };
                let eps_p1 = eps + one;
                let a = C64::new(unm * unm / 2.0, 0.0);
                let b = C64::new(unm * unm, 0.0) * eps_p1 / (sqrt_em1 * tan_phi);
                // The (/tan²) part in C
                let c_cot = C64::new(unm * unm / 4.0 * (2.0 + m * s), 0.0) * eps_p1 * eps_p1
                    / eps_m1
                    - C64::new(unm.powi(4) / (8.0 * m), 0.0) * eps_m1;
                let c = C64::new(unm.powi(4) / 8.0 + unm * unm * m * s / 2.0, 0.0) + c_cot * cot2;
                let d =
                    C64::new(unm.powi(3) / 2.0, 0.0) * (eps * eps + one) / eps_m1 * (one + cot2);
                (a, b, c, d)
            }
        };

        // Assemble: neff = nco·(1 - A·σ² - B·σ³ - C·σ⁴ [+ i·loss_scale·D·σ⁴])
        let s2 = sigma * sigma;
        let s3 = s2 * sigma;
        let s4 = s2 * s2;

        let base = nco * (one - a * s2 - b * s3 - c_coeff * s4);

        if self.loss_on {
            base + nco * C64::new(0.0, self.loss_scale) * d * s4
        } else {
            // Val{false}: force imag part to zero
            C64::new(base.re, 0.0)
        }
    }

    /// In-place batch evaluation over a frequency grid.
    ///
    /// Called by the FFI entry point. `nco_re[i]` / `nco_im[i]` and
    /// `ncl_re[i]` / `ncl_im[i]` are Julia's Sellmeier-evaluated refractive
    /// indices.  All slices must have length `omegas.len()`.
    pub fn neff_vector(
        &self,
        omegas: &[f64],
        nco_re: &[f64],
        nco_im: &[f64],
        ncl_re: &[f64],
        ncl_im: &[f64],
        radius: f64,
        neff_re_out: &mut [f64],
        neff_im_out: &mut [f64],
    ) {
        for i in 0..omegas.len() {
            let nco = C64::new(nco_re[i], nco_im[i]);
            let ncl = C64::new(ncl_re[i], ncl_im[i]);
            let ne = self.neff_one(omegas[i], nco, ncl, radius);
            neff_re_out[i] = ne.re;
            neff_im_out[i] = ne.im;
        }
    }
}

// ─── Marcatili hollow-waveguide neff ─────────────────────────────────────────

/// Batch-evaluates the Marcatili hollow-waveguide effective index
/// `neff = sqrt(εco − nwg)` (`:full` model) or `1 + (εco−1)/2 − nwg` (`:reduced` model)
/// over a packed frequency grid, given per-step gas-index arrays from Julia.
///
/// The waveguide factor `nwg(ω)` depends on the glass index, Bessel root, and
/// core radius, but NOT on the gas filling. For a constant-core-radius mode it
/// is precomputed once at setup time and stored here; Julia only needs to supply
/// `nco(ω; z)` per propagation step (evaluated by its own multi-term Sellmeier).
///
/// Mirrors `Capillary.neff(m::MarcatiliMode, εco, nwg)` with the same clamping
/// (`n.re < 1e-3 → 1e-3`) and loss-flag semantics.
///
/// # Model encoding
/// - `model = 0` → `:full`    model: `neff = sqrt(complex(εco − nwg))`
/// - `model = 1` → `:reduced` model: `neff = 1 + (εco−1)/2 − nwg`
///
/// # Loss encoding
/// - `loss_on = true`  → return complex neff (imaginary part from `nwg`)
/// - `loss_on = false` → force imaginary part to zero
#[derive(Debug, Clone)]
pub struct MarcatiliNeff {
    /// Precomputed `nwg_re[i]` for frequency point `i` in the packed sidcs grid
    nwg_re: Vec<f64>,
    /// Precomputed `nwg_im[i]` for frequency point `i` in the packed sidcs grid
    nwg_im: Vec<f64>,
    model: u8,
    loss_on: bool,
}

impl MarcatiliNeff {
    /// Construct a new handle, taking ownership of the packed `nwg` vectors.
    pub fn new(nwg_re: Vec<f64>, nwg_im: Vec<f64>, model: u8, loss_on: bool) -> Self {
        debug_assert_eq!(
            nwg_re.len(),
            nwg_im.len(),
            "nwg_re and nwg_im must have the same length"
        );
        Self {
            nwg_re,
            nwg_im,
            model,
            loss_on,
        }
    }

    /// In-place batch evaluation.  `nco_re[i]` / `nco_im[i]` are the gas refractive
    /// index at the current propagation position `z` (supplied by Julia Sellmeier).
    /// All four slices must have length `self.nwg_re.len()`.
    pub fn neff_vector(
        &self,
        nco_re: &[f64],
        nco_im: &[f64],
        neff_re_out: &mut [f64],
        neff_im_out: &mut [f64],
    ) {
        let n = self.nwg_re.len();
        for i in 0..n {
            let nco = C64::new(nco_re[i], nco_im[i]);
            let nwg = C64::new(self.nwg_re[i], self.nwg_im[i]);
            let eps = nco * nco; // εco = nco²

            // No clamping here — mirrors Julia's neff(m, εco, nwg) two-argument overload
            // (Capillary.jl:216-234) which returns the bare sqrt/linear expression.
            // Clamping only appears in the per-step neff(m, ω, εco, vn, a) overload.
            let ne: C64 = match self.model {
                0 => {
                    // :full — neff = sqrt(complex(εco − nwg))
                    let raw = (eps - nwg).sqrt();
                    if self.loss_on {
                        raw
                    } else {
                        C64::new(raw.re, 0.0)
                    }
                }
                _ => {
                    // :reduced — neff = 1 + (εco−1)/2 − nwg
                    let one = C64::new(1.0, 0.0);
                    let raw = one + (eps - one) / 2.0 - nwg;
                    if self.loss_on {
                        raw
                    } else {
                        C64::new(raw.re, 0.0)
                    }
                }
            };

            neff_re_out[i] = ne.re;
            neff_im_out[i] = ne.im;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sellmeier_gas_d2n_dw2() {
        let gas = SellmeierGas {
            b1: 1.2,
            c1: 3.4e15,
        };
        let w = 1.0e7;
        let h = 1.0e1;

        let exact = gas.d2n_dω2(w);
        let fd = (gas.dn_dω(w + h) - gas.dn_dω(w - h)) / (2.0 * h);

        assert!((exact - fd).abs() / exact.abs() < 1e-10);
    }

    #[test]
    fn test_sellmeier_gas_dispersion() {
        let gas = SellmeierGas { b1: 0.5, c1: 4.0 };
        let w = 1.0;

        // den = c1 - w*w = 4.0 - 1.0 = 3.0
        // n2 = 1.0 + (b1 * c1) / den = 1.0 + (0.5 * 4.0) / 3.0 = 1.0 + 2.0/3.0 = 5.0/3.0
        let expected_n = (5.0 / 3.0_f64).sqrt();
        let n = gas.refractive_index(w);
        assert!((n - expected_n).abs() < 1e-12);

        // dn_dω:
        // d_n2 = (2.0 * b1 * c1 * w) / (den * den) = (2.0 * 0.5 * 4.0 * 1.0) / (9.0) = 4.0 / 9.0
        // dn_dω = d_n2 / (2.0 * n)
        let expected_dn = (4.0 / 9.0) / (2.0 * expected_n);
        let dn = gas.dn_dω(w);
        assert!((dn - expected_dn).abs() < 1e-12);

        // d2n_dω2:
        // d2_n2 = (2.0 * b1 * c1 * (c1 + 3.0 * w * w)) / (den^3)
        //       = (2.0 * 0.5 * 4.0 * (4.0 + 3.0 * 1.0)) / 27.0 = 28.0 / 27.0
        // d2n_dω2 = (d2_n2 * n - d_n2 * dn) / (2.0 * n^2)
        let d_n2 = 4.0 / 9.0;
        let d2_n2 = 28.0 / 27.0;
        let expected_d2n = (d2_n2 * expected_n - d_n2 * expected_dn) / (2.0 * (5.0 / 3.0));
        let d2n = gas.d2n_dω2(w);
        assert!((d2n - expected_d2n).abs() < 1e-12);
    }
}
