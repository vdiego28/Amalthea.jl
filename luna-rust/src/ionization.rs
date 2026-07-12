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

        Self {
            segments,
            x_min,
            x_max,
        }
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
            alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i]) - (3.0 / h[i - 1]) * (a[i] - a[i - 1]);
        }

        let mut l = vec![1.0f64; n];
        let mut mu = vec![0.0f64; n];
        let mut z = vec![0.0f64; n];

        for i in 1..(n - 1) {
            l[i] = 2.0 * (h[i - 1] + h[i]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        // l[n-1] = 1, z[n-1] = 0 (natural BC: c[n-1] = 0)

        // Back-substitution
        let mut c = vec![0.0f64; n];
        let mut b = vec![0.0f64; n - 1];
        let mut d = vec![0.0f64; n - 1];

        for j in (0..(n - 1)).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
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
    /// or is non-finite (NaN/±inf) — IEEE 754 makes every comparison with NaN
    /// false, so `x < x_min || x > x_max` alone would silently let a NaN
    /// through to the segment lookup below instead of being rejected here.
    pub fn evaluate(&self, x: f64) -> Option<f64> {
        if !x.is_finite() || x < self.x_min || x > self.x_max {
            return None;
        }

        // Binary search to find segment
        let mut low = 0;
        let mut high = self.segments.len() - 1;
        while low < high {
            let mid = (low + high).div_ceil(2);
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
    /// When `false` (the default), `rate()` clamps fields above `e_max` to
    /// `rate(e_max)` — matching Julia's `IonRatePPTAccel`, which the adaptive
    /// stepper relies on when it probes rejected trial points above the
    /// lookup table's upper bound. Set `true` (via [`Self::strict`]) to
    /// restore the old hard-error behaviour for debugging.
    pub strict: bool,
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

        Self {
            spline_lut,
            e_min,
            e_max,
            strict: false,
        }
    }

    /// Opt into the old strict behaviour: `rate()` errors above `e_max`
    /// instead of clamping to `rate(e_max)`. Intended for debugging only.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
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
        let ln_rates: Vec<f64> = rate_values
            .iter()
            .map(|&r| if r > 0.0 { r.ln() } else { -100.0 })
            .collect();
        let spline_lut = CubicSplineLUT::from_samples(e_values, &ln_rates);
        Self {
            spline_lut,
            e_min,
            e_max,
            strict: false,
        }
    }

    /// Calculates the ionization rate. Fields below `e_min` return 0 (Keldysh
    /// calculation diverges there); fields above `e_max` clamp to `rate(e_max)`
    /// (matching Julia's `IonRatePPTAccel` — the adaptive stepper legitimately
    /// probes rejected trial points above the table's upper bound), unless
    /// [`Self::strict`] was set, in which case they return `Err`.
    pub fn rate(&self, field_strength: f64) -> Result<f64, String> {
        let abs_e = field_strength.abs();

        // docs/dev/BACKLOG.md S1 item 2: a NaN/±inf field (e.g. from an overshooting
        // fixed-dt integration far beyond flength) fails both the `< e_min`
        // and `> e_max` comparisons below (IEEE 754: any comparison with NaN
        // is false). Caught here explicitly rather than relying solely on
        // `spline_lut.evaluate`'s own non-finite guard (added alongside this
        // one — `evaluate(NaN)` previously returned `Some(garbage)` instead
        // of `None`, confirmed by `cubic_spline_lut_rejects_out_of_range_without_panicking`
        // failing before that fix). Route non-finite input through the same
        // overflow policy as a too-large-but-finite field (clamp to
        // `rate(e_max)`, or `Err` under `strict`).
        if !abs_e.is_finite() {
            if self.strict {
                return Err(format!(
                    "Electric field amplitude {} V/m is not finite (NaN or ±inf)",
                    field_strength
                ));
            }
            return self.rate(self.e_max);
        }

        if abs_e < self.e_min {
            // Cutoff region: rate is physically negligible and Keldysh calculation diverges
            return Ok(0.0);
        }

        if abs_e > self.e_max {
            if self.strict {
                return Err(format!(
                    "Electric field amplitude {:.2e} V/m exceeds PPT lookup table upper bound {:.2e} V/m",
                    abs_e, self.e_max
                ));
            }
            return self.rate(self.e_max);
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

        if let Some(ctx) = crate::cuda::get_gpu_context()
            && self.rate_vector_gpu(ctx, fields, rates).is_ok()
        {
            return Ok(());
        }

        for i in 0..n_t {
            rates[i] = self.rate(fields[i])?;
        }
        Ok(())
    }

    fn rate_vector_gpu(
        &self,
        ctx: &crate::cuda::GpuContext,
        fields: &[f64],
        rates: &mut [f64],
    ) -> Result<(), String> {
        let n_t = fields.len();
        let num_segments = self.spline_lut.segments.len();

        let d_fields = crate::cuda::GpuBuffer::alloc(std::mem::size_of_val(fields))?;
        let d_rates = crate::cuda::GpuBuffer::alloc(std::mem::size_of_val(fields))?;
        let d_segments =
            crate::cuda::GpuBuffer::alloc(num_segments * std::mem::size_of::<SplineSegment>())?;
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

        let mut strict_val = if self.strict {
            1 as libc::c_int
        } else {
            0 as libc::c_int
        };
        let mut args: [*mut libc::c_void; 9] = [
            &mut d_fields_ptr as *mut _ as *mut libc::c_void,
            &mut d_rates_ptr as *mut _ as *mut libc::c_void,
            &mut d_segments_ptr as *mut _ as *mut libc::c_void,
            &mut e_min_val as *mut _ as *mut libc::c_void,
            &mut e_max_val as *mut _ as *mut libc::c_void,
            &mut num_segments_val as *mut _ as *mut libc::c_void,
            &mut n_t_val as *mut _ as *mut libc::c_void,
            &mut d_err_ptr as *mut _ as *mut libc::c_void,
            &mut strict_val as *mut _ as *mut libc::c_void,
        ];

        let block_size = 256;
        let grid_size = n_t.div_ceil(block_size);

        crate::cuda::activate_context()?;
        let driver = crate::cuda::get_driver_api()?;

        unsafe {
            let res = (driver.cuLaunchKernel)(
                ctx.ppt_fn,
                grid_size as libc::c_uint,
                1,
                1,
                block_size as libc::c_uint,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );

            if res != 0 {
                return Err(format!(
                    "cuLaunchKernel for ppt_ionization_kernel failed: {}",
                    res
                ));
            }
        }

        let mut err_res = [0 as libc::c_int];
        d_err.copy_to_host(&mut err_res)?;
        if err_res[0] != 0 {
            return Err(
                "Electric field amplitude exceeded upper bound of ionization rate lookup table"
                    .to_string(),
            );
        }

        d_rates.copy_to_host(rates)?;
        Ok(())
    }
}

/// ADK (Ammosov-Delone-Krainov) ionization rate — a closed-form analytic
/// formula, unlike PPT's cubic-spline LUT (docs/dev/BACKLOG.md Phase I item 3:
/// `ionisation=:ADK` in `Interface.jl` builds `Ionisation.IonRateADK`, a
/// direct evaluation, not `IonRatePPTAccel`). Precomputed constants are
/// passed in from Julia's own `IonRateADK` (`src/Ionisation.jl:131-174`,
/// same field names) rather than recomputed here, so the two paths use
/// bit-identical `nstar`/`cn_sq`/etc. — the recursion for `cn_sq` involves
/// `gamma(nstar+1)*gamma(nstar))`, which we don't need to reimplement in
/// Rust to stay analytic (this is passing exact constants, not fitting a
/// LUT to samples — no LUT-approximation-of-noise risk here, see
/// `feedback_prefer_analytic_over_lut`).
#[derive(Debug, Clone, Copy)]
pub struct AdkIonizationRate {
    pub occupancy: f64,
    pub omega_p: f64,
    pub cn_sq: f64,
    pub nstar: f64,
    pub omega_t_prefac: f64,
    pub thr: f64,
    /// `avfac` from Julia's `cycle_average` branch; `1.0` when disabled
    /// (the `avfac != 1` branch is skipped in that case, exactly like Julia).
    pub avfac: f64,
}

impl AdkIonizationRate {
    /// Reproduces `(ir::IonRateADK)(E)` (`src/Ionisation.jl:176-189`) exactly:
    /// zero below threshold, otherwise the closed-form ADK rate with the
    /// optional cycle-averaging correction.
    pub fn rate(&self, field_strength: f64) -> Result<f64, String> {
        let a_e = field_strength.abs();
        // Same defensive non-finite guard as PptIonizationRate::rate (docs/dev/BACKLOG.md
        // S1 item 2): `a_e < self.thr` is false for NaN, so without this it
        // would fall through into `x.powf(...)`/`.exp()` and return `Ok(NaN)`
        // silently rather than the documented "never errors, always 0 below
        // threshold" contract.
        if !a_e.is_finite() || a_e < self.thr {
            return Ok(0.0);
        }
        let x = 4.0 * self.omega_p / (self.omega_t_prefac * a_e);
        let mut r = self.occupancy
            * self.omega_p
            * self.cn_sq
            * x.powf(2.0 * self.nstar - 1.0)
            * (-4.0 / 3.0 * self.omega_p / (self.omega_t_prefac * a_e)).exp();
        if self.avfac != 1.0 {
            r *= self.avfac * a_e.sqrt();
        }
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // docs/dev/BACKLOG.md S1 item 2: "a rate query must never be able to segfault
    // regardless of input" — hammer PptIonizationRate::rate/CubicSplineLUT
    // across many orders of magnitude beyond the fitted [e_min, e_max]
    // range, plus non-finite inputs, and assert every call returns cleanly
    // (Ok or a clean Err, never a panic/segfault).

    fn make_ppt(e_min: f64, e_max: f64) -> PptIonizationRate {
        // A simple, well-behaved representative rate function: not the real
        // PPT formula, just something monotonic and strictly positive so
        // the spline is well-conditioned and clamp behavior is checkable.
        PptIonizationRate::new(e_min, e_max, 32, |e| e.abs() + 1.0)
    }

    #[test]
    fn ppt_below_e_min_is_zero() {
        let ppt = make_ppt(1.0, 100.0);
        assert_eq!(ppt.rate(0.0).unwrap(), 0.0);
        assert_eq!(ppt.rate(0.5).unwrap(), 0.0);
        assert_eq!(ppt.rate(-0.5).unwrap(), 0.0); // abs() symmetry
    }

    #[test]
    fn ppt_within_range_is_positive_and_finite() {
        let ppt = make_ppt(1.0, 100.0);
        for e in [1.0, 2.0, 50.0, 99.0, 100.0] {
            let r = ppt.rate(e).unwrap();
            assert!(r.is_finite() && r > 0.0, "rate({e}) = {r}");
        }
    }

    #[test]
    fn ppt_clamps_above_e_max_instead_of_erroring() {
        let ppt = make_ppt(1.0, 100.0);
        let at_max = ppt.rate(100.0).unwrap();
        for factor in [1.01, 2.0, 1e3, 1e6, 1e10] {
            let r = ppt.rate(100.0 * factor).unwrap();
            assert_eq!(r, at_max, "factor={factor}");
            // Negative overshoot (abs() symmetry) must clamp identically.
            let r_neg = ppt.rate(-100.0 * factor).unwrap();
            assert_eq!(r_neg, at_max, "factor={factor} (negative)");
        }
    }

    #[test]
    fn ppt_far_below_e_min_is_still_zero_not_a_panic() {
        let ppt = make_ppt(1.0, 100.0);
        for factor in [0.99, 0.5, 1e-3, 1e-6, 1e-10] {
            assert_eq!(ppt.rate(1.0 * factor).unwrap(), 0.0, "factor={factor}");
        }
    }

    #[test]
    fn ppt_non_finite_inputs_never_panic() {
        let ppt = make_ppt(1.0, 100.0);
        let at_max = ppt.rate(100.0).unwrap();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let r = ppt.rate(bad).unwrap();
            assert_eq!(r, at_max, "bad input {bad}");
        }
    }

    #[test]
    fn ppt_strict_mode_errors_cleanly_instead_of_clamping() {
        let ppt = make_ppt(1.0, 100.0).strict(true);
        assert!(ppt.rate(100.0).is_ok());
        assert!(ppt.rate(101.0).is_err());
        assert!(ppt.rate(f64::NAN).is_err());
        assert!(ppt.rate(f64::INFINITY).is_err());
    }

    #[test]
    fn ppt_extreme_sweep_never_panics() {
        // docs/dev/BACKLOG.md's own reproduction recipe: hammer across ±1e10x the
        // fitted range on a log-spaced sweep, both signs, non-strict.
        let ppt = make_ppt(1e-3, 1e6);
        let mut e = 1e-13; // 1e10x below e_min
        while e <= 1e16 {
            // 1e10x above e_max
            for sign in [1.0, -1.0] {
                let r = ppt.rate(sign * e).unwrap();
                assert!(r.is_finite() && r >= 0.0, "rate({}) = {r}", sign * e);
            }
            e *= 3.7; // irrational-ish step to avoid hitting exact grid knots only
        }
    }

    #[test]
    fn cubic_spline_lut_rejects_out_of_range_without_panicking() {
        let lut = CubicSplineLUT::fit(1.0, 10.0, 8, |x| x * x);
        assert!(lut.evaluate(0.999).is_none());
        assert!(lut.evaluate(10.001).is_none());
        assert!(lut.evaluate(f64::NAN).is_none());
        assert!(lut.evaluate(f64::INFINITY).is_none());
        assert!(lut.evaluate(f64::NEG_INFINITY).is_none());
        // Exactly on the boundary must succeed (inclusive range).
        assert!(lut.evaluate(1.0).is_some());
        assert!(lut.evaluate(10.0).is_some());
    }

    fn make_adk() -> AdkIonizationRate {
        AdkIonizationRate {
            occupancy: 2.0,
            omega_p: 1.0,
            cn_sq: 1.0,
            nstar: 1.0,
            omega_t_prefac: 1.0,
            thr: 1.0,
            avfac: 1.0,
        }
    }

    #[test]
    fn adk_below_threshold_is_zero() {
        let adk = make_adk();
        assert_eq!(adk.rate(0.0).unwrap(), 0.0);
        assert_eq!(adk.rate(0.5).unwrap(), 0.0);
    }

    #[test]
    fn adk_non_finite_inputs_never_panic() {
        let adk = make_adk();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(adk.rate(bad).unwrap(), 0.0, "bad input {bad}");
        }
    }

    #[test]
    fn adk_extreme_sweep_never_panics() {
        let adk = make_adk();
        let mut e = 1e-10;
        while e <= 1e10 {
            for sign in [1.0, -1.0] {
                let r = adk.rate(sign * e).unwrap();
                assert!(r.is_finite(), "rate({}) = {r}", sign * e);
            }
            e *= 3.7;
        }
    }
}
