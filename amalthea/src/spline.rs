//! Natural cubic spline (Numerical-Recipes-style): fit from `(x,y)` samples,
//! evaluate resident with a tridiagonal solve for the second derivatives done
//! once at construction. Self-contained (no external crate) — used to
//! reproduce a Julia scalar function of one variable to near its own
//! precision, given fine-enough sampling. See
//! `docs/dev/native-port/MATH.md` §3.5 for the required-`N` discussion and the
//! self-validation discipline (the caller, not this module, decides `N` and
//! checks the achieved accuracy against held-out Julia values).

pub struct CubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    y2: Vec<f64>,
}

impl CubicSpline {
    /// `x` must be strictly increasing, with `x.len() == y.len() >= 3`.
    pub fn fit(x: Vec<f64>, y: Vec<f64>) -> Self {
        let n = x.len();
        assert!(n >= 3, "CubicSpline needs at least 3 points");
        assert_eq!(x.len(), y.len(), "CubicSpline: x/y length mismatch");
        let mut y2 = vec![0.0; n];
        let mut u = vec![0.0; n];
        // Natural boundary conditions: y2[0] = y2[n-1] = 0.
        for i in 1..n - 1 {
            let sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
            let p = sig * y2[i - 1] + 2.0;
            y2[i] = (sig - 1.0) / p;
            let mut uu =
                (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
            uu = (6.0 * uu / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
            u[i] = uu;
        }
        for k in (0..n - 1).rev() {
            y2[k] = y2[k] * y2[k + 1] + u[k];
        }
        CubicSpline { x, y, y2 }
    }

    /// Evaluate at `xq`. This extrapolates via the natural-spline boundary
    /// polynomial outside `[x_min(), x_max()]`, like any cubic spline — if
    /// flat clamping to the fitted domain is required instead (e.g. to
    /// match a Julia function with a flat boundary), the caller must clamp
    /// `xq` before calling this.
    pub fn eval(&self, xq: f64) -> f64 {
        let n = self.x.len();
        let mut klo = 0usize;
        let mut khi = n - 1;
        while khi - klo > 1 {
            let k = (khi + klo) / 2;
            if self.x[k] > xq {
                khi = k;
            } else {
                klo = k;
            }
        }
        let h = self.x[khi] - self.x[klo];
        let a = (self.x[khi] - xq) / h;
        let b = (xq - self.x[klo]) / h;
        a * self.y[klo]
            + b * self.y[khi]
            + ((a * a * a - a) * self.y2[klo] + (b * b * b - b) * self.y2[khi]) * (h * h) / 6.0
    }

    pub fn x_min(&self) -> f64 {
        self.x[0]
    }
    pub fn x_max(&self) -> f64 {
        *self.x.last().unwrap()
    }
}

/// Exact evaluator for a Julia `Maths.CSpline` — a piecewise-cubic Hermite
/// spline parametrized by per-knot *first derivatives* `D` (solved via a
/// tridiagonal system Julia already did), not the second-derivative
/// ("natural") formulation `CubicSpline` above uses. **Transfers** the
/// already-fitted `(x,y,D)` from Julia rather than re-fitting a *different*
/// spline through sampled values — re-fitting a fresh natural cubic spline
/// through samples of an existing spline is a spline-of-a-spline: the error
/// concentrates at the *original* spline's knots (where the two BCs
/// disagree) and converges only ~`O(h)`, not `O(h⁴)`, so no amount of
/// resampling closes the gap (see PORT_LOG's `PhysData.densityspline`
/// postmortem — this bit real gas density via `CoolProp`, but the same
/// applies to any `Maths.CSpline`-based quantity). Transfer sidesteps the
/// problem entirely: this evaluates the *same* piecewise cubic, so it
/// matches Julia's `dspl(p)` to reassociation tier, not a spline-fit floor.
pub struct HermiteSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    d: Vec<f64>,
}

impl HermiteSpline {
    /// `x` must be strictly increasing (as built by Julia's `densityspline`,
    /// a uniform grid, though this evaluator does not assume uniformity —
    /// binary search matches any strictly-increasing `x`). `y`/`d` are
    /// `Maths.CSpline`'s stored values and first derivatives at each knot.
    pub fn from_parts(x: Vec<f64>, y: Vec<f64>, d: Vec<f64>) -> Self {
        assert!(x.len() >= 2, "HermiteSpline needs at least 2 points");
        assert_eq!(x.len(), y.len(), "HermiteSpline: x/y length mismatch");
        assert_eq!(x.len(), d.len(), "HermiteSpline: x/d length mismatch");
        HermiteSpline { x, y, d }
    }

    /// Evaluate at `x0`, mirroring `Maths.jl`'s `(c::CSpline)(x0)` exactly
    /// (same Hermite basis, same boundary-segment extrapolation for `x0`
    /// outside `[x_min(), x_max()]` — Julia's `ffast` clamps the *index*,
    /// not the value, to the first/last segment).
    pub fn eval(&self, x0: f64) -> f64 {
        let n = self.x.len();
        // Binary search for the segment [x[lo], x[hi]] containing x0,
        // clamped to the first/last segment for out-of-range x0 (matches
        // Julia's `ffast`/`FastFinder` index-clamping behavior).
        let (lo, hi) = if x0 <= self.x[0] {
            (0usize, 1usize)
        } else if x0 >= self.x[n - 1] {
            (n - 2, n - 1)
        } else {
            let mut klo = 0usize;
            let mut khi = n - 1;
            while khi - klo > 1 {
                let k = (khi + klo) / 2;
                if self.x[k] > x0 {
                    khi = k;
                } else {
                    klo = k;
                }
            }
            (klo, khi)
        };
        if x0 == self.x[hi] {
            return self.y[hi];
        }
        if x0 == self.x[lo] {
            return self.y[lo];
        }
        let t = (x0 - self.x[lo]) / (self.x[hi] - self.x[lo]);
        self.y[lo]
            + self.d[lo] * t
            + (3.0 * (self.y[hi] - self.y[lo]) - 2.0 * self.d[lo] - self.d[hi]) * t * t
            + (2.0 * (self.y[lo] - self.y[hi]) + self.d[lo] + self.d[hi]) * t * t * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_a_cubic_in_the_interior() {
        // A natural cubic spline fit to samples of a true global cubic
        // recovers that cubic exactly away from the boundary (where the
        // natural BC's zero-curvature assumption need not match the true
        // polynomial's actual curvature).
        let f = |x: f64| 1.0 + 2.0 * x - 0.5 * x * x + 0.1 * x * x * x;
        let xs: Vec<f64> = (0..61).map(|i| i as f64 * 0.5).collect(); // domain [0,30]
        let ys: Vec<f64> = xs.iter().map(|&x| f(x)).collect();
        let spl = CubicSpline::fit(xs, ys);
        for i in 0..50 {
            let xq = 12.0 + i as f64 * 0.1; // well interior, away from [0,30] ends
            let rel = (spl.eval(xq) - f(xq)).abs() / f(xq).abs();
            assert!(rel < 1e-8, "rel={rel} at xq={xq}");
        }
    }

    #[test]
    fn passes_through_knots() {
        let xs = vec![0.0, 1.0, 2.5, 4.0, 5.0];
        let ys = vec![1.0, 3.0, 2.0, -1.0, 0.5];
        let spl = CubicSpline::fit(xs.clone(), ys.clone());
        for (x, y) in xs.iter().zip(ys.iter()) {
            assert!((spl.eval(*x) - y).abs() < 1e-12);
        }
    }

    #[test]
    fn hermite_spline_matches_julia_cspline_reference() {
        // Literal cross-check against Julia's Maths.CSpline(x,y) for the
        // same x,y — D computed independently in Julia (its own tridiagonal
        // solve for first derivatives), transferred here, not re-derived.
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 3.0, 2.5, 4.0, 3.5];
        let d = vec![
            2.8482142857142856,
            0.30357142857142855,
            0.4375000000000001,
            0.9464285714285713,
            -1.2232142857142856,
        ];
        let spl = HermiteSpline::from_parts(x, y, d);
        let cases = [
            (0.5, 2.318080357142857),
            (1.5, 2.733258928571429),
            (2.5, 3.186383928571429),
            (3.5, 4.021205357142857),
            (-0.5, -0.318080357142857),
            (4.5, 2.978794642857143),
        ];
        for (xq, expected) in cases {
            let got = spl.eval(xq);
            let rel = (got - expected).abs() / expected.abs();
            assert!(
                rel < 1e-13,
                "xq={xq}: got={got}, expected={expected}, rel={rel}"
            );
        }
    }
}
