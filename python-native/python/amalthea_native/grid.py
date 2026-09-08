"""Time grids following Grid.jl, including its FFT ordering and crop rounding."""

import math
import operator

import numpy as np

# CODATA2014 speed of light used by PhysData.jl (exact SI value).
C = 299792458.0


def _positive(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _power_of_two(value):
    return 2 ** max(0, math.ceil(math.log2(value)))


def planck_taper(x, left0, left1, right1, right0):
    """Port of Maths.planck_taper, retaining centered-coordinate arithmetic."""
    x0 = (right0 + left0) / 2
    xc = np.asarray(x, dtype=np.float64) - x0
    width = right0 - left0
    el = abs(left1 - left0) / width
    er = abs(right0 - right1) / width
    x1, x2 = -width / 2, -width / 2 * (1 - 2 * el)
    x3, x4 = width / 2 * (1 - 2 * er), width / 2
    result = np.zeros_like(xc)
    result[(x2 <= xc) & (xc <= x3)] = 1
    # Branch only on interior points to avoid endpoint division by zero.
    with np.errstate(over="ignore"):
        for lo, hi, numerator in ((x1, x2, x2 - x1), (x3, x4, x3 - x4)):
            selected = (lo < xc) & (xc < hi)
            t = xc[selected]
            z = numerator / (t - lo) + numerator / (t - hi)
            result[selected] = 1 / (1 + np.exp(z))
    return result


class _TimeGrid:
    def _inputs(self, zmax, reference_lambda, lambda_lims, trange, delta_t, max_samples):
        self.zmax = _positive("zmax", zmax)
        self.reference_lambda = _positive("reference_lambda", reference_lambda)
        bounds = np.asarray(lambda_lims, dtype=np.float64)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
            raise ValueError("lambda_lims must contain two finite wavelengths")
        if not 0 < bounds[0] < bounds[1]:
            raise ValueError("lambda_lims must be positive and increasing")
        if not bounds[0] < self.reference_lambda < bounds[1]:
            raise ValueError("reference_lambda must lie inside lambda_lims")
        self.lambda_lims = tuple(bounds)
        self.trange = _positive("trange", trange)
        delta_t = _positive("delta_t", delta_t)
        self._max_samples = operator.index(max_samples)
        if self._max_samples < 2:
            raise ValueError("max_samples must be at least two")
        return C / float(bounds[1]), C / float(bounds[0]), delta_t

    def _fine_grid(self, delta_to):
        delta_to = _positive("fine time spacing", delta_to)
        ratio = self.trange / delta_to
        if not math.isfinite(ratio) or ratio > self._max_samples:
            raise ValueError("requested grid exceeds max_samples")
        samples = max(2, _power_of_two(ratio))
        if samples > self._max_samples:
            raise ValueError("rounded grid exceeds max_samples")
        self.to = (np.arange(samples, dtype=float) - samples / 2) * delta_to
        return samples, 2 * math.pi / (delta_to * samples)

    def _windows(self, omega_min, omega_max, omega_max_win):
        self.sidx = (self.omega > omega_min / 2) & (self.omega < omega_max_win)
        self.omega_win = planck_taper(
            self.omega, omega_min / 2, omega_min, omega_max, omega_max_win
        )
        self.twin = planck_taper(
            self.t, self.t.min(), -self.trange / 2, self.trange / 2, self.t.max()
        )
        self.towin = planck_taper(
            self.to, self.to.min(), -self.trange / 2, self.trange / 2, self.to.max()
        )

    @property
    def ω(self):
        return self.omega

    @property
    def ωo(self):
        return self.omega_over

    @property
    def ωwin(self):
        return self.omega_win

    @property
    def referenceλ(self):
        return self.reference_lambda


class RealGrid(_TimeGrid):
    """Carrier-resolved grid; lengths and SI units match Julia's RealGrid.

    ``max_samples`` bounds the oversampled allocation, never its resolution.
    """

    is_real = True

    def __init__(self, zmax, reference_lambda, lambda_lims, trange, delta_t=1.0,
                 *, max_samples=2**24):
        fmin, fmax, delta_t = self._inputs(
            zmax, reference_lambda, lambda_lims, trange, delta_t, max_samples
        )
        delta_to = min(1 / (6 * fmax), delta_t)
        samples, delta_omega = self._fine_grid(delta_to)
        self.omega_over = np.arange(samples // 2 + 1, dtype=float) * delta_omega
        omin, omax = 2 * math.pi * fmin, 2 * math.pi * fmax
        omax_win = 1.1 * omax
        # Julia rounds findfirst's ONE-BASED index, then adds the Nyquist bin.
        index = np.searchsorted(self.omega_over, omax_win, side="right")
        count = _power_of_two(int(index) + 1) + 1
        if count > self.omega_over.size:
            raise ValueError("requested frequency window does not fit the fine grid")
        self.omega = self.omega_over[:count].copy()
        dt = math.pi / self.omega[-1]
        nt = (count - 1) * 2
        self.t = (np.arange(nt, dtype=float) - nt / 2) * dt
        self._windows(omin, omax, omax_win)


class EnvGrid(_TimeGrid):
    """Complex envelope grid in unshifted FFT order, with absolute frequencies."""

    is_real = False

    def __init__(self, zmax, reference_lambda, lambda_lims, trange, *, delta_t=1.0,
                 thg=False, max_samples=2**24):
        fmin, fmax, delta_t = self._inputs(
            zmax, reference_lambda, lambda_lims, trange, delta_t, max_samples
        )
        if not isinstance(thg, (bool, np.bool_)):
            raise TypeError("thg must be a boolean")
        self.omega0 = 2 * math.pi * C / self.reference_lambda
        fmax_win = 1.1 * fmax
        delta_tf = 1 / ((6 if thg else 2) * (
            (fmax if thg else fmax_win) - C / self.reference_lambda
        ))
        delta_to = min(delta_tf, delta_t)
        oversampling = bool(thg) if delta_tf <= delta_t else True
        samples, delta_omega = self._fine_grid(delta_to)
        vo = np.fft.fftshift((np.arange(samples, dtype=float) - samples / 2) * delta_omega)
        self.omega_over = vo + self.omega0
        omin, omax, omax_win = (2 * math.pi * f for f in (fmin, fmax, fmax_win))
        if oversampling:
            matches = np.flatnonzero(self.omega_over >= omax_win - delta_omega)
            if matches.size == 0:
                raise ValueError("requested frequency window does not fit the fine grid")
            count = _power_of_two(int(matches[0]) + 1)
            if 2 * count > samples:
                raise ValueError("coarse spectrum would overlap its negative-frequency half")
            v = np.concatenate((vo[:count], vo[-count:]))
        else:
            v = vo.copy()
        dt = -math.pi / v.min()
        self.t = (np.arange(v.size, dtype=float) - v.size / 2) * dt
        self.omega = v + self.omega0
        self._windows(omin, omax, omax_win)

    @property
    def ω0(self):
        return self.omega0
