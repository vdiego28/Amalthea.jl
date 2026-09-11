"""Private normalized-knot cubic convention shared with Julia's Maths.CSpline."""
import numpy as np
from scipy.linalg import solve_banded


class _NormalizedCubic:
    def __init__(self, x, y):
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            raise ValueError('spline samples must be real')
        self.x = np.array(x, dtype=float, copy=True)
        self.y = np.array(y, dtype=float, copy=True)
        if (self.x.ndim != 1 or self.x.size < 2 or self.y.shape != self.x.shape
                or not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.y))
                or np.any(np.diff(self.x) <= 0)):
            raise ValueError('invalid normalized cubic spline samples')
        rhs = np.empty(self.y.shape)
        rhs[0] = self.y[1]-self.y[0]
        rhs[-1] = self.y[-1]-self.y[-2]
        rhs[1:-1] = self.y[2:]-self.y[:-2]
        rhs *= 3
        matrix = np.ones((3, len(self.y)))
        matrix[1] = 4
        matrix[1, [0, -1]] = 2
        matrix[0, 0] = 0
        matrix[2, -1] = 0
        self.derivative = solve_banded((1, 1), matrix, rhs)

    def __call__(self, value):
        value = np.asarray(value, dtype=float)
        i = np.clip(np.searchsorted(self.x, value, side='right')-1, 0, len(self.x)-2)
        t = (value-self.x[i])/(self.x[i+1]-self.x[i])
        y0, y1 = self.y[i], self.y[i+1]
        d0, d1 = self.derivative[i], self.derivative[i+1]
        result = y0+d0*t+(3*(y1-y0)-2*d0-d1)*t**2+(2*(y0-y1)+d0+d1)*t**3
        result = np.where(value == self.x[i], y0, result)
        return np.where(value == self.x[i+1], y1, result)
