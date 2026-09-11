"""Scaled adaptive stencils matching Maths.derivative/FiniteDifferences.jl.

Adapted from FiniteDifferences.jl 0.12.34 (MIT; see licenses/FiniteDifferences.txt).
"""
from fractions import Fraction
from functools import lru_cache
import math
import operator

import numpy as np


def _sum(values):
    # StaticArrays uses a left fold, including the first zero coefficient.
    values = iter(values)
    total = next(values)
    for value in values:
        total = total + value
    return total


@lru_cache(maxsize=128)
def _coefficients(grid, order):
    n = len(grid)
    matrix = [[Fraction(x)**i for x in grid]+[Fraction(math.factorial(order) if i==order else 0)] for i in range(n)]
    for i in range(n):
        pivot = next(j for j in range(i,n) if matrix[j][i])
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        row = matrix[i]; factor = row[i]
        matrix[i] = [x/factor for x in row]
        for j in range(n):
            if j != i:
                factor = matrix[j][i]
                matrix[j] = [x-factor*y for x,y in zip(matrix[j],matrix[i])]
    return tuple(float(row[-1]) for row in matrix)


@lru_cache(maxsize=32)
def _stencil(points, order):
    half = points//2
    grid = tuple(range(-half,half+1)) if points%2 else tuple(range(-half,0))+tuple(range(1,half+1))
    coefficients = _coefficients(grid,order)
    magnitude = _sum(abs(c*g**points) for c,g in zip(coefficients,grid))/math.factorial(points)
    error = _sum(abs(c) for c in coefficients)
    return grid, coefficients, magnitude, error


def _step(points, order, magnitude, error):
    _,_,mult,emult = _stencil(points,order)
    return (order/(points-order)*(error*emult/(magnitude*mult)))**(1/points)


def _estimate(values, coefficients, step, order):
    return _sum(f*c for f,c in zip(values,coefficients))/step**order


def _evaluate(function, grid, x, step):
    values = [function(x+step*g) for g in grid]
    if not np.all(np.isfinite(values)):
        raise ValueError('derivative callback returned nonfinite values')
    if any(np.ndim(value) != 0 for value in values):
        raise ValueError('derivative callback must return scalar values')
    return values


def derivative(function, x, order=1):
    """Derivative of a scalar function; optional array x is evaluated serially."""
    order = operator.index(order)
    if not 0 <= order <= 7:
        raise ValueError('derivative order must be in 0..7')
    axis = np.asarray(x,dtype=float)
    if not np.all(np.isfinite(axis)):
        raise ValueError('derivative positions must be finite')
    values = [_derivative_scalar(function,float(value),order) for value in axis.flat]
    output = np.array(values).reshape(axis.shape)
    return output.item() if output.ndim==0 else output


def _derivative_scalar(function, x, order):
    if order == 0:
        return _evaluate(function,(0,),x,1.)[0]
    scale = x if abs(x)>0 else 1.
    f = lambda value: function(value*scale)
    center = x/scale
    points = min(order+6,11)
    grid,coefficients,_,_ = _stencil(points,order)
    bound_grid,_,_,_ = _stencil(points+2,points)
    bound_step = _step(points+2,points,10.,np.finfo(float).eps)
    samples = _evaluate(f,bound_grid,center,bound_step)
    bounds = [_estimate(samples,_coefficients(tuple(g+offset for g in bound_grid),points),bound_step,points) for offset in (-1,0,1)]
    magnitude = max(abs(value) if np.isfinite(value) else 0. for value in bounds)
    value_magnitude = max(abs(value) for value in samples)
    default = _step(points,order,10.,np.finfo(float).eps)
    step = default if magnitude==0 or value_magnitude==0 else _step(points,order,magnitude,math.ulp(value_magnitude))
    step = min(step,1000*default)
    result = _estimate(_evaluate(f,grid,center,step),coefficients,step,order)/scale**order
    if not np.isfinite(result):
        raise ValueError('derivative estimate is nonfinite')
    return result
