"""Julia-free setup primitives; propagation is still under development."""

from .grid import EnvGrid, RealGrid
from .solver import SolveResult, solve_precon

__version__ = "0.0.1.dev0"
__all__ = ["EnvGrid", "RealGrid", "SolveResult", "solve_precon"]
