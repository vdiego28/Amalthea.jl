"""luna_rust — Python interface to the Amalthea.jl simulation backend."""
from ._julia import get_julia
from ._kwargs import translate_kwargs
from .output import LunaOutput

def prop_capillary(radius, flength, gas, pressure, **kwargs):
    """Simulate pulse propagation in a hollow-core capillary fibre.

    Accepts both Unicode (λ0=800e-9) and ASCII (lambda0=800e-9) keyword
    arguments. Returns an Output object with numpy-backed arrays.
    """
    jl_kwargs = translate_kwargs(kwargs)
    _jl, Amalthea = get_julia()

    if isinstance(gas, str):
        gas = _jl.Symbol(gas)
    if isinstance(pressure, (tuple, list)):
        pressure = tuple(pressure)

    result = Amalthea.prop_capillary(radius, flength, gas, pressure, **jl_kwargs)
    return LunaOutput(result)

def prop_gnlse(gamma, flength, betas, **kwargs):
    """Simulate pulse propagation using GNLSE."""
    jl_kwargs = translate_kwargs(kwargs)
    _jl, Amalthea = get_julia()
    if isinstance(betas, (list, tuple)):
        betas = tuple(betas)
    result = Amalthea.prop_gnlse(gamma, flength, betas, **jl_kwargs)
    return LunaOutput(result)
