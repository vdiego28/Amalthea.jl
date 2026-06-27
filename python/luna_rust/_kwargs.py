_ASCII_TO_UNICODE = {
    "lambda0":          "λ0",
    "lam0":             "λ0",
    "wavelength":       "λ0",
    "lambda_lims":      "λlims",
    "lam_lims":         "λlims",
    "wavelength_lims":  "λlims",
    "tau_fwhm":         "τfwhm",
    "tfwhm":            "τfwhm",
    "duration":         "τfwhm",
    "tau_w":            "τw",
    "tw":               "τw",
    "phi":              "ϕ",
    "phase":            "ϕ",
    "delta_t":          "δt",
    "dt":               "δt",
    "delta_lambda":     "Δλ",
    "dlambda":          "Δλ",
}

from ._julia import get_julia

def _to_julia_type(k, v):
    _jl, _ = get_julia()
    if isinstance(v, str):
        if k in ("gas", "pulseshape", "polarisation", "model", "plasma", "ramanmodel", "modes"):
            return _jl.Symbol(v)
    elif isinstance(v, (list, tuple)):
        if k == "modes":
            return tuple(_jl.Symbol(x) if isinstance(x, str) else x for x in v)
    return v

def translate_kwargs(kwargs: dict) -> dict:
    """Translate ASCII Python kwargs to Unicode Julia kwargs."""
    out = {}
    for k, v in kwargs.items():
        canonical = _ASCII_TO_UNICODE.get(k, k)
        if canonical in out:
            raise ValueError(f"Duplicate kwarg: '{k}' and '{canonical}' both supplied.")
        v = _to_julia_type(canonical, v)
        out[canonical] = v
    return out
