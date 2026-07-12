import os
import threading

# Resolve absolute path to Amalthea.jl checkout
_this_dir = os.path.dirname(os.path.abspath(__file__))
_luna_path = os.path.abspath(os.path.join(_this_dir, "..", ".."))

_jl = None
Amalthea = None
_init_lock = threading.Lock()


def get_julia():
    """Load Julia and the Amalthea module on first use."""
    global _jl, Amalthea
    if _jl is not None and Amalthea is not None:
        return _jl, Amalthea

    with _init_lock:
        if _jl is None or Amalthea is None:
            import juliapkg

            # Point juliapkg to a Julia executable only if the caller has not already done so.
            if not os.environ.get("PYTHON_JULIAPKG_JULIA"):
                os.environ["PYTHON_JULIAPKG_JULIA"] = os.environ.get("JULIA", "julia")

            juliapkg.require_julia("1.9")
            juliapkg.add("Amalthea", path=_luna_path)

            from juliacall import Main as _main
            _main.seval("using Amalthea")
            _jl = _main
            Amalthea = _main.Amalthea

    return _jl, Amalthea
