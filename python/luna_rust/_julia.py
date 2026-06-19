import os
import threading

# Resolve absolute path to Luna-Rust.jl checkout
_this_dir = os.path.dirname(os.path.abspath(__file__))
_luna_path = os.path.abspath(os.path.join(_this_dir, "..", ".."))

_jl = None
Luna = None
_init_lock = threading.Lock()


def get_julia():
    """Load Julia and the Luna module on first use."""
    global _jl, Luna
    if _jl is not None and Luna is not None:
        return _jl, Luna

    with _init_lock:
        if _jl is None or Luna is None:
            import juliapkg

            # Point juliapkg to a Julia executable only if the caller has not already done so.
            if not os.environ.get("PYTHON_JULIAPKG_JULIA"):
                os.environ["PYTHON_JULIAPKG_JULIA"] = os.environ.get("JULIA", "julia")

            juliapkg.require_julia("1.9")
            juliapkg.add("Luna", path=_luna_path)

            from juliacall import Main as _main
            _main.seval("import Pkg; Pkg.build(\"PyCall\")")
            _main.seval("using Luna")
            _jl = _main
            Luna = _main.Luna

    return _jl, Luna
