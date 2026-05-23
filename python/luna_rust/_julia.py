import os
import juliapkg

# Resolve absolute path to Luna-Rust.jl checkout
_this_dir = os.path.dirname(os.path.abspath(__file__))
_luna_path = os.path.abspath(os.path.join(_this_dir, "..", ".."))

# Point juliapkg to the correct Julia executable
if not os.environ.get("PYTHON_JULIAPKG_JULIA"):
    os.environ["PYTHON_JULIAPKG_JULIA"] = "/home/diego/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia"

juliapkg.require_julia("1.9")
juliapkg.add("Luna", path=_luna_path)

from juliacall import Main as _jl
_jl.seval('using Luna')
Luna = _jl.Luna
