class LunaOutput:
    """Python-side, dict-like view over a Julia `Amalthea.Output` object.

    Wraps whatever `prop_capillary`/`prop_gnlse` returned (a `MemoryOutput`,
    `HDF5Output`, or a nested `Dict`/HDF5 `Group`/`File` reached by indexing
    into one) so it can be used from Python with `output[key]`, `in`, and
    `.keys()` instead of calling into `juliacall` directly. Julia arrays are
    converted to `numpy` on access; nested output-like values are wrapped in
    another `LunaOutput` rather than returned as raw Julia objects.
    """

    def __init__(self, jl_output):
        self._jl_output = jl_output

    def __getitem__(self, key):
        """Index into the underlying Julia output, converting the result to Python."""
        from ._julia import get_julia
        _jl, _ = get_julia()
        try:
            val = self._jl_output[key]
        except Exception as e:
            raise KeyError(key) from e
        return self._to_python(val)

    def _to_python(self, val):
        """Convert a raw Julia value to its Python-side equivalent.

        Dict/Group/File/AbstractOutput values are wrapped in another
        `LunaOutput`; `AbstractArray` values are converted to `numpy` arrays;
        anything else (scalars, strings) is returned via juliacall's default
        auto-conversion.
        """
        from ._julia import get_julia
        import numpy as np
        _jl, _ = get_julia()
        # If it is a Dict, Group, File or MemoryOutput/HDF5Output
        if (_jl.isa(val, _jl.Dict) or
            _jl.isa(val, _jl.Amalthea.Output.HDF5.Group) or
            _jl.isa(val, _jl.Amalthea.Output.HDF5.File) or
            _jl.isa(val, _jl.Amalthea.Output.AbstractOutput)):
            return LunaOutput(val)

        # If it's a Julia Array, convert to numpy
        if _jl.isa(val, _jl.AbstractArray):
            return np.asarray(val)

        return val

    def __contains__(self, key):
        """Return whether `key` exists in the underlying output, dict, or HDF5 group/file."""
        from ._julia import get_julia
        _jl, _ = get_julia()
        if _jl.isa(self._jl_output, _jl.Dict):
            return _jl.haskey(self._jl_output, key)
        elif _jl.isa(self._jl_output, _jl.Amalthea.Output.AbstractOutput):
            return _jl.haskey(self._jl_output, key)
        elif _jl.isa(self._jl_output, _jl.Amalthea.Output.HDF5.Group) or _jl.isa(self._jl_output, _jl.Amalthea.Output.HDF5.File):
            return _jl.haskey(self._jl_output, key)
        return False

    def keys(self):
        """List the available keys, dispatching on the underlying Julia output type.

        Handles plain `Dict`s, in-memory `MemoryOutput` (keys of `.data`),
        on-disk `HDF5Output` (re-opens the file read-only to list top-level
        keys), HDF5 `Group`/`File` handles, and any other `AbstractOutput`
        subtype exposing a `.data` field.
        """
        from ._julia import get_julia
        _jl, _ = get_julia()
        if _jl.isa(self._jl_output, _jl.Dict):
            return list(self._jl_output.keys())
        elif _jl.isa(self._jl_output, _jl.Amalthea.Output.MemoryOutput):
            return list(self._jl_output.data.keys())
        elif _jl.isa(self._jl_output, _jl.Amalthea.Output.HDF5Output):
            fpath = self._jl_output.fpath
            return list(_jl.Amalthea.Output.HDF5.h5open(fpath, "r").keys())
        elif _jl.isa(self._jl_output, _jl.Amalthea.Output.HDF5.Group) or _jl.isa(self._jl_output, _jl.Amalthea.Output.HDF5.File):
            return list(self._jl_output.keys())
        elif _jl.isa(self._jl_output, _jl.Amalthea.Output.AbstractOutput):
            if hasattr(self._jl_output, "data"):
                return list(self._jl_output.data.keys())
        return []
