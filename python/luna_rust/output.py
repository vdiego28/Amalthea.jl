import numpy as np

class LunaOutput:
    def __init__(self, jl_output):
        self._jl_output = jl_output

    def __getitem__(self, key):
        from ._julia import _jl
        try:
            val = self._jl_output[key]
        except Exception as e:
            raise KeyError(key) from e
        return self._to_python(val)

    def _to_python(self, val):
        from ._julia import _jl
        # If it is a Dict, Group, File or MemoryOutput/HDF5Output
        if (_jl.isa(val, _jl.Dict) or 
            _jl.isa(val, _jl.Luna.Output.HDF5.Group) or 
            _jl.isa(val, _jl.Luna.Output.HDF5.File) or 
            _jl.isa(val, _jl.Luna.Output.AbstractOutput)):
            return LunaOutput(val)
        
        # If it's a Julia Array, convert to numpy
        if _jl.isa(val, _jl.AbstractArray):
            return np.asarray(val)
            
        return val

    def __contains__(self, key):
        from ._julia import _jl
        if _jl.isa(self._jl_output, _jl.Dict):
            return _jl.haskey(self._jl_output, key)
        elif _jl.isa(self._jl_output, _jl.Luna.Output.AbstractOutput):
            return _jl.haskey(self._jl_output, key)
        elif _jl.isa(self._jl_output, _jl.Luna.Output.HDF5.Group) or _jl.isa(self._jl_output, _jl.Luna.Output.HDF5.File):
            return _jl.haskey(self._jl_output, key)
        return False

    def keys(self):
        from ._julia import _jl
        if _jl.isa(self._jl_output, _jl.Dict):
            return list(self._jl_output.keys())
        elif _jl.isa(self._jl_output, _jl.Luna.Output.MemoryOutput):
            return list(self._jl_output.data.keys())
        elif _jl.isa(self._jl_output, _jl.Luna.Output.HDF5Output):
            fpath = self._jl_output.fpath
            return list(_jl.Luna.Output.HDF5.h5open(fpath, "r").keys())
        elif _jl.isa(self._jl_output, _jl.Luna.Output.HDF5.Group) or _jl.isa(self._jl_output, _jl.Luna.Output.HDF5.File):
            return list(self._jl_output.keys())
        elif _jl.isa(self._jl_output, _jl.Luna.Output.AbstractOutput):
            if hasattr(self._jl_output, "data"):
                return list(self._jl_output.data.keys())
        return []
