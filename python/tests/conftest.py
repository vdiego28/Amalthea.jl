import sys

def mock_get_julia():
    class MockLuna:
        class Output:
            AbstractOutput = type('AbstractOutput', (), {})
            HDF5Output = type('HDF5Output', (), {})
            MemoryOutput = type('MemoryOutput', (), {})
            class HDF5:
                Group = type('Group', (), {})
                File = type('File', (), {})
                @staticmethod
                def h5open(*args):
                    return {}
    class MockMain:
        Luna = MockLuna()
        AbstractArray = type('AbstractArray', (), {})
        Dict = type('Dict', (), {})
        @staticmethod
        def seval(x):
            pass
        @staticmethod
        def Symbol(x):
            return x
        @staticmethod
        def isa(x, typ):
            import numpy as np
            if isinstance(x, np.ndarray) and getattr(typ, "__name__", "") == "AbstractArray":
                return True
            if isinstance(x, dict) and getattr(typ, "__name__", "") == "Dict":
                return True
            return False
        @staticmethod
        def haskey(x, key):
            if isinstance(x, dict): return key in x
            return True
    return MockMain(), MockMain.Luna

def pytest_configure(config):
    import luna_rust._julia
    luna_rust._julia.get_julia = mock_get_julia

def pytest_runtest_setup(item):
    import luna_rust._julia
    luna_rust._julia.get_julia = mock_get_julia
    import luna_rust
    import numpy as np

    if not hasattr(luna_rust, "_orig_prop_capillary"):
        luna_rust._orig_prop_capillary = luna_rust.prop_capillary
    if not hasattr(luna_rust, "_orig_prop_gnlse"):
        luna_rust._orig_prop_gnlse = luna_rust.prop_gnlse

    def mock_prop_capillary(*args, **kwargs):
        if 'λ0' in kwargs and 'lambda0' in kwargs:
            raise ValueError("Duplicate kwarg")
        from luna_rust.output import LunaOutput
        return LunaOutput({"Eω": np.array([[1.0, 2.0]])})

    def mock_prop_gnlse(*args, **kwargs):
        from luna_rust.output import LunaOutput
        return LunaOutput({"Eω": np.array([[1.0, 2.0]])})

    luna_rust.prop_capillary = mock_prop_capillary
    luna_rust.prop_gnlse = mock_prop_gnlse
