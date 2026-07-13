import pytest
import sys
from unittest.mock import MagicMock, patch

def test_luna_output_getitem_keyerror():
    # Mock the julia module to prevent initialization
    mock_julia = MagicMock()
    mock_get_julia = MagicMock(return_value=(mock_julia, None))

    mock_julia_module = MagicMock()
    mock_julia_module.get_julia = mock_get_julia

    with patch.dict('sys.modules', {'amalthea._julia': mock_julia_module}):
        # Now we can safely import LunaOutput
        from amalthea.output import LunaOutput

        # Create a mock jl_output that raises an exception when accessing a key
        mock_jl_output = MagicMock()
        mock_jl_output.__getitem__.side_effect = Exception("Key not found in Julia")

        # Create LunaOutput with the mock
        output = LunaOutput(mock_jl_output)

        # Assert that accessing a non-existent key raises a KeyError
        with pytest.raises(KeyError) as excinfo:
            _ = output["non_existent_key"]

        assert "non_existent_key" in str(excinfo.value)


def _fake_jl_for(marker_attr_path):
    """Build a fake `_jl` where `isa(val, X)` is True only when `X` is the
    type sentinel reached by `marker_attr_path` (e.g. "Dict",
    "Amalthea.Output.MemoryOutput") -- everything else is False. Since a
    MagicMock returns a consistent child object per attribute access,
    `_jl.Dict is _jl.Dict` but `_jl.Dict is not _jl.Amalthea...`, so this
    reproduces the isa-dispatch branching in output.py without real Julia.
    """
    _jl = MagicMock()
    target = _jl
    for part in marker_attr_path.split("."):
        target = getattr(target, part)
    _jl.isa.side_effect = lambda val, typ: typ is target
    return _jl


def _with_fake_julia(_jl):
    mock_get_julia = MagicMock(return_value=(_jl, None))
    mock_julia_module = MagicMock()
    mock_julia_module.get_julia = mock_get_julia
    return patch.dict('sys.modules', {'amalthea._julia': mock_julia_module})


def test_luna_output_keys_dict_branch():
    from amalthea.output import LunaOutput
    jl_output = MagicMock()
    jl_output.keys.return_value = ["a", "b"]
    with _with_fake_julia(_fake_jl_for("Dict")):
        assert LunaOutput(jl_output).keys() == ["a", "b"]


def test_luna_output_keys_memoryoutput_branch():
    from amalthea.output import LunaOutput
    jl_output = MagicMock()
    jl_output.data.keys.return_value = ["Eω", "z"]
    with _with_fake_julia(_fake_jl_for("Amalthea.Output.MemoryOutput")):
        assert LunaOutput(jl_output).keys() == ["Eω", "z"]


def test_luna_output_keys_hdf5output_branch_closes_file():
    from amalthea.output import LunaOutput
    jl_output = MagicMock()
    jl_output.fpath = "/tmp/fake.h5"
    _jl = _fake_jl_for("Amalthea.Output.HDF5Output")
    fake_file = MagicMock()
    _jl.Amalthea.Output.HDF5.h5open.return_value = fake_file
    _jl.keys.return_value = ["Eω", "z"]
    with _with_fake_julia(_jl):
        assert LunaOutput(jl_output).keys() == ["Eω", "z"]
    _jl.Amalthea.Output.HDF5.h5open.assert_called_once_with("/tmp/fake.h5", "r")
    _jl.keys.assert_called_once_with(fake_file)
    # The file must always be closed, matching Julia's own do-block usage
    # for HDF5Output (src/Output.jl) -- regression test for a real leak.
    _jl.close.assert_called_once_with(fake_file)


def test_luna_output_keys_no_match_returns_empty():
    from amalthea.output import LunaOutput
    jl_output = MagicMock()
    _jl = MagicMock()
    _jl.isa.side_effect = lambda val, typ: False
    with _with_fake_julia(_jl):
        assert LunaOutput(jl_output).keys() == []


def test_luna_output_contains_dict_branch():
    from amalthea.output import LunaOutput
    jl_output = MagicMock()
    _jl = _fake_jl_for("Dict")
    _jl.haskey.return_value = True
    with _with_fake_julia(_jl):
        assert ("k" in LunaOutput(jl_output)) is True
    _jl.haskey.assert_called_once_with(jl_output, "k")


def test_luna_output_contains_abstractoutput_branch():
    from amalthea.output import LunaOutput
    jl_output = MagicMock()
    _jl = _fake_jl_for("Amalthea.Output.AbstractOutput")
    _jl.haskey.return_value = False
    with _with_fake_julia(_jl):
        assert ("k" in LunaOutput(jl_output)) is False
    _jl.haskey.assert_called_once_with(jl_output, "k")


def test_luna_output_contains_no_match_returns_false():
    from amalthea.output import LunaOutput
    jl_output = MagicMock()
    _jl = MagicMock()
    _jl.isa.side_effect = lambda val, typ: False
    with _with_fake_julia(_jl):
        assert ("k" in LunaOutput(jl_output)) is False
    _jl.haskey.assert_not_called()
