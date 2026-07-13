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
