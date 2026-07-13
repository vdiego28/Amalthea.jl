import pytest
from unittest.mock import MagicMock, patch
import amalthea

@patch('amalthea.LunaOutput')
@patch('amalthea.get_julia')
@patch('amalthea.translate_kwargs')
def test_prop_capillary(mock_translate_kwargs, mock_get_julia, mock_LunaOutput):
    """Test prop_capillary string gas and list pressure conversion."""
    mock_translate_kwargs.return_value = {'lambda0': 800e-9}
    mock_jl = MagicMock()
    mock_luna = MagicMock()
    mock_get_julia.return_value = (mock_jl, mock_luna)

    mock_symbol = MagicMock()
    mock_jl.Symbol.return_value = mock_symbol

    radius = 100e-6
    flength = 0.1
    gas = "He"
    pressure = [0.0, 1.0]

    result = amalthea.prop_capillary(radius, flength, gas, pressure, test_arg=1)

    mock_translate_kwargs.assert_called_once_with({'test_arg': 1})
    mock_jl.Symbol.assert_called_once_with("He")
    mock_luna.prop_capillary.assert_called_once_with(
        radius, flength, mock_symbol, (0.0, 1.0), lambda0=800e-9
    )
    mock_LunaOutput.assert_called_once_with(mock_luna.prop_capillary.return_value)
    assert result == mock_LunaOutput.return_value


@patch('amalthea.LunaOutput')
@patch('amalthea.get_julia')
@patch('amalthea.translate_kwargs')
def test_prop_capillary_no_conversion(mock_translate_kwargs, mock_get_julia, mock_LunaOutput):
    """Test prop_capillary with non-string gas and scalar pressure (no conversion)."""
    mock_translate_kwargs.return_value = {}
    mock_jl = MagicMock()
    mock_luna = MagicMock()
    mock_get_julia.return_value = (mock_jl, mock_luna)

    radius = 100e-6
    flength = 0.1
    gas = MagicMock() # Not a string
    pressure = 1.0 # Not a list/tuple

    result = amalthea.prop_capillary(radius, flength, gas, pressure)

    mock_jl.Symbol.assert_not_called()
    mock_luna.prop_capillary.assert_called_once_with(
        radius, flength, gas, 1.0
    )


@patch('amalthea.LunaOutput')
@patch('amalthea.get_julia')
@patch('amalthea.translate_kwargs')
def test_prop_gnlse(mock_translate_kwargs, mock_get_julia, mock_LunaOutput):
    """Test prop_gnlse list betas conversion."""
    mock_translate_kwargs.return_value = {'lambda0': 800e-9}
    mock_jl = MagicMock()
    mock_luna = MagicMock()
    mock_get_julia.return_value = (mock_jl, mock_luna)

    gamma = 0.1
    flength = 0.001
    betas = [0.0, 0.0, -1e-26]

    result = amalthea.prop_gnlse(gamma, flength, betas, test_arg=1)

    mock_translate_kwargs.assert_called_once_with({'test_arg': 1})
    mock_luna.prop_gnlse.assert_called_once_with(
        gamma, flength, (0.0, 0.0, -1e-26), lambda0=800e-9
    )
    mock_LunaOutput.assert_called_once_with(mock_luna.prop_gnlse.return_value)
    assert result == mock_LunaOutput.return_value


@patch('amalthea.LunaOutput')
@patch('amalthea.get_julia')
@patch('amalthea.translate_kwargs')
def test_prop_gnlse_no_conversion(mock_translate_kwargs, mock_get_julia, mock_LunaOutput):
    """Test prop_gnlse with non-list betas (no conversion)."""
    mock_translate_kwargs.return_value = {}
    mock_jl = MagicMock()
    mock_luna = MagicMock()
    mock_get_julia.return_value = (mock_jl, mock_luna)

    gamma = 0.1
    flength = 0.001
    betas = MagicMock() # Not a list/tuple

    result = amalthea.prop_gnlse(gamma, flength, betas)

    mock_luna.prop_gnlse.assert_called_once_with(
        gamma, flength, betas
    )
