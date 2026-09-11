"""Gas data using PhysData conventions (SI, pressure in bar, temperature in K)."""
from functools import lru_cache
import math

import numpy as np

from .grid import C

ROOMTEMP = 293.15
N_A = 6.022140857e23  # CODATA2014, retained for agreement with PhysData.
EPS0 = 1/(4*math.pi*1e-7*C**2)
_GASES = {'He':'He', 'HeJ':'He', 'HeB':'He', 'Ne':'Neon', 'Ar':'Ar',
          'ArB':'Ar', 'Kr':'Krypton', 'Xe':'Xenon', 'Air':'Air', 'N2':'Nitrogen',
          'H2':'Hydrogen', 'D2':'Deuterium', 'O2':'Oxygen', 'CH4':'Methane',
          'SF6':'SulfurHexafluoride', 'N2O':'NitrousOxide'}
# Susceptibility coefficients and squared resonance wavelengths in micrometres.
_SELLMEIER = {
    'He': ((2.16463842e-5, -6.80769781e-4), (2.10561127e-7, 5.13251289e-3), (4.75092720e-5, 3.18621354e-3)),
    'HeB': ((4977.77e-8, 28.54e-6), (1856.94e-8, 7.76e-3)),
    'Ne': ((9154.48e-8, 656.97e-6), (4018.63e-8, 5.728e-3)),
    'Ar': ((.00032323117217767093, .0045416501944977915), (.00011557814904827939, .011120847461156543), (.00010909808164540697, .0006827046691889898)),
    'ArB': ((20332.29e-8, 206.12e-6), (34458.31e-8, 8.066e-3)),
    'Kr': ((26102.88e-8, 2.01e-6), (56946.82e-8, 10.043e-3)),
    'Xe': ((103701.61e-8, 12750e-6), (31228.61e-8, .561e-3)),
    'Air': ((14926.44e-8, 19.36e-6), (41807.57e-8, 7.434e-3)),
    'N2': ((39209.95e-8, 1146.24e-6), (18806.48e-8, 13.476e-3)),
}


def _gas(gas):
    if not isinstance(gas, str) or gas not in _GASES:
        raise ValueError(f'unknown gas: {gas}')
    return gas


def _values(value, name, *, positive=False):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array <= 0 if positive else array < 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return array


def _owned(value):
    value = np.array(value, copy=True)
    if not np.all(np.isfinite(value)):
        raise ValueError('material model returned nonfinite values')
    return value.item() if value.ndim == 0 else value


def _thermo(gas, value, temperature, inverse):
    gas = _gas(gas)
    name = 'density' if inverse else 'pressure'
    values, temperatures = np.broadcast_arrays(_values(value, name), _values(temperature, 'temperature', positive=True))
    output = np.zeros(values.shape)
    selected = values != 0
    if np.any(selected):
        # Lazy import: envelope GNLSE does not need to initialize CoolProp.
        from CoolProp.CoolProp import PropsSI
        if inverse:
            output[selected] = np.asarray(PropsSI('P', 'T', temperatures[selected], 'DMOLAR', values[selected]/N_A, _GASES[gas])).reshape(-1)/1e5
        else:
            output[selected] = np.asarray(PropsSI('DMOLAR', 'T', temperatures[selected], 'P', 1e5*values[selected], _GASES[gas])).reshape(-1)*N_A
    return _owned(output)


def density(gas, pressure=1., temperature=ROOMTEMP):
    """Number density [m^-3] using CoolProp, with exact zero at zero pressure."""
    return _thermo(gas, pressure, temperature, False)


def pressure(gas, number_density, temperature=ROOMTEMP):
    """Pressure [bar] from number density [m^-3]."""
    return _thermo(gas, number_density, temperature, True)


@lru_cache(maxsize=128)
def _reference_density(gas, pressure, temperature):
    return density(gas, pressure, temperature)


def polarizability(gas, wavelength):
    """Single-particle linear susceptibility [m^3], at wavelength in metres."""
    gas = _gas(gas)
    um = _values(wavelength, 'wavelength', positive=True)*1e6
    table_gas = 'He' if gas == 'HeJ' else gas
    if table_gas in _SELLMEIER:
        rho = _reference_density(gas, 1., 273.15)
        value = np.zeros_like(um)
        for b, c in _SELLMEIER[table_gas]:
            value += (b/rho)*um**2/(um**2-c)
    elif gas in ('H2', 'D2'):
        value = ((14895.6e-6/(180.7-1/um**2)+4903.7e-6/(92.-1/um**2)+1)**2-1)/_reference_density(gas, 1.01325, 273.15)
    elif gas == 'O2':
        value = ((1+1.181494e-4+9.708931e-3/(75.4-1/um**2))**2-1)/_reference_density(gas, 1.01325, ROOMTEMP)
    else:
        a,b,c = {'CH4':(3603.09,4.40362e14,1.1741e10),
                 'N2O':(22095.,1.66291e14,6.75226e9),
                 'SF6':(18997.7,8.27663e14,1.56833e10)}[gas]
        # Preserve PhysData's implemented QuanfuHe convention exactly.
        value = (1+1e-8*(a+b/(c-(1e4/um)**2))).astype(complex)/_reference_density(gas,1.01325,288.15)
    return _owned(value)


def refractive_index(gas, wavelength, pressure=1., temperature=ROOMTEMP):
    """Complex refractive index, retaining PhysData's gas model conventions."""
    susceptibility = polarizability(gas,wavelength)*np.asarray(density(gas,pressure,temperature))
    return _owned(np.sqrt(1+np.asarray(susceptibility,dtype=complex)))


@lru_cache(maxsize=32)
def gamma3(gas):
    """Default single-particle third-order hyperpolarizability from PhysData."""
    gas = _gas(gas)
    factors = {'He':1., 'HeB':1., 'HeJ':1., 'Ne':1.8, 'Ar':23.5, 'ArB':23.5,
               'Kr':64., 'Xe':188.2, 'N2':21.1}
    if gas in factors:
        return 4*factors[gas]*3.43e-28/_reference_density(gas,1.01325,273.15)
    if gas in ('H2','D2'):
        return 15.77*gamma3('He')
    if gas in ('CH4','SF6'):
        return {'CH4':2.931,'SF6':1.53}[gas]*gamma3('N2')
    if gas in ('O2','N2O'):
        n0 = refractive_index(gas,800e-9,1.01325,ROOMTEMP)
        rho = _reference_density(gas,1.01325,ROOMTEMP)
        return 4/3*EPS0*C*n0**2/rho*{'O2':8.1e-24,'N2O':17.2e-24}[gas]
    raise ValueError(f'no default gamma3 source for gas: {gas}')


_IONISATION_POTENTIALS = {
    'He':.9036,'HeJ':.9036,'HeB':.9036,'Ne':.7925,'Ar':.5792,'ArB':.5792,
    'Kr':.5142,'Xe':.4458,'H':.5,'N2':.5726,'H2':.5669,'O2':.443553,
    'CH4':.4636,'N2O':.474,'SF6':.5,'D2':.5684,
}


def ionisation_potential(material,unit='SI'):
    """First ionisation potential in joules (SI), eV or atomic units."""
    if not isinstance(material,str) or material not in _IONISATION_POTENTIALS:
        raise ValueError(f'no default ionisation potential for {material}')
    potential=_IONISATION_POTENTIALS[material]
    if unit=='atomic': return potential
    if unit=='eV': return 27.21138602*potential
    if unit=='SI': return 27.21138602*1.6021766208e-19*potential
    raise ValueError('unit must be SI, eV or atomic')
