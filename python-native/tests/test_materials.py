import os
from pathlib import Path
import tomllib

import CoolProp
import numpy as np
import pytest

from amalthea_native import materials as m

GASES = ['Air','He','HeJ','HeB','Ne','Ar','ArB','Kr','Xe','N2','H2','O2','CH4','SF6','N2O','D2']


def relative(a, b):
    return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)


@pytest.mark.parametrize('gas',GASES)
def test_julia_material_setup(gas):
    root = os.environ.get('AMALTHEA_MATERIAL_ORACLE')
    if root is None:
        pytest.skip('set AMALTHEA_MATERIAL_ORACLE for independent gas setup acceptance')
    root = Path(root)
    meta = tomllib.loads((root/'metadata.toml').read_text())
    assert meta['coolprop'] == CoolProp.__version__ == '7.2.0'
    assert meta['N_A'] == m.N_A and meta['roomtemp'] == m.ROOMTEMP
    assert abs(meta['epsilon0']/m.EPS0-1) < 1e-15
    data = np.loadtxt(root/gas/'thermo.txt')
    rho = m.density(gas,data[:,0],data[:,1])
    inverse = m.pressure(gas,data[:,2],data[:,1])
    optics = np.loadtxt(root/gas/'optical.txt')
    polar = m.polarizability(gas,optics[:,0])
    index = m.refractive_index(gas,optics[:,0],1.7,305.)
    errors = [relative(rho,data[:,2]),relative(inverse,data[:,3]),
              relative(polar,optics[:,1]+1j*optics[:,2]),relative(index,optics[:,3]+1j*optics[:,4])]
    if gas != 'Air':
        gamma = np.loadtxt(root/gas/'gamma3.txt')
        errors.append(abs(m.gamma3(gas)/(gamma[0]+1j*gamma[1])-1))
    print(f'{gas} density/pressure/polarizability/index/gamma3: {errors}')
    assert max(errors) < 1e-13
    assert relative(inverse,data[:,0]) < 1e-8  # CoolProp forward/inverse thermodynamic consistency


def test_gas_states_broadcast_and_sensitivity():
    pressures=np.array([0.,1.,4.])[:,None]
    temperatures=np.array([280.,310.])[None,:]
    rho=m.density('Ar',pressures,temperatures)
    assert rho.shape==(3,2) and rho.flags.owndata
    np.testing.assert_array_equal(rho[0],0.)
    np.testing.assert_allclose(m.pressure('Ar',rho,temperatures),np.broadcast_to(pressures,rho.shape),rtol=1e-10,atol=0.)
    assert rho[1,0]/rho[1,1]>1.05
    assert rho[2,0]/rho[1,0]>3.5
    index1=m.refractive_index('Ar',800e-9,1.)
    index4=m.refractive_index('Ar',800e-9,4.)
    assert abs(index4-index1)>1e-4
    assert m.density('He')==m.density('HeJ')==m.density('HeB')
    assert m.polarizability('He',800e-9)==m.polarizability('HeJ',800e-9)
    assert m.polarizability('HeB',800e-9)!=m.polarizability('He',800e-9)
    assert m.gamma3('Ar')==m.gamma3('ArB')
    assert m.refractive_index('N2',[400e-9,800e-9],0.).flags.owndata
    np.testing.assert_array_equal(m.refractive_index('N2',[400e-9,800e-9],0.),1.)
    with pytest.raises(ValueError,match='no default gamma3'):
        m.gamma3('Air')


def test_material_validation_and_zero_pressure():
    for gas in GASES:
        assert m.density(gas,0.)==m.pressure(gas,0.)==0.
    for gas in ['unknown',None]:
        for fn in [lambda:m.density(gas),lambda:m.polarizability(gas,800e-9),lambda:m.gamma3(gas)]:
            with pytest.raises(ValueError): fn()
    for value in [-1.,float('inf'),float('nan')]:
        with pytest.raises(ValueError): m.density('Ar',value)
        with pytest.raises(ValueError): m.pressure('Ar',value)
        with pytest.raises(ValueError): m.refractive_index('Ar',value)
    for value in [0.,-1.,float('nan')]:
        with pytest.raises(ValueError): m.density('Ar',0.,value)
        with pytest.raises(ValueError): m.polarizability('Ar',value)
