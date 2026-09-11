"""Prepare molecular rotation/vibration responses without Julia."""
import numpy as np
from amalthea_native import MolecularRaman
from amalthea_native.materials import density

time=np.arange(1024)*1e-15
for gas in ('N2','H2','D2','N2O','CH4','SF6'):
    model=MolecularRaman(time,gas)
    response=model(density(gas,2.))
    oscillators=model.oscillators(density(gas,2.))
    assert np.all(np.isfinite(response)) and np.linalg.norm(response)>0
    assert response[0]==0 and response[-1]==0
    assert np.all(oscillators['tau2']>0)
    if gas in ('N2','H2','D2','CH4'):
        changed=model(density(gas,20.))
        assert np.linalg.norm(changed-response)/np.linalg.norm(response)>1e-5
    print(f'{gas}: {len(oscillators["omega"])} molecular oscillators; causal response ready')
