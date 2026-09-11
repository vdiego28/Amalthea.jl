"""Complete atomic and molecular mixtures on both field grids."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from amalthea_native import prop_capillary

LENGTH=.0002
OPTIONS=dict(lambda0=800e-9,lambda_lims=(200e-9,1700e-9),trange=300e-15,
             tau_fwhm=20e-15,energy=500e-6,saveN=7,init_dz=.00001,
             min_dz=.00001,max_dz=.00001)


def relative(a,b):return np.linalg.norm(a-b)/np.linalg.norm(b)


with TemporaryDirectory() as directory:
    for envelope in (True,False):
        atomic=prop_capillary(125e-6,LENGTH,('Ar','Ne'),(2.,1.),**OPTIONS,
                              envelope=envelope,plasma=False)
        python=prop_capillary(125e-6,LENGTH,('Ar','Ne'),(2.,1.),**OPTIONS,
                              envelope=envelope,plasma=False,backend='python')
        control=prop_capillary(125e-6,LENGTH,('Ar','Ne'),(2.,0.),**OPTIONS,
                               envelope=envelope,plasma=False)
        effect=relative(atomic.field,control.field)
        assert atomic.metadata['backend']=='native' and effect>1e-5
        assert relative(atomic.field,python.field)<1e-13
        print(f'Ar/Ne, envelope={envelope}: {atomic.field.shape}; native; Ne effect={effect:.6e}')
        radius=lambda z:125e-6*(1+.1*math.sin(math.pi*z/LENGTH))
        pressure=((2.,4.),lambda z:1.+.3*math.sin(math.pi*z/LENGTH))
        species=([{},{}] if envelope else
                 [dict(plasma='PPT',PPT_options=dict(cachedir=directory)),dict(plasma='ADK')])
        options=OPTIONS|dict(envelope=envelope,plasma=False,species_options=species)
        molecular=prop_capillary(radius,LENGTH,('N2','H2'),pressure,**options)
        control=prop_capillary(radius,LENGTH,('N2','H2'),pressure,
                               **(options|dict(raman=False,species_options=[{},{}])))
        effect=relative(molecular.field,control.field)
        assert molecular.metadata['backend']=='python' and effect>1e-5
        output=Path(directory)/'mixture.npz';molecular.save_npz(output)
        with np.load(output,allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['Eomega'],molecular.field)
            assert json.loads(str(saved['parameters']))['gas']==['N2','H2']
        print(f'N2/H2 profiles, envelope={envelope}: {molecular.field.shape}; Python; response effect={effect:.6e}')
