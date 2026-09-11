"""Build and reuse a default-size PPT table without Julia or a runtime download."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from amalthea_native import IonRatePPT,IonRatePPTAccel

with TemporaryDirectory() as directory:
    table=IonRatePPTAccel('Ar',800e-9,cachedir=directory)
    cached=IonRatePPTAccel('Ar',800e-9,cachedir=directory)
    assert len(table.field_nodes)==65536 and cached.cache_hit
    field=np.array([0.,1e10,2e10,4e10])
    np.testing.assert_array_equal(table(field),cached(field))
    assert table(0)==0 and table(4e10)>1e10
    direct=IonRatePPT('Ar',800e-9)(field)
    relative=np.linalg.norm(table(field)-direct)/np.linalg.norm(direct)
    assert relative<1e-6
    assert len(list(Path(directory).glob('*.npz')))==1
    print(f'PPT: {len(table.field_nodes)} local nodes; cache reused; rate at 4e10 V/m={table(4e10):.6e} /s; table error={relative:.3e}')
