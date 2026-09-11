"""Persistent installed-wheel worker; stdout contains only protocol records."""
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
import tomllib

started = time.perf_counter()
import amalthea_native as an
IMPORT_SECONDS = time.perf_counter() - started
import numpy as np
from amalthea_native.gnlse import _Gnlse, PropagationResult
from amalthea_native.capillary import _Capillary, _keywords, needs_modal
from amalthea_native.modal import _ModalCapillary
from amalthea_native.envelope import solve_envelope_model

case_file, case_name, backend = sys.argv[1:]
config = tomllib.loads(Path(case_file).read_text())
case = config['cases'][case_name]
BASE = config['common'] | case['options']
assert Path(an.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), an.__file__
assert backend in ('auto', 'python', 'callback')
assert backend != 'callback' or case_name == 'gnlse-kerr'


def inputs(control=False):
    args = list(case['args']); options = dict(BASE)
    if case['api'] == 'capillary' and isinstance(args[3], list):
        args[3] = tuple(args[3])
    if control:
        feature = case['control']
        if feature == 'gradient': args[3] = args[3][0]
        elif feature == 'kerr' and case['api'] == 'gnlse': args[0] = 0.
        else: options[feature] = False
    if backend == 'callback' and not control:
        # Identical physical coefficient; all fields/context cross the public callback seam.
        gamma = args[0]
        c = 299792458.
        eps0 = 1/(4*np.pi*1e-7*c*c)
        coefficient = (1-.18)*gamma/(2*np.pi/BASE['lambda0'])*eps0**2*c
        def response(field, context):
            return coefficient*field*abs(field)**2
        args[0] = 0.; options['responses'] = response
    options['backend'] = 'python' if backend == 'callback' else backend
    return args, options


def setup(control=False):
    args, options = inputs(control)
    options.update(init_dz=case['step'], max_dz=case['step'], min_dz=1e-15,
                   rtol=1e-9, atol=1e-12)
    if case['api'] == 'gnlse': return _Gnlse(*args, **options)
    kw = _keywords(options)
    constructor = _ModalCapillary if needs_modal(kw) else _Capillary
    return constructor(*args, **kw)


def dump_array(root, name, array):
    value = np.asarray(array)
    dtype = '<c16' if np.iscomplexobj(value) else '<f8'
    data = value.astype(dtype).tobytes(order='F')
    (root/(name+'.bin')).write_bytes(data)
    return {'shape': list(value.shape), 'dtype': dtype, 'sha256': hashlib.sha256(data).hexdigest()}


def measure(operation, root):
    root.mkdir(parents=True, exist_ok=False)
    record = {'operation': operation, 'requested_backend': backend, 'package': an.__file__,
              'python': sys.version, 'import_seconds': IMPORT_SECONDS, 'arrays': {},
              'dependencies': {name: importlib.metadata.version(name)
                               for name in ('numpy','scipy','CoolProp','h5py','amalthea-native')}}
    if operation == 'cold':
        args, options = inputs()
        start = time.perf_counter()
        result = getattr(an, 'prop_'+case['api'])(*args, **options)
        record['first_public_seconds'] = time.perf_counter()-start
    else:
        gc.collect()
        start = time.perf_counter(); model = setup(operation == 'control')
        record['setup_seconds'] = time.perf_counter()-start
        if operation == 'rhs':
            reference = root.parent.parent/'julia'/'check'
            info = tomllib.loads((reference/'result.toml').read_text())['arrays']['initial']
            initial = np.fromfile(reference/'initial.bin', dtype='<c16').reshape(info['shape'], order='F')
            record['arrays']['rhs'] = dump_array(root, 'rhs', model.rhs(0., initial))
            (root/'result.json').write_text(json.dumps(record, indent=2)); return
        start = time.perf_counter(); solved = solve_envelope_model(model)
        if hasattr(model, 'execution_metadata'): solved.metadata.update(model.execution_metadata())
        result = PropagationResult(solved.field, solved.z, model.grid, model.parameters, solved.metadata)
        record['solve_seconds'] = time.perf_counter()-start
        record['complete_seconds'] = record['setup_seconds'] + record['solve_seconds']
        if operation in ('check', 'control'):
            record['arrays']['initial'] = dump_array(root, 'initial', model.initial)
            linop = model.linop(0.) if callable(model.linop) else model.linop
            record['arrays']['linear'] = dump_array(root, 'linear', linop)
    start = time.perf_counter(); copied = result.field.copy()
    record['copy_seconds'] = time.perf_counter()-start
    assert np.array_equal(copied, result.field)
    start = time.perf_counter(); result.save_hdf5(root/'output.h5')
    record['hdf5_seconds'] = time.perf_counter()-start
    record['hdf5_scope'] = 'PropagationResult arrays and JSON grid/parameters/execution'
    record['hdf5_bytes'] = (root/'output.h5').stat().st_size
    for name, value in [('field', result.field), ('z', result.z), ('t', result.grid.t), ('omega', result.grid.omega)]:
        record['arrays'][name] = dump_array(root, name, value)
    record['execution'] = {k: (v.item() if isinstance(v, np.generic) else v)
                           for k,v in result.metadata.items() if isinstance(v,(str,int,float,bool,np.generic))}
    record['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)
    (root/'result.json').write_text(json.dumps(record, indent=2))


print('READY', flush=True)
for line in sys.stdin:
    if line.strip() == 'stop': break
    operation, destination = line.rstrip('\n').split('\t')
    if operation not in ('cold', 'sample', 'check', 'control', 'rhs'): raise ValueError(operation)
    measure(operation, Path(destination))
    print('DONE', flush=True)
