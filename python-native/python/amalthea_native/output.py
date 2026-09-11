"""Completed-result files with unchanged physical arrays and JSON descriptions."""

import json

import numpy as np

from .grid import C


def _json(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f'{type(item).__name__} is not JSON serializable')

    return json.dumps(value, default=convert, ensure_ascii=False, allow_nan=False)


def _payload(result):
    grid = result.grid
    description = dict(
        type='RealGrid' if grid.is_real else 'EnvGrid', is_real=grid.is_real,
        reference_lambda=grid.reference_lambda,
        reference_omega=2 * np.pi * C / grid.reference_lambda,
        lambda_lims=grid.lambda_lims, zmax=grid.zmax, trange=grid.trange,
        field_axes=['frequency', 'mode', 'saved_position'] if result.field.ndim == 3
        else ['frequency', 'saved_position'],
    )
    # Serialize before opening a destination: unsupported user metadata must
    # not truncate a previously saved result.
    return dict(
        Eomega=result.field, z=result.z, t=grid.t, omega=grid.omega,
        to=grid.to, omega_over=grid.omega_over, twin=grid.twin,
        towin=grid.towin, omega_win=grid.omega_win, sidx=grid.sidx,
        format_version=1, grid=_json(description),
        parameters=_json(result.parameters), metadata=_json(result.metadata),
    )


def save_npz(result, path):
    np.savez_compressed(path, **_payload(result))


def save_hdf5(result, path):
    try:
        import h5py
    except ModuleNotFoundError as error:
        if error.name != 'h5py':
            raise
        raise ModuleNotFoundError(
            "HDF5 output requires the optional extra: pip install 'amalthea-native[hdf5]'"
        ) from error

    payload = _payload(result)
    with h5py.File(path, 'w') as destination:
        for name, value in payload.items():
            if isinstance(value, str):
                destination.create_dataset(name, data=value, dtype=h5py.string_dtype('utf-8'))
            else:
                options = dict(compression='gzip') if np.ndim(value) and np.size(value) else {}
                destination.create_dataset(name, data=value, **options)
