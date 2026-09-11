#!/usr/bin/env python3
"""Run complete examples from an installed wheel, optionally inside network isolation."""
import argparse
import contextlib
import ctypes
import errno
import io
import os
from pathlib import Path
import runpy
import socket
import sys

# -I intentionally excludes script directories. Add only the development tools;
# the package itself must still resolve from the selected environment.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _validation import EXAMPLES, save_json, utcnow


def loaded_libraries():
    if sys.platform.startswith('linux'):
        return sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                       if '/' in line})
    if sys.platform == 'darwin':
        library = ctypes.CDLL(None)
        library._dyld_image_count.restype = ctypes.c_uint32
        library._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        library._dyld_get_image_name.restype = ctypes.c_char_p
        return [os.fsdecode(library._dyld_get_image_name(index))
                for index in range(library._dyld_image_count())]
    if sys.platform == 'win32':
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.K32EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
                                                wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        kernel.K32EnumProcessModules.restype = wintypes.BOOL
        kernel.K32GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE,
                                                 wintypes.LPWSTR, wintypes.DWORD]
        kernel.K32GetModuleFileNameExW.restype = wintypes.DWORD
        process = kernel.GetCurrentProcess()
        modules = (wintypes.HMODULE * 4096)()
        needed = wintypes.DWORD()
        if not kernel.K32EnumProcessModules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
            raise ctypes.WinError(ctypes.get_last_error())
        if needed.value > ctypes.sizeof(modules):
            raise RuntimeError('loaded module list exceeds inspection capacity')
        paths = []
        for module in modules[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
            buffer = ctypes.create_unicode_buffer(32768)
            length = kernel.K32GetModuleFileNameExW(process, module, buffer, len(buffer))
            if not length or length >= len(buffer):
                raise RuntimeError('could not read complete loaded module path')
            paths.append(buffer.value)
        return paths
    raise RuntimeError(f'unsupported library inspection platform: {sys.platform}')


def check_isolation(kind):
    if os.environ.get('PATH') != os.path.join(os.path.abspath(os.sep), 'nonexistent'):
        raise RuntimeError('offline check requires an empty toolchain PATH')
    record = {'mechanism': kind}
    if kind == 'linux':
        interfaces = {line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()
                      if ':' in line}
        if interfaces != {'lo'}:
            raise RuntimeError(f'network namespace is not isolated: {interfaces}')
        record['interfaces'] = sorted(interfaces)
    # Reserved TEST-NET address: establish that the OS rejects connection setup,
    # without sending application data or relying on a remote service.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            code = connection.connect_ex(('203.0.113.1', 9))
    except OSError as error:
        code = error.errno
    allowed = {'linux': {errno.ENETUNREACH, errno.EACCES, errno.EPERM},
               'macos': {errno.EACCES, errno.EPERM}, 'windows': {10013}}
    if code not in allowed[kind]:
        raise RuntimeError(f'network denial was not demonstrated: {kind}, error {code}')
    record['connection_error'] = code
    return record


def smoke(examples, output, isolation=None):
    import amalthea_native as an
    from amalthea_native import _native
    from amalthea_native.materials import _GASES, _IONISATION_POTENTIALS, density
    from amalthea_native.ppt import _phi
    record = {'status': 'running', 'started': utcnow(), 'examples': {},
              'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
              'package': an.__file__, 'extension': _native.__file__}
    try:
        for module in (an, _native):
            if not Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
                raise RuntimeError(f'module is outside the installed environment: {module.__file__}')
        if isolation is not None:
            record['isolation'] = check_isolation(isolation)
        save_json(output, record)
        for name in EXAMPLES:
            record['current_example'] = name
            save_json(output, record)
            stream = io.StringIO()
            try:
                with contextlib.redirect_stdout(stream):
                    runpy.run_path(str(examples/f'{name}.py'), run_name='__main__')
            finally:
                record['examples'][name] = stream.getvalue()
            print(f'Passed example: {name}', flush=True)
        for gas in _GASES:
            assert density(gas, 2) > 0, gas
        for material in _IONISATION_POTENTIALS:
            assert an.IonRateADK(material)(4e10) > 0, material
        mode = an.MarcatiliMode(lambda z: 1e-4*(1+z*z), 'N2', lambda z: 1+z)
        assert mode.alpha(2.35e15, z=.137) > 0 and mode.dispersion(1, 2.35e15, z=.137) > 0
        assert _phi(1, 40) > 0
        record.update(gases=len(_GASES), adk_materials=len(_IONISATION_POTENTIALS),
                      high_precision_phi=float(_phi(1, 40)), libraries=loaded_libraries())
        for path in record['libraries']:
            if any(name in Path(path).name.lower() for name in ('julia', 'fftw', 'cubature', 'cuda')):
                raise RuntimeError(f'forbidden runtime library: {path}')
        if any(name.startswith(('juliacall', 'juliapkg')) or name == 'julia' for name in sys.modules):
            raise RuntimeError('Julia runtime was imported')
        record.update(status='passed', current_example=None)
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['finished'] = utcnow()
        save_json(output, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--isolation', choices=['linux', 'macos', 'windows'])
    args = parser.parse_args()
    smoke(args.examples.resolve(), args.output.resolve(), args.isolation)
