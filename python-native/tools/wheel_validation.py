#!/usr/bin/env python3
"""Build portable wheels, then validate actual installed checkout/sdist artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import sysconfig
import tarfile
import zipfile

from _validation import (cpu_environment, digest, pytest_result, run, save_json,
                         utcnow, verify_oracles)
from offline_examples import offline


def host_spec():
    system, machine = platform.system(), platform.machine().lower()
    if platform.python_implementation() != 'CPython' or not (3, 11) <= sys.version_info[:2] <= (3, 14):
        raise RuntimeError('wheel acceptance requires CPython 3.11–3.14')
    if sysconfig.get_config_var('Py_GIL_DISABLED'):
        raise RuntimeError('free-threaded Python is outside this release matrix')
    configurations = {
        ('Linux', 'x86_64'): ('x86_64-unknown-linux-gnu', 'manylinux_2_28_x86_64'),
        ('Linux', 'aarch64'): ('aarch64-unknown-linux-gnu', 'manylinux_2_28_aarch64'),
        ('Darwin', 'arm64'): ('aarch64-apple-darwin', 'macosx_11_0_arm64'),
        ('Windows', 'amd64'): ('x86_64-pc-windows-msvc', 'win_amd64'),
    }
    if (system, machine) not in configurations:
        raise RuntimeError(f'unsupported native wheel-test host: {system}/{machine}')
    target, tag = configurations[system, machine]
    return {'system': system, 'machine': machine, 'target': target, 'tag': tag,
            'python': platform.python_version(), 'executable': sys.executable}


def files_under(directory):
    directory = Path(directory)
    return [path for path in directory.rglob('*') if path.is_file()
            and not any(part in ('target', '__pycache__', '.venv') for part in path.relative_to(directory).parts)
            and path.suffix not in ('.so', '.pyd', '.dll', '.dylib', '.pyc', '.pyo')]


def source_files(repository):
    directories = ('amalthea/src', 'python-native/src', 'python-native/python',
                   'python-native/tests', 'python-native/tools', 'python-native/examples')
    files = [path for name in directories for path in files_under(repository/name)]
    files.extend(repository/name for name in ('amalthea/build.rs', 'amalthea/Cargo.toml',
                 'amalthea/Cargo.lock', 'python-native/Cargo.toml', 'python-native/Cargo.lock',
                 'python-native/pyproject.toml', 'python-native/.cargo/config.toml',
                 'python-native/LICENSE', 'python-native/README.md'))
    return {path.relative_to(repository).as_posix(): digest(path) for path in sorted(files)}


def extract_source(archive, destination, repository):
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        roots = set()
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or '\\' in member.name
                    or ':' in member.name or not (member.isfile() or member.isdir())):
                raise ValueError(f'unsafe source archive member: {member.name}')
            if any(part in ('target', '__pycache__', '.venv', '.git') for part in name.parts) or name.suffix in ('.so', '.dll', '.pyd', '.dylib', '.pyc', '.pyo'):
                raise ValueError(f'compiled or generated source archive member: {member.name}')
            roots.add(name.parts[0])
        if len(roots) != 1:
            raise ValueError('source archive must have exactly one root')
        stream.extractall(destination, filter='data')
    source = destination/roots.pop()
    required = ['pyproject.toml', 'LICENSE', '.cargo/config.toml', 'amalthea/build.rs',
                'amalthea/Cargo.toml', 'amalthea/Cargo.lock', 'python-native/Cargo.toml',
                'python-native/Cargo.lock', 'python/amalthea_native/data/silica.txt',
                'python/amalthea_native/licenses/FiniteDifferences.txt']
    for name in required:
        if not (source/name).is_file():
            raise ValueError(f'incomplete source archive: {name}')
    for directory in ('amalthea/src', 'python-native/src', 'python-native/tests',
                      'python-native/examples', 'python-native/tools', 'python-native/python'):
        for path in files_under(repository/directory):
            relative = path.relative_to(repository).as_posix()
            if relative.startswith('python-native/python/'):
                relative = relative.removeprefix('python-native/')
            copied = source/relative
            if not copied.is_file() or copied.read_bytes() != path.read_bytes():
                raise ValueError(f'source archive differs from checkout: {relative}')
    return source


def verify_wheel(wheel, source, specification):
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if not wheel.name.endswith(f'-{tag}-{tag}-{specification["tag"]}.whl'):
        raise ValueError(f'wheel does not match the required interpreter/platform: {wheel.name}')
    expected = {path.relative_to(source/'python').as_posix(): digest(path)
                for path in files_under(source/'python/amalthea_native')}
    with zipfile.ZipFile(wheel) as archive:
        actual = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                  if name.startswith('amalthea_native/') and not name.endswith(('/', '.so', '.pyd'))}
        if actual != expected:
            raise ValueError('wheel package files differ from the source distribution')
        extensions = [name for name in archive.namelist()
                      if name.startswith('amalthea_native/_native.') and name.endswith(('.so', '.pyd'))]
        if len(extensions) != 1:
            raise ValueError('wheel must contain one compiled Python extension')
    return expected


def build(repository, output):
    if output.is_relative_to(repository):
        raise ValueError('wheel artifacts must be built outside the checkout')
    output.mkdir(parents=True, exist_ok=False)
    specification = host_spec()
    state = {'format_version': 1, 'status': 'building', 'started': utcnow(), 'host': specification,
             'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
             'repository': str(repository), 'sources': source_files(repository), 'wheels': {}}
    manifest = output/'build.json'
    save_json(manifest, state)
    environment = cpu_environment()
    environment.update(CARGO_TARGET_DIR=str(output/'target'), CARGO_BUILD_JOBS='2',
                       ZIG_GLOBAL_CACHE_DIR=str(output/'zig-cache'))
    if specification['system'] == 'Darwin':
        environment['MACOSX_DEPLOYMENT_TARGET'] = '11.0'
    def command(arguments, name, cwd=repository):
        run(arguments, cwd=cwd, environment=environment, log=output/'logs'/f'{name}.log')
    try:
        command([sys.executable, '-m', 'pip', 'list', '--format=json'], 'build-tool-versions')
        command(['maturin', '--version'], 'maturin-version')
        command(['rustc', '-vV'], 'rust-version')
        command(['maturin', 'sdist', '--out', output/'sdist'], 'sdist', repository/'python-native')
        archives = list((output/'sdist').glob('*.tar.gz'))
        if len(archives) != 1:
            raise RuntimeError('expected exactly one source archive')
        source = extract_source(archives[0], output/'source', repository)
        state.update(source=str(source), sdist=str(archives[0]), sdist_sha256=digest(archives[0]))
        for kind, cwd in (('checkout', repository/'python-native'), ('source', source)):
            destination = output/'wheels'/kind
            arguments = ['maturin', 'build', '--release', '--locked', '--target', specification['target'],
                         '--interpreter', sys.executable, '--out', destination]
            if specification['system'] == 'Linux':
                arguments += ['--zig', '--compatibility', 'manylinux_2_28', '--auditwheel', 'check']
            command(arguments, f'{kind}-build', cwd)
            wheels = list(destination.glob('*.whl'))
            if len(wheels) != 1:
                raise RuntimeError(f'expected one {kind} wheel')
            wheel = wheels[0]
            package = verify_wheel(wheel, source, specification)
            if specification['system'] == 'Linux':
                command(['auditwheel', 'show', wheel], f'{kind}-audit')
            elif specification['system'] == 'Windows':
                command([sys.executable, '-m', 'delvewheel', 'show', wheel], f'{kind}-audit')
            else:
                with zipfile.ZipFile(wheel) as archive:
                    extension = next(name for name in archive.namelist() if name.endswith('.so'))
                    binary = output/f'{kind}-extension.so'
                    binary.write_bytes(archive.read(extension))
                command(['otool', '-L', binary], f'{kind}-audit')
            audit = (output/'logs'/f'{kind}-audit.log').read_text().lower()
            if any(name in audit for name in ('libjulia', 'libfftw', 'libcubature', 'libcuda', 'nvcuda')):
                raise RuntimeError('wheel audit found an unwanted runtime dependency')
            state['wheels'][kind] = {'path': str(wheel), 'sha256': digest(wheel), 'package': package}
            save_json(manifest, state)
        if source_files(repository) != state['sources']:
            raise RuntimeError('source changed while wheels were building')
        state['status'] = 'built'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['finished'] = utcnow()
        save_json(manifest, state)
    return manifest


def test(manifest, repository, oracles, *, ephemeral_ci=False):
    output = manifest.parent
    build_record = json.loads(manifest.read_text(encoding='utf-8'))
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    if build_record.get('status') != 'built' or build_record.get('format_version') != 1:
        raise ValueError('test stage requires a completed build manifest')
    if build_record['revision'] != revision or build_record['sources'] != source_files(repository):
        raise ValueError('build source does not match current checkout')
    if build_record['host'] != host_spec():
        raise ValueError('test host/interpreter does not match build host/interpreter')
    environment = cpu_environment() | verify_oracles(oracles, repository, revision)
    source = Path(build_record['source'])
    if digest(build_record['sdist']) != build_record['sdist_sha256']:
        raise ValueError('source archive changed since build')
    # Recheck extracted test/example/engine sources, not only the wheel bytes.
    checked = extract_source(Path(build_record['sdist']), output/'verified-source', repository)
    for path in files_under(checked):
        if not (source/path.relative_to(checked)).is_file() or digest(source/path.relative_to(checked)) != digest(path):
            raise ValueError('extracted source changed since build')
    state = {'status': 'testing', 'started': utcnow(), 'build_manifest': str(manifest),
             'oracle_manifest_sha256': digest(oracles/'manifest.json'), 'artifacts': {}}
    report = output/'validation.json'
    save_json(report, state)
    work = output/'run'
    work.mkdir()
    def command(arguments, name):
        run(arguments, cwd=work, environment=environment, log=output/'logs'/f'{name}.log')
    try:
        for kind in ('checkout', 'source'):
            record = build_record['wheels'][kind]
            wheel = Path(record['path'])
            if digest(wheel) != record['sha256']:
                raise ValueError('wheel changed since build')
            if verify_wheel(wheel, source, build_record['host']) != record['package']:
                raise ValueError('wheel package changed since build')
            directory = output/'environments'/kind
            command([sys.executable, '-m', 'venv', directory], f'{kind}-environment')
            python = directory/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            command([python, '-m', 'pip', 'install', '--only-binary=:all:', f'{wheel}[hdf5]', 'pytest'], f'{kind}-install')
            command([python, '-m', 'pip', 'check'], f'{kind}-dependencies')
            command([python, '-m', 'pip', 'list', '--format=json'], f'{kind}-versions')
            offline_result = offline(python, source/'python-native/examples', output/f'{kind}-offline.json',
                                     environment=environment, cwd=work, ephemeral_ci=ephemeral_ci)
            state['artifacts'][kind] = {'python': str(python), 'wheel_sha256': record['sha256'],
                                        'offline_examples': len(offline_result['examples'])}
            save_json(report, state)
            xml = output/f'{kind}-tests.xml'
            arguments = [python, '-I', '-X', 'utf8', '-m', 'pytest', source/'python-native/tests', '-q', '-s', f'--junitxml={xml}']
            if kind == 'checkout':
                arguments += ['-k', 'output or test_nonlinear_adaptive_rejection_and_dense_output']
            command(arguments, f'{kind}-tests')
            state['artifacts'][kind]['tests'] = pytest_result(xml)
            verify_oracles(oracles, repository, revision)
            save_json(report, state)
        state['status'] = 'passed'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['finished'] = utcnow()
        save_json(report, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    subparsers = parser.add_subparsers(dest='stage', required=True)
    builder = subparsers.add_parser('build')
    builder.add_argument('--output', type=Path, required=True, help='New directory outside the checkout')
    tester = subparsers.add_parser('test')
    tester.add_argument('--manifest', type=Path, required=True)
    tester.add_argument('--oracles', type=Path, required=True)
    tester.add_argument('--ephemeral-ci', action='store_true')
    args = parser.parse_args()
    if args.stage == 'build':
        build(args.repository.resolve(), args.output.resolve())
    else:
        test(args.manifest.resolve(), args.repository.resolve(), args.oracles.resolve(), ephemeral_ci=args.ephemeral_ci)


if __name__ == '__main__':
    main()
