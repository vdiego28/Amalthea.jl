#!/usr/bin/env python3
"""Verify standalone wheels in an isolated, pinned glibc 2.28 userspace."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import subprocess
import sys
import tarfile
import urllib.request

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY/'python-native/tools'))
from _validation import (EXAMPLES, FIXTURES, cpu_environment, digest, pytest_result, run,
                         save_json, utcnow, verify_oracles)
from wheel_validation import source_files

VENDOR = 'debuerreotype/docker-debian-artifacts'
REVISION = '686d9f6eaada08a754bc7abf6f6184c65c5b378f'
ARCHIVE_BLOB = '247843072c4d7e1ee10d4efe42b849b57d9f4d76'
ARCHIVE_SHA256 = '2bc7ec77d5d367039d49548f479aaebab87641f0c51dceb5e8c2e595c5170c32'
MOUNTPOINTS = ('opt/python', 'work', 'evidence', 'oracles', 'wheelhouse',
               'tests', 'examples', 'tools', 'wheels')


def verify_rootfs(rootfs):
    state = json.loads((rootfs.parent/'rootfs.json').read_text())
    if (state.get('status') != 'prepared' or state.get('vendor') != VENDOR
            or state.get('revision') != REVISION or state.get('git_blob') != ARCHIVE_BLOB
            or Path(state.get('rootfs', '')).resolve() != rootfs.resolve()):
        raise ValueError('root filesystem has no matching prepared vendor provenance')
    archive = rootfs.parent/'rootfs.tar.xz'
    if digest(archive) != ARCHIVE_SHA256 or state['files']['rootfs.tar.xz']['sha256'] != ARCHIVE_SHA256:
        raise ValueError('root filesystem archive changed after preparation')
    for name in MOUNTPOINTS:
        if not (rootfs/name).is_dir():
            raise ValueError(f'root filesystem is missing mountpoint: {name}')
    return state


def libc_package_version(manifest):
    packages = dict(line.split('\t') for line in manifest.splitlines() if line)
    version = packages.get('libc6:amd64', packages.get('libc6', ''))
    if not version.startswith('2.28-'):
        raise ValueError('vendor package manifest must contain libc6 2.28')
    return version


def extract_rootfs(archive, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as stream:
        members = []
        for original in stream.getmembers():
            member = copy.copy(original)
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or '\\' in member.name:
                raise ValueError(f'unsafe rootfs member: {member.name}')
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError(f'unsupported rootfs entry: {member.name}')
            if member.issym() and member.linkname.startswith('/'):
                target = posixpath.normpath(member.linkname).lstrip('/')
                member.linkname = posixpath.relpath(target, name.parent.as_posix())
            if member.islnk() and member.linkname.startswith('/'):
                member.linkname = posixpath.normpath(member.linkname).lstrip('/')
            members.append(member)
        # The standard data filter also rejects relative links escaping the root.
        stream.extractall(destination, members=members, filter='data')


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'preparing', 'started': utcnow(), 'vendor': VENDOR,
             'revision': REVISION, 'files': {}}
    try:
        for name in ('rootfs.tar.xz.sha256', 'rootfs.manifest', 'rootfs.os-release',
                     'rootfs.dpkg-arch', 'rootfs.tar.xz'):
            url = f'https://raw.githubusercontent.com/{VENDOR}/{REVISION}/buster/slim/{name}'
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            (output/name).write_bytes(data)
            state['files'][name] = {'url': url, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        archive = output/'rootfs.tar.xz'
        data = archive.read_bytes()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if blob != ARCHIVE_BLOB:
            raise ValueError('root filesystem does not match the pinned vendor Git blob')
        if digest(archive) != (output/'rootfs.tar.xz.sha256').read_text().strip():
            raise ValueError('root filesystem checksum does not match the vendor manifest')
        if (output/'rootfs.dpkg-arch').read_text().strip() != 'amd64':
            raise ValueError('root filesystem must be amd64')
        state['libc_package'] = libc_package_version((output/'rootfs.manifest').read_text())
        extract_rootfs(archive, output/'rootfs')
        for name in MOUNTPOINTS:
            (output/'rootfs'/name).mkdir(parents=True, exist_ok=True)
        state.update(status='prepared', rootfs=str(output/'rootfs'), git_blob=blob,
                     empty_mountpoints=list(MOUNTPOINTS))
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['finished'] = utcnow()
        save_json(output/'rootfs.json', state)


def container_command(rootfs, interpreter, work, evidence, *, bindings=(), environment=None):
    executable = shutil.which('bwrap')
    if not executable:
        raise RuntimeError('bubblewrap is required; host execution is not an alternative')
    command = [executable, '--unshare-user', '--unshare-pid', '--unshare-net', '--unshare-uts',
               '--unshare-ipc', '--die-with-parent', '--new-session', '--ro-bind', str(rootfs), '/',
               '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
               '--ro-bind', str(interpreter), '/opt/python', '--bind', str(work), '/work',
               '--bind', str(evidence), '/evidence', '--chdir', '/work', '--clearenv']
    variables = {'PATH': '/nonexistent', 'HOME': '/tmp', 'LANG': 'C.UTF-8', 'OPENBLAS_NUM_THREADS': '1',
                 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'RAYON_NUM_THREADS': '1',
                 'AMALTHEA_CUDA_BUILD': 'off', 'AMALTHEA_REQUIRE_CUDA_TESTS': '0',
                 'PIP_DISABLE_PIP_VERSION_CHECK': '1'} | (environment or {})
    for key, value in variables.items():
        command += ['--setenv', key, str(value)]
    for host, target in bindings:
        command += ['--ro-bind', str(host), target]
    return command + ['--']


PROBE = '''
import ctypes,json,platform,sys
from pathlib import Path
libc=ctypes.CDLL('libc.so.6');libc.gnu_get_libc_version.restype=ctypes.c_char_p
version=libc.gnu_get_libc_version().decode()
assert version=='2.28',version
libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/' in line})
assert any(path.endswith('/libc-2.28.so') for path in libraries),libraries
assert not any(Path(path).exists() for path in ('/usr/bin/cargo','/usr/bin/julia','/root/.cargo'))
record=dict(glibc=version,python=sys.version,executable=sys.executable,prefix=sys.prefix,
            kernel=platform.release(),libraries=libraries,interfaces=Path('/proc/net/dev').read_text())
Path(sys.argv[1]).write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
'''


def probe(rootfs, interpreter, output):
    verify_rootfs(rootfs)
    output.mkdir(parents=True, exist_ok=False)
    work = output/'work'; work.mkdir()
    prefix = container_command(rootfs, interpreter, work, output)
    run(prefix + ['/opt/python/bin/python3', '-I', '-X', 'utf8', '-c', PROBE, '/evidence/probe.json'],
        cwd=output, environment=cpu_environment(), log=output/'probe.log')


def test(rootfs, interpreter, manifest, oracles, wheelhouse, output):
    output.mkdir(parents=True, exist_ok=False)
    rootfs_provenance = verify_rootfs(rootfs)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True).strip()
    if oracles is not None:
        verify_oracles(oracles, REPOSITORY, revision)
    build = json.loads(manifest.read_text())
    if build['status'] != 'built' or build['revision'] != revision:
        raise ValueError('runtime gate requires the current completed wheel build')
    if build['sources'] != source_files(REPOSITORY) or digest(build['sdist']) != build['sdist_sha256']:
        raise ValueError('build source/archive differs from the current checkout')
    source = Path(build['source'])
    work = output/'work'; work.mkdir()
    env = ({f'AMALTHEA_{key}_ORACLE': f'/oracles/{directory}' for key, (directory, _) in FIXTURES.items()}
           if oracles is not None else {})
    bindings = [(wheelhouse, '/wheelhouse'),
                (source/'python-native/tests', '/tests'), (source/'python-native/examples', '/examples'),
                (source/'python-native/tools', '/tools'), (manifest.parent/'wheels', '/wheels')]
    if oracles is not None:
        bindings.append((oracles, '/oracles'))
    prefix = container_command(rootfs, interpreter, work, output, bindings=bindings, environment=env)
    state = {'status': 'running', 'started': utcnow(), 'build_manifest': str(manifest),
             'build_manifest_sha256': digest(manifest),
             'oracle_manifest_sha256': digest(oracles/'manifest.json') if oracles is not None else None,
             'scope': 'full installed suite' if oracles is not None else 'installation/offline smoke',
             'interpreter': str(interpreter), 'interpreter_sha256': digest(interpreter/'bin/python3'),
             'rootfs_provenance': rootfs_provenance, 'artifacts': {}}
    save_json(output/'validation.json', state)
    def command(arguments, name):
        run(prefix + arguments, cwd=output, environment=cpu_environment(), log=output/f'{name}.log')
    try:
        command(['/opt/python/bin/python3', '-I', '-X', 'utf8', '-c', PROBE, '/evidence/probe.json'], 'probe')
        version = json.loads((output/'probe.json').read_text())['python'].split()[0]
        if version != build['host']['python']:
            raise ValueError(f'container interpreter {version} differs from wheel build {build["host"]["python"]}')
        for kind in ('checkout', 'source'):
            wheel = Path(build['wheels'][kind]['path'])
            if digest(wheel) != build['wheels'][kind]['sha256']:
                raise ValueError('wheel changed since the build gate')
            python = f'/work/{kind}/bin/python'
            command(['/opt/python/bin/python3', '-m', 'venv', f'/work/{kind}'], f'{kind}-venv')
            command([python, '-m', 'pip', 'install', '--no-index', '--only-binary=:all:',
                     '--find-links', '/wheelhouse', f'/wheels/{kind}/{wheel.name}[hdf5]', 'pytest'], f'{kind}-install')
            command([python, '-m', 'pip', 'check'], f'{kind}-dependencies')
            command([python, '-m', 'pip', 'list', '--format=json'], f'{kind}-versions')
            command([python, '-I', '-X', 'utf8', '/tools/installed_smoke.py', '--examples', '/examples',
                     '--output', f'/evidence/{kind}-offline.json', '--isolation', 'linux'], f'{kind}-offline')
            result = json.loads((output/f'{kind}-offline.json').read_text())
            if result['status'] != 'passed' or set(result['examples']) != set(EXAMPLES):
                raise RuntimeError('not all offline examples completed')
            state['artifacts'][kind] = {'wheel_sha256': digest(wheel), 'offline_examples': len(result['examples'])}
            save_json(output/'validation.json', state)
            if oracles is None:
                continue
            report = f'/evidence/{kind}-tests.xml'
            arguments = [python, '-I', '-X', 'utf8', '-m', 'pytest', '/tests', '-q', '-s', '-o', 'cache_dir=/work/pytest-cache', f'--junitxml={report}']
            if kind == 'checkout':
                arguments += ['-k', 'output or test_nonlinear_adaptive_rejection_and_dense_output']
            command(arguments, f'{kind}-tests')
            state['artifacts'][kind]['tests'] = pytest_result(output/f'{kind}-tests.xml')
            verify_oracles(oracles, REPOSITORY, revision)
            save_json(output/'validation.json', state)
        state['status'] = 'passed' if oracles is not None else 'smoke_passed'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['finished'] = utcnow()
        save_json(output/'validation.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_args = commands.add_parser('prepare')
    prepare_args.add_argument('--output', type=Path, required=True)
    for name in ('probe', 'smoke', 'test'):
        subparser = commands.add_parser(name)
        for argument in ('rootfs', 'interpreter', 'output'):
            subparser.add_argument('--'+argument, type=Path, required=True)
        if name in ('smoke', 'test'):
            for argument in ('manifest', 'wheelhouse'):
                subparser.add_argument('--'+argument, type=Path, required=True)
        if name == 'test':
            subparser.add_argument('--oracles', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output.resolve())
    elif args.command == 'probe':
        probe(args.rootfs.resolve(), args.interpreter.resolve(), args.output.resolve())
    else:
        test(args.rootfs.resolve(), args.interpreter.resolve(), args.manifest.resolve(),
             args.oracles.resolve() if args.command == 'test' else None,
             args.wheelhouse.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
