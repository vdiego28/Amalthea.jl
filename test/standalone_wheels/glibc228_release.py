#!/usr/bin/env python3
"""Prepare immutable v1.1.0 inputs for the maintained glibc 2.28 gate."""
import argparse
import copy
import email
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'test'), str(ROOT/'python-native/tools')]
import release
import collect
from _validation import digest, save_json, verify_oracles
from wheel_validation import extract_source, source_files
from pip._vendor.packaging.markers import default_environment
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.tags import compatible_tags, cpython_tags
from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename

REVISION = '41227f43c920af50bfa4324766d4f9636340a7b9'
RUN_ID = 37235178219
TAG = 'v1.1.0'
ORACLE_HASH = 'b295a601e88484543b32efd1da3477a5c381b657f2ea12418ac70c722a36d2b3'
PBS_RELEASE = '20260929'
INTERPRETERS = {
    '3.11': ('3.11.16', 'fbbfd0f2253996486455f44a28c09ac7e2df534b8a4835408d17c99d185b80ec', 48930953),
    '3.12': ('3.12.14', '06c90b93f419b63371c18f20fed0558a1a901f6518c3c24f755077e048447e7f', 66919180),
    '3.13': ('3.13.15', 'd6b4e09474dfc219befabeae16264466f09615a991dcd080ae698833bdb3ed44', 75170574),
    '3.14': ('3.14.7', '0056208b5fdfcec939bc5b3908f5e7c89d11c6e2f1fdf02c9cc6aa98d10c5d05', 74997980),
}
PLATFORMS = [f'manylinux_2_{minor}_x86_64' for minor in range(28, 4, -1)]
PLATFORMS += ['manylinux2014_x86_64', 'manylinux2010_x86_64', 'manylinux1_x86_64']
CHUNK_BYTES = 24 * 1024 * 1024
require = release.require


def read(path):
    return json.loads(path.read_text())


def candidate(repository):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repository, text=True).strip()
    require(git('rev-parse', 'HEAD') == REVISION, 'candidate HEAD differs from release')
    require(git('rev-parse', TAG+'^{commit}') == REVISION, 'candidate tag differs from release')
    require(not git('status', '--porcelain', '--untracked-files=no'), 'candidate tracked files changed')
    release.metadata(repository, TAG)


def api(path):
    headers = {'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'}
    if os.environ.get('GH_TOKEN'):
        headers['Authorization'] = 'Bearer '+os.environ['GH_TOKEN']
    request = urllib.request.Request('https://api.github.com/'+path, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def verify_ci(repository, output):
    candidate(repository)
    prefix = 'repos/vdiego28/Amalthea.jl/actions/runs/'+str(RUN_ID)
    raw = api(prefix)
    require(raw['id'] == RUN_ID and raw['path'] == '.github/workflows/run_tests.yml', 'wrong source workflow')
    jobs = []
    page = 1
    while True:
        data = api(prefix+f'/jobs?per_page=100&page={page}')
        jobs.extend(data['jobs'])
        if len(jobs) >= data['total_count']:
            break
        page += 1
    run = {'databaseId': raw['id'], 'headSha': raw['head_sha'], 'event': raw['event'],
           'status': raw['status'], 'conclusion': raw['conclusion'], 'url': raw['html_url'], 'jobs': jobs}
    release.check_run(run, REVISION, release.required_jobs())
    artifacts = api(prefix+'/artifacts?per_page=100')
    require(artifacts['total_count'] <= 100, 'artifact inventory requires pagination')
    for name in ['python-native-oracles'] + [f'python-native-linux-x86_64-{p}' for p in INTERPRETERS]:
        matching = [a for a in artifacts['artifacts'] if a['name'] == name]
        require(len(matching) == 1 and not matching[0]['expired'], 'missing/expired/ambiguous artifact: '+name)
        require(matching[0]['workflow_run']['head_sha'] == REVISION, 'artifact source revision mismatch')
    save_json(output/'source-run.json', run)
    save_json(output/'source-artifacts.json', artifacts)


def oracle_proof(repository, oracles):
    require(digest(oracles/'manifest.json') == ORACLE_HASH, 'original reference manifest hash mismatch')
    verify_oracles(oracles, repository, REVISION)


def split_archive(archive, output, chunk_bytes=CHUNK_BYTES):
    require(chunk_bytes > 0 and chunk_bytes <= CHUNK_BYTES, 'invalid transport chunk size')
    count = (archive.stat().st_size + chunk_bytes - 1)//chunk_bytes
    require(0 < count <= 8, 'oracle archive exceeds the eight-chunk transport limit')
    records = []
    with archive.open('rb') as stream:
        for index in range(count):
            target = output/f'oracles.part{index:02d}'
            with target.open('xb') as destination:
                destination.write(stream.read(chunk_bytes))
            records.append({'filename': target.name, 'sha256': digest(target), 'bytes': target.stat().st_size})
    return {'archive': archive.name, 'sha256': digest(archive), 'bytes': archive.stat().st_size,
            'chunk_bytes': chunk_bytes, 'chunks': records}


def transport(repository, oracles, output):
    candidate(repository)
    oracle_proof(repository, oracles)
    output.mkdir(parents=True, exist_ok=False)
    # Only regular files are carried; reject links instead of following them.
    files = sorted(oracles.rglob('*'))
    require(all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in files), 'unsafe reference entry')
    archive = output/'python-native-oracles.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in files:
            bundle.add(path, arcname=path.relative_to(oracles).as_posix(), recursive=False)
    record = split_archive(archive, output)
    record.update(status='verified', revision=REVISION, run_id=RUN_ID, oracle_manifest_sha256=ORACLE_HASH,
                  files={p.relative_to(oracles).as_posix(): digest(p) for p in files if p.is_file()})
    save_json(output/'index.json', record)
    archive.unlink()  # Only bounded chunks are uploaded.
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
            stream.write(f'count={len(record["chunks"])}\n')


def verify_extracted_inventory(source, repository):
    for name, hashed in source_files(repository).items():
        if name == 'python-native/pyproject.toml':
            expected = tomllib.loads((repository/name).read_text())
            expected['tool']['maturin']['manifest-path'] = 'python-native/Cargo.toml'
            require(tomllib.loads((source/'pyproject.toml').read_text()) == expected,
                    'source archive build metadata differs from candidate')
            continue
        if name in ('amalthea/Cargo.toml', 'python-native/Cargo.toml'):
            expected = tomllib.loads((repository/name).read_text())
            expected['package']['readme'] = 'README.md'
            require(tomllib.loads((source/name).read_text()) == expected,
                    'source archive Cargo metadata differs: '+name)
            continue
        relative = name.removeprefix('python-native/') if name.startswith('python-native/python/') else name
        require(digest(source/relative) == hashed, 'source archive inventory differs: '+name)
    for name in ('README.md', 'LICENSE', '.cargo/config.toml'):
        require(digest(source/name) == digest(repository/'python-native'/name),
                'source archive root metadata differs: '+name)


def relocate(repository, artifacts, oracles, python, output):
    candidate(repository)
    oracle_proof(repository, oracles)
    version = INTERPRETERS[python][0]
    # Authenticate source bytes, wheels, offline evidence and complete original tests.
    accepted = collect.inspect_cell(artifacts, 'linux-x86_64', python, REVISION,
                                   collect.source_options(repository), ORACLE_HASH)
    original = read(artifacts/'build.json')
    require(original['host']['python'] == version, 'original build Python patch differs')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(artifacts/'build.json', output/'build-original.json')
    shutil.copytree(artifacts/'wheels', output/'wheels')
    shutil.copytree(artifacts/'sdist', output/'sdist')
    sdist = output/'sdist'/collect.basename(original['sdist'])
    release.sdist_metadata(sdist, '0.1.0')
    source = extract_source(sdist, output/'source', repository)
    verify_extracted_inventory(source, repository)
    build = copy.deepcopy(original)
    changes = []
    def change(record, key, new, field):
        changes.append({'field': field, 'old': record[key], 'new': str(new)})
        record[key] = str(new)
    change(build, 'repository', repository, 'repository')
    change(build, 'source', source, 'source')
    change(build, 'sdist', sdist, 'sdist')
    for kind in ('checkout', 'source'):
        path = output/'wheels'/kind/collect.basename(original['wheels'][kind]['path'])
        release.wheel_metadata(path, '0.1.0')
        change(build['wheels'][kind], 'path', path, f'wheels.{kind}.path')
    save_json(output/'build.json', build)
    save_json(output/'relocation.json', {'status': 'verified_and_relocated', 'revision': REVISION,
              'python': version, 'original_manifest_sha256': digest(output/'build-original.json'),
              'relocated_manifest_sha256': digest(output/'build.json'), 'changes': changes,
              'original_acceptance': accepted})


def download(url, destination, expected=None, size=None):
    with urllib.request.urlopen(url, timeout=180) as response, destination.open('xb') as stream:
        shutil.copyfileobj(response, stream)
    if expected is not None:
        require(digest(destination) == expected, 'download hash mismatch: '+destination.name)
    if size is not None:
        require(destination.stat().st_size == size, 'download size mismatch: '+destination.name)


def interpreter(python, output):
    version, hashed, size = INTERPRETERS[python]
    output.mkdir(parents=True, exist_ok=False)
    name = f'cpython-{version}+{PBS_RELEASE}-x86_64-unknown-linux-gnu-install_only.tar.gz'
    url = f'https://github.com/astral-sh/python-build-standalone/releases/download/{PBS_RELEASE}/'+name
    archive = output/name
    download(url, archive, hashed, size)
    with tarfile.open(archive) as bundle:
        bundle.extractall(output, filter='data')
    save_json(output/'provenance.json', {'version': version, 'release': PBS_RELEASE,
              'url': url, 'sha256': hashed, 'bytes': size, 'interpreter': str(output/'python')})


def dependencies(repository, python, output):
    version = INTERPRETERS[python][0]
    abi = 'cp'+python.replace('.', '')
    project = tomllib.loads((repository/'python-native/pyproject.toml').read_text())['project']
    requirements = project['dependencies'] + project['optional-dependencies']['hdf5'] + ['pytest']
    output.mkdir(parents=True, exist_ok=False)
    (output/'requirements.txt').write_text('\n'.join(requirements)+'\n')
    command = [sys.executable, '-m', 'pip', 'download', '--disable-pip-version-check', '--no-cache-dir',
               '--only-binary=:all:', '--index-url', 'https://pypi.org/simple', '--python-version', version,
               '--implementation', 'cp', '--abi', abi, '--dest', str(output), '-r', str(output/'requirements.txt')]
    for platform in PLATFORMS:
        command += ['--platform', platform]
    save_json(output/'download-command.json', command)
    with (output/'download.log').open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    major_minor = tuple(map(int, python.split('.')))
    accepted = set(cpython_tags(major_minor, abis=[abi], platforms=PLATFORMS))
    accepted.update(compatible_tags(major_minor, interpreter=abi, platforms=PLATFORMS))
    environment = default_environment() | dict(python_version=python, python_full_version=version,
        implementation_version=version, implementation_name='cpython', platform_python_implementation='CPython',
        os_name='posix', sys_platform='linux', platform_system='Linux', platform_machine='x86_64', extra='')
    records, selected, closure = [], {}, list(requirements)
    for wheel in sorted(output.glob('*.whl')):
        name, pkg_version, _, tags = parse_wheel_filename(wheel.name)
        require(bool(tags & accepted), 'incompatible dependency wheel: '+wheel.name)
        url = f'https://pypi.org/pypi/{name}/{pkg_version}/json'
        with urllib.request.urlopen(url, timeout=120) as response:
            data = json.load(response)
        published = next(item for item in data['urls'] if item['filename'] == wheel.name)
        require(digest(wheel) == published['digests']['sha256'] and wheel.stat().st_size == published['size'],
                'dependency differs from PyPI: '+wheel.name)
        selected[canonicalize_name(name)] = str(pkg_version)
        record = {'filename': wheel.name, 'sha256': digest(wheel), 'pypi_json_url': url,
                  'python': version, 'accepted_tags': sorted(map(str, tags & accepted)), 'elf_objects': []}
        with zipfile.ZipFile(wheel) as archive:
            names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            require(len(names) == 1, 'ambiguous dependency metadata')
            metadata = email.message_from_bytes(archive.read(names[0]))
            require(version in SpecifierSet(metadata.get('Requires-Python', '')), 'dependency Python mismatch')
            closure.extend(metadata.get_all('Requires-Dist', []))
            for member in archive.infolist():
                if member.is_dir():
                    continue
                with archive.open(member) as stream:
                    if stream.read(4) != b'\x7fELF':
                        continue
                with tempfile.NamedTemporaryFile(dir=output) as target:
                    target.write(archive.read(member)); target.flush()
                    symbols = subprocess.check_output(['readelf', '--version-info', target.name], text=True)
                glibcs = {tuple(map(int, v.split('.'))) for v in re.findall(r'GLIBC_(\d+\.\d+(?:\.\d+)?)', symbols)}
                require(all(v <= (2, 28) for v in glibcs), 'dependency requires newer glibc: '+wheel.name)
                record['elf_objects'].append({'path': member.filename, 'glibc_versions': sorted(glibcs)})
        records.append(record)
    active = []
    for text in closure:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        name = canonicalize_name(requirement.name)
        require(name in selected and selected[name] in requirement.specifier, 'dependency closure failed: '+text)
        active.append(text)
    save_json(output/'provenance.json', {'status': 'verified', 'python': version, 'requirements': requirements,
              'pyproject_sha256': digest(repository/'python-native/pyproject.toml'), 'maximum_glibc': '2.28',
              'active_requirements': sorted(set(active)), 'wheels': records})


def namespace_access(trees, output):
    """Allow namespace UID 0 to traverse public inputs extracted by the runner.

    Python's data extraction filter creates directories with mode 0700. A
    sudo-created user namespace cannot override the host runner's DAC modes.
    Only directory read/search bits change; files and links remain untouched.
    """
    require(not output.exists(), 'namespace access receipt already exists')
    require(all(path.is_dir() and not path.is_symlink() for path in trees),
            'namespace inputs must be real directories')
    changes = []
    for root in trees:
        for path in [root, *sorted(root.rglob('*'))]:
            if path.is_symlink() or not path.is_dir():
                continue
            before = path.stat().st_mode & 0o7777
            after = before | 0o055
            if before != after:
                path.chmod(after)
                changes.append({'tree': str(root), 'path': path.relative_to(root).as_posix(),
                                'before': oct(before), 'after': oct(after)})
    save_json(output, {'status': 'prepared_for_namespace', 'directory_changes': changes,
                      'file_bytes_modified': False, 'links_modified': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('ci', 'transport', 'relocate', 'interpreter', 'dependencies', 'access'))
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--oracles', type=Path)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--python', choices=INTERPRETERS)
    args = parser.parse_args()
    repository, output = args.repository.resolve(), args.output.resolve()
    if args.command == 'ci':
        verify_ci(repository, output)
    elif args.command == 'transport':
        transport(repository, args.oracles.resolve(), output)
    elif args.command == 'relocate':
        relocate(repository, args.artifacts.resolve(), args.oracles.resolve(), args.python, output)
    elif args.command == 'interpreter':
        interpreter(args.python, output)
    elif args.command == 'access':
        candidate(repository)
        runtime, artifacts = args.runtime.resolve(), args.artifacts.resolve()
        source = Path(read(artifacts/'build.json')['source']).resolve()
        require(source.is_relative_to(artifacts/'source'), 'extracted source escaped the build directory')
        namespace_access([runtime/'rootfs/rootfs', runtime/'cpython/python', source], output)
    else:
        candidate(repository)
        dependencies(repository, args.python, output)


if __name__ == '__main__':
    main()
