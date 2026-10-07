#!/usr/bin/env python3
"""Check release metadata, exact-source CI, assets and retained glibc evidence."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tomllib
import tarfile
import zipfile
import zlib
from email.parser import BytesParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'test/standalone_wheels'))
import collect as wheels
import glibc228

LIBRARIES = (
    'libamalthea-x86_64-unknown-linux-gnu.so',
    'libamalthea-aarch64-unknown-linux-gnu.so',
    'libamalthea-aarch64-apple-darwin.dylib',
    'libamalthea-x86_64-pc-windows-msvc.dll',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metadata(repository, tag=None):
    def toml(name):
        return tomllib.loads((repository / name).read_text(encoding='utf-8'))
    julia = toml('Project.toml')['version']
    wrapper = toml('python/pyproject.toml')['project']['version']
    native = toml('python-native/pyproject.toml')['project']['version']
    for version in (julia, wrapper, native):
        require(re.fullmatch(r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)', version),
                f'release version must be X.Y.Z without dev/prerelease suffix: {version}')
    require(julia == wrapper, 'Julia and Julia-backed Python versions differ')
    expected_tag = 'v' + julia
    require(tag is None or tag == expected_tag, f'tag must equal {expected_tag}, got {tag}')
    tree = ast.parse((repository / 'python-native/python/amalthea_native/__init__.py').read_text())
    versions = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == '__version__' for target in node.targets)]
    require(versions == [native], 'standalone import/package versions differ')
    citation = (repository / 'CITATION.cff').read_text(encoding='utf-8')
    require(re.search(r'^version: "' + re.escape(julia) + r'"$', citation, re.M),
            'citation version differs from Julia candidate')
    notes = Path('docs/dev/releases') / (expected_tag + '.md')
    text = (repository / notes).read_text(encoding='utf-8')
    require(text.startswith('# Amalthea.jl ' + expected_tag + '\n'), 'release notes title mismatch')
    require('`amalthea-native` ' + native in text, 'release notes omit standalone version')
    return {'julia_version': julia, 'wrapper_version': wrapper, 'python_version': native,
            'tag': expected_tag, 'notes': notes.as_posix()}


def required_jobs():
    jobs = {'Independent Julia references for standalone wheels',
            'Apple Silicon prepared hardware diagnostic',
            'CPU-only install and FFI smoke (Linux ARM64)',
            'Native-path benchmark (regression guard)', 'Python API Test (ubuntu-latest)'}
    for group in ('physics', 'rust'):
        for os in ('ubuntu-latest', 'windows-2025-vs2026', 'macos-latest'):
            jobs.add(f'Test ({group} - {os} - Julia 1)')
    for version in ('lts', 'pre'):
        jobs.add(f'Test (physics - ubuntu-latest - Julia {version})')
    for group in ('sim-interface', 'sim-multimode', 'sim-propagation', 'io', 'fields', 'examples'):
        jobs.add(f'Test ({group} - ubuntu-latest - Julia 1)')
    jobs.update(f'Standalone Python ({platform} - {python})'
                for platform in wheels.PLATFORMS for python in wheels.PYTHONS)
    return jobs


def check_run(run, revision, jobs):
    require(run.get('headSha') == revision, 'CI revision mismatch')
    require(run.get('event') == 'push', 'release acceptance requires a push run')
    require(run.get('status') == 'completed' and run.get('conclusion') == 'success',
            'exact-source workflow has not succeeded')
    for name in sorted(jobs):
        matching = [job for job in run.get('jobs', []) if job.get('name') == name]
        require(len(matching) == 1 and matching[0].get('conclusion') == 'success',
                f'required job missing, skipped or unsuccessful: {name}')


def gh(*arguments):
    return json.loads(subprocess.check_output(['gh', *arguments], text=True))


def wait_ci(repository, revision, output, timeout):
    require(re.fullmatch('[0-9a-f]{40}', revision), 'CI revision must be a full SHA')
    require(timeout > 0, 'timeout must be positive')
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    pending = {'run_tests.yml': required_jobs(), 'documenter.yml': {'build', 'Deploy documentation'}}
    accepted = {}
    while pending:
        for workflow, jobs in list(pending.items()):
            runs = gh('run', 'list', '--repo', repository, '--workflow', workflow,
                      '--commit', revision, '--event', 'push', '--limit', '30',
                      '--json', 'databaseId,headSha,event,status,conclusion,createdAt')
            exact = [run for run in runs if run.get('headSha') == revision and run.get('event') == 'push']
            if not exact:
                print(f'{workflow}: waiting for a push run at {revision}', flush=True)
                continue
            latest = max(exact, key=lambda run: run['databaseId'])
            run = gh('run', 'view', str(latest['databaseId']), '--repo', repository,
                     '--json', 'databaseId,url,headSha,event,status,conclusion,jobs')
            save(output / (workflow + '.json'), run)
            print(f"{workflow}: run {run['databaseId']} {run['status']} / {run.get('conclusion')}", flush=True)
            if run['status'] == 'completed':
                check_run(run, revision, jobs)
                accepted[workflow] = run['databaseId']
                del pending[workflow]
        if pending:
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'timed out waiting for exact-source CI; retained run metadata')
            time.sleep(min(60, remaining))
    save(output / 'acceptance.json', {'revision': revision, 'status': 'passed', 'runs': accepted})
    return accepted


def wheel_metadata(path, version):
    require(path.name.startswith('amalthea_native-' + version + '-'), 'wheel filename version mismatch')
    with zipfile.ZipFile(path) as archive:
        entries = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        require(len(entries) == 1, 'wheel metadata missing or ambiguous')
        package = BytesParser().parsebytes(archive.read(entries[0]))
    require(package['Name'] == 'amalthea-native' and package['Version'] == version,
            'wheel distribution metadata differs from candidate')


def sdist_metadata(path, version):
    with tarfile.open(path, 'r:gz') as archive:
        entries = [entry for entry in archive.getmembers() if entry.name.endswith('/PKG-INFO')]
        require(len(entries) == 1 and entries[0].isfile(), 'source metadata missing or ambiguous')
        package = BytesParser().parsebytes(archive.extractfile(entries[0]).read())
    require(package['Name'] == 'amalthea-native' and package['Version'] == version,
            'source distribution metadata differs from candidate')


def stage_python(artifacts, repository, run, output):
    candidate = metadata(repository)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    check_run(run, revision, required_jobs())
    require(not output.exists(), 'choose a new staging directory; existing assets are preserved')
    report = wheels.collect(artifacts, repository, run, list(wheels.PLATFORMS))
    require(report['status'] == 'wheel_matrix_passed', 'sixteen-cell wheel acceptance incomplete: ' +
            json.dumps({'errors': report['errors'], 'cells': {name: cell for name, cell in report['cells'].items()
                       if cell['status'] != 'passed'}}))
    selected = []
    for name, cell in sorted(report['cells'].items()):
        record = cell['wheels']['source']
        path = artifacts / ('python-native-' + name) / 'wheels/source' / record['filename']
        wheel_metadata(path, candidate['python_version'])
        selected.append((path, record['sha256'], name))
    archive_cell = artifacts / 'python-native-linux-x86_64-3.11'
    build = read(archive_cell / 'build.json')
    archive = archive_cell / 'sdist' / wheels.basename(build['sdist'])
    require(archive.name == 'amalthea_native-' + candidate['python_version'] + '.tar.gz',
            'source archive version mismatch')
    sdist_metadata(archive, candidate['python_version'])
    selected.append((archive, build['sdist_sha256'], 'linux-x86_64-3.11'))
    require(len({path.name for path, _, _ in selected}) == 17, 'duplicate distribution filenames')
    # Validate everything before creating/copying release assets.
    for path, expected, _ in selected:
        require(wheels.digest(path) == expected, 'selected artifact changed since acceptance')
    output.mkdir(parents=True)
    save(output / 'wheel-evidence.json', report)
    (output / 'wheel-evidence.md').write_text(wheels.markdown(report), encoding='utf-8')
    for path, expected, _ in selected:
        shutil.copyfile(path, output / path.name)
        require(wheels.digest(output / path.name) == expected, 'staged artifact copy differs')
    save(output / 'python-artifacts.json', {'revision': revision, **candidate,
         'run_id': run['databaseId'], 'artifacts': {path.name: {'sha256': expected, 'cell': cell}
                                                  for path, expected, cell in selected}})


def checksums(directory, repository):
    candidate = metadata(repository)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    version = candidate['python_version']
    expected = set(LIBRARIES) | {'amalthea_native-' + version + '.tar.gz',
                               'wheel-evidence.json', 'wheel-evidence.md', 'python-artifacts.json'}
    expected.update(f'amalthea_native-{version}-cp{python.replace(".", "")}-cp{python.replace(".", "")}-{host[3]}.whl'
                    for host in wheels.PLATFORMS.values() for python in wheels.PYTHONS)
    actual = {path.name for path in directory.iterdir()}
    require(actual == expected, f'asset inventory mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    require(all(path.is_file() and not path.is_symlink() and path.stat().st_size > 0 for path in directory.iterdir()),
            'assets must be nonempty regular files')
    provenance = read(directory / 'python-artifacts.json')
    require(provenance['revision'] == revision and provenance['python_version'] == version,
            'staged Python provenance differs from candidate')
    distributions = {name for name in expected if name.endswith(('.whl', '.tar.gz'))}
    require(set(provenance['artifacts']) == distributions, 'staged distribution provenance inventory mismatch')
    for name, record in provenance['artifacts'].items():
        require(wheels.digest(directory / name) == record['sha256'], f'staged distribution changed: {name}')
    report = read(directory / 'wheel-evidence.json')
    require(report['status'] == 'wheel_matrix_passed' and report['revision'] == revision,
            'staged wheel evidence differs from candidate')
    hashes = {name: wheels.digest(directory / name) for name in sorted(expected)}
    save(directory / 'release-manifest.json', {'revision': revision, **candidate, 'assets': hashes,
                                             'cpu_only': True, 'test_run': provenance['run_id']})
    hashes['release-manifest.json'] = wheels.digest(directory / 'release-manifest.json')
    (directory / 'SHA256SUMS.txt').write_text(''.join(f'{hashed}  {name}\n' for name, hashed in sorted(hashes.items())),
                                           encoding='ascii')


def glibc_command(directory, name, expected, inspected):
    """Inspect retained commands without dereferencing their producer paths."""
    log = directory / (name + '.log')
    record_path = log.with_suffix('.log.json')
    inspected[log.name] = wheels.digest(log)
    inspected[record_path.name] = wheels.digest(record_path)
    record = read(record_path)
    require(record.get('exit_code') == 0 and not record.get('error')
            and record.get('started') and record.get('finished'), f'{name}: command did not complete')
    argv = record['argv']
    separator = argv.index('--')
    require(wheels.basename(argv[0]) == 'bwrap' and '--unshare-net' in argv[:separator]
            and '--clearenv' in argv[:separator], f'{name}: missing isolated helper command')
    require(argv[separator + 1:] == expected, f'{name}: unexpected or partial helper command')


def inspect_glibc(directory, build_directory, python, revision, sources, oracle_hash,
                  version, assets, release_assets, inspected):
    def evidence(name):
        inspected[name] = wheels.digest(directory / name)
        return read(directory / name)

    manifest = build_directory / 'build.json'
    inspected['build.json'] = wheels.digest(manifest)
    build = read(manifest)
    require(build.get('format_version') == 1 and build.get('status') == 'built', 'build incomplete')
    require(build['revision'] == revision and build['sources'] == sources,
            'build revision or sources differ from candidate')
    host = build['host']
    require(tuple(host[key] for key in ('system', 'machine', 'target', 'tag')) ==
            wheels.PLATFORMS['linux-x86_64'][:4], 'wrong build platform')
    require(re.fullmatch(re.escape(python) + r'\.\d+', host['python']), 'wrong build Python version')
    state = evidence('validation.json')
    require(state.get('status') == 'passed' and state.get('scope') == 'full installed suite'
            and state.get('finished') and not state.get('error'), 'full glibc validation incomplete')
    require(state['build_manifest_sha256'] == inspected['build.json'], 'build manifest digest mismatch')
    require(state['oracle_manifest_sha256'] == oracle_hash, 'oracle manifest digest mismatch')
    require(set(state['artifacts']) == set(wheels.MINIMUM_TESTS), 'incomplete wheel-kind inventory')
    rootfs = state['rootfs_provenance']
    require(rootfs.get('status') == 'prepared' and rootfs.get('vendor') == glibc228.VENDOR
            and rootfs.get('revision') == glibc228.REVISION and rootfs.get('git_blob') == glibc228.ARCHIVE_BLOB
            and rootfs['files']['rootfs.tar.xz']['sha256'] == glibc228.ARCHIVE_SHA256
            and rootfs['libc_package'].startswith('2.28-'), 'pinned glibc rootfs provenance mismatch')
    probe = evidence('probe.json')
    require(probe['glibc'] == '2.28' and probe['python'].split()[0] == host['python'],
            'runtime glibc or Python version mismatch')
    require(probe['executable'] == '/opt/python/bin/python3' and probe['prefix'] == '/opt/python',
            'unexpected probe interpreter')
    require(any(name.endswith('/libc-2.28.so') for name in probe['libraries']),
            'probe did not load glibc 2.28')
    require({line.split(':')[0].strip() for line in probe['interfaces'].splitlines() if ':' in line} == {'lo'},
            'probe network namespace is not isolated')
    glibc_command(directory, 'probe', ['/opt/python/bin/python3', '-I', '-X', 'utf8', '-c',
                                      glibc228.PROBE, '/evidence/probe.json'], inspected)
    require(read(directory / 'probe.log') == probe, 'raw probe log differs from result')
    archive = build_directory / 'sdist' / wheels.basename(build['sdist'])
    inspected['sdist/' + archive.name] = wheels.digest(archive)
    require(inspected['sdist/' + archive.name] == build['sdist_sha256'], 'source archive digest mismatch')
    require(archive.name == f'amalthea_native-{version}.tar.gz', 'source archive version mismatch')
    sdist_metadata(archive, version)
    result = {'status': 'passed', 'python': host['python'], 'glibc': probe['glibc'], 'wheels': {}}
    for kind, minimum in wheels.MINIMUM_TESTS.items():
        record = build['wheels'][kind]
        wheel = build_directory / 'wheels' / kind / wheels.basename(record['path'])
        hashed = wheels.digest(wheel)
        inspected[f'wheels/{kind}/{wheel.name}'] = hashed
        installed = state['artifacts'][kind]
        require(hashed == record['sha256'] == installed['wheel_sha256'], 'wheel digest mismatch')
        cp = 'cp' + python.replace('.', '')
        require(wheel.name.endswith(f'-{cp}-{cp}-manylinux_2_28_x86_64.whl'), 'wrong wheel interpreter/platform tag')
        wheel_metadata(wheel, version)
        with zipfile.ZipFile(wheel) as archive_file:
            names = archive_file.namelist()
            require(len(names) == len(set(names)), 'duplicate wheel members')
            package = {name: hashlib.sha256(archive_file.read(name)).hexdigest() for name in names
                       if name.startswith('amalthea_native/') and not name.endswith(('/', '.so', '.pyd'))}
            expected = {name.removeprefix('python-native/python/'): hashed
                        for name, hashed in sources.items() if name.startswith('python-native/python/')}
            require(package == record['package'] == expected, 'wheel package inventory mismatch')
            extensions = [name for name in names if name.endswith(('.so', '.pyd'))]
        require(len(extensions) == 1 and extensions[0].startswith('amalthea_native/_native.')
                and extensions[0].endswith('.so'), 'native wheel extension missing or ambiguous')
        interpreter = f'/work/{kind}/bin/python'
        commands = {
            'venv': ['/opt/python/bin/python3', '-m', 'venv', f'/work/{kind}'],
            'install': [interpreter, '-m', 'pip', 'install', '--no-index', '--only-binary=:all:',
                        '--find-links', '/wheelhouse', f'/wheels/{kind}/{wheel.name}[hdf5]', 'pytest'],
            'dependencies': [interpreter, '-m', 'pip', 'check'],
            'versions': [interpreter, '-m', 'pip', 'list', '--format=json'],
            'offline': [interpreter, '-I', '-X', 'utf8', '/tools/installed_smoke.py', '--examples',
                        '/examples', '--output', f'/evidence/{kind}-offline.json', '--isolation', 'linux'],
            'tests': [interpreter, '-I', '-X', 'utf8', '-m', 'pytest', '/tests', '-q', '-s', '-o',
                      'cache_dir=/work/pytest-cache', f'--junitxml=/evidence/{kind}-tests.xml'],
        }
        if kind == 'checkout':
            commands['tests'] += ['-k', 'output or test_nonlinear_adaptive_rejection_and_dense_output']
        for name, command in commands.items():
            glibc_command(directory, f'{kind}-{name}', command, inspected)
        offline = evidence(f'{kind}-offline.json')
        wheels.check_offline(offline, 'Linux', 'linux')
        require(offline['python'].split()[0] == host['python'] and offline['executable'] == interpreter
                and offline['prefix'] == f'/work/{kind}', 'offline interpreter mismatch')
        require(wheels.basename(offline['extension']) == wheels.basename(extensions[0]),
                'loaded extension filename mismatch')
        require(not offline.get('error') and offline.get('finished')
                and installed['offline_examples'] == len(wheels.EXAMPLES), 'incomplete offline examples')
        require(any(name.endswith('/libc-2.28.so') for name in offline['libraries']),
                'offline examples did not load glibc 2.28')
        example_log = (directory / f'{kind}-offline.log').read_text(encoding='utf-8')
        require([line.removeprefix('Passed example: ') for line in example_log.splitlines()
                 if line.startswith('Passed example: ')] == list(wheels.EXAMPLES), 'incomplete raw example log')
        xml = directory / f'{kind}-tests.xml'
        inspected[xml.name] = wheels.digest(xml)
        tests = wheels.pytest_result(xml)
        require(tests == installed['tests'] and tests['passed'] >= minimum, 'incomplete numerical tests')
        if kind == 'source':
            asset = assets / wheel.name
            inspected['release/' + wheel.name] = wheels.digest(asset)
            require(inspected['release/' + wheel.name] == hashed == release_assets[wheel.name],
                    'release source wheel differs from glibc-tested bytes')
        result['wheels'][kind] = {'filename': wheel.name, 'sha256': hashed, 'tests': tests,
                                  'examples': len(wheels.EXAMPLES)}
    return result


def verify_glibc(gates, builds, repository, oracles, assets):
    """Verify retained evidence only; never execute a wheel or alter a producer record."""
    report = {'format_version': 1, 'status': 'incomplete', 'captured': wheels.utcnow(),
              'scope': 'Linux x86_64 glibc 2.28 release-wheel evidence; other release gates remain separate',
              'inputs': {name: str(path.resolve()) for name, path in
                         (('gates', gates), ('builds', builds), ('repository', repository),
                          ('oracles', oracles), ('assets', assets))},
              'sha256': {}, 'errors': [], 'cells': {}}
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
        candidate = metadata(repository)
        report.update(revision=revision, **candidate)
        sources = wheels.source_files(repository)
        wheels.verify_oracles(oracles, repository, revision)
        oracle_hash = wheels.digest(oracles / 'manifest.json')
        report['sha256']['oracle-manifest.json'] = oracle_hash
        manifest = assets / 'release-manifest.json'
        report['sha256']['release-manifest.json'] = wheels.digest(manifest)
        release = read(manifest)
        require(release['revision'] == revision and all(release[key] == value for key, value in candidate.items()),
                'release manifest revision or metadata differs from candidate')
        for python in wheels.PYTHONS:
            inspected = {}
            try:
                cell = inspect_glibc(gates / python, builds / ('linux-x86_64-' + python), python,
                                     revision, sources, oracle_hash, candidate['python_version'],
                                     assets, release['assets'], inspected)
            except (*wheels.INVALID_EVIDENCE, IndexError, EOFError, tarfile.TarError, zlib.error) as error:
                cell = {'status': 'incomplete', 'error': str(error)}
            report['cells'][python] = cell | {'sha256': inspected}
        if all(cell['status'] == 'passed' for cell in report['cells'].values()):
            report['status'] = 'glibc_release_wheels_passed'
    except (*wheels.INVALID_EVIDENCE, IndexError, subprocess.CalledProcessError) as error:
        report['errors'].append(str(error))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    meta = commands.add_parser('metadata')
    meta.add_argument('--repository', type=Path, default=ROOT)
    meta.add_argument('--tag')
    meta.add_argument('--output', type=Path)
    wait = commands.add_parser('wait-ci')
    wait.add_argument('--repo', required=True)
    wait.add_argument('--revision', required=True)
    wait.add_argument('--output', type=Path, required=True)
    wait.add_argument('--timeout', type=int, default=21600)
    stage = commands.add_parser('stage-python')
    stage.add_argument('--repository', type=Path, default=ROOT)
    stage.add_argument('--artifacts', type=Path, required=True)
    stage.add_argument('--run-json', type=Path, required=True)
    stage.add_argument('--output', type=Path, required=True)
    sums = commands.add_parser('checksums')
    sums.add_argument('--repository', type=Path, default=ROOT)
    sums.add_argument('--directory', type=Path, required=True)
    glibc = commands.add_parser('verify-glibc', help='verify four retained glibc 2.28 gates against release wheels offline')
    glibc.add_argument('--repository', type=Path, default=ROOT)
    for name in ('gates', 'builds', 'oracles', 'assets', 'output'):
        glibc.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'metadata':
            result = metadata(args.repository, args.tag)
            if args.output:
                save(args.output, result)
            print(json.dumps(result))
        elif args.command == 'wait-ci':
            wait_ci(args.repo, args.revision, args.output, args.timeout)
        elif args.command == 'stage-python':
            stage_python(args.artifacts, args.repository, read(args.run_json), args.output)
        elif args.command == 'verify-glibc':
            require(not args.output.exists() and not args.output.is_symlink(),
                    'choose a new report path; existing evidence is preserved')
            report = verify_glibc(args.gates, args.builds, args.repository, args.oracles, args.assets)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(report, indent=2) + '\n')
            print(f"{report['status']}: {args.output}")
            return 0 if report['status'] == 'glibc_release_wheels_passed' else 1
        else:
            checksums(args.directory, args.repository)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError, zipfile.BadZipFile, tarfile.TarError) as error:
        print(f'release preparation failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
