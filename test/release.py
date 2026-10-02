#!/usr/bin/env python3
"""Check candidate metadata, gate exact-source CI and stage release assets."""
import argparse
import ast
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
from email.parser import BytesParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'test/standalone_wheels'))
import collect as wheels

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
        else:
            checksums(args.directory, args.repository)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError, zipfile.BadZipFile, tarfile.TarError) as error:
        print(f'release preparation failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
