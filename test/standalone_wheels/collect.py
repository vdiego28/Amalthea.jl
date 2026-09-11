#!/usr/bin/env python3
"""Verify downloaded standalone wheel evidence from one hosted workflow run."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python-native/tools'))
from _validation import EXAMPLES, digest, pytest_result, save_json, utcnow, verify_oracles
from wheel_validation import source_files

PYTHONS = ('3.11', '3.12', '3.13', '3.14')
PLATFORMS = {
    'linux-x86_64': ('Linux', 'x86_64', 'x86_64-unknown-linux-gnu', 'manylinux_2_28_x86_64', 'linux'),
    'linux-arm64': ('Linux', 'aarch64', 'aarch64-unknown-linux-gnu', 'manylinux_2_28_aarch64', 'linux'),
    'macos-arm64': ('Darwin', 'arm64', 'aarch64-apple-darwin', 'macosx_11_0_arm64', 'macos'),
    'windows-x86_64': ('Windows', 'amd64', 'x86_64-pc-windows-msvc', 'win_amd64', 'windows'),
}
MINIMUM_TESTS = {'checkout': 32, 'source': 913}
INVALID_EVIDENCE = (OSError, ValueError, KeyError, RuntimeError, TypeError,
                    AttributeError, zipfile.BadZipFile, ET.ParseError)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_options(repository):
    """Only Windows Git text newline conversion may differ from this checkout."""
    exact = source_files(repository)
    windows = {}
    for name, hashed in exact.items():
        content = (repository / name).read_bytes()
        alternatives = {hashed}
        try:
            content.decode('utf-8')
        except UnicodeDecodeError:
            pass
        else:
            if b'\0' not in content:
                lf = content.replace(b'\r\n', b'\n')
                alternatives.update(hashlib.sha256(data).hexdigest()
                                    for data in (lf, lf.replace(b'\n', b'\r\n')))
        windows[name] = alternatives
    return exact, windows


def basename(producer_path):
    # PureWindowsPath also recognizes forward-slash paths; never access a
    # producer's absolute path on the machine inspecting the downloaded files.
    name = PureWindowsPath(producer_path).name
    require(name not in ('', '.', '..'), 'invalid producer filename')
    return name


def check_offline(data, system, mechanism):
    require(data.get('status') == 'passed' and data.get('current_example') is None,
            'offline examples did not complete')
    require(set(data.get('examples', {})) == set(EXAMPLES), 'incomplete example inventory')
    isolation = data.get('isolation', {})
    require(isolation.get('mechanism') == mechanism, 'wrong isolation mechanism')
    codes = {'linux': {1, 13, 101}, 'macos': {1, 13}, 'windows': {10013}}
    require(isolation.get('connection_error') in codes[mechanism], 'network denial not demonstrated')
    if mechanism == 'linux':
        require(isolation.get('interfaces') == ['lo'], 'network namespace has other interfaces')
    path_type = PureWindowsPath if system == 'Windows' else PurePosixPath
    prefix = path_type(data['prefix'])
    require(prefix.is_absolute(), 'installed prefix must be absolute')
    for key in ('package', 'extension'):
        path = path_type(data[key])
        require(path.is_absolute() and '..' not in path.parts and path.is_relative_to(prefix),
                'module outside installed environment')
    libraries = data.get('libraries')
    require(isinstance(libraries, list) and bool(libraries), 'loaded-library inspection is missing')
    for library in libraries:
        name = path_type(library).name.lower()
        require(not any(word in name for word in ('julia', 'fftw', 'cubature', 'cuda')),
                f'forbidden loaded library: {name}')


def inspect_cell(directory, platform, python, revision, sources, oracle_hash):
    system, machine, target, tag, isolation = PLATFORMS[platform]
    build = read_json(directory / 'build.json')
    validation = read_json(directory / 'validation.json')
    require(build.get('format_version') == 1 and build.get('status') == 'built', 'build incomplete')
    require(build.get('revision') == revision, 'build revision mismatch')
    host = build['host']
    require((host['system'], host['machine'], host['target'], host['tag']) ==
            (system, machine, target, tag), 'wrong native host or target')
    require('.'.join(host['python'].split('.')[:2]) == python, 'wrong Python interpreter')
    exact, windows = sources
    recorded = build['sources']
    require(set(recorded) == set(exact), 'build source inventory mismatch')
    require(all(value in windows[name] if system == 'Windows' else value == exact[name]
                for name, value in recorded.items()), 'build sources differ from checkout')
    require(validation.get('status') == 'passed', 'installed validation incomplete')
    require(validation.get('oracle_manifest_sha256') == oracle_hash, 'reference manifest mismatch')
    sdist = directory / 'sdist' / basename(build['sdist'])
    require(digest(sdist) == build['sdist_sha256'], 'source archive digest mismatch')
    result = {'status': 'passed', 'python': host['python'], 'sdist_sha256': digest(sdist), 'wheels': {}}
    for kind, minimum in MINIMUM_TESTS.items():
        record = build['wheels'][kind]
        installed = validation['artifacts'][kind]
        wheel = directory / 'wheels' / kind / basename(record['path'])
        hashed = digest(wheel)
        require(hashed == record['sha256'] == installed['wheel_sha256'], 'wheel digest mismatch')
        cp = 'cp' + python.replace('.', '')
        require(wheel.name.endswith(f'-{cp}-{cp}-{tag}.whl'), 'wheel filename/tag mismatch')
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), 'duplicate wheel members')
            package = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names
                       if name.startswith('amalthea_native/') and not name.endswith(('/', '.so', '.pyd'))}
            expected = {name.removeprefix('python-native/python/'): hashed
                        for name, hashed in recorded.items() if name.startswith('python-native/python/')}
            require(package == record['package'] == expected, 'wheel package inventory mismatch')
            extensions = [name for name in names if name.endswith(('.so', '.pyd'))]
            require(len(extensions) == 1 and extensions[0].startswith('amalthea_native/_native.')
                    and extensions[0].endswith('.pyd' if system == 'Windows' else '.so'),
                    'native extension missing or ambiguous')
        offline = read_json(directory / f'{kind}-offline.json')
        check_offline(offline, system, isolation)
        require(offline['python'].split()[0] == host['python'], 'offline Python version mismatch')
        require(basename(offline['extension']) == basename(extensions[0]), 'loaded extension filename mismatch')
        require(installed['offline_examples'] == len(EXAMPLES), 'wrong installed example count')
        require(installed['python'] == offline['executable'], 'offline interpreter mismatch')
        tests = pytest_result(directory / f'{kind}-tests.xml')
        require(tests == installed['tests'] and tests['passed'] >= minimum, 'incomplete test inventory')
        for log in (directory / f'{kind}-offline.log.json', directory / 'logs' / f'{kind}-tests.log.json'):
            require(read_json(log).get('exit_code') == 0, 'test/example command failed')
        result['wheels'][kind] = {'filename': wheel.name, 'sha256': hashed,
                                  'examples': len(EXAMPLES), 'tests': tests}
    return result


def collect(artifacts, repository, run, platforms):
    require(bool(platforms) and len(platforms) == len(set(platforms))
            and all(name in PLATFORMS for name in platforms), 'choose distinct supported platforms')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    report = {'status': 'incomplete', 'captured': utcnow(), 'revision': revision,
              'platforms': platforms, 'python_versions': list(PYTHONS), 'expected_cells': 4 * len(platforms),
              'scope': 'downloaded standalone wheel evidence; no release or performance acceptance',
              'workflow': {key: run.get(key) for key in ('databaseId', 'url', 'status', 'conclusion', 'headSha')},
              'workflow_failures': [job['name'] for job in run.get('jobs', [])
                                    if job.get('conclusion') in ('failure', 'cancelled', 'timed_out')],
              'errors': [], 'cells': {}}
    oracles = artifacts / 'python-native-oracles'
    oracle_hash = None
    try:
        require(run.get('headSha') == revision, 'workflow revision mismatch')
        verify_oracles(oracles, repository, revision)
        oracle_hash = digest(oracles / 'manifest.json')
        report['oracle_manifest_sha256'] = oracle_hash
        producer = [job for job in run.get('jobs', [])
                    if job.get('name') == 'Independent Julia references for standalone wheels']
        require(len(producer) == 1 and producer[0].get('conclusion') == 'success', 'reference producer incomplete')
    except INVALID_EVIDENCE as error:
        report['errors'].append(str(error))
    sources = source_options(repository)
    for platform in platforms:
        for python in PYTHONS:
            name = f'{platform}-{python}'
            try:
                cell = inspect_cell(artifacts / f'python-native-{name}', platform, python,
                                    revision, sources, oracle_hash)
                jobs = [job for job in run.get('jobs', [])
                        if job.get('name') == f'Standalone Python ({platform} - {python})']
                require(len(jobs) == 1 and jobs[0].get('conclusion') == 'success', 'hosted wheel job incomplete')
                report['cells'][name] = cell
            except INVALID_EVIDENCE as error:
                report['cells'][name] = {'status': 'incomplete', 'error': str(error)}
    if not report['errors'] and all(cell['status'] == 'passed' for cell in report['cells'].values()):
        report['status'] = 'wheel_matrix_passed'
    return report


def markdown(report):
    lines = ['# Standalone wheel evidence', '', f"Status: **{report['status']}**",
             f"Revision: `{report['revision']}`", '', report['scope'], '',
             f"Requested platforms: {', '.join(report['platforms'])}; {report['expected_cells']} cells", '',
             f"Parent workflow: {report['workflow'].get('status')} / {report['workflow'].get('conclusion')}", '']
    for error in report['errors']:
        lines.append('- Reference/provenance error: ' + error.replace('\n', ' '))
    for name in report['workflow_failures']:
        lines.append('- Separate workflow failure: ' + name)
    lines += ['', '| Platform / Python | Result | Checkout / source tests | Offline examples per wheel |',
              '|---|---|---:|---:|']
    for name, cell in report['cells'].items():
        if cell['status'] == 'passed':
            counts = ' / '.join(str(cell['wheels'][kind]['tests']['passed']) for kind in MINIMUM_TESTS)
            lines.append(f'| {name} | passed | {counts} | {len(EXAMPLES)} |')
        else:
            error = cell['error'].replace('|', '\\|').replace('\n', ' ')
            lines.append(f'| {name} | incomplete: {error} | — | — |')
    for name, cell in report['cells'].items():
        if cell['status'] == 'passed':
            lines += ['', f'## {name}', '', f"Source archive SHA-256: `{cell['sdist_sha256']}`", '']
            lines += [f"- {kind}: `{wheel['filename']}` — `{wheel['sha256']}`"
                      for kind, wheel in cell['wheels'].items()]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--run-json', type=Path, required=True,
                        help='gh run view JSON including headSha, status, conclusion, jobs, url and databaseId')
    parser.add_argument('--platforms', nargs='+', choices=PLATFORMS, default=list(PLATFORMS))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != '.json':
        parser.error('--output must name a .json file')
    report = collect(args.artifacts.resolve(), args.repository.resolve(), read_json(args.run_json),
                     list(dict.fromkeys(args.platforms)))
    save_json(args.output, report)
    args.output.with_suffix('.md').write_text(markdown(report), encoding='utf-8')
    print(f"{report['status']}: {args.output}")
    return 0 if report['status'] == 'wheel_matrix_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
