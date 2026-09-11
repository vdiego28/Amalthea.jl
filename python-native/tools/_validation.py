"""Shared standalone-distribution validation contracts (development only)."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


# Environment key -> (artifact directory, ordered Julia exporters).
FIXTURES = {
    'GRID': ('grid', ('grid',)),
    'SOLVER': ('solver', ('solver',)),
    'GNLSE': ('gnlse', ('gnlse',)),
    'PULSE': ('pulse', ('pulse',)),
    'MATERIAL': ('materials', ('material',)),
    'MODE': ('modes', ('mode',)),
    'CAPILLARY': ('capillary', ('capillary',)),
    'REAL_PULSE': ('real-pulse', ('real_pulse',)),
    'REAL_CAPILLARY': ('real-capillary', ('real_capillary',)),
    'ADK': ('adk', ('adk',)),
    'PPT': ('ppt', ('ppt',)),
    'PLASMA': ('plasma', ('plasma_capillary',)),
    'MOLECULAR': ('molecular', ('molecular_raman',)),
    'RAMAN_CAPILLARY': ('raman-capillary', ('raman_capillary',)),
    'VARIABLE_SOLVER': ('variable-solver', ('variable_solver',)),
    'PROFILE_CAPILLARY': ('profile-capillary', ('profile_capillary', 'profile_limits')),
    'MIXTURE_CAPILLARY': ('mixture-capillary', ('mixture_capillary',)),
    'SPATIAL': ('spatial', ('spatial',)),
    'MODAL_CAPILLARY': ('modal-capillary', ('modal_capillary',)),
    'CALLBACK': ('callback', ('callback',)),
}

EXAMPLES = (
    'solver_analytic', 'gnlse', 'pulses', 'capillary', 'real_pulse',
    'carrier_capillary', 'ppt', 'plasma_capillary', 'molecular_raman',
    'raman_capillary', 'variable_solver', 'profile_capillary', 'mixture_capillary',
    'custom_mode', 'modal_capillary', 'custom_response', 'output_processing',
)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path, *, normalize_lf=False):
    content = Path(path).read_bytes()
    if normalize_lf:
        content = content.replace(b'\r\n', b'\n')
    return hashlib.sha256(content).hexdigest()


def reference_sources(repository):
    repository = Path(repository)
    files = [path for directory in ('src', 'amalthea/src')
             for path in (repository/directory).rglob('*') if path.is_file()]
    files.extend((repository/'python-native/tools').glob('export_*_oracle.jl'))
    files.extend(repository/name for name in ('Project.toml', 'amalthea/Cargo.toml', 'amalthea/Cargo.lock'))
    return {path.relative_to(repository).as_posix(): digest(path, normalize_lf=True)
            for path in sorted(files)}


def fixture_environment(directory):
    directory = Path(directory).resolve()
    return {f'AMALTHEA_{key}_ORACLE': str(directory/value[0]) for key, value in FIXTURES.items()}


def verify_oracles(directory, repository, revision):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete' or manifest.get('format_version') != 1:
        raise ValueError('oracle export has no completed version-1 manifest')
    if manifest.get('revision') != revision:
        raise ValueError('oracle commit does not match this checkout')
    if manifest.get('sources') != reference_sources(repository):
        raise ValueError('oracle source hashes do not match this checkout')
    if set(manifest.get('families', ())) != set(FIXTURES):
        raise ValueError('oracle manifest does not contain all fixture families')
    files = manifest.get('files', {})
    for key, (subdirectory, _) in FIXTURES.items():
        if not any(name.startswith(subdirectory + '/') for name in files):
            raise ValueError(f'oracle family has no files: {key}')
    actual = {path.relative_to(directory).as_posix(): digest(path)
              for subdirectory, _ in FIXTURES.values()
              for path in (directory/subdirectory).rglob('*') if path.is_file()}
    if files != actual:
        raise ValueError('oracle files are missing, changed or unexpected')
    return fixture_environment(directory)


def pytest_result(path):
    """Require actual passing cases; missing or skipped references cannot pass."""
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    failures = sum(any(child.tag in ('failure', 'error') for child in case) for case in cases)
    skipped = sum(any(child.tag == 'skipped' for child in case) for case in cases)
    # A collection error can also be represented directly under a suite.
    suite_errors = sum(int(suite.get('errors', '0')) for suite in root.iter('testsuite'))
    suite_incomplete = any(int(suite.get(key, '0')) for suite in root.iter('testsuite')
                           for key in ('failures', 'errors', 'skipped'))
    if not cases or failures or skipped or suite_errors or suite_incomplete:
        raise RuntimeError(f'incomplete pytest gate: {len(cases)} cases, {failures} failures, '
                           f'{skipped} skips, {suite_errors} suite errors')
    return {'passed': len(cases), 'failures': failures, 'skipped': skipped}


def run(command, *, cwd, environment, log):
    command = list(map(str, command))
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    record = {'argv': command, 'cwd': str(cwd), 'started': utcnow()}
    try:
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=cwd, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT)
        record['exit_code'] = result.returncode
    except OSError as error:
        record['error'] = str(error)
        raise
    finally:
        record['finished'] = utcnow()
        save_json(log.with_suffix(log.suffix + '.json'), record)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}); see {log}')
    print(f'Passed: {log.name}', flush=True)


def cpu_environment():
    environment = os.environ.copy()
    environment.update(AMALTHEA_CUDA_BUILD='off', AMALTHEA_REQUIRE_CUDA_TESTS='0',
                       AMALTHEA_RUST_SKIP_DOWNLOAD='1', RUSTFLAGS='',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', RAYON_NUM_THREADS='1',
                       PIP_DISABLE_PIP_VERSION_CHECK='1')
    for key in ('PYTHONPATH', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_TARGET'):
        environment.pop(key, None)
    return environment
