"""Release refusal/transport regressions; fixtures are not platform acceptance."""
import contextlib
import importlib.util
import io
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_checks', ROOT / 'test/release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
from _validation import reference_sources

SHA = 'a' * 40


def run_record(jobs=None, **overrides):
    return {'databaseId': 1, 'headSha': SHA, 'event': 'push', 'status': 'completed', 'conclusion': 'success',
            'jobs': [{'name': name, 'conclusion': 'success'} for name in
                     (release.required_jobs() if jobs is None else jobs)], **overrides}


def package_bytes(version='0.1.0'):
    return f'Metadata-Version: 2.4\nName: amalthea-native\nVersion: {version}\n\n'.encode()


def fixture(artifacts):
    report = {'status': 'wheel_matrix_passed', 'revision': SHA, 'errors': [], 'cells': {}}
    for platform, host in release.wheels.PLATFORMS.items():
        for python in release.wheels.PYTHONS:
            name = f'{platform}-{python}'
            cell = artifacts / ('python-native-' + name)
            cp = 'cp' + python.replace('.', '')
            filename = f'amalthea_native-0.1.0-{cp}-{cp}-{host[3]}.whl'
            wheel = cell / 'wheels/source' / filename
            wheel.parent.mkdir(parents=True)
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('amalthea_native-0.1.0.dist-info/METADATA', package_bytes())
                archive.writestr('amalthea_native/_native.so', name.encode())
            report['cells'][name] = {'status': 'passed', 'wheels': {'source': {'filename': filename, 'sha256': release.wheels.digest(wheel)}}}
    cell = artifacts / 'python-native-linux-x86_64-3.11'
    sdist = cell / 'sdist/amalthea_native-0.1.0.tar.gz'
    sdist.parent.mkdir()
    with tarfile.open(sdist, 'w:gz') as archive:
        body = package_bytes()
        entry = tarfile.TarInfo('amalthea_native-0.1.0/PKG-INFO')
        entry.size = len(body)
        archive.addfile(entry, io.BytesIO(body))
    release.save(cell / 'build.json', {'sdist': str(sdist), 'sdist_sha256': release.wheels.digest(sdist)})
    return report


class GlibcEvidence:
    """Synthetic retained helper output; no wheels or numerical code are executed."""

    def __init__(self, root):
        self.root = root
        files = {
            'Project.toml': 'version = "1.1.0"\n',
            'python/pyproject.toml': '[project]\nversion = "1.1.0"\n',
            'python-native/pyproject.toml': '[project]\nversion = "0.1.0"\n',
            'python-native/python/amalthea_native/__init__.py': '__version__ = "0.1.0"\n',
            'CITATION.cff': 'version: "1.1.0"\n',
            'docs/dev/releases/v1.1.0.md': '# Amalthea.jl v1.1.0\n\n`amalthea-native` 0.1.0\n',
        }
        for name in ('amalthea/build.rs', 'amalthea/Cargo.toml', 'amalthea/Cargo.lock',
                     'python-native/Cargo.toml', 'python-native/Cargo.lock',
                     'python-native/.cargo/config.toml', 'python-native/LICENSE',
                     'python-native/README.md', 'src/reference.jl', 'amalthea/src/lib.rs',
                     'python-native/src/lib.rs', 'python-native/tests/test_numerical.py',
                     'python-native/tools/export_grid_oracle.jl',
                     'python-native/examples/solver_analytic.py'):
            files[name] = '# synthetic verifier fixture\n'
        for name, body in files.items():
            path = self.root / 'repository' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding='utf-8')
        self.candidate = release.metadata(self.root / 'repository')
        sources = release.wheels.source_files(self.root / 'repository')
        oracle_files = {}
        for family, (directory, _) in release.glibc228.FIXTURES.items():
            path = self.root / 'oracles' / directory / 'fixture.json'
            release.save(path, {'synthetic': family})
            oracle_files[directory + '/fixture.json'] = release.wheels.digest(path)
        release.save(self.root / 'oracles/manifest.json', {
            'format_version': 1, 'status': 'complete', 'revision': SHA,
            'sources': reference_sources(self.root / 'repository'),
            'families': list(release.glibc228.FIXTURES), 'files': oracle_files,
        })
        oracle_hash = release.wheels.digest(self.root / 'oracles/manifest.json')
        release_assets = {}
        for python in release.wheels.PYTHONS:
            build_dir = self.root / 'builds' / ('linux-x86_64-' + python)
            gate = self.root / 'gates' / python
            gate.mkdir(parents=True)
            host = dict(zip(('system', 'machine', 'target', 'tag'),
                            release.wheels.PLATFORMS['linux-x86_64'][:4]))
            host.update(python=python + '.9', executable='/expired/python/bin/python3')
            archive = build_dir / 'sdist/amalthea_native-0.1.0.tar.gz'
            archive.parent.mkdir(parents=True)
            with tarfile.open(archive, 'w:gz') as stream:
                entry = tarfile.TarInfo('amalthea_native-0.1.0/PKG-INFO')
                body = package_bytes(); entry.size = len(body)
                stream.addfile(entry, io.BytesIO(body))
            build = {'format_version': 1, 'status': 'built', 'revision': SHA,
                     'repository': '/expired/checkout', 'source': '/expired/extracted-source',
                     'sources': sources, 'host': host, 'wheels': {},
                     'sdist': '/expired/build/sdist/' + archive.name,
                     'sdist_sha256': release.wheels.digest(archive)}
            state = {'status': 'passed', 'scope': 'full installed suite',
                     'started': '2026-10-05T00:00:00Z', 'finished': '2026-10-05T00:10:00Z',
                     'build_manifest': '/expired/build/build.json',
                     'oracle_manifest_sha256': oracle_hash, 'artifacts': {},
                     'interpreter': '/expired/python', 'interpreter_sha256': '0' * 64,
                     'rootfs_provenance': {
                         'status': 'prepared', 'vendor': release.glibc228.VENDOR,
                         'revision': release.glibc228.REVISION, 'git_blob': release.glibc228.ARCHIVE_BLOB,
                         'rootfs': '/expired/rootfs', 'libc_package': '2.28-10+deb10u1',
                         'files': {'rootfs.tar.xz': {'sha256': release.glibc228.ARCHIVE_SHA256}},
                     }}
            probe = {'glibc': '2.28', 'python': python + '.9 (synthetic verifier fixture)',
                     'executable': '/opt/python/bin/python3', 'prefix': '/opt/python',
                     'libraries': ['/lib/x86_64-linux-gnu/libc-2.28.so'],
                     'interfaces': 'Inter-| Receive | Transmit\n face | bytes | bytes\n lo: 0 0\n'}
            release.save(gate / 'probe.json', probe)
            self.command(gate, 'probe', ['/opt/python/bin/python3', '-I', '-X', 'utf8', '-c',
                                       release.glibc228.PROBE, '/evidence/probe.json'])
            release.save(gate / 'probe.log', probe)
            for kind, count in release.wheels.MINIMUM_TESTS.items():
                cp = 'cp' + python.replace('.', '')
                filename = f'amalthea_native-0.1.0-{cp}-{cp}-{host["tag"]}.whl'
                wheel = build_dir / 'wheels' / kind / filename
                wheel.parent.mkdir(parents=True)
                extension = '_native.cpython-' + python.replace('.', '') + '-x86_64-linux-gnu.so'
                with zipfile.ZipFile(wheel, 'w') as stream:
                    stream.writestr('amalthea_native-0.1.0.dist-info/METADATA', package_bytes())
                    stream.writestr('amalthea_native/__init__.py',
                                    files['python-native/python/amalthea_native/__init__.py'])
                    stream.writestr('amalthea_native/' + extension,
                                    ('synthetic extension: ' + python + '/' + kind).encode())
                wheel_hash = release.wheels.digest(wheel)
                build['wheels'][kind] = {'path': '/expired/build/wheels/' + kind + '/' + filename,
                                        'sha256': wheel_hash, 'package': {
                                            'amalthea_native/__init__.py': sources[
                                                'python-native/python/amalthea_native/__init__.py']}}
                state['artifacts'][kind] = {'wheel_sha256': wheel_hash,
                    'offline_examples': len(release.wheels.EXAMPLES),
                    'tests': {'passed': count, 'failures': 0, 'skipped': 0}}
                prefix = '/work/' + kind
                interpreter = prefix + '/bin/python'
                release.save(gate / f'{kind}-offline.json', {
                    'status': 'passed', 'current_example': None, 'finished': state['finished'],
                    'examples': dict.fromkeys(release.wheels.EXAMPLES, ''),
                    'python': probe['python'], 'executable': interpreter, 'prefix': prefix,
                    'package': prefix + '/lib/amalthea_native/__init__.py',
                    'extension': prefix + '/lib/amalthea_native/' + extension,
                    'libraries': probe['libraries'],
                    'isolation': {'mechanism': 'linux', 'connection_error': 101, 'interfaces': ['lo']},
                })
                (gate / f'{kind}-tests.xml').write_text(
                    f'<testsuite tests="{count}" failures="0" errors="0" skipped="0">' +
                    ''.join(f'<testcase classname="test_numerical" name="case{i}"/>'
                            for i in range(count)) + '</testsuite>', encoding='utf-8')
                commands = {
                    'venv': ['/opt/python/bin/python3', '-m', 'venv', prefix],
                    'install': [interpreter, '-m', 'pip', 'install', '--no-index', '--only-binary=:all:',
                                '--find-links', '/wheelhouse', f'/wheels/{kind}/{filename}[hdf5]', 'pytest'],
                    'dependencies': [interpreter, '-m', 'pip', 'check'],
                    'versions': [interpreter, '-m', 'pip', 'list', '--format=json'],
                    'offline': [interpreter, '-I', '-X', 'utf8', '/tools/installed_smoke.py', '--examples',
                                '/examples', '--output', f'/evidence/{kind}-offline.json', '--isolation', 'linux'],
                    'tests': [interpreter, '-I', '-X', 'utf8', '-m', 'pytest', '/tests', '-q', '-s', '-o',
                              'cache_dir=/work/pytest-cache', f'--junitxml=/evidence/{kind}-tests.xml'],
                }
                if kind == 'checkout':
                    commands['tests'] += ['-k', 'output or test_nonlinear_adaptive_rejection_and_dense_output']
                for name, arguments in commands.items():
                    self.command(gate, kind + '-' + name, arguments)
                (gate / f'{kind}-offline.log').write_text(''.join(
                    'Passed example: ' + name + '\n' for name in release.wheels.EXAMPLES), encoding='utf-8')
                if kind == 'source':
                    (self.root / 'assets').mkdir(exist_ok=True)
                    shutil.copyfile(wheel, self.root / 'assets' / filename)
                    release_assets[filename] = wheel_hash
            release.save(build_dir / 'build.json', build)
            state['build_manifest_sha256'] = release.wheels.digest(build_dir / 'build.json')
            release.save(gate / 'validation.json', state)
        release.save(self.root / 'assets/release-manifest.json',
                     {'revision': SHA, **self.candidate, 'assets': release_assets})

    @staticmethod
    def command(gate, name, arguments):
        prefix = ['/usr/bin/bwrap', '--unshare-user', '--unshare-pid', '--unshare-net', '--unshare-uts',
                  '--unshare-ipc', '--die-with-parent', '--new-session', '--ro-bind', '/expired/rootfs', '/',
                  '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp', '--ro-bind', '/expired/python',
                  '/opt/python', '--bind', '/expired/work', '/work', '--bind', '/expired/gate', '/evidence',
                  '--chdir', '/work', '--clearenv']
        variables = {'PATH': '/nonexistent', 'HOME': '/tmp', 'LANG': 'C.UTF-8',
                     'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                     'RAYON_NUM_THREADS': '1', 'AMALTHEA_CUDA_BUILD': 'off',
                     'AMALTHEA_REQUIRE_CUDA_TESTS': '0', 'PIP_DISABLE_PIP_VERSION_CHECK': '1'}
        variables.update({f'AMALTHEA_{key}_ORACLE': '/oracles/' + directory
                          for key, (directory, _) in release.glibc228.FIXTURES.items()})
        for key, value in variables.items():
            prefix += ['--setenv', key, value]
        for mount in ('wheelhouse', 'tests', 'examples', 'tools', 'wheels', 'oracles'):
            prefix += ['--ro-bind', '/expired/' + mount, '/' + mount]
        prefix.append('--')
        (gate / (name + '.log')).write_text('', encoding='utf-8')
        release.save(gate / (name + '.log.json'), {
            'argv': prefix + arguments, 'cwd': '/expired/gate', 'exit_code': 0,
            'started': '2026-10-05T00:00:00Z', 'finished': '2026-10-05T00:10:00Z',
        })

    def edit(self, name, change):
        path = self.root / name
        data = release.read(path)
        change(data)
        release.save(path, data)

    def rebind_artifact_hashes(self, python='3.11'):
        """Update aggregate hashes after corruption without changing source/package claims."""
        manifest = self.root / 'builds' / ('linux-x86_64-' + python) / 'build.json'
        build = release.read(manifest)
        archive = manifest.parent / 'sdist' / release.wheels.basename(build['sdist'])
        build['sdist_sha256'] = release.wheels.digest(archive)
        for kind, record in build['wheels'].items():
            wheel = manifest.parent / 'wheels' / kind / release.wheels.basename(record['path'])
            record['sha256'] = release.wheels.digest(wheel)
            if kind == 'source':
                shutil.copyfile(wheel, self.root / 'assets' / wheel.name)
                self.edit('assets/release-manifest.json', lambda data:
                          data['assets'].update({wheel.name: record['sha256']}))
        release.save(manifest, build)
        state_path = self.root / 'gates' / python / 'validation.json'
        state = release.read(state_path)
        state['build_manifest_sha256'] = release.wheels.digest(manifest)
        for kind, record in build['wheels'].items():
            state['artifacts'][kind]['wheel_sha256'] = record['sha256']
        release.save(state_path, state)

    def verify(self):
        with patch.object(release.subprocess, 'check_output', return_value=SHA + '\n') as git:
            report = release.verify_glibc(self.root / 'gates', self.root / 'builds',
                                         self.root / 'repository', self.root / 'oracles', self.root / 'assets')
        git.assert_called_once_with(['git', 'rev-parse', 'HEAD'], cwd=self.root / 'repository', text=True)
        return report

    def cli(self, output):
        arguments = ['release.py', 'verify-glibc', '--output', str(output)]
        for name in ('gates', 'builds', 'repository', 'oracles', 'assets'):
            arguments += ['--' + name, str(self.root / name)]
        with patch.object(release.subprocess, 'check_output', return_value=SHA + '\n'), \
                patch.object(sys, 'argv', arguments), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return release.main()

    def snapshot(self):
        return {path.relative_to(self.root).as_posix(): path.read_bytes()
                for path in self.root.rglob('*') if path.is_file()}


class ReleaseChecks(unittest.TestCase):
    def test_metadata_matches_candidate_and_refuses_other_tags(self):
        result = release.metadata(ROOT, 'v1.1.0')
        self.assertEqual(result['python_version'], '0.1.0')
        for tag in ('v1.0.4', 'v1.1.0-rc1', '1.1.0', 'main', 'v01.1.0'):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release.metadata(ROOT, tag)

    def test_metadata_refuses_stale_wrapper_import_citation_and_notes(self):
        paths = ('Project.toml', 'python/pyproject.toml', 'python-native/pyproject.toml',
                 'python-native/python/amalthea_native/__init__.py', 'CITATION.cff', 'docs/dev/releases/v1.1.0.md')
        for name in paths:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                for path in paths:
                    target = root / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / path, target)
                target = root / name
                text = target.read_text().replace('1.1.0', '1.0.4').replace('0.1.0', '0.0.1.dev0')
                target.write_text(text)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    release.metadata(root)

    def test_required_ci_cannot_pass_by_skipping_or_omitting_jobs(self):
        release.check_run(run_record(), SHA, release.required_jobs())
        for name in release.required_jobs():
            for conclusion in ('skipped', 'failure', 'cancelled', ''):
                with self.subTest(name=name, conclusion=conclusion):
                    run = run_record()
                    next(job for job in run['jobs'] if job['name'] == name)['conclusion'] = conclusion
                    with self.assertRaises(ValueError): release.check_run(run, SHA, release.required_jobs())
            with self.assertRaises(ValueError):
                release.check_run(run_record(release.required_jobs() - {name}), SHA, release.required_jobs())

    def test_wrong_revision_event_and_workflow_outcome_refused(self):
        for changes in ({'headSha': 'b'*40}, {'event': 'pull_request'}, {'status': 'in_progress'},
                        {'conclusion': 'failure'}, {'conclusion': 'cancelled'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                release.check_run(run_record(**changes), SHA, release.required_jobs())

    def test_latest_ci_run_failure_is_not_hidden_by_old_success(self):
        latest = run_record(databaseId=2, conclusion='failure')
        with tempfile.TemporaryDirectory() as temp, patch.object(release, 'gh', side_effect=[
                [run_record(), latest], latest]), patch.object(release.time, 'sleep') as sleep:
            with self.assertRaises(ValueError): release.wait_ci('owner/repo', SHA, Path(temp), 100)
            self.assertEqual(release.read(Path(temp) / 'run_tests.yml.json')['databaseId'], 2)
            sleep.assert_not_called()

    def test_missing_ci_times_out_and_success_keeps_both_workflows(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(release, 'gh', return_value=[]), \
                patch.object(release.time, 'monotonic', side_effect=[0, 101]):
            with self.assertRaises(ValueError): release.wait_ci('owner/repo', SHA, Path(temp), 100)
        docs = run_record({'build', 'Deploy documentation'}, databaseId=2)
        with tempfile.TemporaryDirectory() as temp, patch.object(release, 'gh', side_effect=[
                [run_record()], run_record(), [docs], docs]):
            accepted = release.wait_ci('owner/repo', SHA, Path(temp), 100)
            self.assertEqual(accepted, {'run_tests.yml': 1, 'documenter.yml': 2})
            self.assertEqual(release.read(Path(temp) / 'acceptance.json')['status'], 'passed')

    def stage(self, artifacts, report, output):
        with patch.object(release.wheels, 'collect', return_value=report), \
                patch.object(release.wheels, 'markdown', return_value='synthetic test evidence\n'), \
                patch.object(release.subprocess, 'check_output', return_value=SHA+'\n'):
            release.stage_python(artifacts, ROOT, run_record(), output)

    def test_stage_uses_complete_source_wheels_and_checksum_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); artifacts = root / 'artifacts'; output = root / 'staged'
            report = fixture(artifacts)
            self.stage(artifacts, report, output)
            self.assertEqual(len(list(output.glob('*.whl'))), 16)
            for name in release.LIBRARIES: (output / name).write_bytes(name.encode())
            with patch.object(release.subprocess, 'check_output', return_value=SHA+'\n'):
                release.checksums(output, ROOT)
            lines = (output / 'SHA256SUMS.txt').read_text().splitlines()
            self.assertEqual(len(lines), 25)
            for line in lines:
                hashed, name = line.split('  ')
                self.assertEqual(release.wheels.digest(output / name), hashed)
            manifest = release.read(output / 'release-manifest.json')
            self.assertEqual(manifest['revision'], SHA)
            self.assertTrue(manifest['cpu_only'])

    def test_stage_refuses_corruption_or_missing_metadata_before_copying(self):
        for kind in ('digest', 'metadata', 'report', 'sdist'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); artifacts = root / 'artifacts'; output = root / 'staged'
                report = fixture(artifacts)
                record = next(iter(report['cells'].values()))['wheels']['source']
                wheel = next(artifacts.rglob(record['filename']))
                if kind == 'digest': wheel.write_bytes(b'corrupt')
                elif kind == 'metadata':
                    with zipfile.ZipFile(wheel, 'w') as archive:
                        archive.writestr('amalthea_native-0.1.0.dist-info/METADATA', package_bytes('0.0.1.dev0'))
                    record['sha256'] = release.wheels.digest(wheel)
                elif kind == 'report': report.update(status='incomplete', errors=['missing cell'])
                else: next(artifacts.rglob('*.tar.gz')).write_bytes(b'broken archive')
                with self.assertRaises((ValueError, zipfile.BadZipFile, tarfile.TarError)):
                    self.stage(artifacts, report, output)
                self.assertFalse(output.exists())

    def test_existing_stage_and_changed_staged_distributions_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); artifacts = root / 'artifacts'; output = root / 'staged'
            report = fixture(artifacts); self.stage(artifacts, report, output)
            before = (output / 'python-artifacts.json').read_bytes()
            with self.assertRaises(ValueError): self.stage(artifacts, report, output)
            self.assertEqual(before, (output / 'python-artifacts.json').read_bytes())
            for name in release.LIBRARIES: (output / name).write_bytes(b'library')
            next(output.glob('*.whl')).write_bytes(b'changed after staging')
            with patch.object(release.subprocess, 'check_output', return_value=SHA+'\n'), self.assertRaises(ValueError):
                release.checksums(output, ROOT)
            self.assertFalse((output / 'SHA256SUMS.txt').exists())

    def test_missing_or_unexpected_assets_cannot_get_release_checksums(self):
        for kind in ('missing', 'extra', 'symlink', 'provenance'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); artifacts = root / 'artifacts'; output = root / 'staged'
                report = fixture(artifacts); self.stage(artifacts, report, output)
                for name in release.LIBRARIES: (output / name).write_bytes(b'library')
                if kind == 'missing': (output / release.LIBRARIES[0]).unlink()
                elif kind == 'extra': (output / 'unexpected.whl').write_bytes(b'unknown')
                elif kind == 'symlink':
                    (output / release.LIBRARIES[0]).unlink()
                    (output / release.LIBRARIES[0]).symlink_to(output / release.LIBRARIES[1])
                else:
                    data = release.read(output / 'python-artifacts.json'); data['artifacts'] = {}
                    release.save(output / 'python-artifacts.json', data)
                with patch.object(release.subprocess, 'check_output', return_value=SHA+'\n'), self.assertRaises(ValueError):
                    release.checksums(output, ROOT)


class GlibcReleaseChecks(unittest.TestCase):
    def rejected(self, change, expected='', *, global_error=False):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = GlibcEvidence(Path(temporary) / 'evidence')
            change(fixture)
            before = fixture.snapshot()
            report = fixture.verify()
            self.assertEqual(report['status'], 'incomplete')
            if global_error:
                self.assertTrue(report['errors'])
                self.assertIn(expected, ' '.join(report['errors']))
            else:
                self.assertFalse(report['errors'])
                self.assertEqual(set(report['cells']), set(release.wheels.PYTHONS))
                self.assertEqual(report['cells']['3.11']['status'], 'incomplete')
                self.assertIn(expected, report['cells']['3.11']['error'])
                for python in release.wheels.PYTHONS[1:]:
                    self.assertEqual(report['cells'][python]['status'], 'passed', report['cells'][python])
            self.assertEqual(fixture.snapshot(), before)

    def test_relocated_full_evidence_preserves_inputs_and_hashes_inspected_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / 'producer'
            fixture = GlibcEvidence(original)
            before = fixture.snapshot()
            moved = Path(temporary) / 'downloaded'
            original.rename(moved); fixture.root = moved
            report = fixture.verify()
            self.assertEqual(report['status'], 'glibc_release_wheels_passed', report)
            self.assertEqual(report['revision'], SHA)
            self.assertEqual(set(report['cells']), set(release.wheels.PYTHONS))
            self.assertEqual(fixture.snapshot(), before)
            self.assertFalse(original.exists())
            for python, cell in report['cells'].items():
                gate = moved / 'gates' / python
                build = moved / 'builds' / ('linux-x86_64-' + python)
                for name in ('validation.json', 'probe.json', 'probe.log', 'source-tests.xml',
                             'source-tests.log', 'source-tests.log.json', 'source-offline.json',
                             'source-offline.log', 'checkout-venv.log'):
                    self.assertEqual(cell['sha256'][name], release.wheels.digest(gate / name))
                self.assertEqual(cell['sha256']['build.json'], release.wheels.digest(build / 'build.json'))
                for kind, minimum in release.wheels.MINIMUM_TESTS.items():
                    self.assertEqual(cell['wheels'][kind]['tests']['passed'], minimum)
                    self.assertEqual(cell['wheels'][kind]['examples'], 17)
                filename = cell['wheels']['source']['filename']
                self.assertEqual(cell['sha256']['release/' + filename],
                                 release.wheels.digest(moved / 'assets' / filename))

    def test_missing_failed_running_and_smoke_only_gates_are_rejected(self):
        for status in ('failed', 'running', 'smoke_passed'):
            with self.subTest(status=status):
                self.rejected(lambda f: f.edit('gates/3.11/validation.json',
                    lambda data: data.update(status=status)), 'full glibc validation incomplete')
        changes = (
            ('scope', lambda data: data.update(scope='installation/offline smoke')),
            ('unfinished', lambda data: data.pop('finished')),
            ('error', lambda data: data.update(error='interrupted after passing')),
            ('missing wheel', lambda data: data['artifacts'].pop('source')),
            ('build digest', lambda data: data.update(build_manifest_sha256='0' * 64)),
            ('reference digest', lambda data: data.update(oracle_manifest_sha256='0' * 64)),
            ('wheel digest', lambda data: data['artifacts']['source'].update(wheel_sha256='0' * 64)),
        )
        for name, change in changes:
            with self.subTest(name=name):
                self.rejected(lambda f: f.edit('gates/3.11/validation.json', change))
        self.rejected(lambda f: shutil.rmtree(f.root / 'gates/3.11'))

    def test_build_revision_source_version_platform_and_completion_are_required(self):
        changes = (
            ('revision', lambda data: data.update(revision='b' * 40)),
            ('source', lambda data: data['sources'].update({'amalthea/src/lib.rs': '0' * 64})),
            ('source inventory', lambda data: data['sources'].pop('amalthea/src/lib.rs')),
            ('Python', lambda data: data['host'].update(python='3.12.9')),
            ('prerelease Python', lambda data: data['host'].update(python='3.11.9rc1')),
            ('platform', lambda data: data['host'].update(machine='aarch64')),
            ('status', lambda data: data.update(status='building')),
            ('format', lambda data: data.update(format_version=2)),
        )
        for name, change in changes:
            with self.subTest(name=name):
                self.rejected(lambda f: f.edit('builds/linux-x86_64-3.11/build.json', change))

    def test_candidate_release_and_real_oracle_verification_are_required(self):
        for field in ('revision', 'julia_version', 'wrapper_version', 'python_version', 'tag', 'notes'):
            with self.subTest(field=field):
                self.rejected(lambda f: f.edit('assets/release-manifest.json',
                    lambda data: data.update({field: 'stale'})), 'release manifest', global_error=True)
        changes = (
            ('status', lambda data: data.update(status='running')),
            ('revision', lambda data: data.update(revision='b' * 40)),
            ('sources', lambda data: data['sources'].update({'src/reference.jl': '0' * 64})),
            ('family inventory', lambda data: data['families'].pop()),
            ('file inventory', lambda data: data['files'].pop('grid/fixture.json')),
        )
        for name, change in changes:
            with self.subTest(name=name):
                self.rejected(lambda f: f.edit('oracles/manifest.json', change), global_error=True)
        for path in ('oracles/grid/fixture.json', 'repository/src/reference.jl',
                     'repository/python-native/python/amalthea_native/__init__.py'):
            with self.subTest(path=path):
                self.rejected(lambda f: (f.root / path).write_text('changed\n'), global_error=True)

    def test_changed_or_missing_wheels_archives_and_release_assets_are_rejected(self):
        for location in ('builds/linux-x86_64-3.11/wheels/source',
                         'builds/linux-x86_64-3.11/sdist', 'assets'):
            for action in ('corrupt', 'missing'):
                with self.subTest(location=location, action=action):
                    def change(f):
                        directory = f.root / location
                        path = next(directory.glob('*.tar.gz' if location.endswith('sdist') else '*cp311*.whl'))
                        if action == 'corrupt': path.write_bytes(b'changed artifact')
                        else: path.unlink()
                    self.rejected(change)
        self.rejected(lambda f: f.edit('assets/release-manifest.json', lambda data:
            data['assets'].update({next(name for name in data['assets'] if '-cp311-' in name): '0' * 64})),
            'release source wheel differs')

    def test_wrong_wheel_metadata_is_rejected_even_when_all_hashes_match(self):
        def change(f):
            build_path = f.root / 'builds/linux-x86_64-3.11/build.json'
            build = release.read(build_path)
            record = build['wheels']['source']
            wheel = build_path.parent / 'wheels/source' / release.wheels.basename(record['path'])
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('amalthea_native-0.1.0.dist-info/METADATA', package_bytes('0.0.9'))
                archive.writestr('amalthea_native/_native.cpython-311-x86_64-linux-gnu.so', b'synthetic')
            record['sha256'] = release.wheels.digest(wheel)
            release.save(build_path, build)
            f.edit('gates/3.11/validation.json', lambda data: (
                data.update(build_manifest_sha256=release.wheels.digest(build_path)),
                data['artifacts']['source'].update(wheel_sha256=record['sha256'])))
            shutil.copyfile(wheel, f.root / 'assets' / wheel.name)
            f.edit('assets/release-manifest.json', lambda data: data['assets'].update({wheel.name: record['sha256']}))
        self.rejected(change, 'wheel distribution metadata differs')

    def test_wrong_source_archive_metadata_is_rejected_with_matching_digest(self):
        def change(f):
            build_path = f.root / 'builds/linux-x86_64-3.11/build.json'
            build = release.read(build_path)
            archive = build_path.parent / 'sdist' / release.wheels.basename(build['sdist'])
            with tarfile.open(archive, 'w:gz') as stream:
                entry = tarfile.TarInfo('amalthea_native-0.1.0/PKG-INFO')
                body = package_bytes('0.0.9'); entry.size = len(body)
                stream.addfile(entry, io.BytesIO(body))
            build['sdist_sha256'] = release.wheels.digest(archive)
            release.save(build_path, build)
            f.edit('gates/3.11/validation.json', lambda data:
                data.update(build_manifest_sha256=release.wheels.digest(build_path)))
        self.rejected(change, 'source distribution metadata differs')

    def test_wheel_package_corruption_missing_record_and_duplicates_are_rejected(self):
        for damage, expected in (('changed package', 'wheel package inventory mismatch'),
                                 ('missing package record', 'package'),
                                 ('duplicate member', 'duplicate wheel members')):
            with self.subTest(damage=damage):
                def change(f):
                    wheel = next((f.root / 'builds/linux-x86_64-3.11/wheels/source').glob('*.whl'))
                    package = 'amalthea_native/__init__.py'
                    if damage == 'changed package':
                        with zipfile.ZipFile(wheel) as archive:
                            entries = {name: archive.read(name) for name in archive.namelist()}
                        entries[package] += b'\n# changed after the candidate build\n'
                        with zipfile.ZipFile(wheel, 'w') as archive:
                            for name, body in entries.items():
                                archive.writestr(name, body)
                    elif damage == 'missing package record':
                        f.edit('builds/linux-x86_64-3.11/build.json', lambda data:
                               data['wheels']['source'].pop('package'))
                    else:
                        with zipfile.ZipFile(wheel, 'a') as archive:
                            body = archive.read(package)
                            with self.assertWarnsRegex(UserWarning, 'Duplicate name'):
                                archive.writestr(package, body)
                    f.rebind_artifact_hashes()
                self.rejected(change, expected)

    def test_malformed_compressed_archive_retains_cli_failure_report(self):
        for damage, exception in (('truncated gzip', EOFError),
                                  ('invalid deflate', (zlib.error, tarfile.ReadError))):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); fixture = GlibcEvidence(root / 'evidence')
                archive = next((fixture.root / 'builds/linux-x86_64-3.11/sdist').glob('*.tar.gz'))
                original = archive.read_bytes()
                # A truncated gzip header raises EOFError. A reserved deflate block
                # type raises zlib.error, wrapped in ReadError by some tarfile versions.
                damaged = original[:8] if damage == 'truncated gzip' else (
                    b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\x06' + b'\x00' * 8)
                archive.write_bytes(damaged)
                fixture.rebind_artifact_hashes()
                with self.assertRaises(exception):
                    release.sdist_metadata(archive, '0.1.0')
                before = fixture.snapshot()
                output = root / 'reports/malformed-archive.json'
                self.assertEqual(fixture.cli(output), 1)
                report = release.read(output)
                self.assertEqual(report['status'], 'incomplete')
                self.assertFalse(report['errors'])
                cell = report['cells']['3.11']
                self.assertEqual(cell['status'], 'incomplete')
                self.assertTrue(cell['error'])
                self.assertEqual(cell['sha256']['sdist/' + archive.name], release.wheels.digest(archive))
                for python in release.wheels.PYTHONS[1:]:
                    self.assertEqual(report['cells'][python]['status'], 'passed')
                saved = output.read_bytes()
                self.assertEqual(fixture.cli(output), 1)
                self.assertEqual(output.read_bytes(), saved)
                self.assertEqual(fixture.snapshot(), before)

    def test_pinned_rootfs_and_actual_loaded_glibc_probe_are_required(self):
        for field, value in (('status', 'preparing'), ('vendor', 'other'), ('revision', 'b' * 40),
                             ('git_blob', 'b' * 40), ('libc_package', '2.39-1')):
            with self.subTest(field=field):
                self.rejected(lambda f: f.edit('gates/3.11/validation.json', lambda data:
                    data['rootfs_provenance'].update({field: value})), 'rootfs provenance mismatch')
        self.rejected(lambda f: f.edit('gates/3.11/validation.json', lambda data:
            data['rootfs_provenance']['files']['rootfs.tar.xz'].update(sha256='0' * 64)),
            'rootfs provenance mismatch')
        for field, value in (('glibc', '2.39'), ('python', '3.11.8'), ('prefix', '/host/python'),
                             ('executable', '/usr/bin/python3'), ('libraries', ['/lib/libc-2.39.so']),
                             ('interfaces', 'lo: 0 0\neth0: 0 0\n')):
            with self.subTest(field=field):
                self.rejected(lambda f: f.edit('gates/3.11/probe.json', lambda data: data.update({field: value})))
        self.rejected(lambda f: f.edit('gates/3.11/probe.log', lambda data: data.update(glibc='2.39')),
                      'raw probe log differs')

    def test_raw_junit_failures_skips_collection_errors_and_truncation_are_rejected(self):
        replacements = {
            'failed': '<testsuite><testcase><failure/></testcase></testsuite>',
            'skipped': '<testsuite><testcase><skipped/></testcase></testsuite>',
            'error': '<testsuite><testcase><error/></testcase></testsuite>',
            'collection error': '<testsuite errors="1"><testcase/></testsuite>',
            'incomplete': '<testsuite tests="913"><testcase/></testsuite>',
            'empty': '<testsuite tests="913"/>', 'malformed': '<testsuite>',
        }
        for name, text in replacements.items():
            with self.subTest(name=name):
                self.rejected(lambda f: (f.root / 'gates/3.11/source-tests.xml').write_text(text))
        self.rejected(lambda f: f.edit('gates/3.11/validation.json', lambda data:
            data['artifacts']['source']['tests'].update(passed=914)), 'incomplete numerical tests')
        self.rejected(lambda f: (f.root / 'gates/3.11/checkout-tests.xml').write_text(
            '<testsuite><testcase/></testsuite>'), 'incomplete numerical tests')
        for kind in release.wheels.MINIMUM_TESTS:
            with self.subTest(kind=kind, matching_truncated_summary=True):
                def change(f):
                    (f.root / 'gates/3.11' / (kind + '-tests.xml')).write_text(
                        '<testsuite tests="1"><testcase/></testsuite>')
                    f.edit('gates/3.11/validation.json', lambda data:
                        data['artifacts'][kind]['tests'].update(passed=1))
                self.rejected(change, 'incomplete numerical tests')

    def test_seventeen_examples_and_installed_runtime_must_be_complete(self):
        changes = (
            ('missing example', lambda data: data['examples'].pop(release.wheels.EXAMPLES[-1])),
            ('extra example', lambda data: data['examples'].update(unexpected='')),
            ('failed', lambda data: data.update(status='failed')),
            ('partial', lambda data: data.update(current_example=release.wheels.EXAMPLES[-1])),
            ('unfinished', lambda data: data.pop('finished')),
            ('Python patch', lambda data: data.update(python='3.11.8')),
            ('interpreter', lambda data: data.update(executable='/usr/bin/python3')),
            ('package', lambda data: data.update(package='/checkout/amalthea_native/__init__.py')),
            ('extension', lambda data: data.update(extension='/work/source/lib/_native.wrong.so')),
            ('glibc', lambda data: data.update(libraries=['/lib/libc-2.39.so'])),
            ('forbidden library', lambda data: data['libraries'].append('/lib/libjulia.so')),
            ('network', lambda data: data['isolation'].update(connection_error=0)),
            ('interfaces', lambda data: data['isolation'].update(interfaces=['lo', 'eth0'])),
        )
        for name, change in changes:
            with self.subTest(name=name):
                self.rejected(lambda f: f.edit('gates/3.11/source-offline.json', change))
        self.rejected(lambda f: (f.root / 'gates/3.11/source-offline.log').write_text(
            ''.join('Passed example: ' + name + '\n' for name in release.wheels.EXAMPLES[:-1])),
            'incomplete raw example log')
        self.rejected(lambda f: f.edit('gates/3.11/validation.json', lambda data:
            data['artifacts']['source'].update(offline_examples=16)), 'incomplete offline examples')

    def test_command_records_cannot_hide_partial_failed_or_missing_execution(self):
        changes = (
            ('failed', lambda data: data.update(exit_code=1)),
            ('exception', lambda data: data.update(error='interrupted')),
            ('unfinished', lambda data: data.pop('finished')),
            ('selected cases', lambda data: data['argv'].extend(['-k', 'output'])),
            ('selected marker', lambda data: data['argv'].extend(['-m', 'smoke'])),
            ('different command', lambda data: data.update(argv=['/usr/bin/bwrap', '--unshare-net', '--clearenv', '--', 'true'])),
            ('network', lambda data: data['argv'].remove('--unshare-net')),
            ('environment', lambda data: data['argv'].remove('--clearenv')),
            ('malformed command', lambda data: data.update(argv=[])),
        )
        for name, change in changes:
            with self.subTest(name=name):
                self.rejected(lambda f: f.edit('gates/3.11/source-tests.log.json', change))
        for name in ('probe', 'source-venv', 'source-install', 'source-dependencies', 'source-versions',
                     'source-offline', 'source-tests', 'checkout-tests'):
            with self.subTest(name=name):
                self.rejected(lambda f: (f.root / 'gates/3.11' / (name + '.log')).unlink())
        self.rejected(lambda f: (f.root / 'gates/3.11/source-tests.log.json').unlink())

    def test_cli_keeps_failure_report_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture = GlibcEvidence(root / 'evidence')
            success = root / 'reports/success.json'
            self.assertEqual(fixture.cli(success), 0)
            self.assertEqual(release.read(success)['status'], 'glibc_release_wheels_passed')
            (fixture.root / 'gates/3.11/source-tests.xml').write_text('<broken')
            before = fixture.snapshot()
            failure = root / 'reports/failure.json'
            self.assertEqual(fixture.cli(failure), 1)
            report = release.read(failure)
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['cells']['3.11']['status'], 'incomplete')
            self.assertIn('source-tests.xml', report['cells']['3.11']['sha256'])
            saved = failure.read_bytes()
            self.assertEqual(fixture.cli(failure), 1)
            self.assertEqual(failure.read_bytes(), saved)
            link = root / 'reports/dangling.json'; target = root / 'never-created.json'
            link.symlink_to(target)
            self.assertEqual(fixture.cli(link), 1)
            self.assertTrue(link.is_symlink()); self.assertFalse(target.exists())
            self.assertEqual(fixture.snapshot(), before)


if __name__ == '__main__':
    unittest.main()
