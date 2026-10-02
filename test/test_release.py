"""Release refusal/transport regressions; fixtures are not platform acceptance."""
import importlib.util
import io
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_checks', ROOT / 'test/release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
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


if __name__ == '__main__':
    unittest.main()
