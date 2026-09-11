"""Artifact-transport regressions; synthetic wheels are not platform evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wheel_collection', ROOT / 'test/standalone_wheels/collect.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def hashed(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def fixture(directory, platform='linux-x86_64', python='3.14', crlf=False):
    system, machine, target, tag, mechanism = module.PLATFORMS[platform]
    body = b'x = 1\n'
    packaged = body.replace(b'\n', b'\r\n') if crlf else body
    source_name = 'python-native/python/amalthea_native/__init__.py'
    sources = ({source_name: hashed(body)}, {source_name: {hashed(body), hashed(body.replace(b'\n', b'\r\n'))}})
    build = {'format_version': 1, 'status': 'built', 'revision': 'abc',
             'sources': {source_name: hashed(packaged)},
             'host': dict(system=system, machine=machine, target=target, tag=tag, python=python+'.9'),
             'sdist': '/original/sdist/amalthea.tar.gz', 'sdist_sha256': hashed(b'source'), 'wheels': {}}
    (directory / 'sdist').mkdir(parents=True)
    (directory / 'sdist/amalthea.tar.gz').write_bytes(b'source')
    validation = {'status': 'passed', 'oracle_manifest_sha256': 'oracle', 'artifacts': {}}
    windows = system == 'Windows'
    for kind, count in module.MINIMUM_TESTS.items():
        cp = 'cp'+python.replace('.', '')
        name = f'amalthea_native-0.0.1.dev0-{cp}-{cp}-{tag}.whl'
        wheel = directory / 'wheels' / kind / name
        wheel.parent.mkdir(parents=True)
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr('amalthea_native/__init__.py', packaged)
            archive.writestr('amalthea_native/_native.'+('pyd' if windows else 'so'), b'synthetic extension')
        wheel_hash = module.digest(wheel)
        producer = f'C:\\producer\\wheels\\{kind}\\{name}' if windows else f'/producer/wheels/{kind}/{name}'
        build['wheels'][kind] = {'path': producer, 'sha256': wheel_hash,
                                  'package': {'amalthea_native/__init__.py': hashed(packaged)}}
        prefix = 'C:\\env\\'+kind if windows else '/env/'+kind
        python_exe = prefix+('\\Scripts\\python.exe' if windows else '/bin/python')
        isolation = {'mechanism': mechanism, 'connection_error': {'linux': 101, 'macos': 1, 'windows': 10013}[mechanism]}
        if mechanism == 'linux':
            isolation['interfaces'] = ['lo']
        offline = {'status': 'passed', 'current_example': None, 'examples': dict.fromkeys(module.EXAMPLES, 'ok'),
                   'python': python+'.9 (synthetic transport fixture)',
                   'prefix': prefix, 'package': prefix+'/lib/amalthea_native/__init__.py',
                   'extension': prefix+'/lib/amalthea_native/_native.'+('pyd' if windows else 'so'),
                   'executable': python_exe, 'libraries': [prefix+'/lib/system'], 'isolation': isolation}
        save(directory / f'{kind}-offline.json', offline)
        save(directory / f'{kind}-offline.log.json', {'exit_code': 0})
        save(directory / 'logs' / f'{kind}-tests.log.json', {'exit_code': 0})
        (directory / f'{kind}-tests.xml').write_text('<testsuite tests="'+str(count)+'">'+
            ''.join(f'<testcase name="case{i}"/>' for i in range(count))+'</testsuite>')
        validation['artifacts'][kind] = {'wheel_sha256': wheel_hash, 'python': python_exe,
                                        'offline_examples': len(module.EXAMPLES),
                                        'tests': {'passed': count, 'failures': 0, 'skipped': 0}}
    save(directory / 'build.json', build)
    save(directory / 'validation.json', validation)
    return sources


class CollectionChecks(unittest.TestCase):
    def inspect(self, directory, sources, platform='linux-x86_64'):
        return module.inspect_cell(directory, platform, '3.14', 'abc', sources, 'oracle')

    def test_transported_native_platform_records(self):
        for platform in module.PLATFORMS:
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                sources = fixture(directory, platform, crlf=platform.startswith('windows'))
                result = self.inspect(directory, sources, platform)
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(result['wheels']['source']['tests']['passed'], 913)

    def test_windows_newlines_do_not_permit_other_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'text').write_bytes(b'a\nb\n')
            (root / 'binary').write_bytes(b'\0a\nb\n')
            original = {name: module.digest(root / name) for name in ('text', 'binary')}
            with patch.object(module, 'source_files', return_value=original):
                exact, windows = module.source_options(root)
            self.assertEqual(exact, original)
            self.assertIn(hashed(b'a\r\nb\r\n'), windows['text'])
            self.assertNotIn(hashed(b'changed\r\nb\r\n'), windows['text'])
            self.assertEqual(windows['binary'], {original['binary']})

    def test_empty_platform_selection_cannot_pass(self):
        with self.assertRaises(ValueError): module.collect(Path('.'), ROOT, {}, [])

    def test_changed_revision_source_target_or_oracle_rejected(self):
        for kind in ('revision', 'source', 'target', 'python', 'oracle', 'status'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary); sources = fixture(directory)
                path = directory / ('validation.json' if kind in ('oracle', 'status') else 'build.json')
                data = module.read_json(path)
                if kind == 'revision': data['revision'] = 'other'
                elif kind == 'source': data['sources'][next(iter(data['sources']))] = 'other'
                elif kind in ('target', 'python'): data['host'][kind] = 'other'
                elif kind == 'oracle': data['oracle_manifest_sha256'] = 'other'
                else: data['status'] = 'testing'
                save(path, data)
                with self.assertRaises(ValueError): self.inspect(directory, sources)

    def test_changed_or_missing_artifacts_rejected(self):
        for kind in ('wheel', 'sdist', 'xml', 'log'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary); sources = fixture(directory)
                if kind == 'wheel': next((directory / 'wheels/source').glob('*.whl')).write_bytes(b'changed')
                elif kind == 'sdist': (directory / 'sdist/amalthea.tar.gz').write_bytes(b'changed')
                elif kind == 'xml': (directory / 'source-tests.xml').unlink()
                else: save(directory / 'logs/source-tests.log.json', {'exit_code': 1})
                with self.assertRaises(module.INVALID_EVIDENCE): self.inspect(directory, sources)

    def test_failed_skipped_or_truncated_tests_rejected(self):
        for child in ('<failure/>', '<skipped/>', ''):
            with self.subTest(child=child), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary); sources = fixture(directory)
                (directory / 'source-tests.xml').write_text(f'<testsuite><testcase>{child}</testcase></testsuite>')
                with self.assertRaises((ValueError, RuntimeError)): self.inspect(directory, sources)

    def test_incomplete_or_wrong_offline_execution_rejected(self):
        for kind in ('example', 'network', 'package', 'library', 'interpreter'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary); sources = fixture(directory)
                path = directory / 'source-offline.json'; data = module.read_json(path)
                if kind == 'example': data['examples'].pop(module.EXAMPLES[0])
                elif kind == 'network': data['isolation']['connection_error'] = 0
                elif kind == 'package': data['package'] = '/checkout/amalthea_native/__init__.py'
                elif kind == 'library': data['libraries'].append('/lib/libjulia.so')
                else: data['executable'] = '/other/python'
                save(path, data)
                with self.assertRaises(ValueError): self.inspect(directory, sources)

    def test_report_continues_after_missing_and_malformed_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); artifacts = root / 'artifacts'; artifacts.mkdir()
            sources = fixture(artifacts / 'python-native-linux-x86_64-3.14')
            damaged = artifacts / 'python-native-linux-x86_64-3.13'
            fixture(damaged, python='3.13')
            (damaged / 'source-tests.xml').write_text('<broken')
            producer = {'name': 'Independent Julia references for standalone wheels', 'conclusion': 'success'}
            job = {'name': 'Standalone Python (linux-x86_64 - 3.14)', 'conclusion': 'success'}
            run = {'headSha': 'abc', 'status': 'completed', 'conclusion': 'failure',
                   'jobs': [producer, job, {'name': 'Apple diagnostic', 'conclusion': 'failure'}]}
            (artifacts / 'python-native-oracles').mkdir()
            (artifacts / 'python-native-oracles/manifest.json').write_text('reference')
            oracle_hash = module.digest(artifacts / 'python-native-oracles/manifest.json')
            for directory in (artifacts / 'python-native-linux-x86_64-3.14', damaged):
                path = directory / 'validation.json'; data = module.read_json(path)
                data['oracle_manifest_sha256'] = oracle_hash; save(path, data)
            with patch.object(module.subprocess, 'check_output', return_value='abc\n'), \
                 patch.object(module, 'verify_oracles'), patch.object(module, 'source_options', return_value=sources):
                report = module.collect(artifacts, root, run, ['linux-x86_64'])
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(len(report['cells']), 4)
            self.assertEqual(report['cells']['linux-x86_64-3.14']['status'], 'passed')
            self.assertEqual(report['cells']['linux-x86_64-3.13']['status'], 'incomplete')
            text = module.markdown(report)
            self.assertIn('Separate workflow failure: Apple diagnostic', text)
            self.assertIn('no release or performance acceptance', text)


if __name__ == '__main__':
    unittest.main()
