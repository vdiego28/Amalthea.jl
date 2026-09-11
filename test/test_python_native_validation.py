"""Reference transport and gate regressions; synthetic fixtures are not physics evidence."""
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import tarfile
import unittest
from unittest.mock import patch
import zipfile

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY/'python-native/tools'))
import _validation as validation
import check_validation
import export_oracles
import offline_examples
import wheel_validation


class ReferenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = self.root/'checkout'
        for name in ('src/Grid.jl', 'src/data/table.txt', 'amalthea/src/lib.rs',
                     'python-native/tools/export_grid_oracle.jl', 'Project.toml',
                     'amalthea/Cargo.toml', 'amalthea/Cargo.lock'):
            path = self.repository/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic reference source\n', encoding='utf-8')
        self.artifact = self.root/'oracle'
        self.manifest = {'format_version': 1, 'status': 'complete', 'revision': 'test-revision',
                         'sources': validation.reference_sources(self.repository),
                         'families': list(validation.FIXTURES), 'files': {}}
        for directory, _ in validation.FIXTURES.values():
            path = self.artifact/directory/'fixture.txt'
            path.parent.mkdir(parents=True)
            path.write_text('synthetic fixture\n', encoding='utf-8')
            self.manifest['files'][path.relative_to(self.artifact).as_posix()] = validation.digest(path)
        self.write_manifest()

    def write_manifest(self):
        validation.save_json(self.artifact/'manifest.json', self.manifest)

    def verify(self):
        return validation.verify_oracles(self.artifact, self.repository, 'test-revision')

    def test_complete_artifact_and_windows_line_endings(self):
        expected = validation.fixture_environment(self.artifact)
        self.assertEqual(self.verify(), expected)
        for path in self.repository.rglob('*'):
            if path.is_file():
                path.write_bytes(path.read_bytes().replace(b'\n', b'\r\n'))
        self.assertEqual(self.verify(), expected)

    def test_rejects_incomplete_stale_or_wrong_version_manifest(self):
        for key, wrong in [('status', 'running'), ('format_version', 2),
                           ('revision', 'old-revision'), ('families', ['GRID'])]:
            with self.subTest(key=key):
                original = self.manifest[key]
                self.manifest[key] = wrong
                self.write_manifest()
                with self.assertRaises(ValueError):
                    self.verify()
                self.manifest[key] = original

    def test_changed_source_or_data_invalidates_provenance(self):
        for name in ('src/Grid.jl', 'src/data/table.txt', 'amalthea/src/lib.rs',
                     'python-native/tools/export_grid_oracle.jl', 'Project.toml'):
            with self.subTest(name=name):
                path = self.repository/name
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with self.assertRaisesRegex(ValueError, 'source hashes'):
                    self.verify()
                path.write_bytes(original)

    def test_missing_changed_extra_or_empty_fixture_fails(self):
        path = self.artifact/'grid/fixture.txt'
        original = path.read_bytes()
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'files are'):
            self.verify()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'files are'):
            self.verify()
        path.write_bytes(original)
        extra = self.artifact/'grid/unexpected.txt'
        extra.write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'files are'):
            self.verify()
        extra.unlink()
        del self.manifest['files']['grid/fixture.txt']
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'family has no files'):
            self.verify()

    def test_manifest_required_and_environment_not_written_on_failure(self):
        github_env = self.root/'github-env'
        github_env.write_text('EXISTING=value\n')
        (self.artifact/'manifest.json').unlink()
        with patch.object(check_validation.subprocess, 'check_output', return_value='test-revision\n'):
            with self.assertRaises(FileNotFoundError):
                check_validation.checked_environment(self.artifact, self.repository, github_env)
        self.assertEqual(github_env.read_text(), 'EXISTING=value\n')

    def test_environment_export_preserves_existing_entries(self):
        github_env = self.root/'github-env'
        github_env.write_text('EXISTING=value\n')
        with patch.object(check_validation.subprocess, 'check_output', return_value='test-revision\n'):
            result = check_validation.checked_environment(self.artifact, self.repository, github_env)
        self.assertEqual(github_env.read_text().splitlines(), ['EXISTING=value'] +
                         [f'{key}={value}' for key, value in result.items()])

    def test_newline_path_rejected_before_environment_append(self):
        github_env = self.root/'github-env'
        with patch.object(check_validation.subprocess, 'check_output', return_value='test-revision'), \
             patch.object(check_validation, 'verify_oracles', return_value={'KEY': '/tmp/path\nINJECT=value'}):
            with self.assertRaisesRegex(ValueError, 'newlines'):
                check_validation.checked_environment(self.artifact, self.repository, github_env)
        self.assertFalse(github_env.exists())

    def test_inventory_matches_actual_consumers_exporters_and_examples(self):
        keys = set()
        for path in (REPOSITORY/'python-native/tests').glob('test_*.py'):
            keys.update(re.findall(r'AMALTHEA_([A-Z_]+)_ORACLE', path.read_text()))
        self.assertEqual(keys, set(validation.FIXTURES))
        self.assertEqual(len({value[0] for value in validation.FIXTURES.values()}), len(keys))
        for _, exporters in validation.FIXTURES.values():
            for name in exporters:
                self.assertTrue((REPOSITORY/'python-native/tools'/f'export_{name}_oracle.jl').is_file())
        self.assertEqual(set(validation.EXAMPLES),
                         {path.stem for path in (REPOSITORY/'python-native/examples').glob('*.py')})

    def fake_export(self, command, *, cwd, environment, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        self.assertEqual(environment['AMALTHEA_USE_RUST_NATIVE'], '0')
        self.assertEqual(environment['AMALTHEA_USE_RUST_QDHT'], '0')
        self.assertNotIn('AMALTHEA_MODAL_CASE', environment)
        if '--version' in command:
            log.write_text('julia version synthetic\n')
            return
        directory = Path(command[-1])
        directory.mkdir(exist_ok=True)
        (directory/'fixture.txt').write_text('synthetic exporter output')
        if directory.name == 'materials':
            (directory/'metadata.toml').write_text('coolprop = "7.2.0"\n')
        log.write_text('synthetic exporter log\n')

    def test_exporter_completes_all_families_and_clears_developer_selectors(self):
        output = self.root/'new-export'
        with patch.object(export_oracles.subprocess, 'check_output', return_value='test-revision'), \
             patch.object(export_oracles, 'run', side_effect=self.fake_export), \
             patch.dict(os.environ, {'AMALTHEA_MODAL_CASE': 'one', 'AMALTHEA_USE_RUST_NATIVE': '1'}):
            export_oracles.export(self.repository, output, 'julia')
        self.assertEqual(validation.verify_oracles(output, self.repository, 'test-revision'),
                         validation.fixture_environment(output))
        self.assertEqual(json.loads((output/'export-state.json').read_text())['status'], 'complete')

    def test_failed_export_has_no_completion_manifest_and_retains_state(self):
        output = self.root/'failed-export'
        def fail(command, **kwargs):
            if str(command[-1]).endswith('/gnlse'):
                raise RuntimeError('export failed')
            self.fake_export(command, **kwargs)
        with patch.object(export_oracles.subprocess, 'check_output', return_value='test-revision'), \
             patch.object(export_oracles, 'run', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'export failed'):
                export_oracles.export(self.repository, output, 'julia')
        self.assertFalse((output/'manifest.json').exists())
        state = json.loads((output/'export-state.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['families'], ['GRID', 'SOLVER'])
        self.assertTrue((output/'logs/grid.log').exists())

    def test_producer_refuses_existing_artifact(self):
        before = (self.artifact/'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):
            export_oracles.export(self.repository, self.artifact, 'julia')
        self.assertEqual((self.artifact/'manifest.json').read_bytes(), before)

    def test_export_cannot_publish_wrong_thermodynamics_or_changing_source(self):
        for defect in ('coolprop', 'source', 'empty'):
            with self.subTest(defect=defect):
                output = self.root/defect
                source = self.repository/'src/Grid.jl'
                original = source.read_bytes()
                def corrupt(command, **kwargs):
                    self.fake_export(command, **kwargs)
                    directory = Path(command[-1])
                    if directory.name == 'callback':
                        if defect == 'coolprop':
                            (output/'materials/metadata.toml').write_text('coolprop = "wrong"\n')
                        elif defect == 'source':
                            source.write_bytes(b'changed while exporting')
                        else:
                            (directory/'fixture.txt').unlink()
                with patch.object(export_oracles.subprocess, 'check_output', return_value='test-revision'), \
                     patch.object(export_oracles, 'run', side_effect=corrupt):
                    with self.assertRaises(RuntimeError):
                        export_oracles.export(self.repository, output, 'julia')
                self.assertFalse((output/'manifest.json').exists())
                self.assertEqual(json.loads((output/'export-state.json').read_text())['status'], 'failed')
                source.write_bytes(original)

    def test_junit_acceptance_fails_closed(self):
        report = self.root/'tests.xml'
        for body in ('<testsuites/>', '<testsuite tests="0"/>',
                     '<testsuite><testcase><failure/></testcase></testsuite>',
                     '<testsuite><testcase><error/></testcase></testsuite>',
                     '<testsuite><testcase><skipped/></testcase></testsuite>',
                     '<testsuite errors="1"><testcase/></testsuite>',
                     '<testsuite failures="1"><testcase/></testsuite>',
                     '<testsuite skipped="1"><testcase/></testsuite>'):
            with self.subTest(body=body):
                report.write_text(body)
                with self.assertRaises(RuntimeError):
                    validation.pytest_result(report)
        report.write_text('<testsuites><testsuite tests="2"><testcase/><testcase/></testsuite></testsuites>')
        self.assertEqual(validation.pytest_result(report), {'passed': 2, 'failures': 0, 'skipped': 0})

    def test_real_process_wrapper_records_failure_without_shell_interpretation(self):
        log = self.root/'command.log'
        literal = 'spaces; $(not-a-command) `literal`'
        command = [sys.executable, '-c', 'import sys; print(sys.argv[1]); sys.exit(7)', literal]
        with self.assertRaisesRegex(RuntimeError, 'command failed'):
            validation.run(command, cwd=self.root, environment=validation.cpu_environment(), log=log)
        self.assertEqual(log.read_text().strip(), literal)
        metadata = json.loads(log.with_suffix('.log.json').read_text())
        self.assertEqual(metadata['argv'], command)
        self.assertEqual(metadata['exit_code'], 7)

    def test_real_process_wrapper_records_missing_executable(self):
        log = self.root/'missing.log'
        with self.assertRaises(OSError):
            validation.run([self.root/'does-not-exist'], cwd=self.root,
                           environment=validation.cpu_environment(), log=log)
        self.assertIn('error', json.loads(log.with_suffix('.log.json').read_text()))


class WheelValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_native_platform_mapping_and_rejection(self):
        for system, machine, target in [('Linux', 'x86_64', 'x86_64-unknown-linux-gnu'),
                ('Linux', 'aarch64', 'aarch64-unknown-linux-gnu'),
                ('Darwin', 'arm64', 'aarch64-apple-darwin'),
                ('Windows', 'AMD64', 'x86_64-pc-windows-msvc')]:
            with self.subTest(system=system, machine=machine), \
                 patch.object(wheel_validation.platform, 'system', return_value=system), \
                 patch.object(wheel_validation.platform, 'machine', return_value=machine):
                self.assertEqual(wheel_validation.host_spec()['target'], target)
        with patch.object(wheel_validation.platform, 'machine', return_value='unsupported'):
            with self.assertRaisesRegex(RuntimeError, 'unsupported native'):
                wheel_validation.host_spec()
        with patch.object(wheel_validation.sysconfig, 'get_config_var', return_value=1):
            with self.assertRaisesRegex(RuntimeError, 'free-threaded'):
                wheel_validation.host_spec()

    def test_archive_rejects_traversal_links_and_compiled_products(self):
        for index, name in enumerate(('../outside', '/absolute', 'root\\outside',
                'root/target/file', 'root/package/module.so', 'root/link', 'root/C:/file',
                'root/package/old.pyo')):
            with self.subTest(name=name):
                archive = self.root/f'{index}.tar.gz'
                with tarfile.open(archive, 'w:gz') as stream:
                    member = tarfile.TarInfo(name)
                    if name.endswith('/link'):
                        member.type = tarfile.SYMTYPE
                        member.linkname = '/outside'
                    stream.addfile(member)
                with self.assertRaises(ValueError):
                    wheel_validation.extract_source(archive, self.root/f'extracted-{index}', REPOSITORY)
        self.assertFalse((self.root/'outside').exists())

    def test_source_inventory_exclusions_are_relative_to_selected_root(self):
        root = self.root/'target/project'
        root.mkdir(parents=True)
        (root/'source.py').write_bytes(b'source')
        (root/'old.pyo').write_bytes(b'old bytecode')
        (root/'target').mkdir()
        (root/'target/build.rs').write_bytes(b'generated')
        self.assertEqual(wheel_validation.files_under(root), [root/'source.py'])

    def test_source_archive_cannot_omit_required_engine(self):
        archive = self.root/'empty.tar.gz'
        with tarfile.open(archive, 'w:gz') as stream:
            stream.addfile(tarfile.TarInfo('root/placeholder'))
        with self.assertRaisesRegex(ValueError, 'incomplete source'):
            wheel_validation.extract_source(archive, self.root/'extracted', REPOSITORY)

    def test_wheel_must_have_correct_tag_package_and_one_extension(self):
        source = self.root/'source'
        package = source/'python/amalthea_native'
        package.mkdir(parents=True)
        (package/'__init__.py').write_bytes(b'# synthetic package\n')
        spec = wheel_validation.host_spec()
        py = f'cp{sys.version_info.major}{sys.version_info.minor}'
        wheel = self.root/f'amalthea_native-0.0.1-{py}-{py}-{spec["tag"]}.whl'
        def write(content=b'# synthetic package\n', extension=True, extra=False):
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('amalthea_native/__init__.py', content)
                if extension:
                    archive.writestr('amalthea_native/_native.fake.so', b'synthetic extension')
                if extra:
                    archive.writestr('amalthea_native/unexpected.py', b'extra')
        write()
        self.assertEqual(len(wheel_validation.verify_wheel(wheel, source, spec)), 1)
        for kwargs in ({'content': b'changed'}, {'extension': False}, {'extra': True}):
            write(**kwargs)
            with self.assertRaises(ValueError):
                wheel_validation.verify_wheel(wheel, source, spec)
        with self.assertRaisesRegex(ValueError, 'interpreter/platform'):
            wheel_validation.verify_wheel(wheel, source, spec | {'tag': 'wrong'})

    def test_test_stage_refuses_unfinished_build_before_environment_creation(self):
        manifest = self.root/'build.json'
        validation.save_json(manifest, {'status': 'building', 'format_version': 1})
        with self.assertRaisesRegex(ValueError, 'completed build'):
            wheel_validation.test(manifest, REPOSITORY, self.root/'oracles')
        self.assertFalse((self.root/'environments').exists())

    def test_windows_firewall_isolation_refuses_non_hosted_execution(self):
        for ephemeral, environment in [(False, {}), (True, {}),
                (True, {'GITHUB_ACTIONS': 'true', 'RUNNER_ENVIRONMENT': 'self-hosted'})]:
            with self.subTest(ephemeral=ephemeral, environment=environment), \
                 patch.object(offline_examples.sys, 'platform', 'win32'), \
                 patch.object(offline_examples, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'restricted'):
                    offline_examples.offline(Path(sys.executable), self.root, self.root/'result.json',
                        environment=environment, cwd=self.root, ephemeral_ci=ephemeral)
                run.assert_not_called()

    def test_missing_linux_isolation_cannot_pass(self):
        with patch.object(offline_examples.sys, 'platform', 'linux'), \
             patch.object(offline_examples.shutil, 'which', return_value=None), \
             patch.object(offline_examples, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'unshare is required'):
                offline_examples.offline(Path(sys.executable), self.root, self.root/'result.json',
                    environment={}, cwd=self.root)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
