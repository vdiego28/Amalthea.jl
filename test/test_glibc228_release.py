"""Durable runtime transport checks; synthetic fixtures are not numerical evidence."""
import hashlib
import importlib.util
import json
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('durable_glibc', ROOT/'test/standalone_wheels/glibc228_release.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
from _validation import FIXTURES, reference_sources, save_json, digest


class DurablePreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def oracles(self):
        repository, oracles = self.root/'repository', self.root/'oracles'
        for name in ('src/Grid.jl', 'amalthea/src/lib.rs', 'python-native/tools/export_grid_oracle.jl',
                     'Project.toml', 'amalthea/Cargo.toml', 'amalthea/Cargo.lock'):
            path = repository/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic source\n')
        files = {}
        for directory, _ in FIXTURES.values():
            path = oracles/directory/'fixture.csv'
            path.parent.mkdir(parents=True)
            path.write_text('1,2,3\n')
            files[path.relative_to(oracles).as_posix()] = digest(path)
        save_json(oracles/'manifest.json', {'status': 'complete', 'format_version': 1,
                  'revision': gate.REVISION, 'sources': reference_sources(repository),
                  'families': list(FIXTURES), 'files': files})
        return repository, oracles

    def test_transport_reconstructs_every_byte_and_preserves_original(self):
        repository, oracles = self.oracles()
        original = {p.relative_to(oracles): p.read_bytes() for p in oracles.rglob('*') if p.is_file()}
        output = self.root/'transport'
        with patch.object(gate, 'candidate'), patch.object(gate, 'ORACLE_HASH', digest(oracles/'manifest.json')):
            gate.transport(repository, oracles, output)
        index = json.loads((output/'index.json').read_text())
        archive = b''.join((output/c['filename']).read_bytes() for c in index['chunks'])
        self.assertEqual(hashlib.sha256(archive).hexdigest(), index['sha256'])
        self.assertEqual(len(archive), index['bytes'])
        self.assertEqual(original, {p.relative_to(oracles): p.read_bytes() for p in oracles.rglob('*') if p.is_file()})
        for record in index['chunks']:
            self.assertLessEqual(record['bytes'], 24*1024*1024)
            self.assertEqual(digest(output/record['filename']), record['sha256'])
        self.assertFalse((output/index['archive']).exists())

    def test_transport_rejects_wrong_manifest_and_mutated_oracle_before_writing(self):
        repository, oracles = self.oracles()
        output = self.root/'transport'
        with patch.object(gate, 'candidate'):
            with self.assertRaisesRegex(ValueError, 'manifest hash'):
                gate.transport(repository, oracles, output)
            with patch.object(gate, 'ORACLE_HASH', digest(oracles/'manifest.json')):
                (oracles/'grid/fixture.csv').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'files are'):
                    gate.transport(repository, oracles, output)
        self.assertFalse(output.exists())

    def test_transport_rejects_external_symlink(self):
        repository, oracles = self.oracles()
        (oracles/'outside').symlink_to(repository/'Project.toml')
        with patch.object(gate, 'candidate'), patch.object(gate, 'ORACLE_HASH', digest(oracles/'manifest.json')):
            with self.assertRaisesRegex(ValueError, 'unsafe reference'):
                gate.transport(repository, oracles, self.root/'transport')

    def test_chunk_limit_is_checked_before_partial_output(self):
        archive = self.root/'archive'
        archive.write_bytes(b'012345678')
        output = self.root/'chunks'; output.mkdir()
        with self.assertRaisesRegex(ValueError, 'eight-chunk'):
            gate.split_archive(archive, output, chunk_bytes=1)
        self.assertEqual(list(output.iterdir()), [])
        result = gate.split_archive(archive, output, chunk_bytes=4)
        self.assertEqual([x['bytes'] for x in result['chunks']], [4, 4, 1])
        with self.assertRaises(FileExistsError):
            gate.split_archive(archive, output, chunk_bytes=4)

    def test_source_ci_rejects_missing_required_job_and_wrong_artifact_revision(self):
        raw = {'id': gate.RUN_ID, 'path': '.github/workflows/run_tests.yml', 'head_sha': gate.REVISION,
               'event': 'push', 'status': 'completed', 'conclusion': 'success', 'html_url': 'test'}
        jobs = [{'name': name, 'conclusion': 'success'} for name in gate.release.required_jobs()]
        artifacts = [{'name': name, 'expired': False, 'workflow_run': {'head_sha': gate.REVISION}}
                     for name in ['python-native-oracles']+[f'python-native-linux-x86_64-{p}' for p in gate.INTERPRETERS]]
        def responses():
            return [raw, {'jobs': jobs, 'total_count': len(jobs)}, {'artifacts': artifacts, 'total_count': len(artifacts)}]
        with patch.object(gate, 'candidate'), patch.object(gate, 'api', side_effect=responses()):
            gate.verify_ci(self.root, self.root/'proof')
        jobs[0]['conclusion'] = 'skipped'
        with patch.object(gate, 'candidate'), patch.object(gate, 'api', side_effect=responses()):
            with self.assertRaisesRegex(ValueError, 'required job'):
                gate.verify_ci(self.root, self.root/'bad')
        self.assertFalse((self.root/'bad').exists())
        jobs[0]['conclusion'] = 'success'
        artifacts[0]['workflow_run']['head_sha'] = 'wrong'
        with patch.object(gate, 'candidate'), patch.object(gate, 'api', side_effect=responses()):
            with self.assertRaisesRegex(ValueError, 'artifact source revision'):
                gate.verify_ci(self.root, self.root/'bad')

    def test_explicit_release_identity_rejects_malformed_selectors(self):
        self.assertEqual(gate.release_identity(), {'revision': gate.REVISION,
                         'run_id': gate.RUN_ID, 'oracle_hash': gate.ORACLE_HASH})
        for kwargs in ({'revision': 'main'}, {'revision': 'a'*39},
                       {'run_id': 0}, {'run_id': -1}, {'run_id': True},
                       {'oracle_hash': 'wrong'}, {'oracle_hash': 'a'*63}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gate.release_identity(**kwargs)

    def test_candidate_binds_explicit_head_tag_and_clean_source(self):
        selected = gate.release_identity('a'*40, 123, 'b'*64)
        with patch.object(gate.release, 'metadata'), \
             patch.object(gate.subprocess, 'check_output', side_effect=['a'*40, 'a'*40, '']):
            gate.candidate(self.root, selected)
        for outputs, message in [(['c'*40], 'HEAD'),
                                 (['a'*40, gate.REVISION], 'tag'),
                                 (['a'*40, 'a'*40, ' M src/RK45.jl'], 'tracked')]:
            with self.subTest(message=message), patch.object(gate.release, 'metadata'), \
                 patch.object(gate.subprocess, 'check_output', side_effect=outputs):
                with self.assertRaisesRegex(ValueError, message):
                    gate.candidate(self.root, selected)

    def test_transport_binds_explicit_candidate_and_reference_manifest(self):
        repository, oracles = self.oracles()
        manifest = json.loads((oracles/'manifest.json').read_text())
        manifest['revision'] = 'a'*40
        save_json(oracles/'manifest.json', manifest)
        selected = gate.release_identity('a'*40, 123, digest(oracles/'manifest.json'))
        with patch.object(gate, 'candidate') as candidate:
            gate.transport(repository, oracles, self.root/'selected-transport', selected)
            candidate.assert_called_once_with(repository, selected)
        index = json.loads((self.root/'selected-transport/index.json').read_text())
        self.assertEqual(index['revision'], selected['revision'])
        self.assertEqual(index['run_id'], selected['run_id'])
        self.assertEqual(index['oracle_manifest_sha256'], selected['oracle_hash'])
        with patch.object(gate, 'candidate'):
            with self.assertRaisesRegex(ValueError, 'manifest hash'):
                gate.transport(repository, oracles, self.root/'wrong-transport')
        self.assertFalse((self.root/'wrong-transport').exists())

    def test_explicit_source_run_rejects_other_revision_and_run(self):
        selected = gate.release_identity('a'*40, 123, 'b'*64)
        raw = {'id': 123, 'path': '.github/workflows/run_tests.yml', 'head_sha': 'a'*40,
               'event': 'push', 'status': 'completed', 'conclusion': 'success', 'html_url': 'test'}
        jobs = [{'name': name, 'conclusion': 'success'} for name in gate.release.required_jobs()]
        artifacts = [{'name': name, 'expired': False, 'workflow_run': {'head_sha': 'a'*40}}
                     for name in ['python-native-oracles']+[f'python-native-linux-x86_64-{p}' for p in gate.INTERPRETERS]]
        def responses():
            return [raw, {'jobs': jobs, 'total_count': len(jobs)},
                    {'artifacts': artifacts, 'total_count': len(artifacts)}]
        with patch.object(gate, 'candidate'), patch.object(gate, 'api', side_effect=responses()) as api:
            gate.verify_ci(self.root, self.root/'selected-proof', selected)
            self.assertTrue(api.call_args_list[0].args[0].endswith('/123'))
        raw['head_sha'] = gate.REVISION
        with patch.object(gate, 'candidate'), patch.object(gate, 'api', side_effect=responses()):
            with self.assertRaisesRegex(ValueError, 'revision'):
                gate.verify_ci(self.root, self.root/'wrong-revision', selected)
        self.assertFalse((self.root/'wrong-revision').exists())
        raw['head_sha'], raw['id'] = 'a'*40, gate.RUN_ID
        with patch.object(gate, 'candidate'), patch.object(gate, 'api', side_effect=responses()):
            with self.assertRaisesRegex(ValueError, 'workflow'):
                gate.verify_ci(self.root, self.root/'wrong-run', selected)
        self.assertFalse((self.root/'wrong-run').exists())

    def test_relocation_rejects_original_incomplete_validation_before_mutation(self):
        # The original-cell validator is separately exercised by test_wheel_collection.
        # Ensure this orchestration cannot bypass a rejection and still write a relocated build.
        with patch.object(gate, 'candidate'), patch.object(gate, 'oracle_proof'), \
             patch.object(gate.collect, 'source_options', return_value=({}, {})), \
             patch.object(gate.collect, 'inspect_cell', side_effect=ValueError('installed validation incomplete')):
            with self.assertRaisesRegex(ValueError, 'installed validation incomplete'):
                gate.relocate(self.root, self.root, self.root, '3.11', self.root/'relocated')
        self.assertFalse((self.root/'relocated').exists())

    def test_root_sdist_metadata_is_bound_to_candidate(self):
        repository, source = self.root/'repository', self.root/'source'
        for name in ('amalthea/Cargo.lock', 'python-native/README.md', 'python-native/LICENSE',
                     'python-native/.cargo/config.toml', 'python-native/python/amalthea_native/__init__.py'):
            a = repository/name; a.parent.mkdir(parents=True, exist_ok=True); a.write_text('source\n')
            relative = name.removeprefix('python-native/') if name.startswith('python-native/python/') else name
            b = source/relative; b.parent.mkdir(parents=True, exist_ok=True); b.write_bytes(a.read_bytes())
        for name in ('README.md', 'LICENSE', '.cargo/config.toml'):
            path = source/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('source\n')
        project = repository/'python-native/pyproject.toml'
        project.write_text('[tool.maturin]\npython-source = "python"\n')
        (source/'pyproject.toml').write_text(project.read_text()+'manifest-path = "python-native/Cargo.toml"\n')
        cargo = repository/'amalthea/Cargo.toml'
        cargo.write_text('[package]\nname = "synthetic"\nversion = "0.1.0"\n')
        normalized_cargo = cargo.read_text()+'readme = "README.md"\n'
        (source/'amalthea/Cargo.toml').write_text(normalized_cargo)
        sources = {p.relative_to(repository).as_posix(): digest(p) for p in repository.rglob('*') if p.is_file()}
        with patch.object(gate, 'source_files', return_value=sources):
            gate.verify_extracted_inventory(source, repository)
            (source/'amalthea/Cargo.toml').write_text(normalized_cargo.replace('0.1.0', '9.9.9'))
            with self.assertRaisesRegex(ValueError, 'Cargo metadata differs'):
                gate.verify_extracted_inventory(source, repository)
            (source/'amalthea/Cargo.toml').write_text(normalized_cargo)
            (source/'amalthea/Cargo.lock').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'inventory differs'):
                gate.verify_extracted_inventory(source, repository)
            (source/'amalthea/Cargo.lock').write_text('source\n')
            (source/'pyproject.toml').write_text(project.read_text()+'manifest-path = "other/Cargo.toml"\n')
            with self.assertRaisesRegex(ValueError, 'build metadata'):
                gate.verify_extracted_inventory(source, repository)

    def test_runtime_recipe_keeps_original_defaults_and_selects_exact_build_patch(self):
        manifest = self.root/'build.json'
        for minor, (version, hashed, size) in gate.INTERPRETERS.items():
            self.assertEqual(gate.interpreter_recipe(minor),
                             (version, gate.PBS_RELEASE, hashed, size))
            save_json(manifest, {'host': {'python': version}})
            self.assertEqual(gate.interpreter_recipe(minor, manifest),
                             gate.interpreter_recipe(minor))
        for version in ('3.11.17', '3.12.15', '3.13.16', '3.14.8'):
            minor = version.rsplit('.', 1)[0]
            save_json(manifest, {'host': {'python': version}})
            self.assertEqual(gate.interpreter_recipe(minor, manifest)[0:2],
                             (version, '20261009'))

    def test_runtime_rejects_unknown_patch_or_wrong_minor_before_writing(self):
        manifest, output = self.root/'build.json', self.root/'runtime'
        for version, message in [('3.11.999', 'unauthenticated'), ('3.12.15', 'minor'),
                                 ('3.11.17-custom', 'minor'), (None, 'minor')]:
            save_json(manifest, {'host': {'python': version}})
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, message):
                gate.interpreter('3.11', output, manifest)
            self.assertFalse(output.exists())
        save_json(self.root/'build.json', {'host': {'python': '3.11.999'}})
        with patch.object(gate, 'candidate'), patch.object(gate, 'oracle_proof'), \
             patch.object(gate.collect, 'source_options', return_value=({}, {})), \
             patch.object(gate.collect, 'inspect_cell', return_value={'status': 'passed'}):
            with self.assertRaisesRegex(ValueError, 'unauthenticated'):
                gate.relocate(self.root, self.root, self.root, '3.11', self.root/'relocated')
        self.assertFalse((self.root/'relocated').exists())

    def test_selected_interpreter_records_build_digest_and_authenticated_download(self):
        manifest, output = self.root/'build.json', self.root/'runtime'
        save_json(manifest, {'host': {'python': '3.14.8'}})
        def download(url, destination, expected, size):
            self.assertIn('/20261009/cpython-3.14.8+20261009-', url)
            self.assertEqual((expected, size), gate.INTERPRETER_RECIPES['3.14.8'][1:])
            with tarfile.open(destination, 'w:gz') as archive:
                member = tarfile.TarInfo('python/synthetic')
                member.size = 7
                archive.addfile(member, io.BytesIO(b'fixture'))
        with patch.object(gate, 'download', side_effect=download):
            gate.interpreter('3.14', output, manifest)
        provenance = json.loads((output/'provenance.json').read_text())
        self.assertEqual(provenance['version'], '3.14.8')
        self.assertEqual(provenance['release'], '20261009')
        self.assertEqual(provenance['build_manifest_sha256'], digest(manifest))
        self.assertEqual((output/'python/synthetic').read_bytes(), b'fixture')

    def test_cli_rejects_build_from_another_candidate_before_runtime_preparation(self):
        manifest, output = self.root/'build.json', self.root/'runtime'
        save_json(manifest, {'revision': 'c'*40, 'host': {'python': '3.14.8'}})
        args = ['helper', 'interpreter', '--repository', str(self.root),
                '--output', str(output), '--python', '3.14', '--build-manifest', str(manifest),
                '--revision', 'a'*40, '--run-id', '123', '--oracle-sha256', 'b'*64]
        with patch.object(sys, 'argv', args), patch.object(gate, 'candidate'), \
             patch.object(gate, 'interpreter') as prepare:
            with self.assertRaisesRegex(ValueError, 'runtime build revision'):
                gate.main()
            prepare.assert_not_called()
        self.assertFalse(output.exists())

    def test_dependency_resolution_uses_selected_exact_patch(self):
        project = self.root/'python-native/pyproject.toml'
        project.parent.mkdir()
        project.write_text('[project]\ndependencies = []\n[project.optional-dependencies]\nhdf5 = []\n')
        manifest, output = self.root/'build.json', self.root/'dependencies'
        save_json(manifest, {'host': {'python': '3.14.8'}})
        with patch.object(gate.subprocess, 'run', side_effect=RuntimeError('stop before download')):
            with self.assertRaisesRegex(RuntimeError, 'stop before download'):
                gate.dependencies(self.root, '3.14', output, manifest)
        command = json.loads((output/'download-command.json').read_text())
        self.assertEqual(command[command.index('--python-version') + 1], '3.14.8')
        self.assertIn('--only-binary=:all:', command)

    def test_runtime_archive_download_rejects_wrong_hash_and_size(self):
        source = self.root/'fixture'
        source.write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            gate.download(source.as_uri(), self.root/'bad-hash', '0'*64, 7)
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            gate.download(source.as_uri(), self.root/'bad-size', digest(source), 8)


if __name__ == '__main__':
    unittest.main()
