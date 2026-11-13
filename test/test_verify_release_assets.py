"""Transport/byte-binding tests; synthetic fixtures are not release acceptance."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import test_release as fixtures
import verify_release_assets as verification

release = fixtures.release
SHA = fixtures.SHA
ORACLE = 'b' * 64
ROOT = Path(__file__).resolve().parents[1]


class ActualAssets(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        artifacts = self.root / 'artifacts'
        report = fixtures.fixture(artifacts)
        report.update(workflow=fixtures.run_record(), workflow_failures=[], oracle_manifest_sha256=ORACLE)
        self.assets = self.root / 'assets'
        with patch.object(release.subprocess, 'check_output', return_value=SHA + '\n'), \
                patch.object(release.wheels, 'collect', return_value=report), \
                patch.object(release.wheels, 'markdown', return_value='synthetic evidence fixture\n'):
            release.stage_python(artifacts, ROOT, fixtures.run_record(), self.assets)
        for name in release.LIBRARIES:
            (self.assets / name).write_bytes(b'synthetic library ' + name.encode())
        with patch.object(release.subprocess, 'check_output', return_value=SHA + '\n'):
            release.checksums(self.assets, ROOT)

    def verify(self, run=1, oracle=ORACLE):
        with patch.object(verification.release.subprocess, 'check_output', return_value=SHA + '\n'):
            return verification.verify_assets(self.assets, ROOT, run, oracle)

    def test_all_actual_bytes_and_sixteen_cells_are_bound(self):
        proof = self.verify()
        self.assertEqual(proof['wheel_cells'], 16)
        self.assertEqual(len(proof['sha256']), 26)

    def test_altered_wheel_library_and_evidence_are_rejected(self):
        for name in [release.LIBRARIES[0], 'wheel-evidence.json', next(self.assets.glob('*.whl')).name]:
            path = self.assets / name
            before = path.read_bytes()
            path.write_bytes(before + b'changed')
            with self.subTest(name=name), self.assertRaises(ValueError): self.verify()
            path.write_bytes(before)

    def test_extra_missing_symlink_and_duplicate_checksum_are_rejected(self):
        path = self.assets / 'extra'
        path.write_text('unexpected')
        with self.assertRaises(ValueError): self.verify()
        path.unlink()
        path = self.assets / release.LIBRARIES[0]
        before = path.read_bytes(); path.unlink()
        with self.assertRaises(ValueError): self.verify()
        other = self.root / 'library'; other.write_bytes(before); path.symlink_to(other)
        with self.assertRaises(ValueError): self.verify()
        path.unlink(); path.write_bytes(before)
        sums = self.assets / 'SHA256SUMS.txt'
        sums.write_text(sums.read_text() + sums.read_text().splitlines()[0] + '\n')
        with self.assertRaises(ValueError): self.verify()

    def test_wrong_source_run_and_oracle_are_rejected(self):
        with self.assertRaises(ValueError): self.verify(run=2)
        with self.assertRaises(ValueError): self.verify(oracle='c' * 64)

    def test_producer_identity_and_bytes_cannot_be_substituted(self):
        proof = self.verify()
        produced = self.root / 'produced'; shutil.copytree(self.assets, produced)
        jobs = {'Exact-source release acceptance', 'Assemble checked release assets',
                'Create reviewable GitHub release draft'}
        jobs.update('Build (' + t + ')' for t in ('x86_64-unknown-linux-gnu',
            'aarch64-unknown-linux-gnu', 'aarch64-apple-darwin', 'x86_64-pc-windows-msvc'))
        run = fixtures.run_record(jobs)
        verification.verify_produced_assets(produced, proof, run)
        with self.assertRaises(ValueError):
            verification.verify_produced_assets(produced, proof, run | {'headSha': 'c' * 40})
        (produced / release.LIBRARIES[0]).write_bytes(b'substituted')
        with self.assertRaises(ValueError): verification.verify_produced_assets(produced, proof, run)

    def test_runtime_receipts_are_relocated_without_rewriting(self):
        fixture = fixtures.GlibcEvidence(self.root / 'fixture')
        before = fixture.snapshot()
        raw = self.root / 'raw'
        for p in release.wheels.PYTHONS:
            target = raw / ('glibc228-linux-x86_64-' + p) / 'evidence'
            shutil.copytree(fixture.root / 'builds' / ('linux-x86_64-' + p), target / 'build')
            shutil.copytree(fixture.root / 'gates' / p, target / 'gate')
        output = self.root / 'relocated'
        verification.arrange(raw, output)
        with patch.object(release.subprocess, 'check_output', return_value=SHA + '\n'):
            result = release.verify_glibc(output / 'gates', output / 'builds',
                fixture.root / 'repository', fixture.root / 'oracles', fixture.root / 'assets')
        self.assertEqual(result['status'], 'glibc_release_wheels_passed', result)
        self.assertEqual(fixture.snapshot(), before)


if __name__ == '__main__':
    unittest.main()
