"""Guard evidence gates in the separate performance runner; no synthetic physics."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('python_performance',HERE/'python_performance/run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class EvidenceGates(unittest.TestCase):
    def test_stability_requires_samples_and_precision(self):
        self.assertFalse(runner.statistics([1.]*9)['stable'])
        self.assertTrue(runner.statistics([1.]*10)['stable'])
        self.assertFalse(runner.statistics([1.,2.]*15)['stable'])
        for values in ([],[0.],[-1.],[float('nan')],[float('inf')],[[1.,1.]]):
            with self.subTest(values=values),self.assertRaises(ValueError):runner.statistics(values)

    def test_relative_requires_same_shape_and_finite(self):
        self.assertEqual(runner.relative(np.ones(3),np.ones(3)),0.)
        for left in (np.ones((1,3)),np.array([np.nan,1.,1.])):
            with self.assertRaises(ValueError):runner.relative(left,np.ones(3))

    def test_field_hash_and_fortran_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);value=np.arange(6).reshape(2,3)+1j
            (root/'field.bin').write_bytes(value.astype('<c16').tobytes(order='F'))
            record={'arrays':{'field':{'shape':[2,3],'dtype':'<c16','sha256':runner.digest(root/'field.bin')}}}
            np.testing.assert_array_equal(runner.read_array(root,record,'field'),value)
            with (root/'field.bin').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):runner.read_array(root,record,'field')

    def test_installed_bytes_must_match_wheel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package=root/'amalthea_native';package.mkdir()
            (package/'__init__.py').write_text('correct')
            wheel=root/'test.whl'
            with zipfile.ZipFile(wheel,'w') as archive:archive.writestr('amalthea_native/__init__.py','correct')
            record={'package':str(package/'__init__.py')}
            runner.installed_matches(record,wheel)
            (package/'__init__.py').write_text('changed')
            with self.assertRaises(ValueError):runner.installed_matches(record,wheel)


class ReportGates(unittest.TestCase):
    @staticmethod
    def module():
        spec=importlib.util.spec_from_file_location('performance_report',HERE/'python_performance/report.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    @staticmethod
    def fixture():
        def entry(seconds):
            sample=dict(complete_seconds=seconds,setup_seconds=seconds/4,solve_seconds=seconds*.75,
                        copy_seconds=.001,hdf5_seconds=.01,peak_rss_bytes=1024,
                        accepted_steps=10,rejected_steps=1)
            return dict(samples=[dict(sample) for _ in range(10)],cold={})
        return dict(status='passed',smoke=False,cases={'case':dict(status='passed',
                    backends={'julia':entry(2.),'auto':entry(1.)},correctness={})})

    def test_diagnostic_snapshot_never_claims_accepted_speedup(self):
        module=self.module()
        for status,smoke in [('smoke_passed',True),('failed',False),('unstable',False)]:
            snapshot=self.fixture();snapshot.update(status=status,smoke=smoke)
            text=module.render(snapshot)
            self.assertIn('Diagnostic data only',text)
            self.assertNotIn('speedup versus',text)
            self.assertNotIn('## Measured Python auto workload costs',text)

    def test_recompute_raw_stability_and_exclude_inadmissible(self):
        module=self.module();snapshot=self.fixture()
        self.assertIn('**2.000×**',module.render(snapshot))
        altered=copy.deepcopy(snapshot)
        for sample in altered['cases']['case']['backends']['julia']['samples'][::2]:
            sample['complete_seconds']=20.
        altered['cases']['case']['backends']['julia']['statistics']={'complete_seconds':{'stable':True}}
        self.assertNotIn('speedup versus julia',module.render(altered))
        excluded=copy.deepcopy(snapshot)
        excluded['cases']['case']['backends']['julia']['status']='inadmissible: ADE'
        self.assertNotIn('speedup versus julia',module.render(excluded))
        self.assertIn('inadmissible: ADE',module.render(excluded))


if __name__=='__main__':unittest.main()
