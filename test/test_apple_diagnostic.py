"""Acceptance transport checks; synthetic data is never Apple hardware evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('apple_diagnostic',ROOT/'test/standalone_wheels/apple_diagnostic.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():
    rows=[dict(threads=t,seconds=.1,thread_relative_error=1e-14,cross_build_relative_error=1e-14) for t in (1,2,4)]
    lever=dict(correct=True,relative_error=1e-14,max_thread_relative_error=1e-14,max_cross_build_relative_error=1e-14,
               portable_threads=rows,native_threads=copy.deepcopy(rows))
    return dict(dry_run=False,host=dict(system='Darwin',machine='arm64'),lto_recommendation='candidate only',
                levers=dict(neon_raman=lever,configured_blas_qdht=copy.deepcopy(lever),
                    process_thread_topology=dict(correct=True,modal_by_threads={f'modal_t{t}':dict(exact=True) for t in (1,2,4)})))


class AppleAcceptance(unittest.TestCase):
    def test_complete_schema_preserves_diagnostic_scope(self):
        result=module.validate(fixture())
        self.assertEqual(result['status'],'passed');self.assertIn('no LTO promotion',result['scope'])

    def test_native_hardware_and_real_run_required(self):
        for change in ({'dry_run':True},{'host':{'system':'Linux','machine':'aarch64'}},{'host':{'system':'Darwin','machine':'x86_64'}}):
            with self.subTest(change=change),self.assertRaises(ValueError):module.validate(fixture()|change)

    def test_false_missing_or_nonfinite_correctness_rejected(self):
        for key,value in [('correct',False),('relative_error',2e-6),('relative_error',float('nan'))]:
            data=fixture();data['levers']['neon_raman'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):module.validate(data)
        data=fixture();del data['levers']['neon_raman']
        with self.assertRaises(ValueError):module.validate(data)

    def test_every_thread_and_topology_required(self):
        for change in ('missing','time','field','topology','modal'):
            data=fixture();lever=data['levers']['neon_raman']
            if change=='missing':lever['native_threads'].pop()
            elif change=='time':lever['native_threads'][0]['seconds']=float('inf')
            elif change=='field':lever['native_threads'][0]['thread_relative_error']=1e-3
            elif change=='topology':data['levers']['process_thread_topology']['correct']=False
            else:data['levers']['process_thread_topology']['modal_by_threads']['modal_t1']['exact']=False
            with self.subTest(change=change),self.assertRaises(ValueError):module.validate(data)

    @unittest.skipIf(platform.system()=='Darwin' and platform.machine()=='arm64','failure-path check runs on non-Apple hosts')
    def test_real_non_apple_invocation_records_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'result'
            result=subprocess.run([sys.executable,str(ROOT/'test/standalone_wheels/apple_diagnostic.py'),'--output',str(output)],
                                  capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            data=json.loads((output/'validation.json').read_text())
            self.assertEqual(data['status'],'failed');self.assertIn('native Apple Silicon',data['error'])
            self.assertFalse((output/'runner.log').exists())


if __name__=='__main__':unittest.main()
