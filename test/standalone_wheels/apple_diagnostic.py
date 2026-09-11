#!/usr/bin/env python3
"""Collect and validate the prepared Apple quick diagnostic on native hardware."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python-native/tools'))
from _validation import cpu_environment,digest,run,save_json,utcnow


def validate(data):
    if data.get('dry_run') is not False or data.get('host',{}).get('system')!='Darwin' or data['host'].get('machine')!='arm64':
        raise ValueError('actual native Apple Silicon results are required')
    levers=data.get('levers',{})
    for name in ('neon_raman','configured_blas_qdht'):
        lever=levers.get(name,{})
        if lever.get('correct') is not True:raise ValueError(f'{name} correctness failed or missing')
        for key in ('relative_error','max_thread_relative_error','max_cross_build_relative_error'):
            error=lever[key]
            if not math.isfinite(error) or not 0<=error<=1e-6:raise ValueError(f'{name} invalid {key}')
        for build in ('portable','native'):
            rows=lever[build+'_threads']
            if sorted(row['threads'] for row in rows)!=[1,2,4]:raise ValueError('incomplete thread series')
            for row in rows:
                if not math.isfinite(row['seconds']) or row['seconds']<=0:raise ValueError('invalid timing')
                for key in ('thread_relative_error','cross_build_relative_error'):
                    if not math.isfinite(row[key]) or not 0<=row[key]<=1e-6:raise ValueError('thread/build field check failed')
    topology=levers.get('process_thread_topology',{})
    if topology.get('correct') is not True:raise ValueError('modal/scan topology failed')
    for threads in (1,2,4):
        if topology['modal_by_threads'][f'modal_t{threads}'].get('exact') is not True:
            raise ValueError('modal thread identity failed')
    return {'status':'passed','scope':'native Apple diagnostic; no LTO promotion',
            'levers':list(levers),'lto_recommendation':data['lto_recommendation']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    state={'status':'running','started':utcnow(),'host':{'system':platform.system(),'machine':platform.machine()},
           'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    sources=[ROOT/'test/standalone_wheels/apple_diagnostic.py',*sorted((ROOT/'test/performance_audit').glob('*.py')),
             *sorted((ROOT/'test/performance_audit').glob('*.jl')),ROOT/'test/performance_audit/workloads.toml']
    state['sources']={str(path.relative_to(ROOT)):digest(path) for path in sources}
    save_json(output/'validation.json',state)
    try:
        if platform.system()!='Darwin' or platform.machine()!='arm64':raise RuntimeError('run on native Apple Silicon')
        environment=cpu_environment()
        for name in list(environment):
            if name.startswith('AMALTHEA_USE_RUST_'):environment[name]='0'
        environment.update(AMALTHEA_NATIVE_GPU='off',AMALTHEA_USE_RUST_CUDA_NATIVE='0',
                           VECLIB_MAXIMUM_THREADS='1',JULIA_NUM_THREADS='1')
        run([sys.executable,ROOT/'test/performance_audit/run_apple_quick_test.py',
             '--output',output/'apple-quick.json'],cwd=ROOT,environment=environment,log=output/'runner.log')
        result=json.loads((output/'apple-quick.json').read_text())
        state.update(validate(result))
        if state['sources']!={str(path.relative_to(ROOT)):digest(path) for path in sources}:
            raise ValueError('diagnostic source changed during run')
        state['artifacts']={name:digest(output/name) for name in ('apple-quick.json','apple-quick.md','runner.log')}
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        state['finished']=utcnow();save_json(output/'validation.json',state)


if __name__=='__main__':main()
