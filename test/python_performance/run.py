#!/usr/bin/env python3
"""Separate installed Python / current Julia / resident Rust performance snapshot."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import tomllib
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
sys.path.insert(0, str(REPOSITORY/'python-native/tools'))
from _validation import cpu_environment, digest, save_json, verify_oracles
from wheel_validation import source_files


def source_hashes(repository, files):
    return {p.relative_to(repository).as_posix(): digest(p, normalize_lf=True) for p in sorted(files)}


def relative(a, b):
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('field shape/finiteness mismatch')
    return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)), 1e-300))


def statistics(values):
    x = np.asarray(values, dtype=float)
    if x.ndim!=1 or not x.size or not np.all(np.isfinite(x)) or np.any(x <= 0): raise ValueError('invalid timing sample')
    median = float(np.median(x))
    rng = np.random.default_rng(731)
    boot = np.median(rng.choice(x, size=(4000, len(x))), axis=1)
    low, high = np.quantile(boot, [.025, .975])
    result = dict(median=median, relative_mad=float(np.median(abs(x-median))/median),
                  ci95=[float(low),float(high)], relative_halfwidth=float((high-low)/(2*median)))
    result['stable'] = len(x)>=10 and result['relative_mad']<=.03 and result['relative_halfwidth']<=.05
    return result


def read_array(root, record, name):
    info = record['arrays'][name]; path = root/(name+'.bin')
    if digest(path) != info['sha256']: raise ValueError('changed field artifact')
    return np.fromfile(path,dtype=info['dtype']).reshape(info['shape'],order='F')


def installed_matches(record, wheel):
    package = Path(record['package']).resolve().parent
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('amalthea_native/') and not name.endswith('/'):
                path = package.parent/name
                if not path.is_file() or path.read_bytes()!=archive.read(name):
                    raise ValueError(f'installed package differs from wheel: {name}')


class Worker:
    def __init__(self, backend, case, output, args, environment):
        self.backend = backend; self.root = output/case/backend
        self.root.mkdir(parents=True)
        self.log = (self.root/'worker.log').open('w')
        env = dict(environment)
        if backend in ('julia','rust'):
            env['AMALTHEA_USE_RUST_NATIVE'] = '1' if backend=='rust' else '0'
            argv = [args.julia,'--startup-file=no',f'--project={REPOSITORY}',
                    str(HERE/'julia_worker.jl'),str(HERE/'cases.toml'),case,backend]
        else:
            argv = [str(args.python),'-I','-X','utf8',str(HERE/'python_worker.py'),str(HERE/'cases.toml'),case,backend]
        start=time.perf_counter()
        self.process=subprocess.Popen(argv,cwd=self.root,env=env,text=True,encoding='utf-8',
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)
        try:
            self.await_line('READY')
        except BaseException:
            self.close(); raise
        self.startup_seconds=time.perf_counter()-start

    def await_line(self, expected):
        for line in self.process.stdout:
            self.log.write(line); self.log.flush()
            if line.strip()==expected:return
        raise RuntimeError(f'{self.backend} worker ended ({self.process.wait()}); see {self.root}/worker.log')

    def request(self, operation, name=None):
        root=self.root/(name or operation)
        self.process.stdin.write(operation+'\t'+str(root)+'\n'); self.process.stdin.flush()
        self.await_line('DONE')
        if (root/'result.json').exists(): record=json.loads((root/'result.json').read_text())
        else: record=tomllib.loads((root/'result.toml').read_text())
        return record, root

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('stop\n'); self.process.stdin.flush(); self.process.wait(timeout=10)
            except (BrokenPipeError,subprocess.TimeoutExpired):
                self.process.terminate(); self.process.wait(timeout=10)
        self.log.close()


def busy_validation():
    active=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try: command=path.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (OSError,ProcessLookupError):continue
        if ('-m pytest' in command or 'export_oracles.py --output' in command) and int(path.parent.name)!=os.getpid():
            active.append({'pid':int(path.parent.name),'command':command[:300]})
    return active


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--julia',default='julia')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--oracles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--smoke',action='store_true',help='correctness/development only; timings are not accepted')
    args=parser.parse_args(); args.output=args.output.resolve(); args.python=args.python.absolute()
    args.output.mkdir(parents=True,exist_ok=False)
    if args.output.is_relative_to(REPOSITORY): raise ValueError('snapshot outputs must be outside checkout')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPOSITORY,text=True).strip()
    verify_oracles(args.oracles.resolve(),REPOSITORY,revision)
    build=json.loads(args.manifest.read_text())
    if build['status']!='built' or build['revision']!=revision or build['sources']!=source_files(REPOSITORY):
        raise ValueError('completed matching build manifest required')
    wheel=Path(build['wheels']['source']['path'])
    if digest(wheel)!=build['wheels']['source']['sha256']:raise ValueError('wheel changed')
    busy=busy_validation()
    if busy and not args.smoke:raise RuntimeError('heavy validation is active; use smoke or wait for completion')
    affinity=sorted(os.sched_getaffinity(0));cpu=affinity[-1];os.sched_setaffinity(0,{cpu})
    environment=cpu_environment()
    for key in list(environment):
        if key.startswith('AMALTHEA_USE_RUST_'): environment[key]='0'
    environment.update(AMALTHEA_NATIVE_GPU='off',AMALTHEA_NATIVE_FFTW_WISDOM='0',
        JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='1')
    sources=source_hashes(REPOSITORY,[*HERE.glob('*.py'),*HERE.glob('*.jl'),HERE/'cases.toml'])
    record=dict(status='running',revision=revision,source_hashes=build['sources'],harness_hashes=sources,
        build_manifest_sha256=digest(args.manifest),wheel_sha256=digest(wheel),
        oracle_manifest_sha256=digest(args.oracles/'manifest.json'),
        julia_manifest_sha256=digest(REPOSITORY/'Manifest.toml'),
        library_sha256=digest(REPOSITORY/'amalthea/target/release/libamalthea.so'),
        smoke=args.smoke,busy_validation=busy,affinity_before=affinity,affinity=[cpu],
        cpu=Path('/proc/cpuinfo').read_text().split('\n\n')[0],kernel=os.uname().release,
        rustc=subprocess.check_output(['rustc','-Vv'],text=True),
        julia_command=args.julia,python_command=str(args.python),
        endian=sys.byteorder,
        threads={k:environment[k] for k in ('JULIA_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')},cases={})
    turbo=Path('/sys/devices/system/cpu/cpufreq/boost')
    record['turbo_boost']=turbo.read_text().strip() if turbo.exists() else None
    for leaf in ('cpufreq/scaling_governor','cpufreq/scaling_cur_freq','topology/thread_siblings_list','microcode/version'):
        path=Path(f'/sys/devices/system/cpu/cpu{cpu}')/leaf
        record[leaf]=path.read_text().strip() if path.exists() else None
    def save():save_json(args.output/'snapshot.json',record)
    save();config=tomllib.loads((HERE/'cases.toml').read_text())
    selected=args.cases or list(config['cases'])
    try:
        for case in selected:
            if case not in config['cases']:raise ValueError('unknown case '+case)
            workers={};cell={'inputs':config['common']|config['cases'][case],'backends':{},'correctness':{}}
            record['cases'][case]=cell;save()
            names=['julia','rust','auto','python']+(['callback'] if case=='gnlse-kerr' else [])
            try:
                checks={};cold={}
                for name in names:
                    print(f'{case}: {name} cold and correctness',flush=True)
                    worker=Worker(name,case,args.output,args,environment);workers[name]=worker
                    cold[name]=worker.request('cold');check,root=worker.request('check')
                    if name in ('auto','python','callback'):installed_matches(check,wheel)
                    checks[name]=(check,root)
                    entry={'startup_seconds':worker.startup_seconds,'cold':cold[name][0], 'check':check,'samples':[]}
                    cell['backends'][name]=entry
                    if name=='rust' and check['native_eligible'] != config['cases'][case].get('julia_native',True):
                        raise AssertionError(f'unexpected Julia native eligibility: {case} {check["actual_stepper"]}')
                    if name=='rust' and not check['native_eligible']:
                        entry['status']='unavailable: Julia native eligibility fallback';worker.close();del workers[name]
                reference,refroot=checks['julia'];field=read_array(refroot,reference,'field')
                control,controlroot=workers['julia'].request('control')
                effect=relative(field,read_array(controlroot,control,'field'))
                if effect<=1e-5:raise AssertionError(f'vacuous {case} {effect}')
                cell['correctness']['feature_effect']=effect
                for name,worker in list(workers.items()):
                    check,root=checks[name]
                    errors={key:relative(read_array(root,check,key),read_array(refroot,reference,key))
                            for key in ('field','initial','linear','t','omega','z')}
                    cold_record,coldroot=cold[name];cold_ref,coldrefroot=cold['julia']
                    errors['public_field']=relative(read_array(coldroot,cold_record,'field'),read_array(coldrefroot,cold_ref,'field'))
                    if name=='rust' and errors['field']>=1e-6 and config['cases'][case].get('allow_rust_method_exclusion',False):
                        cell['backends'][name].update(status='inadmissible: existing ADE versus FFT discretization',errors=errors)
                        worker.close();del workers[name];continue
                    for key in ('field','public_field','linear'):
                        if errors[key]>=1e-6:raise AssertionError((case,name,key,errors[key]))
                    for key in ('initial','t','omega','z'):
                        if errors[key]>=1e-13:raise AssertionError((case,name,key,errors[key]))
                    if name in ('auto','python','callback'):
                        rhs,rhsroot=worker.request('rhs')
                        errors['same_input_rhs']=relative(read_array(rhsroot,rhs,'rhs'),read_array(refroot,reference,'rhs'))
                        # Modal integrals use independently converged global-error budgets.
                        limit=1e-6 if case.startswith('modal-') else 1e-13
                        if errors['same_input_rhs']>=limit:raise AssertionError((case,name,'rhs',errors['same_input_rhs']))
                    cell['correctness'][name]=errors
                if 'callback' in workers:
                    custom,croot=checks['callback']; builtin,broot=checks['python']
                    error=relative(read_array(croot,custom,'field'),read_array(broot,builtin,'field'))
                    if error>=1e-13 or custom['execution'].get('custom_response_calls',0)<=0:
                        raise AssertionError(('custom callback equivalence',error))
                    cell['correctness']['callback_vs_python']=error
                save()
                order=random.Random(731)
                for index in range(1 if args.smoke else 32):
                    names=list(workers);order.shuffle(names)
                    for name in names:
                        sample,root=workers[name].request('sample',f'sample-{index:02}')
                        error=relative(read_array(root,sample,'field'),field)
                        if error>=1e-6:raise AssertionError((case,name,'sample trajectory',error))
                        sample['trajectory_error']=error
                        if index>=2 or args.smoke:cell['backends'][name]['samples'].append(sample)
                    for name in names:
                        samples=cell['backends'][name]['samples']
                        if samples:
                            cell['backends'][name]['statistics']={metric:statistics([s[metric] for s in samples])
                                for metric in ('complete_seconds','setup_seconds','solve_seconds','copy_seconds','hdf5_seconds')}
                    save()
                    if not args.smoke and index>=11 and all(cell['backends'][n]['statistics']['complete_seconds']['stable'] for n in names):break
                cell['status']='smoke_passed' if args.smoke else ('passed' if all(
                    cell['backends'][n]['statistics']['complete_seconds']['stable'] for n in workers) else 'unstable')
                save()
            finally:
                for worker in workers.values():worker.close()
        if build['sources']!=source_files(REPOSITORY) or sources!=source_hashes(REPOSITORY,[*HERE.glob('*.py'),*HERE.glob('*.jl'),HERE/'cases.toml']):
            raise ValueError('source changed during snapshot')
        record['excluded_comparisons']=[{'case':name,'backend':backend,'status':entry['status']}
            for name,cell in record['cases'].items() for backend,entry in cell['backends'].items()
            if entry.get('status','').startswith('inadmissible')]
        if record['library_sha256']!=digest(REPOSITORY/'amalthea/target/release/libamalthea.so'):
            raise ValueError('shared library changed during snapshot')
        record['status']='smoke_passed' if args.smoke else ('passed' if all(c['status']=='passed' for c in record['cases'].values()) else 'unstable')
        if record['excluded_comparisons']:record['status']+='_with_exclusions'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
