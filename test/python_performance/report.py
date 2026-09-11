#!/usr/bin/env python3
"""Render accepted timings or explicitly labeled diagnostic snapshot data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import statistics


def series(entry, metric):
    return [sample[metric] for sample in entry.get('samples',[])]


def admitted(snapshot, case, entry):
    return (not snapshot.get('smoke',True) and snapshot.get('status') in ('passed','passed_with_exclusions')
            and case.get('status')=='passed' and not entry.get('status')
            and statistics(series(entry,'complete_seconds'))['stable'])


def counts(entry):
    pairs=set()
    for sample in entry.get('samples',[]):
        count=sample.get('execution',sample)
        pairs.add((count['accepted_steps'],count['rejected_steps']))
    return ', '.join(f'{a}/{r}' for a,r in sorted(pairs)) or '—'


def milliseconds(entry, metric):
    values=series(entry,metric)
    return f"{statistics(values)['median']*1000:.4f}" if values else '—'


def render(snapshot):
    accepted=(not snapshot.get('smoke',True) and snapshot.get('status') in ('passed','passed_with_exclusions')
              and all(admitted(snapshot,case,entry) for case in snapshot.get('cases',{}).values()
                      for entry in case['backends'].values() if not entry.get('status')))
    lines=['# Standalone Python post-repair snapshot','',f"Status: **{snapshot['status']}**.",'']
    lines += ['Controlled timings passed the recorded acceptance gates.' if accepted else
              '**Diagnostic data only. No accepted speedups or bottleneck ranking.**','']
    lines += [f"Revision: `{snapshot.get('revision','unknown')}`. Source wheel SHA-256: `{snapshot.get('wheel_sha256','unknown')}`.",
              f"CPU affinity: `{snapshot.get('affinity',[])}`; kernel `{snapshot.get('kernel','unknown')}`.",
              'The complete-workload metric includes fresh frontend setup and solve, including plans, accepted filtering and dense saved fields.',
              'Import/startup/first-public metrics are single-process observations. Peak RSS is the process lifetime high-water mark, including correctness checks.',
              'HDF5 writing is separate: Python writes complete result/grid/JSON metadata; the Julia worker writes numeric field and positions. Those payloads are not equal work.','']
    ranking=[]
    for name,case in snapshot.get('cases',{}).items():
        lines += [f'## {name}','',f"Nonlinear/profile control: `{case.get('correctness',{}).get('feature_effect','unavailable')}`.",'',
                  '| Path | Complete ms | Setup ms | Solve ms | Copy ms | HDF5 ms | Peak MiB | Accepted/rejected |',
                  '|---|---:|---:|---:|---:|---:|---:|---|']
        for backend,entry in case['backends'].items():
            if entry.get('status'):
                lines += [f"| {backend}: {entry['status']} | — | — | — | — | — | — | — |"]
                continue
            samples=entry.get('samples',[])
            rss=max((s['peak_rss_bytes'] for s in samples),default=0)/2**20
            metrics=[milliseconds(entry,k) for k in ('complete_seconds','setup_seconds','solve_seconds','copy_seconds','hdf5_seconds')]
            lines += [f"| {backend} | {' | '.join(metrics)} | {rss:.1f} | {counts(entry)} |"]
        lines += ['','| Path | Startup s | Import s | First public simulation s |','|---|---:|---:|---:|']
        for backend,entry in case['backends'].items():
            cold=entry.get('cold',{})
            values=[entry.get('startup_seconds'),cold.get('import_seconds'),cold.get('first_public_seconds')]
            formatted=['—' if value is None else f'{value:.4f}' for value in values]
            lines += [f"| {backend} | {' | '.join(formatted)} |"]
        lines += ['']
        auto=case['backends'].get('auto')
        if accepted and auto and admitted(snapshot,case,auto):
            total=statistics(series(auto,'complete_seconds'))['median']
            setup=statistics(series(auto,'setup_seconds'))['median']
            ranking.append((total,name,setup/total))
            for baseline in ('julia','rust','python'):
                entry=case['backends'].get(baseline)
                if entry and admitted(snapshot,case,entry):
                    left,right=series(entry,'complete_seconds'),series(auto,'complete_seconds')
                    if len(left)!=len(right):raise ValueError('unpaired timing samples')
                    ratio=statistics([a/b for a,b in zip(left,right)])
                    low,high=ratio['ci95']
                    if ratio['stable']:
                        lines += [f"Python auto speedup versus {baseline}: **{ratio['median']:.3f}×**, paired bootstrap 95% CI [{low:.3f}, {high:.3f}]."]
                    else:lines += [f'Paired ratio versus {baseline} is unstable; no accepted speedup.']
        lines += ['', '<details><summary>Exact inputs and numerical checks</summary>','', '```json',
                  json.dumps({'inputs':case.get('inputs',{}),'correctness':case.get('correctness',{})},indent=2), '```','', '</details>','']
    if accepted and ranking:
        lines += ['## Measured Python auto workload costs','',
                  'Descending complete-workload time identifies where further profiling is useful. These are different physical workloads; their ratios do not establish an optimization.','']
        for total,name,fraction in sorted(ranking,reverse=True):
            lines += [f'- {name}: {total*1000:.4f} ms; frontend setup accounts for approximately {fraction:.1%} of that cost.']
    lines += ['', 'Frozen CPU audit artifacts were not changed. Raw per-sample records, fields, hashes and worker logs remain next to the source snapshot.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    content=render(json.loads(args.snapshot.read_text()))
    with args.output.open('x') as stream:stream.write(content)


if __name__=='__main__':main()
