"""Read-only final verification of downloaded release assets and glibc evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import release


def verify_assets(assets, repository, source_run, oracle_hash):
    candidate = release.metadata(repository)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    manifest = release.read(assets / 'release-manifest.json')
    release.require(manifest['revision'] == revision and manifest['test_run'] == source_run
                    and manifest['cpu_only'] is True, 'release source/run identity mismatch')
    release.require(all(manifest[k] == v for k, v in candidate.items()), 'release metadata mismatch')
    version = candidate['python_version']
    expected = set(release.LIBRARIES) | {
        f'amalthea_native-{version}.tar.gz', 'wheel-evidence.json',
        'wheel-evidence.md', 'python-artifacts.json'}
    expected.update(f'amalthea_native-{version}-cp{p.replace(".", "")}-cp{p.replace(".", "")}-{host[3]}.whl'
                    for host in release.wheels.PLATFORMS.values() for p in release.wheels.PYTHONS)
    release.require(set(manifest['assets']) == expected, 'release manifest inventory mismatch')
    inventory = expected | {'release-manifest.json', 'SHA256SUMS.txt'}
    release.require({p.name for p in assets.iterdir()} == inventory, 'downloaded asset inventory mismatch')
    hashes = {}
    for name in inventory:
        path = assets / name
        release.require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                        'asset is missing, empty or symlinked: ' + name)
        hashes[name] = release.wheels.digest(path)
    release.require(all(hashes[name] == hashed for name, hashed in manifest['assets'].items()),
                    'downloaded release bytes differ from manifest')
    sums = {}
    for line in (assets / 'SHA256SUMS.txt').read_text(encoding='ascii').splitlines():
        hashed, name = line.split('  ')
        release.require(name not in sums, 'duplicate checksum entry')
        sums[name] = hashed
    release.require(sums == {k: v for k, v in hashes.items() if k != 'SHA256SUMS.txt'},
                    'downloaded checksum manifest mismatch')
    evidence = release.read(assets / 'wheel-evidence.json')
    cells = {f'{platform}-{p}' for platform in release.wheels.PLATFORMS for p in release.wheels.PYTHONS}
    release.require(evidence['status'] == 'wheel_matrix_passed' and evidence['revision'] == revision
                    and evidence['workflow']['databaseId'] == source_run
                    and evidence['workflow']['headSha'] == revision
                    and evidence['workflow']['status'] == 'completed'
                    and evidence['workflow']['conclusion'] == 'success'
                    and not evidence['errors'] and not evidence['workflow_failures']
                    and evidence['oracle_manifest_sha256'] == oracle_hash
                    and set(evidence['cells']) == cells
                    and all(c['status'] == 'passed' for c in evidence['cells'].values()),
                    'sixteen-wheel evidence mismatch or incomplete')
    provenance = release.read(assets / 'python-artifacts.json')
    release.require(provenance['revision'] == revision and provenance['run_id'] == source_run,
                    'Python artifact provenance mismatch')
    distributions = {n for n in expected if n.endswith(('.whl', '.tar.gz'))}
    release.require(set(provenance['artifacts']) == distributions and
                    all(r['sha256'] == hashes[n] for n, r in provenance['artifacts'].items()),
                    'distributed wheel/source bytes differ from accepted provenance')
    return {'revision': revision, 'source_run': source_run, 'wheel_cells': len(cells), 'sha256': hashes}


def verify_produced_assets(produced, proof, run):
    jobs = {'Exact-source release acceptance', 'Assemble checked release assets',
            'Create reviewable GitHub release draft'}
    jobs.update('Build (' + triple + ')' for triple in (
        'x86_64-unknown-linux-gnu', 'aarch64-unknown-linux-gnu',
        'aarch64-apple-darwin', 'x86_64-pc-windows-msvc'))
    release.check_run(run, proof['revision'], jobs)
    release.require({p.name for p in produced.iterdir()} == set(proof['sha256']),
                    'producer/downloaded asset inventories differ')
    release.require(all((produced / n).is_file() and not (produced / n).is_symlink()
                        and release.wheels.digest(produced / n) == h
                        for n, h in proof['sha256'].items()),
                    'actual draft bytes differ from the release producer')


def arrange(raw, output):
    """Copy producer receipts unchanged into the existing verifier's layout."""
    for minor in release.wheels.PYTHONS:
        root = raw / ('glibc228-linux-x86_64-' + minor) / 'evidence'
        shutil.copytree(root / 'build', output / 'builds' / ('linux-x86_64-' + minor))
        shutil.copytree(root / 'gate', output / 'gates' / minor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'artifacts', 'oracles', 'assets', 'produced-assets', 'release-run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--source-run', type=int, required=True)
    parser.add_argument('--oracle-sha256', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'incomplete', 'revision': args.revision, 'source_run': args.source_run,
               'runtime_run': os.environ.get('VERIFIED_GLIBC_RUN'),
               'verification_revision': os.environ.get('GITHUB_SHA')}
    try:
        proof = verify_assets(args.assets, args.repository, args.source_run, args.oracle_sha256)
        release.require(proof['revision'] == args.revision, 'selected candidate differs from checkout')
        release.require(release.wheels.digest(args.oracles / 'manifest.json') == args.oracle_sha256,
                        'selected oracle manifest digest mismatch')
        release_run = release.read(args.release_run)
        verify_produced_assets(args.produced_assets, proof, release_run)
        summary['release_run'] = release_run['databaseId']
        release.save(args.output / 'assets.json', proof)
        arrange(args.artifacts, args.output)
        result = release.verify_glibc(args.output / 'gates', args.output / 'builds', args.repository,
                                      args.oracles, args.assets)
        release.save(args.output / 'glibc.json', result)
        release.require(result['status'] == 'glibc_release_wheels_passed', 'full glibc verification incomplete')
        summary.update(status='release_assets_and_glibc_passed', wheel_cells=proof['wheel_cells'],
                       assets_sha256=proof['sha256'],
                       glibc={p: {'status': c['status'], 'wheels': c['wheels']}
                              for p, c in result['cells'].items()})
    except (OSError, ValueError, KeyError, TypeError, shutil.Error) as error:
        summary['error'] = str(error)
    release.save(args.output / 'summary.json', summary)
    print('RELEASE_VERIFICATION_JSON=' + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary['status'] == 'release_assets_and_glibc_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
