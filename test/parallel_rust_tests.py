#!/usr/bin/env python3
"""Run the `rust` Julia test group as load-balanced parallel worker processes.

Splits the rust-tagged test files across at most 10 worker processes using
LPT (longest-processing-time-first) bin packing, keyed on historical
per-file durations in rust_test_timings.txt, so workers finish at roughly
the same time instead of one long file straggling behind many short ones.
Each worker runs its assigned files in one julia process via
run_rust_bucket.jl (required: @run_package_tests resolves the package root
from its own file location, so it must live in test/, not a scratch dir).

Usage: julia --project must be runnable from the repo root.
    python3 test/parallel_rust_tests.py [--max-workers N] [--update-timings]

Exit code is 0 iff every worker's Pass == Total (no failures/errors).
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "test"
TIMINGS_FILE = TEST_DIR / "rust_test_timings.txt"
BUCKET_RUNNER = TEST_DIR / "run_rust_bucket.jl"
DEFAULT_MAX_WORKERS = 10


def discover_rust_files():
    files = []
    for path in sorted(TEST_DIR.glob("*.jl")):
        if path.name in ("run_rust_bucket.jl",):
            continue
        if "tags=[:rust]" in path.read_text():
            files.append(path.name)
    return files


def load_timings():
    timings = {}
    if TIMINGS_FILE.exists():
        for line in TIMINGS_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, secs = line.rsplit(maxsplit=1)
            timings[name] = float(secs)
    return timings


def lpt_bins(files, timings, max_workers):
    median = sorted(timings.values())[len(timings) // 2] if timings else 10.0
    weighted = sorted(
        ((timings.get(f, median), f) for f in files), reverse=True
    )
    n_bins = max(1, min(max_workers, len(files)))
    bins = [[] for _ in range(n_bins)]
    loads = [0.0] * n_bins
    for dur, f in weighted:
        i = loads.index(min(loads))
        bins[i].append(f)
        loads[i] += dur
    return bins, loads


SUMMARY_RE = re.compile(r"^Package\s*\|\s*(?:(\d+)\s+)?(\d+)\s")


def run_bucket(bucket_id, files, log_path):
    with open(log_path, "w") as log:
        proc = subprocess.run(
            ["julia", "--project", str(BUCKET_RUNNER), *files],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return bucket_id, proc.returncode


def parse_summary(log_path):
    text = log_path.read_text()
    for line in text.splitlines():
        m = SUMMARY_RE.match(line.strip())
        if m:
            passed = int(m.group(1)) if m.group(1) is not None else int(m.group(2))
            total = int(m.group(2))
            return passed, total
    return None, None


def write_timings(durations):
    header = TIMINGS_FILE.read_text().splitlines() if TIMINGS_FILE.exists() else []
    header = [l for l in header if l.strip().startswith("#")]
    lines = header + [f"{name} {secs:.1f}" for name, secs in sorted(durations.items())]
    TIMINGS_FILE.write_text("\n".join(lines) + "\n")


def update_timings(files, max_workers, log_dir):
    """Re-measure each file's wall-clock duration in its own process (one
    file per bucket, run up to max_workers at a time), then overwrite
    rust_test_timings.txt. Run on an otherwise-idle machine for stable
    numbers — measurements are wall-clock, so contention from other bins
    running concurrently inflates them."""
    durations = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run_one(name):
        log_path = log_dir / f"timing_{name}.log"
        start = time.time()
        with open(log_path, "w") as log:
            subprocess.run(
                ["julia", "--project", str(BUCKET_RUNNER), name],
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return name, time.time() - start

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(run_one, f): f for f in files}
        for fut in as_completed(futs):
            name, dur = fut.result()
            durations[name] = dur
            print(f"  measured {name}: {dur:.1f}s")

    write_timings(durations)
    print(f"Updated {TIMINGS_FILE} with {len(durations)} fresh timings.")
    return durations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    ap.add_argument("--log-dir", default=str(REPO_ROOT / ".rust_test_logs"))
    ap.add_argument("--update-timings", action="store_true",
                     help="Re-measure each file's duration individually "
                          "(one file per process) and overwrite "
                          "rust_test_timings.txt before scheduling.")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    files = discover_rust_files()
    if not files:
        print("No rust-tagged test files found.", file=sys.stderr)
        return 1

    if args.update_timings:
        print(f"Re-measuring {len(files)} files individually "
              f"(max {args.max_workers} concurrent)...")
        timings = update_timings(files, args.max_workers, log_dir)
    else:
        timings = load_timings()
    bins, loads = lpt_bins(files, timings, args.max_workers)

    print(f"Distributing {len(files)} files across {len(bins)} workers "
          f"(max {args.max_workers}):")
    for i, (b, load) in enumerate(zip(bins, loads)):
        print(f"  worker {i}: {len(b)} files, est. {load:.1f}s")

    start = time.time()
    results = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=len(bins)) as pool:
        futs = {
            pool.submit(run_bucket, i, b, log_dir / f"worker{i}.log"): i
            for i, b in enumerate(bins) if b
        }
        for fut in as_completed(futs):
            results.append(fut.result())
    elapsed = time.time() - start

    total_pass = 0
    total_all = 0
    any_fail = False
    for bucket_id, rc in sorted(results):
        log_path = log_dir / f"worker{bucket_id}.log"
        passed, total = parse_summary(log_path)
        ok = rc == 0 and passed is not None and passed == total
        any_fail = any_fail or not ok
        print(f"worker {bucket_id}: rc={rc} pass={passed} total={total} "
              f"{'OK' if ok else 'FAIL'} (log: {log_path})")
        if passed is not None:
            total_pass += passed
            total_all += total

    print(f"\nTOTAL: {total_pass}/{total_all} in {elapsed:.1f}s wall-clock "
          f"across {len(bins)} workers")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
