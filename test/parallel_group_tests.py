#!/usr/bin/env python3
"""Run any CI test group as load-balanced parallel worker processes.

Splits a group's tagged test files across at most 10 worker processes using
LPT (longest-processing-time-first) bin packing, keyed on historical
per-file durations in test/<group>_test_timings.txt, so workers finish at
roughly the same time instead of one long file straggling behind many short
ones. Each worker runs its assigned files in one julia process via
run_group_bucket.jl (required: @run_package_tests resolves the package root
from its own file location, so it must live in test/, not a scratch dir).

Groups with few files (e.g. sim-interface, 2 files) get correspondingly
little benefit from this — parallelism is capped at the file count, not
--max-workers.

Usage: julia --project must be runnable from the repo root.
    python3 test/parallel_group_tests.py --group rust [--max-workers N] [--update-timings]

Exit code is 0 iff every worker's Pass == Total (no failures/errors).
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "test"
BUCKET_RUNNER = TEST_DIR / "run_group_bucket.jl"
DEFAULT_MAX_WORKERS = 10

# `rust`'s timings file predates this generic script and is kept under its
# original name for backward compatibility (CLAUDE.md, existing tooling).
TIMINGS_FILE_OVERRIDE = {"rust": TEST_DIR / "rust_test_timings.txt"}


def timings_file_for(group):
    return TIMINGS_FILE_OVERRIDE.get(group, TEST_DIR / f"{group}_test_timings.txt")


def tag_sym(group):
    return group.replace("-", "_")


def discover_group_files(group):
    tag = tag_sym(group)
    needle = f":{tag}"
    files = []
    for path in sorted(TEST_DIR.glob("*.jl")):
        if path.name in ("run_group_bucket.jl", "run_rust_bucket.jl"):
            continue
        text = path.read_text()
        if re.search(rf"tags\s*=\s*\[[^\]]*{re.escape(needle)}\b", text):
            files.append(path.name)
    return files


def load_timings(timings_file):
    timings = {}
    if timings_file.exists():
        for line in timings_file.read_text().splitlines():
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


def run_bucket(group, bucket_id, files, log_path):
    env = {**os.environ, "LUNA_BUCKET_TAG": tag_sym(group), "LUNA_BUCKET_FILES": "\n".join(files)}
    with open(log_path, "w") as log:
        proc = subprocess.run(
            ["julia", "--project", str(BUCKET_RUNNER)],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return bucket_id, proc.returncode


def parse_summary(log_path):
    """Reads the TestItemRunner "Test Summary: | <cols...>" header and the
    following "Package | <values...>" row. Column set varies (Pass/Fail/
    Error/Broken/Total/Time, only the ones that occurred appear) — parse by
    column name, not position, since e.g. a Broken column shifts Total's
    index. `ok` requires zero Fail/Error, not passed==total: Broken tests
    are intentional expected-failures (see BACKLOG/CLAUDE.md's documented
    "1645/1657+12 broken" physics baseline), not a real gate failure."""
    lines = log_path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Test Summary") and "|" in line:
            cols = line.split("|", 1)[1].split()
            if i + 1 >= len(lines):
                return None, None, False
            vals_part = lines[i + 1].split("|", 1)
            if len(vals_part) != 2:
                continue
            vals = vals_part[1].split()
            data = dict(zip(cols, vals))
            total = int(data.get("Total", 0))
            passed = int(data.get("Pass", 0))
            fail = int(data.get("Fail", 0)) + int(data.get("Error", 0))
            return passed, total, fail == 0
    return None, None, False


def write_timings(timings_file, durations):
    header = timings_file.read_text().splitlines() if timings_file.exists() else []
    header = [l for l in header if l.strip().startswith("#")]
    lines = header + [f"{name} {secs:.1f}" for name, secs in sorted(durations.items())]
    timings_file.write_text("\n".join(lines) + "\n")


def update_timings(group, files, max_workers, log_dir, timings_file):
    """Re-measure each file's wall-clock duration in its own process (one
    file per bucket, run up to max_workers at a time), then overwrite the
    group's timings file. Run on an otherwise-idle machine for stable
    numbers — measurements are wall-clock, so contention from other bins
    running concurrently inflates them."""
    durations = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run_one(name):
        log_path = log_dir / f"timing_{name}.log"
        env = {**os.environ, "LUNA_BUCKET_TAG": tag_sym(group), "LUNA_BUCKET_FILES": name}
        start = time.time()
        with open(log_path, "w") as log:
            subprocess.run(
                ["julia", "--project", str(BUCKET_RUNNER)],
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        return name, time.time() - start

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(run_one, f): f for f in files}
        for fut in as_completed(futs):
            name, dur = fut.result()
            durations[name] = dur
            print(f"  measured {name}: {dur:.1f}s")

    write_timings(timings_file, durations)
    print(f"Updated {timings_file} with {len(durations)} fresh timings.")
    return durations


def run_group(group, max_workers, log_dir, do_update_timings):
    log_dir.mkdir(parents=True, exist_ok=True)
    timings_file = timings_file_for(group)

    files = discover_group_files(group)
    if not files:
        print(f"No {group!r}-tagged test files found.", file=sys.stderr)
        return 1, 0, 0, 0.0

    if do_update_timings:
        print(f"[{group}] Re-measuring {len(files)} files individually "
              f"(max {max_workers} concurrent)...")
        timings = update_timings(group, files, max_workers, log_dir, timings_file)
    else:
        timings = load_timings(timings_file)
    bins, loads = lpt_bins(files, timings, max_workers)

    print(f"[{group}] Distributing {len(files)} files across {len(bins)} workers "
          f"(max {max_workers}):")
    for i, (b, load) in enumerate(zip(bins, loads)):
        print(f"  worker {i}: {len(b)} files, est. {load:.1f}s")

    start = time.time()
    results = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=len(bins)) as pool:
        futs = {
            pool.submit(run_bucket, group, i, b, log_dir / f"{group}_worker{i}.log"): i
            for i, b in enumerate(bins) if b
        }
        for fut in as_completed(futs):
            results.append(fut.result())
    elapsed = time.time() - start

    total_pass = 0
    total_all = 0
    any_fail = False
    for bucket_id, rc in sorted(results):
        log_path = log_dir / f"{group}_worker{bucket_id}.log"
        passed, total, summary_ok = parse_summary(log_path)
        ok = rc == 0 and passed is not None and summary_ok
        any_fail = any_fail or not ok
        print(f"[{group}] worker {bucket_id}: rc={rc} pass={passed} total={total} "
              f"{'OK' if ok else 'FAIL'} (log: {log_path})")
        if passed is not None:
            total_pass += passed
            total_all += total

    print(f"[{group}] TOTAL: {total_pass}/{total_all} in {elapsed:.1f}s wall-clock "
          f"across {len(bins)} workers\n")
    return (1 if any_fail else 0), total_pass, total_all, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="rust")
    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    ap.add_argument("--log-dir", default=str(REPO_ROOT / ".rust_test_logs"))
    ap.add_argument("--update-timings", action="store_true",
                     help="Re-measure each file's duration individually "
                          "(one file per process) and overwrite the "
                          "group's timings file before scheduling.")
    args = ap.parse_args()

    rc, total_pass, total_all, elapsed = run_group(
        args.group, args.max_workers, Path(args.log_dir), args.update_timings
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
