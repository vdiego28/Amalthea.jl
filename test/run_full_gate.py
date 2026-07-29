#!/usr/bin/env python3
"""Run all 8 maintained CI test groups, each load-balanced across parallel workers.

Large groups run alone; smaller groups are batched while the combined worker
count stays near the machine's core count.

Usage: python3 test/run_full_gate.py [--max-workers N] [--update-timings]

Exit code is 0 iff every group's every worker's Pass == Total.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parallel_group_tests import (
    prepare_group_bins, run_groups, DEFAULT_MAX_WORKERS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GROUPS_FILE = REPO_ROOT / "test" / "test_groups.txt"

GROUPS = [
    line.strip()
    for line in GROUPS_FILE.read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

# Batches of groups to run *concurrently* (one batch at a time, batches
# themselves sequential). Per-worker BLAS threads are capped against each
# batch's *combined* worker count (see run_groups/_blas_threads_for), so
# batching is safe as long as the sum of workers across a batch stays near
# `os.cpu_count()`.
#
# physics and rust each saturate DEFAULT_MAX_WORKERS on their own. Item-level
# scheduling exposes eight interface items, so the concurrent batches also
# need per-group caps; otherwise the third batch would launch 19 Julia
# processes on the 12-core reference machine. The measured two-/four-worker
# caps below keep each combined batch at 10 or fewer processes while their LPT
# loads finish at roughly the same time.
DEFAULT_BATCHES = [
    ["physics"],
    ["rust"],
    ["sim_multimode", "sim_interface", "sim_propagation"],
    ["io", "fields", "examples"],
]
DEFAULT_BATCH_WORKERS = {
    "sim_multimode": 4,
    "sim_interface": 2,
    "sim_propagation": 4,
    "io": 4,
    "fields": 4,
    "examples": 1,
}


def _batches_for(groups):
    """Filter DEFAULT_BATCHES down to the requested --groups subset,
    preserving batching; any requested group missing from the default
    schedule (e.g. a custom hyphenated name) runs in its own solo batch."""
    known = {g for batch in DEFAULT_BATCHES for g in batch}
    batches = [[g for g in batch if g in groups] for batch in DEFAULT_BATCHES]
    batches = [b for b in batches if b]
    extra = [g for g in groups if g not in known]
    batches += [[g] for g in extra]
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    ap.add_argument("--log-dir", default=str(REPO_ROOT / ".rust_test_logs"))
    ap.add_argument("--update-timings", action="store_true")
    ap.add_argument("--groups", nargs="+", default=GROUPS,
                     help="Subset of groups to run (default: all 8).")
    ap.add_argument("--no-batch", action="store_true",
                     help="Run every requested group sequentially and solo "
                          "instead of using DEFAULT_BATCHES' concurrent pairing.")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    batches = [[g] for g in args.groups] if args.no_batch else _batches_for(args.groups)

    start = time.time()
    any_fail = False
    summary = []
    for batch in batches:
        group_bins = {}
        for group in batch:
            worker_limit = (
                args.max_workers
                if args.no_batch or len(batch) == 1
                else min(args.max_workers, DEFAULT_BATCH_WORKERS.get(group, args.max_workers))
            )
            bins = prepare_group_bins(group, worker_limit, log_dir, args.update_timings)
            if bins is not None:
                group_bins[group] = bins
        if not group_bins:
            continue
        results, elapsed = run_groups(group_bins, log_dir)
        for group, (rc, passed, total) in results.items():
            any_fail = any_fail or rc != 0
            summary.append((group, rc, passed, total, elapsed))

    total_elapsed = time.time() - start
    print("=" * 60)
    print("FULL GATE SUMMARY")
    for group, rc, passed, total, elapsed in summary:
        status = "OK" if rc == 0 else "FAIL"
        print(f"  {group:16s} {passed:>6}/{total:<6} {elapsed:7.1f}s  {status}")
    print(f"\nTOTAL wall-clock: {total_elapsed:.1f}s across {len(summary)} groups "
          f"in {len(batches)} batches")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
