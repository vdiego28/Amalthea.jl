#!/usr/bin/env python3
"""Run all 7 CI test groups, each load-balanced across parallel workers.

Groups run one at a time (not concurrently with each other) so each gets
the full worker budget (--max-workers, default 10) without oversubscribing
the machine's cores — running two groups' worker pools simultaneously would
mean 20 Julia processes competing for e.g. 12 cores.

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

GROUPS = [
    "physics", "rust", "sim_interface", "sim_multimode",
    "sim_propagation", "io", "fields",
]

# Batches of groups to run *concurrently* (one batch at a time, batches
# themselves sequential). Per-worker BLAS threads are capped against each
# batch's *combined* worker count (see run_groups/_blas_threads_for), so
# batching is safe as long as the sum of workers across a batch stays near
# `os.cpu_count()`.
#
# physics and rust each already saturate DEFAULT_MAX_WORKERS (10) workers
# on their own, so they stay solo. The other five groups have few enough
# files that their worker counts (2-6) sum well within a 12-core budget:
# sim-multimode's single 334s file dominates its group's wall-clock
# regardless of pairing (its other 3 workers finish in under 130s), so
# running sim-interface and sim-propagation alongside it is close to free
# — and io+fields is the pairing suggested directly by the user.
DEFAULT_BATCHES = [
    ["physics"],
    ["rust"],
    ["sim_multimode", "sim_interface", "sim_propagation"],
    ["io", "fields"],
]


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
                     help="Subset of groups to run (default: all 7).")
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
            bins = prepare_group_bins(group, args.max_workers, log_dir, args.update_timings)
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
