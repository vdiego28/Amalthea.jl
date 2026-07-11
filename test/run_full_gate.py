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
from parallel_group_tests import run_group, DEFAULT_MAX_WORKERS

REPO_ROOT = Path(__file__).resolve().parent.parent

GROUPS = [
    "physics", "rust", "sim_interface", "sim_multimode",
    "sim_propagation", "io", "fields",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    ap.add_argument("--log-dir", default=str(REPO_ROOT / ".rust_test_logs"))
    ap.add_argument("--update-timings", action="store_true")
    ap.add_argument("--groups", nargs="+", default=GROUPS,
                     help="Subset of groups to run (default: all 7).")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    start = time.time()
    any_fail = False
    summary = []
    for group in args.groups:
        rc, passed, total, elapsed = run_group(
            group, args.max_workers, log_dir, args.update_timings
        )
        any_fail = any_fail or rc != 0
        summary.append((group, rc, passed, total, elapsed))

    total_elapsed = time.time() - start
    print("=" * 60)
    print("FULL GATE SUMMARY")
    for group, rc, passed, total, elapsed in summary:
        status = "OK" if rc == 0 else "FAIL"
        print(f"  {group:16s} {passed:>6}/{total:<6} {elapsed:7.1f}s  {status}")
    print(f"\nTOTAL wall-clock: {total_elapsed:.1f}s across {len(args.groups)} groups")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
