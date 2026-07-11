#!/usr/bin/env python3
"""Run the `rust` Julia test group as load-balanced parallel worker processes.

Thin backward-compatible wrapper around parallel_group_tests.py (group=rust)
— kept as its own entry point because CLAUDE.md and existing tooling invoke
it by this name, and `rust`'s timings file predates the generic script.
See parallel_group_tests.py for the load-balancing/measurement logic itself.

Usage: julia --project must be runnable from the repo root.
    python3 test/parallel_rust_tests.py [--max-workers N] [--update-timings]

Exit code is 0 iff every worker's Pass == Total (no failures/errors).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parallel_group_tests import run_group, DEFAULT_MAX_WORKERS

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    ap.add_argument("--log-dir", default=str(REPO_ROOT / ".rust_test_logs"))
    ap.add_argument("--update-timings", action="store_true",
                     help="Re-measure each file's duration individually "
                          "(one file per process) and overwrite "
                          "rust_test_timings.txt before scheduling.")
    args = ap.parse_args()

    rc, _, _, _ = run_group("rust", args.max_workers, Path(args.log_dir), args.update_timings)
    return rc


if __name__ == "__main__":
    sys.exit(main())
