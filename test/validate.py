#!/usr/bin/env python3
"""Build and validate this checkout, retaining a separate evidence bundle per run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from parallel_group_tests import discover_group_items
from run_full_gate import GROUPS, REPO_ROOT


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--groups", nargs="+", help="Maintained groups (default: rust)")
    selection.add_argument("--all", action="store_true", help="Run all eight Julia groups")
    parser.add_argument("--cuda", action="store_true", help="Require CUDA build and hardware tests")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--log-dir", type=Path,
                        default=REPO_ROOT / ".rust_test_logs" / "validation",
                        help="Parent directory for unique run directories")
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    args.groups = list(dict.fromkeys(
        GROUPS if args.all else [g.replace("-", "_") for g in (args.groups or ["rust"]) ]
    ))
    unknown = set(args.groups) - set(GROUPS)
    if unknown:
        parser.error("unknown groups: " + ", ".join(sorted(unknown)))
    if args.cuda and "rust" not in args.groups:
        args.groups.append("rust")
    return args


def validation_environment(cuda):
    env = os.environ.copy()
    env.update(CARGO_TARGET_DIR=str(REPO_ROOT / "amalthea" / "target"),
               AMALTHEA_CUDA_BUILD="required" if cuda else "off",
               AMALTHEA_REQUIRE_CUDA_TESTS="1" if cuda else "0",
               AMALTHEA_RUST_SKIP_DOWNLOAD="1",
               RUSTFLAGS=env.get("RUSTFLAGS", ""), PYTHONUNBUFFERED="1")
    # Encoded flags take precedence over RUSTFLAGS; use one explicit flag source.
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    env.pop("CARGO_BUILD_TARGET", None)
    return env


def recorded_environment(env):
    exact = {"PATH", "RUSTFLAGS", "CARGO_TARGET_DIR", "JULIA_NUM_THREADS",
             "JULIA_DEPOT_PATH", "JULIA_LOAD_PATH", "OPENBLAS_NUM_THREADS",
             "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS",
             "CUDA_HOME", "CUDA_PATH", "NVCC", "HDF5_USE_FILE_LOCKING"}
    return {k: v for k, v in sorted(env.items())
            if k in exact or k.startswith(("AMALTHEA_", "LUNA_"))}


def preflight_code():
    return r'''
using Amalthea, Libdl
root = realpath(ARGS[1])
@assert realpath(pathof(Amalthea)) == realpath(joinpath(root, "src", "Amalthea.jl")) "Wrong package checkout"
name = Sys.iswindows() ? "amalthea.dll" : Sys.isapple() ? "libamalthea.dylib" : "libamalthea.so"
expected = realpath(joinpath(root, "amalthea", "target", "release", name))
@assert realpath(Amalthea.RK45._LIBAMALTHEA_RK45) == expected "Wrong native library"
handle = Libdl.dlopen(expected)
try
    Libdl.dlsym(handle, :init_native_sim)
    Libdl.dlsym(handle, :native_step)
finally
    Libdl.dlclose(handle)
end
println("Validated package: ", pathof(Amalthea))
println("Validated native library: ", expected)
'''


class Evidence:
    def __init__(self, directory, metadata, env):
        self.directory = directory
        self.metadata = {**metadata, "status": "running", "commands": []}
        self.env = env
        self.save()

    def save(self):
        temporary = self.directory / "summary.json.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "summary.json")

    def run(self, name, command, cwd=REPO_ROOT):
        log = self.directory / (name + ".log")
        record = {"name": name, "argv": command, "cwd": str(cwd),
                  "log": str(log), "exit_code": None}
        self.metadata["commands"].append(record)
        self.save()
        print(f"Running {name}; log: {log}", flush=True)
        start = time.monotonic()
        try:
            with log.open("w", encoding="utf-8") as output:
                result = subprocess.run(command, cwd=cwd, env=self.env,
                                        stdout=output, stderr=subprocess.STDOUT)
            record["exit_code"] = result.returncode
            return result.returncode
        finally:
            record["elapsed_seconds"] = time.monotonic() - start
            self.save()


def main(argv=None):
    args = arguments(argv)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    directory = Path(tempfile.mkdtemp(prefix=stamp, dir=args.log_dir)).resolve()
    env = validation_environment(args.cuda)
    evidence = Evidence(directory, {
        "started_utc": stamp.rstrip("-"), "repo_root": str(REPO_ROOT),
        "platform": platform.platform(), "python": sys.version,
        "groups": args.groups, "cuda_required": args.cuda,
        "max_workers": args.max_workers, "environment": recorded_environment(env),
    }, env)
    print(f"Validation evidence: {directory}", flush=True)
    code = 1
    try:
        for name, cmd in [
            ("revision", ["git", "rev-parse", "HEAD"]),
            ("worktree", ["git", "status", "--porcelain=v1"]),
            ("rust-version", ["rustc", "-vV"]),
            ("cargo-version", ["cargo", "--version"]),
            ("julia-version", ["julia", "--startup-file=no", "--version"]),
        ]:
            if evidence.run(name, cmd):
                raise RuntimeError(f"{name} failed")
        evidence.metadata["revision"] = (directory / "revision.log").read_text().strip()
        evidence.metadata["worktree_status"] = (directory / "worktree.log").read_text()
        inventory = {group: discover_group_items(group) for group in args.groups}
        if any(not items for items in inventory.values()):
            raise RuntimeError("Requested group has no discovered test items")
        (directory / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
        if evidence.run("build", ["cargo", "build", "--release"], REPO_ROOT / "amalthea"):
            raise RuntimeError("Release build failed; tests were not launched")
        name = "amalthea.dll" if sys.platform == "win32" else (
            "libamalthea.dylib" if sys.platform == "darwin" else "libamalthea.so")
        library = REPO_ROOT / "amalthea" / "target" / "release" / name
        evidence.metadata["library"] = {"path": str(library.resolve()),
                                        "sha256": hashlib.sha256(library.read_bytes()).hexdigest()}
        if evidence.run("julia-preflight", ["julia", "--startup-file=no", "--project=" + str(REPO_ROOT),
                                           "-e", preflight_code(), str(REPO_ROOT)]):
            raise RuntimeError("Checkout/library preflight failed; tests were not launched")
        rust_rc = evidence.run("cargo-tests", ["cargo", "test", "--release"], REPO_ROOT / "amalthea")
        hdf5_rc = evidence.run("rust-hdf5-tests", ["julia", "--startup-file=no",
                                "--project=" + str(REPO_ROOT),
                                str(REPO_ROOT / "test" / "run_rust_hdf5.jl")])
        julia_rc = evidence.run("julia-tests", [sys.executable, str(REPO_ROOT / "test" / "run_full_gate.py"),
                               "--groups", *args.groups, "--max-workers", str(args.max_workers),
                               "--log-dir", str(directory / "workers")])
        code = 1 if rust_rc or hdf5_rc or julia_rc else 0
    except KeyboardInterrupt:
        evidence.metadata["error"] = "Interrupted"
        code = 130
    except (OSError, RuntimeError) as exc:
        evidence.metadata["error"] = str(exc)
        print(str(exc), file=sys.stderr)
    finally:
        evidence.metadata.update(status="passed" if code == 0 else "failed",
                                 exit_code=code,
                                 finished_utc=datetime.now(timezone.utc).isoformat())
        evidence.save()
        print(f"Validation {evidence.metadata['status']}: {directory / 'summary.json'}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
