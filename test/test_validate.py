"""Regression tests for gate orchestration; no Julia/Cargo/CUDA needed."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate


class ValidationTests(unittest.TestCase):
    def test_worker_records_actual_invocation_and_thread_overrides(self):
        import parallel_group_tests as scheduler
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "worker.log"
            with patch.object(scheduler.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)):
                bucket, rc = scheduler.run_bucket("rust", 2, ["file.jl::item"], log, n_workers=4)
            metadata = json.loads(log.with_suffix(".json").read_text())
            self.assertEqual((bucket, rc), (2, 7))
            self.assertEqual(metadata["exit_code"], 7)
            self.assertEqual(metadata["items"], ["file.jl::item"])
            self.assertEqual(metadata["environment_overrides"]["JULIA_NUM_THREADS"],
                             str(scheduler._blas_threads_for(4)))
            self.assertEqual(metadata["argv"], scheduler.julia_bucket_command(log))

    def test_rejects_empty_or_unknown_selection_and_invalid_workers(self):
        for argv in (["--groups", "typo"], ["--groups"],
                     ["--max-workers", "0"], ["--all", "--groups", "rust"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    validate.arguments(argv)
                self.assertEqual(error.exception.code, 2)

    def test_normalizes_groups_and_requires_cuda_coverage(self):
        args = validate.arguments(["--groups", "sim-interface", "sim_interface", "--cuda"])
        self.assertEqual(args.groups, ["sim_interface", "rust"])
        self.assertEqual(validate.arguments(["--all"]).groups, validate.GROUPS)

    def test_environment_is_explicit_and_does_not_record_unrelated_secrets(self):
        with patch.dict(os.environ, {"RUSTFLAGS": "-C opt-level=2",
                                    "CARGO_ENCODED_RUSTFLAGS": "unexpected",
                                    "CARGO_BUILD_TARGET": "foreign-target",
                                    "PRIVATE_TOKEN": "secret",
                                    "AMALTHEA_REQUIRE_CUDA_TESTS": "1"}, clear=True):
            cpu = validate.validation_environment(False)
            cuda = validate.validation_environment(True)
        self.assertEqual(cpu["AMALTHEA_CUDA_BUILD"], "off")
        self.assertEqual(cpu["AMALTHEA_REQUIRE_CUDA_TESTS"], "0")
        self.assertEqual(cuda["AMALTHEA_CUDA_BUILD"], "required")
        self.assertEqual(cuda["AMALTHEA_REQUIRE_CUDA_TESTS"], "1")
        self.assertEqual(cpu["RUSTFLAGS"], "-C opt-level=2")
        self.assertNotIn("CARGO_ENCODED_RUSTFLAGS", cpu)
        self.assertNotIn("CARGO_BUILD_TARGET", cpu)
        self.assertNotIn("PRIVATE_TOKEN", validate.recorded_environment(cpu))

    def simulate(self, failure=None, empty=False, repeat=False, missing_library=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "amalthea" / "target" / "release").mkdir(parents=True)
        if not missing_library:
            for name in ("libamalthea.so", "libamalthea.dylib", "amalthea.dll"):
                (root / "amalthea" / "target" / "release" / name).write_bytes(b"local test library")
        calls = []

        def run(command, *, cwd, env, stdout, stderr):
            calls.append(command)
            name = Path(stdout.name).stem
            stdout.write("retained output for " + name + "\n")
            if failure == "interrupt" and name == "cargo-tests":
                raise KeyboardInterrupt
            if failure == "missing-tool" and name == "julia-version":
                raise FileNotFoundError("julia not found")
            return subprocess.CompletedProcess(command, 7 if name == failure else 0)

        with patch.object(validate, "REPO_ROOT", root), \
                patch.object(validate.platform, "platform", return_value="test platform"), \
                patch.object(validate, "discover_group_items", return_value=[] if empty else ["item"]), \
                patch.object(validate.subprocess, "run", side_effect=run), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = validate.main(["--log-dir", str(root / "logs")])
            if repeat:
                self.assertEqual(validate.main(["--log-dir", str(root / "logs")]), rc)
        summaries = sorted((root / "logs").glob("*/summary.json"))
        return rc, [json.loads(p.read_text()) for p in summaries], calls

    def test_success_retains_output_and_uses_unique_directories(self):
        rc, summaries, calls = self.simulate(repeat=True)
        self.assertEqual(rc, 0)
        self.assertEqual(len(summaries), 2)
        for summary in summaries:
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(len(summary["library"]["sha256"]), 64)
            for command in summary["commands"]:
                self.assertEqual(command["exit_code"], 0)
                self.assertIn("retained output", Path(command["log"]).read_text())
            preflight = next(c for c in summary["commands"] if c["name"] == "julia-preflight")
            self.assertIn("realpath(pathof(Amalthea))", preflight["argv"][-2])
            self.assertIn("realpath(Amalthea.RK45._LIBAMALTHEA_RK45)", preflight["argv"][-2])

    def test_build_and_preflight_failure_stop_dependent_tests(self):
        for failure in ("build", "julia-preflight", "missing-tool"):
            with self.subTest(failure=failure):
                rc, summaries, calls = self.simulate(failure=failure)
                self.assertNotEqual(rc, 0)
                self.assertEqual(summaries[0]["status"], "failed")
                self.assertFalse(any("test" in cmd for cmd in calls))

    def test_cargo_failure_still_collects_julia_results(self):
        rc, summaries, calls = self.simulate(failure="cargo-tests")
        self.assertEqual(rc, 1)
        self.assertEqual(summaries[0]["commands"][-1]["name"], "julia-tests")
        self.assertEqual(summaries[0]["status"], "failed")

    def test_required_hdf5_failure_cannot_pass(self):
        rc, summaries, calls = self.simulate(failure="rust-hdf5-tests")
        self.assertEqual(rc, 1)
        self.assertEqual(summaries[0]["status"], "failed")
        stage = next(c for c in summaries[0]["commands"] if c["name"] == "rust-hdf5-tests")
        self.assertEqual(stage["exit_code"], 7)
        self.assertTrue(stage["argv"][-1].endswith("test/run_rust_hdf5.jl"))
        self.assertEqual(summaries[0]["commands"][-1]["name"], "julia-tests")

    def test_julia_failure_and_interrupt_cannot_pass(self):
        for failure, expected in (("julia-tests", 1), ("interrupt", 130)):
            rc, summaries, calls = self.simulate(failure=failure)
            self.assertEqual(rc, expected)
            self.assertEqual(summaries[0]["status"], "failed")

    def test_empty_discovery_cannot_pass(self):
        rc, summaries, calls = self.simulate(empty=True)
        self.assertEqual(rc, 1)
        self.assertIn("no discovered test items", summaries[0]["error"])
        self.assertFalse(any("build" in cmd for cmd in calls))

    def test_missing_library_cannot_be_reported_as_skipped_success(self):
        rc, summaries, calls = self.simulate(missing_library=True)
        self.assertEqual(rc, 1)
        self.assertEqual(summaries[0]["status"], "failed")
        self.assertEqual(summaries[0]["commands"][-1]["name"], "build")


if __name__ == "__main__":
    unittest.main()
