"""Unit coverage for the local/GitHub LPT test scheduler."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCHEDULER_PATH = Path(__file__).with_name("parallel_group_tests.py")
SPEC = importlib.util.spec_from_file_location("parallel_group_tests", SCHEDULER_PATH)
SCHEDULER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCHEDULER)

FULL_GATE_PATH = Path(__file__).with_name("run_full_gate.py")
FULL_GATE_SPEC = importlib.util.spec_from_file_location("run_full_gate", FULL_GATE_PATH)
FULL_GATE = importlib.util.module_from_spec(FULL_GATE_SPEC)
FULL_GATE_SPEC.loader.exec_module(FULL_GATE)


class SchedulerTests(unittest.TestCase):
    def test_declaration_discovery_reads_utf8_explicitly(self):
        source = '@testitem "Unicode ω" tags=[:physics] begin\nend\n'
        path = Path("unicode_test.jl")

        with mock.patch.object(Path, "read_text", return_value=source) as read_text:
            self.assertEqual(
                SCHEDULER._testitems_in(path),
                [("Unicode ω", {"physics"})],
            )

        read_text.assert_called_once_with(encoding="utf-8")

    def test_multi_item_files_are_independently_schedulable(self):
        items = SCHEDULER.discover_group_items("sim-interface")
        interface_items = [
            item for item in items if SCHEDULER.item_file(item) == "test_interface.jl"
        ]

        self.assertEqual(len(interface_items), 7)
        self.assertTrue(
            all(SCHEDULER.ITEM_SEPARATOR in item for item in interface_items)
        )
        self.assertIn("test_greek_aliases.jl", items)

    def test_legacy_file_timing_is_divided_between_items(self):
        items = ["test_many.jl::first", "test_many.jl::second", "test_one.jl"]
        resolved = SCHEDULER.resolved_item_timings(
            items, {"test_many.jl": 20.0, "test_one.jl": 3.0}
        )

        self.assertEqual(
            resolved,
            {
                "test_many.jl::first": 10.0,
                "test_many.jl::second": 10.0,
                "test_one.jl": 3.0,
            },
        )

    def test_safe_log_stems_cannot_escape_or_alias(self):
        identities = [
            "amalthea/tests/test_gpu_cuda.jl",
            "test_interface.jl::Interface LunaPulse",
            "test_interface.jl::Interface/LunaPulse",
        ]
        stems = [SCHEDULER.safe_log_stem(identity) for identity in identities]

        self.assertEqual(len(stems), len(set(stems)))
        self.assertTrue(all("/" not in stem and "\\" not in stem for stem in stems))

    def test_ci_command_preserves_julia_runtest_safety_and_coverage_flags(self):
        log_path = Path("/tmp/physics_worker0.log")
        command = SCHEDULER.julia_bucket_command(log_path, ci=True)

        self.assertIn("--check-bounds=yes", command)
        self.assertIn("--depwarn=yes", command)
        self.assertIn("--compiled-modules=yes", command)
        self.assertIn("--inline=yes", command)
        self.assertIn("--code-coverage=user", command)
        self.assertIn(
            "--code-coverage=/tmp/physics_worker0.coverage.info", command
        )
        self.assertEqual(command[-1], str(SCHEDULER.BUCKET_RUNNER))
        self.assertNotIn("--check-bounds=yes",
                         SCHEDULER.julia_bucket_command(log_path))

    def test_lpt_places_heaviest_items_on_different_workers(self):
        items = ["heavy-a", "heavy-b", "small-a", "small-b"]
        timings = {
            "heavy-a": 10.0,
            "heavy-b": 9.0,
            "small-a": 2.0,
            "small-b": 1.0,
        }

        bins, loads = SCHEDULER.lpt_bins(items, timings, 2)

        self.assertTrue(
            all(len(set(bucket) & {"heavy-a", "heavy-b"}) == 1 for bucket in bins)
        )
        self.assertLessEqual(max(loads) - min(loads), 2.0)

    def test_failed_timing_probe_does_not_overwrite_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            timings_path = tmp_path / "timings.txt"
            timings_path.write_text("existing.jl 12.0\n", encoding="utf-8")
            failed = mock.Mock(returncode=1)

            with mock.patch.object(SCHEDULER.subprocess, "run", return_value=failed):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(RuntimeError):
                        SCHEDULER.update_timings(
                            "rust", ["broken.jl"], 1, tmp_path, timings_path
                        )

            self.assertEqual(
                timings_path.read_text(encoding="utf-8"),
                "existing.jl 12.0\n",
            )

    def test_default_local_batches_do_not_oversubscribe_reference_budget(self):
        for batch in FULL_GATE.DEFAULT_BATCHES:
            if len(batch) == 1:
                continue
            worker_count = sum(
                min(
                    FULL_GATE.DEFAULT_MAX_WORKERS,
                    FULL_GATE.DEFAULT_BATCH_WORKERS[group],
                    len(SCHEDULER.discover_group_items(group)),
                )
                for group in batch
            )
            self.assertLessEqual(worker_count, FULL_GATE.DEFAULT_MAX_WORKERS)


if __name__ == "__main__":
    unittest.main()
