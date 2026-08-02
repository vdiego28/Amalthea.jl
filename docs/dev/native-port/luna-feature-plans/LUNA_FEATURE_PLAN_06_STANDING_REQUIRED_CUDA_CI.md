# Luna feature plan 06 — Add standing required-CUDA CI

Status: externally gated. Source: current resume queue item 2 and open GPU CI coverage.

## Required prerequisite

An approved CUDA-equipped runner or hosted GPU service, its repository label,
and authorization to register/use it must exist. If those are absent, stop and
report the exact missing external prerequisite; do not create a workflow that
can never run and call the backlog item complete.

The local RTX 5060 Ti is available for manual strict verification, sometimes
only through approved out-of-sandbox execution. That hardware baseline is not
itself permission to register this workstation as a CI runner and does not
satisfy this prerequisite.

## Outcome

A scheduled and manually dispatchable CI job runs real PTX and the resident
CUDA numerical suite in required-hardware mode, where initialization, kernel
load, or backend-selection failures fail rather than skip.

## Implementation

1. Add a narrowly permissioned GPU workflow/job using the approved runner
   label. Pin the same Julia/Rust setup conventions as `run_tests.yml`.
2. Set `AMALTHEA_RUST_SKIP_DOWNLOAD=1` and
   `AMALTHEA_REQUIRE_CUDA_TESTS=1`; build release Rust with real PTX.
3. Run strict `cargo test` and the maintained resident CUDA Julia items,
   including Kerr, PPT, ADK, Raman, dense-output fallback, dispatch, and FFI
   lifecycle tests. Reuse the canonical test manifest/scheduler rather than a
   hand-maintained incomplete filename list.
4. Emit `nvidia-smi`, driver/toolkit, PTX marker, item assignments, and live
   heartbeats. Preserve complete console-safe logs on failure.
5. Ensure concurrency prevents overlapping jobs from exhausting one GPU, and
   add a reasonable timeout.
6. Keep workflow permissions read-only unless artifact/log upload requires a
   narrower explicit write permission.
7. Add a manifest/meta-test proving every `AMALTHEA_REQUIRE_CUDA_TESTS` item is
   included in the standing job.
8. Update backlog/GPU/testing docs only after a real workflow run is green.

## Acceptance

- A real scheduled or manual run completes on the approved GPU and contains no
  CUDA self-skips.
- Deliberately test strictness before finalizing: a controlled missing-kernel
  or disabled-dispatch reproduction must make the job fail, then revert that
  temporary change and rerun green.
- Existing CPU-only workflows remain green and unchanged in behavior.
- Record workflow/run IDs, runner model, versions, test totals, and strictness
  proof in `PORT_LOG.md`.

## Non-goals

Do not register the developer workstation without explicit authorization, do
not expose secrets, and do not make releases depend on an unproven runner.
