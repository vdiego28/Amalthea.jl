# Luna feature-plan pack

These files turn the live GPU backlog and the 2026-08-02 review findings into
bounded implementation runs. Give **one file at a time** to the Luna model.
Each plan is intended to finish one independently reviewable feature, its
tests, documentation, and mandatory `PORT_LOG.md` entry in the same run.

Suggested instruction:

> Implement `docs/dev/native-port/luna-feature-plans/<PLAN>.md`. Read
> `AGENTS.md` and this plan-pack index first, verify every listed dependency is
> already present, then complete only that plan's implementation, tests,
> documentation, and `PORT_LOG.md` entry. You are responsible for authoring
> every required theory argument, derivation, mathematical contract, and hard
> empirical conclusion; leave none of that work for the reviewing model. Do
> not commit or push.

The plans do not authorize commits, pushes, workflow registration, release
publication, or destructive cleanup. `AGENTS.md` remains authoritative.

## Luna authorship, larger-model verification

The Luna agent implementing a plan must author the substantive reasoning that
makes its implementation reviewable. This includes any new or adapted theory,
equations and derivations, normalization/scaling arguments, invariants, error
and tolerance arguments, kernel-capacity or resource arguments, benchmark
design, raw measurements, and conclusions drawn from those measurements.

Write that material into the relevant design/support document **before source
code**, as required by `AGENTS.md`, and summarize the exact result in the
plan's completion section and `PORT_LOG.md`. Identify assumptions, dimensions,
units, reference equations or oracle code paths, numerical tolerances, hardware
and software conditions, and any unresolved uncertainty. Existing repository
math may be cited instead of re-derived, but the Luna agent must explain why it
applies to the new feature and record any changed indexing, layout, scaling, or
precision contract.

The later larger-model pass is a verifier, not the original author of missing
reasoning. It should independently check derivations against the Julia oracle
and repository math, inspect code-to-equation correspondence, challenge
assumptions and non-vacuity, reproduce important tests or benchmarks, and
either accept the result or return concrete corrections. If a Luna run leaves
theory, math, or a difficult empirical claim undocumented, that run is
incomplete and should be sent back for completion rather than having the
verifier silently supply the missing result.

## Local CUDA execution environment

This computer has the real manual-verification target used by the existing
GPU work: an NVIDIA RTX 5060 Ti with the CUDA 13.3 toolchain (the recorded
driver baseline is 610.43.02). CUDA access is environment-dependent and may
fail inside the normal filesystem/process sandbox even though the hardware is
available. For every plan that requires real CUDA:

1. try the documented strict command normally;
2. if it fails with driver discovery, device access, permission, or another
   sandbox-shaped CUDA error, rerun that same bounded command with the
   approved elevated/out-of-sandbox execution mechanism;
3. keep `AMALTHEA_REQUIRE_CUDA_TESTS=1` set so initialization, PTX, kernel
   loading, and dispatch failures fail instead of becoming skips; and
4. do not report “CUDA unavailable” until the elevated strict run has also
   failed and its concrete error has been recorded.

The already-approved command shapes include strict `cargo test`, strict
focused CUDA Julia items, and the full Rust group. Use the narrowest approved
command that verifies the current plan. Manual access to this GPU does **not**
close plan 06: standing CI still requires an explicitly authorized and
registered runner or hosted GPU service.

## What kind of errors the review found

| Review finding | Error class | Risk | Owning plan |
|---|---|---|---|
| EnvGrid + plasma passes `_gpu_kernel_supports`, but the CUDA EnvGrid RHS never applies plasma | capability-predicate over-acceptance / silently omitted physics | correctness | 01 |
| Julia accepts 49-50 oscillator N2 rotational Raman while CUDA rejects more than 32 | declared-capability / implementation-capacity mismatch | forced-GPU fallback and false support claim | 02 |
| Fallback tests can only see `RustNativeStepper`, not whether its resident backend is CPU or CUDA | observability and test-oracle gap | false-positive dispatch tests | 03 |
| EnvGrid Kerr inherits a threshold measured only for RealGrid | benchmark-domain extrapolation | performance-policy regression | 04 |
| Raman is correct under `:on` but has no measured `:auto` policy | missing retained benchmark | feature remains manual-only | 05 |
| Pure eligibility/fallback checks are inside a successful-CUDA branch | test-topology gap | CPU-only CI does not protect dispatch | 03 |
| GPU/support docs overclaim EnvGrid plasma and rotational scope | derived-documentation drift | misleading user/agent handoff | 01 and 02 |

## Backlog coverage and execution order

The live backlog has two explicit themes: standing required-CUDA CI and
broader GPU physics/geometries. The review adds the correctness and dispatch
repairs at the head of this queue.

| Order | File | Atomic outcome | Depends on |
|---:|---|---|---|
| 1 | `LUNA_FEATURE_PLAN_01_GPU_ENVGRID_PLASMA_CONTRACT.md` | EnvGrid plasma cannot be silently selected | current tree |
| 2 | `LUNA_FEATURE_PLAN_02_GPU_ROTATIONAL_RAMAN_CAPACITY.md` | N2 rotational Raman really runs on CUDA | 01 recommended |
| 3 | `LUNA_FEATURE_PLAN_03_GPU_BACKEND_OBSERVABILITY.md` | tests can prove CPU-vs-CUDA selection without hardware | current tree |
| 4 | `LUNA_FEATURE_PLAN_04_GPU_ENVGRID_AUTO_POLICY.md` | measured EnvGrid Kerr `:auto` policy | 03 |
| 5 | `LUNA_FEATURE_PLAN_05_GPU_RAMAN_AUTO_POLICY.md` | measured Raman `:auto` policy | 02, 03 |
| 6 | `LUNA_FEATURE_PLAN_06_STANDING_REQUIRED_CUDA_CI.md` | strict CUDA tests run continuously | approved runner |
| 7 | `LUNA_FEATURE_PLAN_07_GPU_MODEAVG_SIO2_RAMAN.md` | mode-averaged EnvGrid `:SiO2` Raman (complete 2026-08-02) | 03, preferably 06 |
| 8 | `LUNA_FEATURE_PLAN_08_GPU_RADIAL_REAL_KERR.md` | radial RealGrid Kerr foundation (complete 2026-08-02) | 03, preferably 06 |
| 9 | `LUNA_FEATURE_PLAN_09_GPU_RADIAL_ENV_KERR.md` | radial EnvGrid Kerr (complete 2026-08-02) | 08 |
| 10 | `LUNA_FEATURE_PLAN_10_GPU_RADIAL_PPT.md` | radial RealGrid PPT plasma (complete 2026-08-02) | 08 |
| 11 | `LUNA_FEATURE_PLAN_11_GPU_RADIAL_ADK.md` | radial RealGrid thresholded ADK (complete 2026-08-02) | 10 |
| 12 | `LUNA_FEATURE_PLAN_12_GPU_RADIAL_REAL_RAMAN.md` | radial RealGrid SDO Raman | 02, 08 |
| 13 | `LUNA_FEATURE_PLAN_13_GPU_RADIAL_ENV_RAMAN.md` | radial EnvGrid SDO Raman | 09, 12 |
| 14 | `LUNA_FEATURE_PLAN_14_GPU_MODAL_REAL_KERR.md` | modal RealGrid Kerr | 03, preferably 06 |
| 15 | `LUNA_FEATURE_PLAN_15_GPU_MODAL_ENV_KERR.md` | modal EnvGrid Kerr | 14 |
| 16 | `LUNA_FEATURE_PLAN_16_GPU_MODAL_REAL_RAMAN.md` | modal RealGrid scalar SDO Raman | 02, 14 |
| 17 | `LUNA_FEATURE_PLAN_17_GPU_FREE_REAL_KERR.md` | free-space RealGrid Kerr | 03, preferably 06 |
| 18 | `LUNA_FEATURE_PLAN_18_GPU_FREE_ENV_KERR.md` | free-space EnvGrid Kerr | 17 |
| 19 | `LUNA_FEATURE_PLAN_19_GPU_FREE_PPT.md` | free-space RealGrid PPT plasma | 17 |
| 20 | `LUNA_FEATURE_PLAN_20_GPU_FREE_ADK.md` | free-space RealGrid thresholded ADK | 19 |
| 21 | `LUNA_FEATURE_PLAN_21_GPU_FREE_REAL_RAMAN.md` | free-space RealGrid SDO Raman | 02, 17 |

Plans 08-21 deliberately leave each new geometry on
`AMALTHEA_NATIVE_GPU=on`. Automatic dispatch is a separate measured feature,
to be planned only after a production-shaped implementation benchmark exists.

## Deliberately not converted into implementation plans

- `StepIndexMode` multi-mode remains explicitly parked for lack of a consumer.
- The cold-start standalone CLI, SoA rewrite, direct PPT series, direct error
  coefficients, and short-kernel Raman were measured or studied and rejected.
- `UPSTREAM_TRIAGE.md` candidates are proposals, not live backlog work.
- Arbitrary Julia closures cannot become resident GPU kernels without
  reintroducing the callback boundary.
- GPU shot noise, mixtures, and z-dependent geometries are still exclusions,
  but the current resume queue does not prioritize them ahead of the concrete
  geometry/physics parity slices above.

## Definition of a successful plan run

Every implementation run must:

1. read `AGENTS.md`, `ARCHITECTURE.md`, `MATH.md`, `TESTING.md`, the relevant
   GPU/support documents, and the latest `PORT_LOG.md` entries before editing;
2. author every required theory, derivation, mathematical contract, tolerance
   argument, and hard empirical result before asking the larger model to review;
3. preserve the Julia and CPU-native paths as explicit oracles/fallbacks;
4. prove the feature changes the oracle by more than its asserted tolerance;
5. include a direct/single-stage comparison and a fixed-step trajectory;
6. include rejected-step/adaptive coverage when RK/controller behavior is
   touched;
7. run strict real-CUDA tests for CUDA code and fail rather than skip under
   `AMALTHEA_REQUIRE_CUDA_TESTS=1`;
8. run affected CPU/native regressions and `git diff --check`;
9. update `GPU.md`, `NATIVE_SUPPORT_MATRIX.md`, `BACKLOG.md`, and the plan
   status where the support claim changes; and
10. append the mandatory `PORT_LOG.md` entry with exact commands and measured
   tolerances.
