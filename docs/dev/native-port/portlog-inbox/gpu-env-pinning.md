# BACKLOG item 9 — `AMALTHEA_NATIVE_GPU=on` rerouted CPU-native tests onto the GPU

**Date:** 2026-07-26. **Status:** fixed and verified on real hardware
(RTX 5060 Ti, driver 610.43.02).

## The defect

`RK45._gpu_native_eligible(f!, linop, n)` decides per *construction* whether a
`RustNativeStepper` builds the CPU-resident or the GPU-resident backend. Under
`AMALTHEA_NATIVE_GPU=on` it returns `true` for **any** config
`_gpu_kernel_supports` accepts — mode-averaged `TransModeAvg`, `RealGrid`,
constant `Array{ComplexF64}` linop, one plain Kerr response, at most one
`PlasmaCumtrapz` with a PPT rate, no shot noise.

That is exactly the shape of the CPU-native equivalence tests. They construct a
`RustNativeStepper`, compare it against the Julia `PreconStepper` oracle at
CPU-native tolerance tiers (~1e-13 single-step, ~1e-6/~1e-10 full-solve), and
never state which backend they mean — so running the whole `rust` group under
`AMALTHEA_USE_RUST_CUDA_NATIVE=1 AMALTHEA_NATIVE_GPU=on` silently measured the
GPU backend against CPU tolerances. 18 failures, none of them a GPU
correctness problem. The practical cost: "run the suite on the GPU" was
unusable as a verification technique — which is precisely how a
zero-nonlinearity GPU backend survived two weeks (item 1).

Two subtler instances the backlog's failure count did *not* capture, because
they passed:

- `test_native_phase8.jl` passed by tolerance luck. Both `out_native_explicit`
  and `out_default` rerouted to the GPU, so their bit-identity assertion still
  held, and the native-vs-Julia comparison came out at 1.7e-9 against an
  expected ~1.6e-11 — under the loose 1e-8 bound, so green.
- `test_native_dense_order5.jl`'s GPU testitem built its `s_cpu` reference
  without pinning, so under `on` the "GPU `apply_prop` matches the CPU backend"
  check compared GPU against GPU — a tautology.

## The fix

Wrap each vulnerable `RustNativeStepper` construction in
`withenv("AMALTHEA_NATIVE_GPU" => "off")` and assert the decision immediately
after with a counted
`@test !RK45._gpu_native_eligible(transform, linop, length(Eω))`, so a future
dispatch-policy change fails loudly instead of rerouting silently. `@test`
rather than `@assert` deliberately: an `@assert` aborts the testitem as an
uncaught Error and never enters the Pass/Total tally, and this check is a
contract worth counting.

Five files: `test_native_phase1.jl`, `test_native_phase2.jl`,
`test_native_phase8.jl`, `test_native_fftw_wisdom.jl`,
`test_native_dense_order5.jl`.

`test_native_cuda.jl` and `test_native_gpu_dispatch.jl` already set their
backend explicitly and are untouched — they are supposed to run on the GPU.

Calling `_gpu_native_eligible` from a test is safe on a CPU-only machine: it is
pure config/type inspection and returns `false` at its first line when
`AMALTHEA_USE_RUST_CUDA_NATIVE` is unset. It never initialises a CUDA context,
so the new guard cannot make the default path depend on CUDA being installed.

**Why only five files, when ~30 test files construct a `RustNativeStepper`:**
the rest are GPU-*ineligible* configs — radial/modal/free-space geometry,
`EnvGrid`, Raman, mixtures, ADK ionisation, shot noise, z-dependent linops —
so `_gpu_kernel_supports` rejects them and `on` cannot reroute them. That
reasoning is an argument, not a proof; the whole-group GPU run below is the
proof.

## Verification (measured, not asserted)

Both runs are the full `rust` group from a clean worktree, GPU present:

| Run | Result |
|---|---|
| `AMALTHEA_USE_RUST_CUDA_NATIVE=1 AMALTHEA_NATIVE_GPU=on LUNA_TEST_GROUP=rust` | **42269 pass, 1 broken, 42270 total, 0 failures**, exit 0, 12m27.7s |
| default env, `LUNA_TEST_GROUP=rust` | **42269 pass, 1 broken, 42270 total, 0 failures**, exit 0, 9m55.3s |

Before the fix, the first of those produced 18 failures —
`test_native_phase1.jl` 6, `test_native_dense_order5.jl` 8,
`test_native_fftw_wisdom.jl` 3, `test_native_phase2.jl` 1. **That before-state
split, and the phase8 "1.7e-9 vs ~1.6e-11" figure above, are agent-measured and
were not re-run by the lead session** — only the post-fix state was verified on
hardware here. The 18 total matches what item 9 recorded independently on
2026-07-25. Nothing in the fix's validity rests on those numbers: the claim
that matters is 0 failures under the GPU env, and that one is lead-measured.

The two runs have **identical totals**, which is the check that matters against
the obvious way to fake this fix: pinning `off` did not skip, disable or
short-circuit any test, and the GPU tests still ran on the GPU under the GPU
env — the log shows the GPU-resident stepper selected, "Successfully verified
dynamic GPU CUDA dispatch from Julia FFI", stage derivatives
`CPU=1230.5720437772707` vs `GPU=1230.5720437772711`, GPU full-solve
`rel_solve = 3.5e-16`, and Kerr+plasma `1.8e-16`.

The single `Broken` is the pre-existing documented one, unchanged between runs.

## What this does *not* do

- It does not add GPU CI (item 2). Standing GPU coverage is still a manual,
  recorded hardware run — this change only makes that run trustworthy.
- It does not touch the GPU `err` weak-norm placeholder (item 8).
- The guard is per-construction, by imitation: a *new* CPU-native equivalence
  test written without the `withenv` will be silently reroutable again. A
  structural fix (e.g. a helper that constructs CPU-native steppers) was
  deliberately not built — it would be a framework for five call sites.

## Provenance

Test edits by a Sonnet agent in an isolated worktree (commits `299e1aa`,
`580e350`), reviewed before merge. It stopped before verifying or documenting;
the two whole-group runs above, this record, and the BACKLOG update were done
by the lead session.
