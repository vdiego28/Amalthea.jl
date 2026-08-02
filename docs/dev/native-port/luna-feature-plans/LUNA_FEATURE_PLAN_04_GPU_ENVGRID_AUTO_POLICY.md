# Luna feature plan 04 — Establish an evidence-based EnvGrid Kerr auto policy

Status: complete 2026-08-02. Source: review benchmark-domain finding.

## Outcome

`AMALTHEA_NATIVE_GPU=auto` has an explicit, measured policy for mode-averaged
EnvGrid Kerr. It must not reuse the RealGrid threshold by accident.

## Benchmark evidence and retained policy

The benchmark was run before changing dispatch on the local RTX 5060 Ti
(driver 610.43.02, CUDA 13.3.73, Julia 1.12.6). Each row uses the same
mode-averaged `TransModeAvg` + `EnvGrid` + scalar-density + plain `Kerr_env`
configuration, fixed `dt=0.01`, `JULIA_NUM_THREADS=1`, and the same resident
`native_step` path reached through `step!`. The CPU and CUDA steppers were
constructed separately from identical `Eω`/linop/transform data. After two
warm-up steps, three five-step batches were timed; entries are microseconds per
step and GPU speedup is CPU/GPU.

| `length(Eω)` | CPU batches (µs/step) | GPU batches (µs/step) | speedup batches |
|---:|---:|---:|---:|
| 2,048  | 217.5, 212.3, 212.6 | 821.0, 695.8, 692.0 | 0.26x, 0.31x, 0.31x |
| 4,096  | 511.7, 500.9, 497.9 | 1,089.3, 1,596.0, 1,085.1 | 0.47x, 0.31x, 0.46x |
| 8,192  | 1,262.0, 1,195.0, 1,257.0 | 2,268.1, 1,720.6, 1,641.3 | 0.56x, 0.69x, 0.77x |
| 16,384 | 2,246.1, 2,178.5, 2,181.6 | 1,250.2, 1,587.3, 1,274.1 | 1.80x, 1.37x, 1.71x |
| 32,768 | 4,508.2, 4,434.4, 4,429.7 | 1,361.6, 1,264.7, 1,113.4 | 3.31x, 3.51x, 3.98x |
| 65,536 | 41,291.6, 9,333.6, 9,774.1 | 2,308.3, 1,881.1, 1,811.4 | 17.89x, 4.96x, 5.40x |

The 16,384 row has one marginal 1.37x batch, so it is not retained as the
first stable threshold. The first size with a substantial win in every batch
and a clear margin over the crossover is 32,768. Therefore `_GPU_ENV_KERR_N_THRESHOLD`
will be retained at `32768`; the existing RealGrid Kerr threshold remains
`16384`. The 65,536 first CPU batch is a warm-up/outlier, but does not affect
the threshold decision because all three 32,768 batches already clear the
retention gate.

Benchmark command:

```text
JULIA_DEPOT_PATH=/tmp/luna-julia-depot:/home/diego/.julia \
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
julia --project /tmp/luna_envgrid_auto_bench.jl
```

The command was run outside the normal sandbox because CUDA driver discovery
inside the sandbox reports `cuInit failed: 100`.

## Implementation

1. Before changing dispatch, benchmark the existing CPU-native and CUDA
   EnvGrid Kerr step on the same RTX-class hardware across a size sweep that
   brackets 2k through at least 64k spectral points. Use fixed-step
   `native_step`, warm up both paths, and record at least three batches.
2. Keep RealGrid and EnvGrid thresholds as separate named constants and
   branches even if the measured values happen to match.
3. Apply the repository retention rule: select the first measured size with a
   substantial, stable win and margin above crossover. If no tested size
   clears the bar, implement an explicit “EnvGrid remains manual `:on`” branch
   rather than inheriting RealGrid behavior.
4. Add pure dispatch tests immediately below/at the retained threshold, or
   tests proving `:auto` remains false when no threshold is retained.
5. Add one hardware assertion that `:auto` constructs the expected backend at
   a retained size; numerical correctness stays covered by the existing
   EnvGrid CUDA tests.
6. Update the threshold docstrings, `GPU.md`, README, backlog, timing evidence,
   and support wording.

## Acceptance

The result is complete whether measurement retains or declines automatic
selection, provided the code no longer extrapolates the RealGrid threshold and
the decision is regression-tested. Run focused dispatch and EnvGrid hardware
tests, CPU/native regressions, the Rust group, and `git diff --check`.

## Non-goals

No new EnvGrid physics, plasma, Raman, or generic benchmark framework.

## Handoff

Record the full size/timing table, hardware/software versions, warmup method,
chosen policy, and exact tests in `PORT_LOG.md`.

## Completion evidence

`src/RK45.jl:1093-1142` keeps the RealGrid Kerr threshold at 16,384 and adds
the separately measured EnvGrid threshold at 32,768; the branch is explicit at
`src/RK45.jl:1210-1223`. Pure coverage is in
`test/test_native_gpu_dispatch.jl:122-192`, and the real-hardware assertion
that EnvGrid `:auto` constructs `:cuda` at 32,768 is in
`test/test_native_cuda_raman.jl:182-195`. The support wording and timing
metadata were updated in `GPU.md`, `amalthea/README.md`,
`NATIVE_SUPPORT_MATRIX.md`, `BACKLOG.md`, and `test/rust_test_timings.txt`.

The benchmark table above was recorded on the RTX 5060 Ti (driver 610.43.02,
CUDA 13.3.73, Julia 1.12.6) with two warm-up steps and three five-step fixed
`native_step` batches at 2,048–65,536 points. The first stable substantial
EnvGrid win was 32,768; the 16,384 row included one marginal 1.37x batch.
The focused no-hardware dispatch item passed 56/56. The strict CUDA suite
(`test_native_cuda.jl`, `test_native_cuda_raman.jl`, and
`test_native_gpu_dispatch.jl`) passed 259/259. The full Rust group passed
42,689 assertions with one expected sandbox CUDA-driver broken item
(`cuInit failed: 100`). `cargo build --release` and `git diff --check` passed.
