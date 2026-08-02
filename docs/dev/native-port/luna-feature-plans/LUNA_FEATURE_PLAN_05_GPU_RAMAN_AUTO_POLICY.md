# Luna feature plan 05 — Establish automatic dispatch for supported Raman

Status: complete 2026-08-02 after plans 02–04. Source: `PORT_LOG.md` next
step and GPU backlog.

## Outcome

Supported mode-averaged SDO Raman configurations receive measured,
geometry-specific `:auto` dispatch thresholds, or remain explicitly manual
with evidence that the retention bar was not met.

## Benchmark matrix

Measure these separately because their pipelines differ:

1. RealGrid `RamanPolarField`, `thg=true`, vibration-only N2;
2. RealGrid `thg=false`, including resident Hilbert c2c work;
3. EnvGrid `RamanPolarEnv`;
4. one rotational N2 case from plan 02 to detect oscillator-count scaling.

Use production-shaped fields, fixed steps, warmups, at least three batches,
and a size sweep bracketing each crossover. Benchmark CPU native against CUDA,
not Julia orchestration.

## Measured result

The benchmark used the production-shaped N₂ capillary from
`test/test_native_cuda_raman.jl` (`λ₀=800 nm`, 125 µm radius, 1 atm, 5 cm,
20 fs FWHM, 5 µJ, `dt=0.01`), the resident `RustNativeStepper`, two warm-up
steps, and three five-step batches. CPU and CUDA were each constructed with
`AMALTHEA_NATIVE_GPU=off/on`; the reported entries below are the three
CPU-time/GPU-time ratios for each batch. The retention bar is the existing
stable substantial-win gate: every retained threshold must clear 1.4× with
margin.

| pipeline | SDOs | `Nω` sweep | GPU/CPU speedup by batch (in sweep order) | decision |
|---|---:|---|---|---|
| RealGrid, `thg=true`, vibration | 1 | 1,025; 2,049; 4,097; 8,193; 16,385; 32,769 | (0.033, 0.036, 0.035); (0.963, 0.925, 0.979); (0.950, 0.979, 1.024); (0.985, 0.992, 1.097); (1.114, 1.080, 1.052); (1.104, 1.114, 1.063) | manual |
| RealGrid, `thg=false`, vibration | 1 | 1,025; 2,049; 4,097; 8,193; 16,385; 32,769 | (0.844, 0.893, 0.843); (0.902, 0.895, 0.937); (1.056, 1.015, 0.994); (1.130, 1.071, 1.036); (1.091, 1.108, 1.118); (1.130, 1.018, 1.139) | manual |
| EnvGrid, vibration | 1 | 1,024; 2,048; 4,096; 8,192; 16,384; 32,768 | (0.925, 0.613, 0.673); (0.856, 0.890, 0.808); (0.963, 0.891, 0.994); (0.912, 0.952, 0.989); (1.030, 1.081, 1.076); (1.059, 1.124, 1.141) | manual |
| RealGrid, `thg=true`, rotation+vibration | 50 | 1,025; 2,049; 4,097; 8,193; 16,385 | (0.995, 1.001, 0.994); (0.998, 1.002, 0.998); (0.999, 1.003, 1.003); (1.001, 1.001, 1.003); (1.001, 1.002, 1.001) | manual |
| RealGrid, `thg=false`, rotation+vibration | 50 | 1,025; 2,049; 4,097; 8,193; 16,385 | (0.004, 0.004, 0.004); (1.000, 0.999, 0.995); (0.997, 0.998, 0.989); (1.001, 1.000, 1.001); (0.998, 0.998, 0.989) | manual |
| EnvGrid, rotation+vibration | 50 | 1,024; 2,048; 4,096; 8,192; 16,384; 32,768 | (1.019, 1.001, 0.966); (0.970, 1.060, 0.986); (1.019, 1.033, 1.028); (1.011, 0.986, 0.988); (0.991, 1.015, 0.949); (1.009, 1.016, 1.001) | manual |

No batch in any class approached 1.4×; the largest observed speedup was
1.141×. The first long run terminated while allocating the unprinted
32,769-point 50-oscillator RealGrid `thg=true` point; the bounded follow-up
completed the other rotational class and EnvGrid through 32,768, while the
completed 16,385-point `thg=true` rotational row was 1.001–1.002×. This
termination does not support a threshold and is retained as a gotcha rather
than silently presented as a measurement.

The result is therefore a deliberate no-go for all Raman classes. Four named
policy constants are retained as `nothing` in `RK45.jl`—RealGrid THG on,
RealGrid THG off, EnvGrid, and multi-oscillator/rotational Raman—so a future
benchmark can enable one class without sharing or inheriting a generic Kerr
threshold. `:off` and explicit `:on` retain their existing meanings; only
`:auto` remains CPU-native for supported Raman.

## Implementation

1. Introduce separate named policy thresholds for the Raman pipeline classes;
   the measured no-go leaves each threshold unset rather than forcing a common
   number or inheriting a Kerr threshold.
2. Extend `_gpu_native_eligible` so Raman `:auto` uses only those thresholds.
3. Preserve `:off` and explicit `:on` behavior and all support predicates.
4. Add below/at-threshold pure dispatch tests and real-hardware backend-kind
   assertions.
5. Keep unsupported `:SiO2`, over-capacity, mixtures, and other geometries on
   CPU under every policy.
6. Update docs with complete tables and the retained/rejected decisions.

## Acceptance

Each enabled threshold must clear the established performance bar with margin,
and its configuration must pass the existing non-vacuous stage, fixed,
rejection/retry, and adaptive correctness suite. A measured “remain manual”
decision is valid for an individual class; accidental fallback to the generic
Kerr threshold is not. Run strict CUDA, focused Raman/dispatch, Rust group, and
`git diff --check`; append all measurements to `PORT_LOG.md`.

## Non-goals

No new Raman response type or geometry and no threshold guessed from FFT size.
