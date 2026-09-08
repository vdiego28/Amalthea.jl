# Native-Rust Backend Port — Testing & Equivalence

> Status: design doc for the phased port. Phases 0-8 are implemented and
> passing (see `docs/dev/native-port/PORT_LOG.md`) — the native-Rust backend
> port is complete, and the follow-on scope phases (BACKLOG.md Phases D-I,
> all ✅ 2026-07-08) extended it to essentially every configuration the
> high-level API can construct. The testing discipline below (tolerance
> tiers, fixed-step full-solve, non-vacuousness / triangulation) applied
> unchanged to all of them.
> Companion docs: [ARCHITECTURE.md](ARCHITECTURE.md), [MATH.md](MATH.md),
> [PORT_LOG.md](PORT_LOG.md).

Every phase ships **only** when an equivalence test proves the native path
reproduces the Julia path within the tolerance tier justified below. The Julia
path is the **oracle** — this is the central reason it is retained as a fallback
(ARCHITECTURE §4.3).

## 1. How to write an equivalence `@testitem`

Mirror `test/test_stepper_rust.jl`. Every Rust test is a `@testitem` tagged
`:rust` with a **skip-guard** so a fresh clone without the built `.so` skips
(not fails):

```julia
using TestItems

@testitem "Native <phase> equivalence" tags=[:rust] begin
    import Test: @test, @testset
    using Amalthea
    import Logging: with_logger, NullLogger
    import LinearAlgebra: norm

    # ── skip guard (copy verbatim) ──────────────────────────────────────────
    libname = if Sys.iswindows(); "amalthea.dll"
              elseif Sys.isapple(); "libamalthea.dylib"
              else; "libamalthea.so"; end
    libpath = joinpath(@__DIR__, "..", "amalthea", "target", "release", libname)
    if !isfile(libpath)
        @warn "Skipping: shared library not found at $libpath. " *
              "Build with `cargo build --release` in amalthea/."
        return
    end

    # ── run BOTH paths and compare the final spectrum ───────────────────────
    @testset "<geometry> equivalence" begin
        args = (radius, L, gas, pres)
        kw = (; λ0, τfwhm=τ, energy, modes=:HE11, loss=false,
                saveN=2, trange=0.5e-12, λlims=(200e-9, 4e-6))

        out_julia = withenv("AMALTHEA_USE_RUST_NATIVE" => "0") do
            with_logger(NullLogger()) do
                prop_capillary(args...; kw...)
            end
        end
        out_rust = withenv("AMALTHEA_USE_RUST_NATIVE" => "1") do
            with_logger(NullLogger()) do
                prop_capillary(args...; kw...)
            end
        end

        rel = norm(out_rust["Eω"][:,end] - out_julia["Eω"][:,end]) /
              norm(out_julia["Eω"][:,end])
        @test rel < 1e-6        # tier — see §2, justify per phase
    end
end
```

Notes:

- Toggle **both** paths explicitly via `withenv`, never by relying on the
  default or mutating global state. Native defaults on, so omitting the
  `"0"` around the oracle produces a vacuous native-vs-native comparison.
- `prop_capillary` **requires** `λlims`; it does **not** accept `stepfun`,
  `rtol`, or `atol` as kwargs (learned the hard way — see PORT_LOG seed).
- Compare the final-z spectrum `Eω[:,end]`. For stronger checks also compare an
  intermediate save and a derived observable (e.g. spectral energy).
- Prove the feature under test changes the explicit Julia oracle by more than
  the comparison tolerance. The CUDA test failed this rule: a zero-nonlinearity
  backend passed because its `1e-3` tolerance exceeded the entire `4.5e-4`
  nonlinear effect.
- Place the file in `test/` and register any new testitem in the corresponding
  `test/*_test_timings.txt` manifest used by the parallel CI scheduler.
  The serial `@run_package_tests` runner discovers testitems automatically.

### Backend observability and dispatch gates

`RustNativeStepper` exposes the internal diagnostic
`RK45._native_backend(s)`, which returns exactly `:cpu` or `:cuda` from the
backend selected at construction. Use this accessor when a test needs to
prove dispatch; `s isa RustNativeStepper` proves only that the resident API
was used, not which resident implementation owns the field. The accessor does
not query CUDA or change eligibility.

Pure eligibility, unsupported-response, capacity-boundary, and CPU-fallback
assertions must run before any CUDA-device gate. A supported configuration may
be constructed with `gpu_dispatch=:off` or a below-threshold `:auto` policy to
prove `:cpu` on every host. A `gpu_dispatch=:on` test on a CPU-only host should
assert only the pure eligibility result; it must not attempt a supported CUDA
construction. After the hardware gate succeeds, assert `:cuda` before reading
GPU stages or comparing trajectories. Z-dependent native constructors are
CPU-only and should assert `:cpu` explicitly.

## 2. Tolerance tiers (and the reason each applies)

Pick the **tightest** tier the math justifies. A test that passes at a looser
tier than its math allows is hiding a bug.

| Tier | Threshold | When it applies | Example (wired) |
|------|-----------|-----------------|-----------------|
| **bitwise** | `== 0.0` | identical IEEE-754 formula **and** identical Float64 inputs | Marcatili neff (`test_dispersion_rust.jl`) |
| **reassociation** | `~1e-13` | same formula, summation/BLAS order differs (FFTW parity, QDHT, dot products) | QDHT (`test_qdht_rust.jl`), Zeisberger (~1e-12) |
| **method/spline** | `~1e-8` | LUT/spline interpolation or a different-but-equivalent algorithm | PPT ionization (`test_ionisation_rust.jl`) |
| **FFT-method + floor** | `~1e-6` | FFT method differences **and** the run-to-run nondeterminism floor (§3) | RK45 stepper (`test_stepper_rust.jl`) |
| **deliberate divergence** | `~1e-4` (config-dependent) | Rust computes a *more accurate* value than Julia's own oracle on purpose, and the resulting small systematic (non-random) offset accumulates coherently over propagation length/bandwidth | Phase 7 β1(z) (`test_native_zdep_linop.jl`, see `BETA1_ANALYTIC.md`) |
| **different backend** | `~1e-8`-`1e-6` (config-dependent) | Two configs in the *same* comparison legitimately execute on different steppers (one `NativeIneligible`, one not) — as of Phase 8 this is possible for the first time, since native is the default rather than opt-in | `test_mixtures.jl` (mixture vs single-gas), `test_tapers.jl` (Function-radius vs constant-radius) |

This last tier is different in kind from the others: it is not "we haven't
converged Rust to match Julia yet," it is "Rust and Julia will never
converge further, because Rust is right and Julia's own value has a real,
repeatable, tiny error against the true derivative." Before reaching for
this tier, do the two checks that prove it isn't secretly the other three
tiers in disguise: (1) a `kerr=false`/linear-only control run should show
the *same* magnitude as the full nonlinear run (proves it's not a bug in
some other piece of the RHS), and (2) an independent BigFloat/higher-precision
ground truth should confirm Rust's value, not just Julia's, is the accurate
one. See `BETA1_ANALYTIC.md` §4 for a worked example of both checks.

**Per-phase target.** Because the native port binds the **same FFTW** (so
transforms are bit-parity) and copies Julia's coefficient arrays in, most phases
should land in the **reassociation tier (~1e-13)** for a single deterministic
step. Whole-`solve` comparisons that accumulate many adaptive steps fall to the
**~1e-6 floor tier** (§3). State both numbers in the PORT_LOG entry: the
single-step tightness (proves the math) and the full-run tolerance (proves the
trajectory).

## 3. The run-to-run nondeterminism floor (critical)

**The Julia stepper alone varies ~2e-8 run-to-run** for a typical capillary
setup, even single-threaded, because FFTW's summation order is not reproducible
across invocations. This is a hard floor: **no equivalence threshold for a full
`solve` can sit below ~2e-8**, regardless of how perfect the native code is.
This is why `test_stepper_rust.jl` uses `1e-6` (comfortably above the floor),
not `1e-10` (numerically impossible).

Two consequences for the port:
1. **Full-run equivalence tests use the ~1e-6 tier.** Do not tighten them below
   the floor; a "failing" test there is measuring FFTW noise, not a port bug.
2. **For tight per-step checks**, compare a **single deterministic RHS/step
   evaluation** (same input field, one `fbar!`/one `step!`) rather than a full
   adaptive run. A single evaluation has no accumulated step-sequence divergence
   and should hit ~1e-13. This is the test that actually proves the math is right.

### Tighter local checks
For local debugging, pin FFTW to one thread and `:estimate` planning to reduce
(not eliminate) variance:

```julia
import FFTW
FFTW.set_num_threads(1)
# tests already use :estimate planning (see CLAUDE.md)
```

The step controller also matters: once two paths' `err` estimates differ by even
1 ULP near an accept/reject boundary, they take different step sequences and
diverge within the tolerance band (MATH §2.2). Single-step comparison sidesteps
this entirely.

**Phase 2 postmortem — this bit us for real.** Phase 2a's full-solve test
initially failed at 9.64e-5 (vs the 1e-6 tier) despite the single-step test
passing at <1e-13. Root cause: the embedded RK45 error estimate is a
near-total cancellation (`b5-b4=0` in the Butcher tableau) — early in a weakly
nonlinear propagation, `err` sits at the ~1e-15 floor, where FP-summation-order
noise between Julia and Rust shows up as a ~20% *relative* disagreement in
`err`. The PI controller amplifies that into a different `dtn` choice, and the
two adaptive integrators diverge onto different step sequences that land at
different z — so the "full-solve" comparison was comparing the field at two
different points in space, not detecting a state-accumulation bug. Confirmed
by forcing `max_dt=min_dt=dt` on both steppers: agreement collapsed to
~1e-17 all the way to `flength`.

**Recommended full-run test shape (adopted in `test_native_phase{1,2}.jl`):**
construct both steppers with `max_dt=dt, min_dt=dt` for the full-solve
testset specifically (leave the single-step testset as-is). This forces an
identical step-size sequence — sidestepping the adaptive-path-divergence
confound entirely — while still exercising genuine multi-step state
accumulation, which is what the full-run tier is supposed to test. Apply this
to every future phase's full-solve test, not just the ones where it happens to
bite (Phase 1/2b's `err` values were "healthy" — far from the cancellation
floor — so their raw-`yn` full-solve tests happened to pass anyway; that's
coincidence of regime, not immunity to the same mechanism).

## 4. Per-phase acceptance criteria

| Phase | Status | What to test | Test file | Single-step tier | Full-run tier |
|-------|--------|--------------|-----------|------------------|---------------|
| 0 | ✅ done | set/get round-trip bit-exact; no-op RHS reproduces Julia stepper | `test/test_native_phase0.jl` | bitwise (round-trip) | ~1e-6 |
| 1 | ✅ done | mode-avg + Kerr `prop_capillary(:HE11)`, RealGrid | `test/test_native_phase1.jl` | <1e-13 (achieved) | 2.75e-16 (fixed dt) |
| 2 | ✅ done | EnvGrid Kerr (2a) + plasma/RealGrid (2b) | `test/test_native_phase2.jl` | <1e-13 (achieved) | 3.19e-17 / 2.73e-16 (fixed dt) |
| 3 | ✅ done | radial + resident QDHT (RealGrid + scalar Kerr) | `test/test_native_radial.jl` | 1.1e-17 (achieved) | 1.3e-16 (fixed dt) |
| 4 | ✅ done | Raman (carrier SDO, thg=true, all-SDO eligibility) | `test/test_native_raman.jl` | 0.0 (see note) | 4.2e-8 (achieved) |
| 5 | ✅ done | modal + overlap cubature (`HE,n=1`, `full=false`, Kerr-only) | `test/test_native_modal.jl` | 1.4e-19 (achieved; ~1e-10 tier) | 4.0e-16 (achieved; fixed dt) |
| 6 | ✅ done | free-space 3-D FFT (RealGrid, const_norm_free, Kerr-only) | `test/test_native_free.jl` | 7.05e-18 (achieved) | 5.01e-17 (achieved; fixed dt) |
| 7 | ✅ done | z-dependent linop (mode-avg, graded-core, two-point pressure gradient) | `test/test_native_zdep_linop.jl` | <1e-9 (β1 vs BigFloat truth, achieved); ~1e-12 (`dtn`/`err`, achieved) | <1e-3 tier (measured ~2.7e-7 post-Phase-8-precision-fix, deliberate-divergence, see §2) |
| 8 | ✅ done | default-flip: existing suite green with native as default | `test/test_native_phase8.jl` + full suite | — | 46590 pass / 0 fail / 0 error / 12 broken (pre-existing), 46602 total |
| S5.3 | ✅ done | order-5 dense output + deferred FSAL carry on Julia/CPU-native; measured CUDA order-4 fallback | `test/test_native_dense_order5.jl` | CPU order → 5; CUDA order → 4; native-vs-Julia ~1e-17 | full gate green |

Phase 5's single-step tier is documented looser (~1e-10) than the FFTW-only
phases, not because cubature node placement is algorithm-dependent — it binds
the *same* `libcubature` C library Julia calls (`Cubature.jl` is a thin
`ccall` wrapper, confirmed via `Cubature.Cubature_jll.libcubature` and
`nm -D libcubature.so`), so node placement is bit-identical by construction.
The tier is looser because the per-node mode-field synthesis needs
`besselj(0,·)`, and the existing Rust `j0` (`diffraction.rs`) agrees with
`SpecialFunctions.besselj` to ~1e-15 *absolute*, not bitwise — verified
standalone before implementing, not assumed. In practice the achieved
number (1.4e-19) came in far under even the tight tier at this problem size;
the ~1e-10 ceiling is the honest one to keep asserting.

Phase 4's single-step result (`0.0`, exact) is **not a vacuous test** — verified
via a three-cell diagnostic (PORT_LOG 2026-07-01): Raman's raw per-step RHS
contribution is ~2e-16 relative to Kerr's at these test parameters (right at
the FP floor for a single 1cm z-step — physically expected, since
Raman-induced spectral changes are cumulative over propagation, unlike Kerr
self-phase-modulation). The full-solve testset is the meaningful gate and is
self-validating: it independently asserts Raman changes the Julia oracle's
result (1.1e-4, far above any noise floor) *before* asserting Rust matches
Julia on that changed result (4.2e-8). Any future Raman-adjacent full-solve
test should include the same "does this feature actually change the
reference result" sanity assertion — a passing comparison between two paths
that both silently exclude the feature under test proves nothing.

Phase 7's full-run tier (~1e-3, measured ~2.7e-7 post-Phase-8-precision-fix,
see `BETA1_ANALYTIC.md` §6) is the widest of any phase, and is the **only**
phase where the widening is deliberate rather
than a limitation to work around — see the "deliberate divergence" tier in
§2 and `BETA1_ANALYTIC.md`. The single-step tier is still tight (β1 itself
is verified to <1e-9 against a BigFloat ground truth, and `dtn`/`err`
agreement is ~1e-12); it is specifically the *coherent accumulation* of
β1's tiny systematic offset from Julia's own value, over a broad-bandwidth,
multi-step propagation, that produces the wider full-run number. A
narrower-bandwidth or shorter-fibre config would show a smaller number
without any code change.

Phase 8's gate is the widest in *scope* (the entire suite, not a phase-specific
subset) but not in *tolerance* — most of its failures turned out to be real
bugs (see PORT_LOG), fixed properly rather than tolerance-widened. Only two
tests legitimately needed a tolerance change, and for a reason specific to
Phase 8: a config comparison where the two sides now execute on genuinely
different backends (native vs `NativeIneligible`-fallback Julia) for the
first time — see the "different backend" tier above. Before reaching for
that tier on any future test failure, check first whether both sides of the
comparison are actually eligible for the same backend; if they are, a
failure is a real bug, not a tolerance problem.

**GPU-specific acceptance rule.** A self-skipping CUDA test passing on a
CPU-only CI runner proves only that the skip guard works. Every GPU correctness
slice requires all of:

1. a CPU-native and Julia-oracle control showing the intended nonlinear effect;
2. a GPU stage-derivative check whose scale is comparable to CPU native, not
   approximately zero;
3. a full-solve tolerance tighter than that measured nonlinear effect;
4. a recorded run on real CUDA hardware; and
5. eventually, a standing CUDA CI job so the path cannot rot silently.

Adaptive-controller changes additionally require a deliberately rejected
trial whose field remains unchanged, a controller-selected retry, and an
adaptive trajectory against CPU native/Julia. The 2026-07-27
`test_native_cuda.jl` extension is the reference: both Kerr and Kerr+PPT reject
and retry, with full adaptive trajectory differences of `5.42e-15` and
`2.24e-15` on the RTX 5060 Ti.

### CPU optimization/concurrency gate (2026-08-24)

**Embedded fourth-order mode (PLANS §18).** Test both propagation modes against
an independent nonlinear analytic solution and test each solver against a fresh
restart from its own accepted endpoint. False-mode interpolation must converge
to y4 from inside the interval and retain fourth-order dense convergence.
Exercise nonzero linear operators, rejected attempts/retries, and repeated
dense queries before the next step. A Julia/native parity check alone cannot
detect shared invalid k7 reuse. The y4/y5 gap must exceed the continuity test's
tolerance. Default fifth-order FSAL and dense-order gates remain required.

Allocation removal is accepted only with the existing rejected-step,
`locextrap=false`, dense-output, and callback/window lifecycle suites green.
The matched medium audit additionally records Julia-visible allocations: the
optimized native fixed-step cells are 96 bytes each, down from
16,616–787,000 bytes; complete adaptive solves are 480–1,088 bytes, down from
49,608–18,885,744 bytes.

QDHT policy coverage must exercise real and complex forward/inverse transforms,
round trips, invalid FFI modes, and a resident radial trajectory under
`off`/`auto`/`on` plus deterministic override. On the current workload,
`auto == on`, `off == deterministic`, and the two kernels agree at
`rtol=1e-12` while differing bitwise as expected from summation order.

Raman SIMD coverage compares the scalar oracle with the dispatched kernel at
1, 2, 3, 4, 5, 49, 50, and 65 oscillators, multiple time lengths, and
adversarial signs at `2e-13`. AArch64 code must cross-compile cleanly and the
Apple quick test supplies the runtime NEON gate on Apple hardware.

Julia modal batching requires exact sequential/four-thread callback arrays
under `FFTW_ESTIMATE`, forced-GC repeats, and proof that stateful closures stay
sequential. Queue tests require simultaneous scans, exact-once result files,
failure marking, stable queue removal, one-thread worker topology, concurrent
resident handles, and no leaked `Distributed` workers.

## 5. Commands

Use the recorded local gate for work-item validation:

```bash
# CPU build + Cargo tests + Rust/native Julia group
python3 test/validate.py

# Include affected groups, or run all eight
python3 test/validate.py --groups rust physics --max-workers 4
python3 test/validate.py --all --max-workers 4

# Required CUDA build/tests: run on the host, outside the agent sandbox
PATH=/usr/local/cuda-13.3/bin:$PATH python3 test/validate.py --cuda
```

The wrapper builds into `amalthea/target/release`, disables prebuilt downloads,
and verifies Julia loads both the package and RK45 library from this checkout.
CPU mode sets `AMALTHEA_CUDA_BUILD=off` and strict-CUDA tests off. `--cuda`
sets the build to `required` and `AMALTHEA_REQUIRE_CUDA_TESTS=1`, and always
includes the Rust group: missing real PTX or GPU dispatch fails validation.
It does not globally force GPU dispatch for CPU control tests. CUDA commands
still require host execution; the wrapper does not request sandbox escalation.
Explicit RUSTFLAGS are preserved; unset means empty for portability. Encoded
Cargo flags and a cross-compilation target override are removed so this gate
builds a host-loadable library with the recorded RUSTFLAGS.

Every run creates a unique directory under `.rust_test_logs/validation/`
(or under the parent supplied with `--log-dir`). `summary.json` records the
revision, dirty status, selected groups, platform, effective relevant environment,
library SHA-256, commands, durations, exit codes, and final status. Version
logs, `inventory.json`, command logs, and `workers/` retain complete output.
Each worker JSON sidecar records its exact Julia command, assigned items,
thread overrides, duration, and exit code.
Build or preflight failure stops dependent tests; Cargo test failure still
allows Julia tests to collect evidence. A failed or interrupted run is not a
passing gate. Logs are ignored local artifacts: retain/share the evidence
bundle when handing work to another machine, and record concise measured
results in PORT_LOG. Exit codes do not establish numerical tolerances or
replace feature-sensitivity and actual-backend assertions.

To check the validation tooling itself without Julia or CUDA, run
`python3 -m unittest discover -s test -p 'test_*py'`. These tests also run in
the existing Python CI job.

Lower-level commands remain useful for focused diagnostics (they do not create
the wrapper's evidence bundle):

```bash
AMALTHEA_CUDA_BUILD=off cargo build --release --manifest-path amalthea/Cargo.toml
LUNA_TEST_GROUP=rust julia --project test/runtests.jl
python3 test/parallel_group_tests.py --group rust --max-workers 2
python3 test/run_full_gate.py
python3 test/parallel_group_tests.py --group rust --max-workers 10 --update-timings-only
julia --project -e 'using Pkg; Pkg.test("Amalthea")'
```

Main-gate `LUNA_TEST_GROUP` values: `physics`, `rust`, `sim-interface`,
`sim-multimode`, `sim-propagation`, `io`, `fields`, `examples`, `All`
(default). `test/test_groups.txt` is the canonical maintained-group list.
`test/run_full_gate.py` and GitHub Actions both use the same item-level LPT
scheduler; the serial `runtests.jl` command remains the simplest oracle when
checking aggregate assertion counts.

GitHub Actions adds the scheduler's `--ci` option, which preserves the former
`julia-actions/julia-runtest` bounds-check, deprecation-warning,
compiled-module, inlining, and user-code-coverage settings. Each worker writes
a distinct LCOV trace beside its log. Local timing and full-gate runs omit
`--ci` to retain their existing lower-overhead command; add it locally only
when reproducing the hosted Julia invocation:

```bash
python3 test/parallel_group_tests.py --group physics --max-workers 2 --ci
```

## 6. Definition of done for a native work item

### Internal Python foundation gate

The separate `python-native/` package tests grids, portable FFTs, and the
low-level callback solver described below. High-level optical propagation APIs
remain unfinished. Build/install it in an isolated
Python environment with `RUSTFLAGS="" maturin build --release` from that
directory, then install the resulting wheel. From the repository root export
independent setup fixtures:

```bash
julia --startup-file=no --project python-native/tools/export_grid_oracle.jl /tmp/amalthea-grid-oracle
AMALTHEA_GRID_ORACLE=/tmp/amalthea-grid-oracle python -m pytest python-native/tests -q -s
```

Use the environment's Python executable for pytest. No environment variable
means the Julia-fixture test skips; an acceptance run must supply the exported
fixtures. It asserts all axis/window max-relative errors below `1e-13`, exact
selection masks, and matching shapes. Portable FFT checks cover odd/even
lengths, direction/normalization, DC/Nyquist, input ownership, repeated use,
Hilbert phase, and causal convolution. The documented final propagation gates
still apply when the driver/physics are implemented.

Also build an sdist (`maturin sdist`), extract outside the checkout, rebuild
with `CARGO_NET_OFFLINE=true` after development dependencies are cached, and
run that extracted suite against its installed wheel. Record the actual wheel
tag: a locally produced `manylinux_2_34` wheel is not evidence for the planned
`manylinux_2_28` release baseline. These are internal artifact checks.

### Native work-item acceptance

For the standalone callback driver, additionally export and supply its solver
oracle directory:

```bash
julia --startup-file=no --project python-native/tools/export_solver_oracle.jl /tmp/amalthea-solver-oracle
AMALTHEA_GRID_ORACLE=/tmp/amalthea-grid-oracle AMALTHEA_SOLVER_ORACLE=/tmp/amalthea-solver-oracle python -m pytest python-native/tests -q -s
```

These development-only exporters use the Julia fallback, not the new Python
code. Driver acceptance covers first-interval dense equivalence (`1e-13`),
fixed/adaptive trajectories (respectively `1e-13`/`1e-6`), independent nonlinear
analytic accuracy, order-4/order-5 dense convergence, rejected trials, restart,
nonuniform nonlinear filters, exception identity, and field-array ownership.
Exact saved-position and filtered-output checks guard against NumPy/Julia
range rounding placing nominal boundary samples on opposite sides of a filter.
An exported-oracle skip is not a passing acceptance gate. Rebuild/install the
sdist outside the checkout and run its nested `python-native/tests` suite and
`python-native/examples/solver_analytic.py` against the installed wheel.

A native work item is complete when **all** hold:

1. The native path is selected by its toggle and runs the full geometry with no
   Julia callback in the hot loop (verify: no `@cfunction` round-trip for that path).
2. A single-step equivalence test passes at the tier in §4.
3. A full-`solve` equivalence test passes at the ~1e-6 floor tier.
4. The pre-existing Julia-path tests still pass (no regression).
5. A `PORT_LOG.md` entry records both achieved tolerances, the FFI symbols added,
   and any gotchas.
6. The test is non-vacuous: the feature changes the oracle by more than the
   asserted equivalence tolerance, or another direct assertion proves the
   relevant intermediate quantity is present.
